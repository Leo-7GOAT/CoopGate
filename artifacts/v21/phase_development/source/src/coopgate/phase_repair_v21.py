"""Full joint repair with an additional sub-period arrival-order condition.

All original cost/time conditions and candidate rankings remain exact. A fresh
per-call command cache is shared by complete forecasts; there is no short horizon,
future-noise input, approximate value or deadline truncation.
"""
import time

import numpy as np
from numba import njit

from .arrival_phase_v20 import crossing_time
from .cached_portfolio_v19 import cached_command, cached_disturbed_suffix, cached_nominal_suffix, new_command_cache
from .fast_repair_v20 import FastRepairPortfolioGate
from .fast_v8 import transition_values
from .repair_portfolio_v20 import expanded_compositions
from .robust_v9 import robust_certificate, validate
from .temporal_portfolio_v20 import BANK_STREAM


@njit(cache=True, fastmath=False)
def cached_arrival_phase(state, previous, step, tape, packed, first_command, use_first, cache):
    x, prev = state.copy(), previous.copy()
    arrivals = np.zeros(3)
    for i in range(3):
        if x[i, 0] < packed[11]:
            arrivals[i] = np.nan
    substeps = round(packed[1] / packed[0])
    transitions, internal = 0, 0
    for k in range(step, round(packed[2] / packed[1])):
        if np.all(x[:, 0] >= packed[11]):
            break
        if use_first and k == step:
            command = first_command.copy()
        else:
            command, work = cached_command(x, prev, k, packed, cache)
            internal += work
        for sub in range(substeps):
            positions = x[:, 0].copy()
            for i in range(3):
                acceleration = command[i] + tape[k * substeps + sub, i]
                x[i, 1] = min(packed[3], max(0., x[i, 1] + acceleration * packed[0]))
                x[i, 0] += x[i, 1] * packed[0]
                if np.isnan(arrivals[i]):
                    arrivals[i] = crossing_time(positions[i], x[i, 0], packed[11],
                                                (k-step)*packed[1] + sub*packed[0], packed[0])
            if not transition_values(positions[0], positions[1], positions[2], x[0, 0], x[1, 0], x[2, 0], packed):
                raise RuntimeError('Unsafe arrival-phase forecast transition')
        prev = command
        transitions += 1
    return arrivals, transitions, internal


class PhaseRepairGate(FastRepairPortfolioGate):
    def __init__(self, model, phase_tolerance=.05):
        if phase_tolerance is not None and (isinstance(phase_tolerance, (bool, np.bool_)) or
                not np.isfinite(phase_tolerance) or not 0. <= phase_tolerance <= .5):
            raise ValueError('Phase tolerance must be None or finite seconds in [0, 0.5]')
        super().__init__(model, bounded=False, bank='sampled')
        self.phase_tolerance = phase_tolerance

    def _phase(self, x, prev, command, path, cache, use_first):
        result, count, work = cached_arrival_phase(x, prev, self.step, self.tapes[path], self.packed,
                                                   command, use_first, cache)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise RuntimeError('Incomplete or nonfinite arrival-phase forecast')
        return result, count, work

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, .5)
        if self.step >= round(self.c.horizon/self.c.decision_dt):
            raise ValueError('Controller finite horizon exhausted')
        cache = new_command_cache()
        base, baseline, transitions, internal = cached_nominal_suffix(x, prev, self.step, self.packed, cache)
        self.primary.step = self.step
        _, actor_info = self.primary.select(x, prev)
        candidates, original_count, generation_errors = expanded_compositions(
            x, prev, np.asarray(actor_info['proposal']), baseline, self.packed, True)
        nominal, errors = [], []
        for code, weight, command in candidates:
            try:
                future, count, work, _, _ = self._future(x, prev, command, 0, base, cache)
            except RuntimeError as error:
                errors.append(dict(composition=code, weight=weight, path='clean', error=str(error)))
                continue
            transitions += count
            internal += work
            difference = future-base
            if not np.isfinite(difference).all():
                errors.append(dict(composition=code, weight=weight, path='clean', error='Nonfinite nominal forecast'))
                continue
            if np.all(difference <= 1e-10) and difference[:, 0].mean() < -1e-9:
                nominal.append((float(difference[:, 0].mean()), -weight, code, command, difference))
        nominal.sort(key=lambda item:item[:3])
        base_values, base_phases, base_errors = {0:base}, {}, {}
        chosen, selected_code, selected_weight = baseline, 0, None
        winner, winner_phase, evidence = [], [], []
        phase_enabled = self.phase_tolerance is not None
        for _, negative_weight, code, command, nominal_delta in nominal:
            weight = -negative_weight
            differences, phases = [], []
            failed, phase_failed, error_record = 0, 0, None
            for path in range(len(self.tapes)):
                if path in base_errors:
                    error_record = base_errors[path]
                    break
                stage = 'candidate_cost'
                try:
                    if path == 0:
                        difference = nominal_delta
                    else:
                        if path not in base_values:
                            # The same canonical full baseline is used by _future.
                            stage = 'baseline_cost'
                            value, count, work = cached_disturbed_suffix(x, prev, self.step, self.tapes[path], self.packed, cache)
                            transitions += count
                            internal += work
                            if not np.isfinite(value).all():
                                raise RuntimeError('Nonfinite baseline forecast')
                            base_values[path] = value
                        stage = 'candidate_cost'
                        value, count, work, _, _ = self._future(x, prev, command, path, base_values[path], cache)
                        transitions += count
                        internal += work
                        difference = value-base_values[path]
                    if not np.isfinite(difference).all():
                        raise RuntimeError('Nonfinite candidate forecast')
                    differences.append(difference.tolist())
                    if np.any(difference > 1e-10):
                        failed = 1
                        break
                    if phase_enabled:
                        if path not in base_phases:
                            stage = 'baseline_phase'
                            value, count, work = self._phase(x, prev, baseline, path, cache, False)
                            transitions += count
                            internal += work
                            base_phases[path] = value
                        stage = 'candidate_phase'
                        value, count, work = self._phase(x, prev, command, path, cache, True)
                        transitions += count
                        internal += work
                        delta_phase = value-base_phases[path]
                        phases.append(delta_phase.tolist())
                        if np.any(delta_phase > self.phase_tolerance + 1e-10):
                            phase_failed = 1
                            break
                except RuntimeError as error:
                    error_record = dict(path=self.names[path], stage=stage, error=str(error))
                    # A failing baseline cannot authorize any candidate on this path.
                    if stage in ('baseline_cost','baseline_phase'):
                        base_errors[path] = error_record
                    break
            evidence.append(dict(composition=code, weight=weight, command=command.tolist(),
                checked_scenarios=len(differences), failed_scenarios=failed, differences=differences,
                phase_checked_scenarios=len(phases), phase_failed_scenarios=phase_failed,
                phase_differences=phases, forecast_error=error_record))
            if error_record is not None:
                errors.append(dict(composition=code, weight=weight, **error_record))
                continue
            if not failed and not phase_failed:
                if len(differences)!=33 or (phase_enabled and len(phases)!=33):
                    raise RuntimeError('Incomplete full-bank evidence cannot authorize learning')
                chosen, selected_code, selected_weight = command, code, weight
                winner, winner_phase = differences, phases
                break
        self.step += 1
        admitted = selected_code != 0
        if not robust_certificate(x, prev, chosen, self.packed, .5):
            raise RuntimeError('Chosen phase-gated command failed the physical certificate')
        return chosen, dict(in_set=True, certified=True, intervention=float(not admitted),
            nominal_joint_action_admitted=admitted, portfolio_joint_action_admitted=admitted,
            repair_action_admitted=bool(admitted and selected_weight<1.), composition=selected_code,
            repair_weight=selected_weight, baseline_command=baseline.tolist(), actor_proposal=actor_info['proposal'],
            candidate_count=len(candidates), original_candidate_count=original_count,
            repair_candidate_count=len(candidates)-original_count, nominal_candidate_count=len(nominal),
            candidate_evidence=evidence, selected_bank_differences=winner,
            selected_phase_differences=winner_phase, phase_tolerance_seconds=self.phase_tolerance,
            phase_filter_enabled=phase_enabled, max_failed_scenarios=0,
            temporal_bank='sampled', forecast_bank_size=33, forecast_bank_stream=BANK_STREAM,
            generation_errors=generation_errors, forecast_errors=errors, work_counts_complete=not errors,
            gate_ms=(time.perf_counter()-started)*1000., performance_forecast_transitions=transitions,
            performance_model_transitions=internal, exact_cache_entries=len(cache),
            out_of_bank_risk_certified=False,
            performance_claim='Original full-bank component conditions plus the declared phase tolerance; no out-of-bank risk certificate')
