"""Reuse exact deterministic baseline commands across consecutive gate calls.

The V21 policy remains the reference. The copied selection method differs only
in its name and cache source; candidate generation, ordering, every path, full
horizon, physical checks and evidence are unchanged. A cache entry depends on
all state/previous/step key components and the fixed packed configuration.
No approximate keys, trajectory splicing, new risk certificate or deadline
truncation is introduced. This implementation is outside active calibration.
"""
import numbers
import time

import numpy as np
from numba import njit

from .phase_repair_v21 import (
    BANK_STREAM, PhaseRepairGate, cached_disturbed_suffix, cached_nominal_suffix,
    expanded_compositions, new_command_cache, robust_certificate, validate,
)


@njit(cache=True, fastmath=False)
def retain_future_commands(cache, current_step, retained_limit):
    before = len(cache)
    for key in list(cache.keys()):
        if key[0] < current_step:
            del cache[key]
    if len(cache) > retained_limit:
        # Keep nearer future times first. Eviction affects work, never values.
        keys = sorted(list(cache.keys()))
        for key in keys[retained_limit:]:
            del cache[key]
    return before - len(cache)


class PersistentPhaseRepairGate(PhaseRepairGate):
    def __init__(self, model, phase_tolerance=.05, max_cache_entries=8192, reuse=True):
        if (isinstance(max_cache_entries, (bool, np.bool_))
                or not isinstance(max_cache_entries, numbers.Integral)
                or max_cache_entries < 1):
            raise ValueError('max_cache_entries must be a positive integer')
        if type(reuse) is not bool:
            raise ValueError('reuse must be a boolean')
        super().__init__(model, phase_tolerance)
        self.max_cache_entries = int(max_cache_entries)
        self.reuse = reuse
        self.clear_cache()

    def clear_cache(self):
        self._command_cache = new_command_cache()
        self._cache_configuration = None
        self._last_step = None

    def _prepare_cache(self):
        packed = np.asarray(self.packed)
        signature = (packed.dtype.str, packed.shape, packed.tobytes())
        reason = 'reuse'
        if not self.reuse:
            reason = 'disabled'
        elif signature != self._cache_configuration:
            reason = 'initial_or_changed_configuration'
        elif self._last_step is not None and self.step <= self._last_step:
            reason = 'nonincreasing_clock'
        if reason != 'reuse':
            self._command_cache = new_command_cache()
        self._cache_configuration = signature
        return reason

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        reason = self._prepare_cache()
        before = len(self._command_cache)
        try:
            command, info = self._select_cached(state, previous, unused)
            self._last_step = self.step - 1
            peak = len(self._command_cache)
            if self.reuse:
                evicted = retain_future_commands(self._command_cache, self.step, self.max_cache_entries)
            else:
                evicted = peak
                self._command_cache = new_command_cache()
            info.update(persistent_cache_enabled=self.reuse,
                        persistent_cache_reset_reason=reason,
                        persistent_cache_entries_before=before,
                        persistent_cache_entries_peak=peak,
                        persistent_cache_entries_after=len(self._command_cache),
                        persistent_cache_evicted=evicted,
                        persistent_cache_retained_limit=self.max_cache_entries)
            # Include cache lifecycle, eviction and result construction costs.
            info['gate_ms'] = (time.perf_counter() - started) * 1000.
            return command, info
        except BaseException:
            self.clear_cache()
            raise

    def _select_cached(self, state, previous, unused=None):
        started = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, .5)
        if self.step >= round(self.c.horizon/self.c.decision_dt):
            raise ValueError('Controller finite horizon exhausted')
        cache = self._command_cache
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
