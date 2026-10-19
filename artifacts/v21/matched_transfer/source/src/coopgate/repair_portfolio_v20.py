"""Complete joint portfolio with physically reprojected attenuation candidates.

Original commands are retained exactly. A candidate is admitted only after all
vehicle cost/time components pass the same finite disturbance bank. Additional
forecast failures reject a candidate; they never authorize execution.
"""
import time

import numpy as np

from .action_repair_v20 import attenuated_candidates
from .advantage_gate_v18 import action_then_baseline, baseline_suffix
from .disturbed_advantage_v19 import disturbed_suffix, disturbed_transition
from .portfolio_gate_v19 import joint_compositions
from .robust_v9 import robust_certificate, validate
from .temporal_portfolio_v20 import BANK_STREAM, TemporalPortfolioGate


def expanded_compositions(state, previous, raw_proposal, baseline, packed, with_repair=True):
    originals = joint_compositions(state, previous, raw_proposal, baseline, packed)
    candidates = [(code, 1., command.copy()) for code, command, _ in originals]
    seen = [np.asarray(baseline)] + [command for _, _, command in candidates]
    errors = []
    if with_repair:
        for code, command, _ in originals:
            try:
                extensions = attenuated_candidates(state, previous, command, baseline, packed)
            except RuntimeError as error:
                errors.append(dict(composition=code, error=str(error)))
                continue
            for weight, candidate in extensions:
                if weight == 1. or any(np.array_equal(candidate, old) for old in seen):
                    continue
                seen.append(candidate.copy())
                candidates.append((code, weight, candidate.copy()))
    return candidates, len(originals), errors


class RepairPortfolioGate(TemporalPortfolioGate):
    def __init__(self, model, bank='sampled', with_repair=True):
        if type(with_repair) is not bool:
            raise ValueError('with_repair must be a boolean')
        super().__init__(model, bank)
        self.with_repair = with_repair

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, .5)
        if self.step >= round(self.c.horizon / self.c.decision_dt):
            raise ValueError('Controller finite horizon exhausted')
        base, baseline_command, transitions, internal = baseline_suffix(x, prev, self.step, self.packed)
        self.primary.step = self.step
        _, actor_info = self.primary.select(x, prev)
        candidates, original_count, generation_errors = expanded_compositions(
            x, prev, np.asarray(actor_info['proposal']), baseline_command, self.packed, self.with_repair)
        nominal, forecast_errors = [], []
        for code, weight, command in candidates:
            try:
                future, count, work = action_then_baseline(x, prev, self.step, command, self.packed)
            except RuntimeError as error:
                forecast_errors.append(dict(composition=code, weight=weight, path='clean', error=str(error)))
                continue
            transitions += count
            internal += work
            difference = future - base
            if not np.isfinite(difference).all():
                forecast_errors.append(dict(composition=code, weight=weight, path='clean',
                                            error='Nonfinite nominal forecast'))
                continue
            if np.all(difference <= 1e-10) and difference[:, 0].mean() < -1e-9:
                nominal.append((float(difference[:, 0].mean()), -weight, code, command, difference))
        # Original-only mode reproduces the original gain/composition ordering.
        nominal.sort(key=lambda candidate: candidate[:3])
        base_cache, base_errors = {}, {}
        command, selected_code, selected_weight = baseline_command, 0, None
        winner, evidence = [], []
        for _, negative_weight, code, candidate, nominal_difference in nominal:
            weight = -negative_weight
            differences, failed, error_record = [nominal_difference.tolist()], 0, None
            for path in range(1, len(self.tapes)):
                tape = self.tapes[path]
                if path in base_errors:
                    error_record = base_errors[path]
                    break
                if path not in base_cache:
                    try:
                        value, count, work = disturbed_suffix(x, prev, self.step, tape, self.packed)
                    except RuntimeError as error:
                        error_record = dict(path=self.names[path], stage='baseline_forecast', error=str(error))
                        base_errors[path] = error_record
                        break
                    transitions += count
                    internal += work
                    if not np.isfinite(value).all():
                        error_record = dict(path=self.names[path], stage='baseline_forecast',
                                            error='Nonfinite baseline forecast')
                        base_errors[path] = error_record
                        break
                    base_cache[path] = value
                following = x.copy()
                try:
                    disturbed_transition(following, candidate, self.step, tape, self.packed)
                    value, count, work = disturbed_suffix(following, candidate, self.step + 1, tape, self.packed)
                except RuntimeError as error:
                    error_record = dict(path=self.names[path], stage='candidate_forecast', error=str(error))
                    break
                transitions += count + 1
                internal += work
                for vehicle in range(3):
                    if x[vehicle, 0] < self.c.goal:
                        value[vehicle, 0] += self.c.decision_dt + .05 * abs(candidate[vehicle] - prev[vehicle])
                        value[vehicle, 1] += self.c.decision_dt
                difference = value - base_cache[path]
                if not np.isfinite(difference).all():
                    error_record = dict(path=self.names[path], stage='candidate_forecast',
                                        error='Nonfinite disturbance forecast')
                    break
                differences.append(difference.tolist())
                failed += int(np.any(difference > 1e-10))
                if failed:
                    break
            evidence.append(dict(composition=code, weight=weight, command=candidate.tolist(),
                                 checked_scenarios=len(differences), failed_scenarios=failed,
                                 differences=differences, forecast_error=error_record))
            if error_record is not None:
                forecast_errors.append(dict(composition=code, weight=weight, **error_record))
                continue
            if not failed:
                if len(differences) != len(self.tapes):
                    raise RuntimeError('Incomplete bank cannot authorize a repair')
                command, selected_code, selected_weight, winner = candidate, code, weight, differences
                break
        self.step += 1
        admitted = selected_code != 0
        if not robust_certificate(x, prev, command, self.packed, .5):
            raise RuntimeError('Final repaired command failed the physical certificate')
        return command, dict(
            in_set=True, certified=True, intervention=float(not admitted),
            nominal_joint_action_admitted=admitted, portfolio_joint_action_admitted=admitted,
            repair_action_admitted=bool(admitted and selected_weight < 1.),
            composition=selected_code, repair_weight=selected_weight,
            baseline_command=baseline_command.tolist(), actor_proposal=actor_info['proposal'],
            candidate_count=len(candidates), original_candidate_count=original_count,
            repair_candidate_count=len(candidates) - original_count,
            nominal_candidate_count=len(nominal), candidate_evidence=evidence,
            selected_bank_differences=winner, max_failed_scenarios=0,
            temporal_bank=self.temporal_bank, forecast_bank_size=len(self.tapes), forecast_bank_stream=BANK_STREAM,
            generation_errors=generation_errors, forecast_errors=forecast_errors,
            work_counts_complete=not forecast_errors, gate_ms=(time.perf_counter() - started) * 1000.,
            performance_forecast_transitions=transitions, performance_model_transitions=internal,
            out_of_bank_risk_certified=False,
            performance_claim='All original joint candidates retained; every executed candidate passes the complete fixed bank; no out-of-bank risk certificate')
