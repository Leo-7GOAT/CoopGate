"""Exact command reuse and optional rejection-only bounds for full action repair.

The reference repair policy remains unchanged. Every admitted action still uses
the complete finite bank; bounds can only reject a provably losing candidate.
"""
import time

import numpy as np

from .bounded_portfolio_v19 import bounded_action_then_baseline
from .cached_portfolio_v19 import (
    cached_action_then_baseline, cached_disturbed_suffix, cached_nominal_suffix, new_command_cache,
)
from .disturbed_advantage_v19 import disturbed_transition
from .repair_portfolio_v20 import RepairPortfolioGate, expanded_compositions
from .robust_v9 import robust_certificate, validate
from .temporal_portfolio_v20 import BANK_STREAM


class FastRepairPortfolioGate(RepairPortfolioGate):
    def __init__(self, model, bounded=False, bank='sampled'):
        if type(bounded) is not bool:
            raise ValueError('bounded must be a boolean')
        super().__init__(model, bank, True)
        self.bounded = bounded

    def _future(self, x, prev, command, path, base, cache):
        if self.bounded:
            return bounded_action_then_baseline(x, prev, self.step, command, self.tapes[path],
                                                self.packed, base, cache, path == 0)
        if path == 0:
            value, count, work = cached_action_then_baseline(x, prev, self.step, command, self.packed, cache)
        else:
            following = x.copy()
            disturbed_transition(following, command, self.step, self.tapes[path], self.packed)
            value, count, work = cached_disturbed_suffix(following, command, self.step + 1,
                                                         self.tapes[path], self.packed, cache)
            count += 1
            for vehicle in range(3):
                if x[vehicle, 0] < self.c.goal:
                    value[vehicle, 0] += self.c.decision_dt + .05 * abs(command[vehicle] - prev[vehicle])
                    value[vehicle, 1] += self.c.decision_dt
        return value, count, work, False, 0

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, .5)
        if self.step >= round(self.c.horizon / self.c.decision_dt):
            raise ValueError('Controller finite horizon exhausted')
        cache = new_command_cache()
        base, baseline_command, transitions, internal = cached_nominal_suffix(x, prev, self.step, self.packed, cache)
        self.primary.step = self.step
        _, actor_info = self.primary.select(x, prev)
        candidates, original_count, generation_errors = expanded_compositions(
            x, prev, np.asarray(actor_info['proposal']), baseline_command, self.packed, True)
        nominal, forecast_errors, nominal_pruned = [], [], []
        optimistic_work, bank_pruned = 0, 0
        for code, weight, command in candidates:
            try:
                future, count, work, pruned, optimistic = self._future(x, prev, command, 0, base, cache)
            except RuntimeError as error:
                forecast_errors.append(dict(composition=code, weight=weight, path='clean', error=str(error)))
                continue
            transitions += count
            internal += work
            optimistic_work += optimistic
            difference = future - base
            if not np.isfinite(difference).all():
                forecast_errors.append(dict(composition=code, weight=weight, path='clean', error='Nonfinite nominal forecast'))
                continue
            if pruned:
                nominal_pruned.append(dict(composition=code, weight=weight, command=command.tolist(),
                                           lower_difference=difference.tolist()))
            elif np.all(difference <= 1e-10) and difference[:, 0].mean() < -1e-9:
                nominal.append((float(difference[:, 0].mean()), -weight, code, command, difference))
        nominal.sort(key=lambda candidate: candidate[:3])
        base_cache, base_errors = {}, {}
        command, selected_code, selected_weight = baseline_command, 0, None
        winner, evidence = [], []
        for _, negative_weight, code, candidate, nominal_difference in nominal:
            weight = -negative_weight
            differences, failed, error_record, lower_indices = [nominal_difference.tolist()], 0, None, []
            for path in range(1, len(self.tapes)):
                if path in base_errors:
                    error_record = base_errors[path]
                    break
                if path not in base_cache:
                    try:
                        value, count, work = cached_disturbed_suffix(x, prev, self.step, self.tapes[path], self.packed, cache)
                    except RuntimeError as error:
                        error_record = dict(path=self.names[path], stage='baseline_forecast', error=str(error))
                        base_errors[path] = error_record
                        break
                    transitions += count
                    internal += work
                    if not np.isfinite(value).all():
                        error_record = dict(path=self.names[path], stage='baseline_forecast', error='Nonfinite baseline forecast')
                        base_errors[path] = error_record
                        break
                    base_cache[path] = value
                try:
                    value, count, work, pruned, optimistic = self._future(x, prev, candidate, path, base_cache[path], cache)
                except RuntimeError as error:
                    error_record = dict(path=self.names[path], stage='candidate_forecast', error=str(error))
                    break
                transitions += count
                internal += work
                optimistic_work += optimistic
                difference = value - base_cache[path]
                if not np.isfinite(difference).all():
                    error_record = dict(path=self.names[path], stage='candidate_forecast', error='Nonfinite disturbance forecast')
                    break
                if pruned:
                    if not np.any(difference > 1e-10):
                        raise RuntimeError('An inconclusive bound cannot reject a candidate')
                    lower_indices.append(path)
                    bank_pruned += 1
                differences.append(difference.tolist())
                failed += int(np.any(difference > 1e-10))
                if failed:
                    break
            evidence.append(dict(composition=code, weight=weight, command=candidate.tolist(),
                                 checked_scenarios=len(differences), failed_scenarios=failed,
                                 differences=differences, forecast_error=error_record,
                                 lower_bound_scenarios=lower_indices))
            if error_record is not None:
                forecast_errors.append(dict(composition=code, weight=weight, **error_record))
                continue
            if not failed:
                if len(differences) != len(self.tapes) or lower_indices:
                    raise RuntimeError('Only the full exact bank can authorize a repair')
                command, selected_code, selected_weight, winner = candidate, code, weight, differences
                break
        self.step += 1
        admitted = selected_code != 0
        if not robust_certificate(x, prev, command, self.packed, .5):
            raise RuntimeError('Final repaired command failed the physical certificate')
        return command, dict(
            in_set=True, certified=True, intervention=float(not admitted),
            nominal_joint_action_admitted=admitted, portfolio_joint_action_admitted=admitted,
            repair_action_admitted=bool(admitted and selected_weight < 1.), composition=selected_code,
            repair_weight=selected_weight, baseline_command=baseline_command.tolist(), actor_proposal=actor_info['proposal'],
            candidate_count=len(candidates), original_candidate_count=original_count,
            repair_candidate_count=len(candidates) - original_count, nominal_candidate_count=len(nominal),
            candidate_evidence=evidence, selected_bank_differences=winner, max_failed_scenarios=0,
            temporal_bank=self.temporal_bank, forecast_bank_size=len(self.tapes), forecast_bank_stream=BANK_STREAM,
            generation_errors=generation_errors, forecast_errors=forecast_errors,
            work_counts_complete=not forecast_errors, gate_ms=(time.perf_counter() - started) * 1000.,
            performance_forecast_transitions=transitions, performance_model_transitions=internal,
            nominal_pruned_bounds=nominal_pruned, bank_pruned_count=bank_pruned,
            optimistic_forecast_transitions=optimistic_work, exact_cache_entries=len(cache),
            out_of_bank_risk_certified=False,
            performance_claim='Same full candidate family and accepted finite-bank evidence; bounds only reject candidates')
