"""Parallel complete disturbance continuations with serial decision semantics.

Every task owns its command cache. Candidate and scenario ordering, stopping
conditions, arithmetic and admission thresholds match BoundedJointPortfolioGate.
All submitted work is joined before return and included in wall time. Extra
speculative work is counted rather than hidden by early logical rejection.
"""
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from numba import njit

from .bounded_portfolio_v19 import BoundedJointPortfolioGate, bounded_action_then_baseline
from .cached_portfolio_v19 import cached_disturbed_suffix, cached_nominal_suffix, new_command_cache
from .portfolio_gate_v19 import joint_compositions
from .robust_v9 import robust_certificate, validate


@njit(cache=True, nogil=True, fastmath=False)
def evaluate_disturbed_candidate(state, previous, step, command, tape, packed, cache, base_ready, previous_base):
    """Private mutable cache; all state/tape/configuration inputs are read-only."""
    base, transitions, internal = previous_base, 0, 0
    if not base_ready:
        base, transitions, internal = cached_disturbed_suffix(state, previous, step, tape, packed, cache)
    value, count, work, pruned, optimistic = bounded_action_then_baseline(
        state, previous, step, command, tape, packed, base, cache, False)
    return base, value - base, transitions + count, internal + work, pruned, optimistic


class ParallelJointPortfolioGate(BoundedJointPortfolioGate):
    def __init__(self, model, max_failed_scenarios=0, workers=2):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError('One to four CPU workers required')
        super().__init__(model, max_failed_scenarios)
        self.workers = workers

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, .5)
        if self.step >= round(self.c.horizon / self.c.decision_dt):
            raise ValueError('Controller finite horizon exhausted')
        cache = new_command_cache()
        base, baseline_command, transitions, internal = cached_nominal_suffix(x, prev, self.step, self.packed, cache)
        self.primary.step = self.step
        _, actor_info = self.primary.select(x, prev)
        candidates = joint_compositions(x, prev, np.array(actor_info['proposal']), baseline_command, self.packed)
        nominal_candidates, nominal_pruned = [], []
        optimistic_work, bank_pruned = 0, 0
        for code, command, vertex in candidates:
            future, count, work, pruned, optimistic = bounded_action_then_baseline(
                x, prev, self.step, command, self.tapes[0], self.packed, base, cache, True)
            transitions += count
            internal += work
            optimistic_work += optimistic
            difference = future - base
            if pruned:
                nominal_pruned.append(dict(composition=code, command=command.tolist(), lower_difference=difference.tolist()))
            elif np.all(difference <= 1e-10) and difference[:, 0].mean() < -1e-9:
                nominal_candidates.append((float(difference[:, 0].mean()), code, command, difference, vertex))
        nominal_candidates.sort(key=lambda row: (row[0], row[1]))
        command, selected_code = baseline_command, 0
        evidence, winner_differences = [], []
        base_cache, scenario_caches = {}, {}
        computed, unused_results, unused_errors = 0, 0, 0
        if nominal_candidates:
            # Pool startup and shutdown are inside the actual timing scope.
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for _, code, candidate, nominal_difference, vertex in nominal_candidates:
                    differences, lower_indices = [nominal_difference.tolist()], []
                    failed, index = 0, 1
                    while index < len(self.tapes) and not failed:
                        # Reject cheap first-scenario failures before speculation.
                        width = 1 if index == 1 else self.workers
                        batch = list(range(index, min(index + width, len(self.tapes))))
                        futures = []
                        for scenario in batch:
                            if scenario not in scenario_caches:
                                # Values are immutable float tuples. Copies are
                                # constructed on the caller thread before submit;
                                # no dictionary is mutated by concurrent tasks.
                                scenario_caches[scenario] = cache.copy()
                            ready = scenario in base_cache
                            future = pool.submit(evaluate_disturbed_candidate,
                                x, prev, self.step, candidate, self.tapes[scenario], self.packed,
                                scenario_caches[scenario], ready,
                                base_cache[scenario] if ready else np.zeros((3, 2)))
                            futures.append((scenario, future))
                        results = {}
                        for scenario, future in futures:
                            try:
                                result = future.result()
                            except Exception as error:
                                # An exception in a logically checked scenario
                                # is re-raised below. A later unused scenario
                                # cannot change the original rejection decision.
                                results[scenario] = error
                                continue
                            baseline, difference, count, work, pruned, optimistic = result
                            base_cache[scenario] = baseline
                            transitions += count
                            internal += work
                            optimistic_work += optimistic
                            computed += 1
                            results[scenario] = result
                        for scenario in batch:
                            result = results[scenario]
                            if failed:
                                unused_results += 1
                                unused_errors += int(isinstance(result, Exception))
                                continue
                            if isinstance(result, Exception):
                                raise result
                            _, difference, _, _, pruned, _ = result
                            differences.append(difference.tolist())
                            if pruned:
                                lower_indices.append(scenario)
                                bank_pruned += 1
                                assert np.any(difference > 1e-10)
                            failed += int(np.any(difference > 1e-10))
                        index += len(batch)
                    evidence.append(dict(composition=code, command=candidate.tolist(), checked_scenarios=len(differences),
                                         failed_scenarios=failed, differences=differences, lower_bound_scenarios=lower_indices))
                    if not failed:
                        assert not lower_indices and len(differences) == len(self.tapes)
                        command, selected_code, winner_differences = candidate, code, differences
                        break
        self.step += 1
        accepted = selected_code != 0
        if not robust_certificate(x, prev, command, self.packed, .5):
            raise RuntimeError('Final composition failed physical certificate')
        cache_entries = len(cache) + sum(len(value) for value in scenario_caches.values())
        return command, dict(
            in_set=True, certified=True, intervention=float(not accepted),
            nominal_joint_action_admitted=accepted, portfolio_joint_action_admitted=accepted,
            composition=selected_code, candidate_count=len(candidates), nominal_candidate_count=len(nominal_candidates),
            candidate_evidence=evidence, selected_bank_differences=winner_differences,
            nominal_pruned_bounds=nominal_pruned, bank_pruned_count=bank_pruned,
            max_failed_scenarios=0, gate_ms=(time.perf_counter() - started) * 1000.,
            performance_forecast_transitions=transitions, performance_model_transitions=internal,
            optimistic_forecast_transitions=optimistic_work,
            exact_cache_entries=cache_entries,
            parallel_workers=self.workers, computed_disturbed_scenarios=computed,
            unused_speculative_scenarios=unused_results, unused_speculative_errors=unused_errors,
            work_counts_complete=unused_errors == 0,
            performance_claim='Same finite-bank componentwise admission; all parallel work joined before return')
