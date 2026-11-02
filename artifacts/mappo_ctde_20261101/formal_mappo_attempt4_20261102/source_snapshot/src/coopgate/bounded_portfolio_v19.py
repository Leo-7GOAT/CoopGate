"""Reject only when an optimistic physical lower bound already fails the gate.

For a fixed forecast tape, unconstrained maximum acceleration (with the same
command slew/speed limits) yields an upper envelope of each vehicle's progress.
Its decision-grid arrival time lower-bounds any baseline continuation. Add it
to the accumulated cost/time and first-action stage cost. A component exceeding
the full baseline potential by >1e-9 proves rejection at the original 1e-10
threshold. Accepted candidates still receive every complete original forecast.
"""
import time

import numpy as np
from numba import njit

from .cached_portfolio_v19 import (
    cached_command, cached_disturbed_suffix, cached_nominal_suffix, new_command_cache,
)
from .disturbed_advantage_v19 import disturbed_transition
from .fast_v8 import advance_one
from .portfolio_gate_v19 import JointPortfolioGate, joint_compositions
from .robust_v9 import robust_certificate, validate


@njit(cache=True, fastmath=False)
def optimistic_arrival_lower_bound(state, previous, step, tape, packed, nominal):
    """Ignore interactions while respecting individual maximum acceleration.

    The result is capped at the same finite horizon as the reference potential.
    A vehicle already finished has zero remaining cost. No collision certificate
    is attached to this optimistic trajectory; it is a mathematical bound only.
    """
    x, command = state.copy(), previous.copy()
    remaining = np.zeros(3)
    count = 0
    substeps = round(packed[1] / packed[0])
    for k in range(step, round(packed[2] / packed[1])):
        active = x[:, 0] < packed[11]
        if not np.any(active):
            break
        for i in range(3):
            command[i] = min(packed[5], command[i] + packed[6] * packed[1])
            if active[i]:
                remaining[i] += packed[1]
        for sub in range(substeps):
            for i in range(3):
                error = 0. if nominal else tape[k * substeps + sub, i]
                x[i, 1] = min(packed[3], max(0., x[i, 1] + (command[i] + error) * packed[0]))
                x[i, 0] += x[i, 1] * packed[0]
        count += 1
    return remaining, count


@njit(cache=True, fastmath=False)
def bounded_action_then_baseline(state, previous, step, command, tape, packed, base, cache, nominal):
    if not robust_certificate(state, previous, command, packed, .5):
        raise ValueError('Uncertified first command')
    x, prev = state.copy(), command.copy()
    if nominal:
        if not advance_one(x, command, packed):
            raise RuntimeError('Unsafe nominal first transition')
    else:
        disturbed_transition(x, command, step, tape, packed)
    stage = np.zeros((3, 2))
    for i in range(3):
        if state[i, 0] < packed[11]:
            stage[i, 0] = packed[1] + .05 * abs(command[i] - previous[i])
            stage[i, 1] = packed[1]
    # Retain reference summation order: suffix first, initial stage added last.
    values = np.zeros((3, 2))
    transitions, internal, optimistic_work = 1, 0, 0
    for k in range(step + 1, round(packed[2] / packed[1])):
        active = x[:, 0] < packed[11]
        if not np.any(active):
            break
        arrival, count = optimistic_arrival_lower_bound(x, prev, k, tape, packed, nominal)
        optimistic_work += count
        lower = values + stage
        for i in range(3):
            lower[i, 0] += arrival[i]
            lower[i, 1] += arrival[i]
        # Conservative numerical cushion; never relax the original admission
        # threshold. A complete value is still used whenever this cannot prune.
        if np.any(lower > base + 1e-9):
            return lower, transitions, internal, True, optimistic_work
        following, count = cached_command(x, prev, k, packed, cache)
        internal += count
        for i in range(3):
            if active[i]:
                values[i, 0] += packed[1] + .05 * abs(following[i] - prev[i])
                values[i, 1] += packed[1]
        if nominal:
            if not advance_one(x, following, packed):
                raise RuntimeError('Unsafe nominal baseline suffix')
        else:
            disturbed_transition(x, following, k, tape, packed)
        prev = following
        transitions += 1
    return values + stage, transitions, internal, False, optimistic_work


class BoundedJointPortfolioGate(JointPortfolioGate):
    def __init__(self, model, max_failed_scenarios=0):
        if max_failed_scenarios != 0:
            raise ValueError('This optimization is validated for the strict gate only')
        super().__init__(model, max_failed_scenarios)

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
        base_cache = {}
        command, selected_code = baseline_command, 0
        evidence, winner_differences = [], []
        for _, code, candidate, nominal_difference, vertex in nominal_candidates:
            differences, lower_indices = [nominal_difference.tolist()], []
            failed = 0
            for index in range(1, len(self.tapes)):
                tape = self.tapes[index]
                if index not in base_cache:
                    value, count, work = cached_disturbed_suffix(x, prev, self.step, tape, self.packed, cache)
                    transitions += count
                    internal += work
                    base_cache[index] = value
                value, count, work, pruned, optimistic = bounded_action_then_baseline(
                    x, prev, self.step, candidate, tape, self.packed, base_cache[index], cache, False)
                transitions += count
                internal += work
                optimistic_work += optimistic
                difference = value - base_cache[index]
                differences.append(difference.tolist())
                if pruned:
                    lower_indices.append(index)
                    bank_pruned += 1
                    assert np.any(difference > 1e-10)
                failed += int(np.any(difference > 1e-10))
                if failed:
                    break
            evidence.append(dict(composition=code, command=candidate.tolist(), checked_scenarios=len(differences),
                                 failed_scenarios=failed, differences=differences, lower_bound_scenarios=lower_indices))
            if not failed:
                assert not lower_indices
                command, selected_code, winner_differences = candidate, code, differences
                break
        self.step += 1
        accepted = selected_code != 0
        if not robust_certificate(x, prev, command, self.packed, .5):
            raise RuntimeError('Final composition failed physical certificate')
        return command, dict(
            in_set=True, certified=True, intervention=float(not accepted),
            nominal_joint_action_admitted=accepted, portfolio_joint_action_admitted=accepted,
            composition=selected_code, candidate_count=len(candidates), nominal_candidate_count=len(nominal_candidates),
            candidate_evidence=evidence, selected_bank_differences=winner_differences,
            nominal_pruned_bounds=nominal_pruned, bank_pruned_count=bank_pruned,
            max_failed_scenarios=0, gate_ms=(time.perf_counter() - started) * 1000.,
            performance_forecast_transitions=transitions, performance_model_transitions=internal,
            optimistic_forecast_transitions=optimistic_work, exact_cache_entries=len(cache),
            performance_claim='Finite-bank componentwise pathwise guarantee; optimistic bounds only authorize rejection')
