"""Behavior-preserving work reduction for the frozen V19 joint portfolio gate.

Cache only exact deterministic baseline commands within one select call. State,
previous command and step form the key; the configuration is fixed for that
call. No quantization, stale cross-step cache, shortened horizon or approximate
suffix is used. The original V19 implementation remains the reference oracle.
"""
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict

from .deduplicated_planner_v19 import deduplicated_joint_rollout
from .disturbed_advantage_v19 import disturbed_transition
from .fast_v8 import advance_one
from .portfolio_gate_v19 import JointPortfolioGate, joint_compositions
from .robust_v9 import robust_certificate, validate


def new_command_cache():
    return Dict.empty(key_type=types.UniTuple(types.float64, 10), value_type=types.UniTuple(types.float64, 3))


@njit(cache=True, fastmath=False)
def cached_command(x, prev, step, packed, cache):
    key = (float(step), x[0, 0], x[0, 1], x[1, 0], x[1, 1], x[2, 0], x[2, 1], prev[0], prev[1], prev[2])
    if key in cache:
        value = cache[key]
        return np.array(value), 0
    command, _, _, count = deduplicated_joint_rollout(x, prev, step, packed)
    cache[key] = (command[0], command[1], command[2])
    return command, count


@njit(cache=True, fastmath=False)
def cached_nominal_suffix(state, previous, step, packed, cache):
    x, prev = state.copy(), previous.copy()
    values = np.zeros((3, 2))
    first, transitions, internal = previous.copy(), 0, 0
    for k in range(step, round(packed[2] / packed[1])):
        active = x[:, 0] < packed[11]
        if not np.any(active):
            break
        command, count = cached_command(x, prev, k, packed, cache)
        internal += count
        if k == step:
            first = command.copy()
        for i in range(3):
            if active[i]:
                values[i, 0] += packed[1] + .05 * abs(command[i] - prev[i])
                values[i, 1] += packed[1]
        if not advance_one(x, command, packed):
            raise RuntimeError('Unsafe nominal baseline suffix')
        prev = command
        transitions += 1
    return values, first, transitions, internal


@njit(cache=True, fastmath=False)
def cached_action_then_baseline(state, previous, step, command, packed, cache):
    if not robust_certificate(state, previous, command, packed, .5):
        raise ValueError('Uncertified first command')
    active = state[:, 0] < packed[11]
    following = state.copy()
    if not advance_one(following, command, packed):
        raise RuntimeError('Unsafe nominal first transition')
    values, _, transitions, internal = cached_nominal_suffix(following, command, step + 1, packed, cache)
    for i in range(3):
        if active[i]:
            values[i, 0] += packed[1] + .05 * abs(command[i] - previous[i])
            values[i, 1] += packed[1]
    return values, transitions + 1, internal


@njit(cache=True, fastmath=False)
def cached_disturbed_suffix(state, previous, step, tape, packed, cache):
    x, prev = state.copy(), previous.copy()
    values = np.zeros((3, 2))
    transitions, internal = 0, 0
    for k in range(step, round(packed[2] / packed[1])):
        active = x[:, 0] < packed[11]
        if not np.any(active):
            break
        command, count = cached_command(x, prev, k, packed, cache)
        internal += count
        for i in range(3):
            if active[i]:
                values[i, 0] += packed[1] + .05 * abs(command[i] - prev[i])
                values[i, 1] += packed[1]
        disturbed_transition(x, command, k, tape, packed)
        prev = command
        transitions += 1
    return values, transitions, internal


class CachedJointPortfolioGate(JointPortfolioGate):
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
        nominal_candidates = []
        for code, command, vertex in candidates:
            future, count, work = cached_action_then_baseline(x, prev, self.step, command, self.packed, cache)
            transitions += count
            internal += work
            difference = future - base
            if np.all(difference <= 1e-10) and difference[:, 0].mean() < -1e-9:
                nominal_candidates.append((float(difference[:, 0].mean()), code, command, difference, vertex))
        nominal_candidates.sort(key=lambda row: (row[0], row[1]))
        base_cache = {}
        command, selected_code = baseline_command, 0
        evidence, winner_differences = [], []
        for _, code, candidate, nominal_difference, vertex in nominal_candidates:
            differences, failed = [nominal_difference.tolist()], 0
            for index in range(1, len(self.tapes)):
                tape = self.tapes[index]
                if index not in base_cache:
                    value, count, work = cached_disturbed_suffix(x, prev, self.step, tape, self.packed, cache)
                    transitions += count
                    internal += work
                    base_cache[index] = value
                following = x.copy()
                disturbed_transition(following, candidate, self.step, tape, self.packed)
                value, count, work = cached_disturbed_suffix(following, candidate, self.step + 1, tape, self.packed, cache)
                transitions += count + 1
                internal += work
                for i in range(3):
                    if x[i, 0] < self.c.goal:
                        value[i, 0] += self.c.decision_dt + .05 * abs(candidate[i] - prev[i])
                        value[i, 1] += self.c.decision_dt
                difference = value - base_cache[index]
                differences.append(difference.tolist())
                failed += int(np.any(difference > 1e-10))
                if failed > self.max_failed_scenarios:
                    break
            evidence.append(dict(composition=code, command=candidate.tolist(), checked_scenarios=len(differences),
                                 failed_scenarios=failed, differences=differences))
            if failed <= self.max_failed_scenarios:
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
            max_failed_scenarios=self.max_failed_scenarios, gate_ms=(time.perf_counter() - started) * 1000.,
            performance_forecast_transitions=transitions, performance_model_transitions=internal,
            exact_cache_entries=len(cache),
            performance_claim=('Finite-bank componentwise pathwise guarantee' if self.max_failed_scenarios == 0
                               else 'Nominal guarantee only; relaxed bank is NOT a risk certificate'))
