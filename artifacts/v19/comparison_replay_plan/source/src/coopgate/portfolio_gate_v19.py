"""Jointly recheck actor/baseline action compositions against baseline potentials.

The portfolio changes ONLY how actor proposals are arbitrated. The actors are
the frozen, separately trained MARL policies. A mixed action is always evaluated
jointly, after the common physical projection. Per-car isolated acceptance is
never used as a certificate for the combined action.
"""
import time

import numpy as np

from .advantage_gate_v18 import action_then_baseline, baseline_suffix
from .cooperative_v15 import JointPolicyGate
from .disturbed_advantage_v19 import disturbed_suffix, disturbed_transition
from .fast_v8 import parameters
from .marl_v12 import project_joint
from .merge_v4 import MergeConfig
from .robust_v9 import robust_certificate, validate
from .stress_prediction_v17 import fixed_stress_bank


def joint_compositions(state, previous, raw_proposal, baseline_command, packed):
    """At most seven nonbaseline commands; identical projections count once."""
    candidates = []
    seen = [np.asarray(baseline_command)]
    for code in range(1, 8):
        mask = np.array([bool(code & (1 << i)) for i in range(3)])
        proposal = np.where(mask, raw_proposal, baseline_command)
        command, vertex, valid = project_joint(state, previous, proposal, packed, .5)
        if not valid:
            raise RuntimeError('Joint composition outside physical contract')
        if any(np.array_equal(command, old) for old in seen):
            continue
        seen.append(command.copy())
        candidates.append((code, command.copy(), vertex))
    return candidates


class JointPortfolioGate:
    """Highest nominal gain among jointly feasible complete action candidates.

    max_failed_scenarios=0 retains the finite-bank pathwise guarantee. A positive
    budget is an uncalibrated development candidate and emits NO such guarantee.
    It needs complete-trajectory, actual-adoption conditional risk certification.
    """
    def __init__(self, model, max_failed_scenarios=0):
        if type(max_failed_scenarios) is not int or not 0 <= max_failed_scenarios <= 16:
            raise ValueError('Integer scenario-failure budget 0..16 required')
        self.c = MergeConfig()
        self.packed = parameters(self.c)
        self.primary = JointPolicyGate(model)
        self.step = 0
        self.names, self.tapes = fixed_stress_bank()
        self.max_failed_scenarios = max_failed_scenarios

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, .5)
        if self.step >= round(self.c.horizon / self.c.decision_dt):
            raise ValueError('Controller finite horizon exhausted')
        base, baseline_command, transitions, internal = baseline_suffix(x, prev, self.step, self.packed)
        self.primary.step = self.step
        _, actor_info = self.primary.select(x, prev)
        candidates = joint_compositions(x, prev, np.array(actor_info['proposal']), baseline_command, self.packed)
        nominal_candidates = []
        for code, command, vertex in candidates:
            future, count, work = action_then_baseline(x, prev, self.step, command, self.packed)
            transitions += count
            internal += work
            difference = future - base
            if np.all(difference <= 1e-10) and difference[:, 0].mean() < -1e-9:
                nominal_candidates.append((float(difference[:, 0].mean()), code, command, difference, vertex))
        nominal_candidates.sort(key=lambda row: (row[0], row[1]))
        base_cache = {}
        command = baseline_command
        selected_code = 0
        evidence = []
        winner_differences = []
        for _, code, candidate, nominal_difference, vertex in nominal_candidates:
            differences = [nominal_difference.tolist()]
            failed = 0
            for index in range(1, len(self.tapes)):
                tape = self.tapes[index]
                if index not in base_cache:
                    value, count, work = disturbed_suffix(x, prev, self.step, tape, self.packed)
                    transitions += count
                    internal += work
                    base_cache[index] = value
                following = x.copy()
                disturbed_transition(following, candidate, self.step, tape, self.packed)
                value, count, work = disturbed_suffix(following, candidate, self.step + 1, tape, self.packed)
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
            evidence.append(dict(composition=code, command=candidate.tolist(),
                                 checked_scenarios=len(differences), failed_scenarios=failed,
                                 differences=differences))
            if failed <= self.max_failed_scenarios:
                command = candidate
                selected_code = code
                winner_differences = differences
                break
        self.step += 1
        accepted = selected_code != 0
        if not robust_certificate(x, prev, command, self.packed, .5):
            raise RuntimeError('Final composition failed physical certificate')
        return command, dict(
            in_set=True, certified=True, intervention=float(not accepted),
            nominal_joint_action_admitted=accepted, portfolio_joint_action_admitted=accepted,
            composition=selected_code, candidate_count=len(candidates),
            nominal_candidate_count=len(nominal_candidates),
            candidate_evidence=evidence, selected_bank_differences=winner_differences,
            max_failed_scenarios=self.max_failed_scenarios,
            gate_ms=(time.perf_counter() - started) * 1000.,
            performance_forecast_transitions=transitions,
            performance_model_transitions=internal,
            performance_claim=('Finite-bank componentwise pathwise guarantee' if self.max_failed_scenarios == 0
                               else 'Nominal guarantee only; relaxed bank is NOT a risk certificate'))
