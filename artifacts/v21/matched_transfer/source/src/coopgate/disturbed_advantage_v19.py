"""Joint baseline-relative admission against a finite disturbance scenario bank.

For EACH fixed tape, the potential is the cost of the SAME feedback baseline
under the remaining tape. Checking every component of every Bellman difference
therefore implies pathwise performance improvement if the actual tape is in the
bank. No claim is made for arbitrary bounded errors, or for an estimated risk.
Statistical validation of out-of-bank closed-loop outcomes is a separate step.
"""
import time

import numpy as np
from numba import njit

from .advantage_gate_v18 import action_then_baseline, baseline_suffix
from .cooperative_v15 import JointPolicyGate
from .fast_v8 import parameters, transition_values
from .joint_planner_v15 import joint_rollout
from .merge_v4 import MergeConfig
from .robust_v9 import robust_certificate, validate
from .stress_prediction_v17 import fixed_stress_bank


@njit(cache=True, fastmath=False)
def disturbed_transition(state, command, step, tape, packed):
    """Execute a forecast tape, never an error tape supplied by the simulator."""
    substeps = round(packed[1] / packed[0])
    for sub in range(substeps):
        p0, p1, p2 = state[0, 0], state[1, 0], state[2, 0]
        for i in range(3):
            acceleration = command[i] + tape[step * substeps + sub, i]
            state[i, 1] = min(packed[3], max(0., state[i, 1] + acceleration * packed[0]))
            state[i, 0] += state[i, 1] * packed[0]
        if not transition_values(p0, p1, p2, state[0, 0], state[1, 0], state[2, 0], packed):
            raise RuntimeError('Unsafe disturbance-bank forecast transition')


@njit(cache=True, fastmath=False)
def disturbed_suffix(state, previous, step, tape, packed):
    x = state.copy()
    prev = previous.copy()
    values = np.zeros((3, 2))
    transitions = 0
    internal = 0
    for k in range(step, round(packed[2] / packed[1])):
        active = x[:, 0] < packed[11]
        if not np.any(active):
            break
        command, _, _, count = joint_rollout(x, prev, k, packed)
        internal += count
        for i in range(3):
            if active[i]:
                values[i, 0] += packed[1] + .05 * abs(command[i] - prev[i])
                values[i, 1] += packed[1]
        disturbed_transition(x, command, k, tape, packed)
        prev = command
        transitions += 1
    return values, transitions, internal


@njit(cache=True, fastmath=False)
def disturbed_action_difference(state, previous, step, command, tape, packed):
    base, transitions, internal = disturbed_suffix(state, previous, step, tape, packed)
    following = state.copy()
    disturbed_transition(following, command, step, tape, packed)
    candidate, count, work = disturbed_suffix(following, command, step + 1, tape, packed)
    for i in range(3):
        if state[i, 0] < packed[11]:
            candidate[i, 0] += packed[1] + .05 * abs(command[i] - previous[i])
            candidate[i, 1] += packed[1]
    return candidate - base, transitions + count + 1, internal + work


class DisturbanceBankGate:
    """A finite-scenario performance gate sharing V18's actor and physical shield.

    Early rejection evaluates the nominal condition first, then fixed scenarios.
    This is an exact AND of the full bank, not data-dependent scenario selection.
    There is no simulator seed, disturbance mode, future error, or outcome input.
    """
    def __init__(self, model, bank='bias_pulse'):
        if bank not in ('bias', 'bias_pulse'):
            raise ValueError('Unknown fixed disturbance bank')
        self.c = MergeConfig()
        self.packed = parameters(self.c)
        self.primary = JointPolicyGate(model)
        self.step = 0
        names, tapes = fixed_stress_bank()
        count = 9 if bank == 'bias' else 17
        self.names = names[:count]
        self.tapes = tapes[:count].copy()
        self.bank = bank

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        x, prev, _ = validate(state, previous, self.c, .5)
        if self.step >= round(self.c.horizon / self.c.decision_dt):
            raise ValueError('Controller finite horizon exhausted')
        base, baseline_command, transitions, internal = baseline_suffix(x, prev, self.step, self.packed)
        self.primary.step = self.step
        proposed, _ = self.primary.select(x, prev)
        accepted = False
        checked = []
        differences = []
        if not np.array_equal(proposed, baseline_command):
            candidate, count, work = action_then_baseline(x, prev, self.step, proposed, self.packed)
            transitions += count
            internal += work
            difference = candidate - base
            checked.append(self.names[0])
            differences.append(difference.tolist())
            accepted = bool(np.all(difference <= 1e-10) and difference[:, 0].mean() < -1e-9)
            if accepted:
                for index in range(1, len(self.tapes)):
                    difference, count, work = disturbed_action_difference(
                        x, prev, self.step, proposed, self.tapes[index], self.packed)
                    transitions += count
                    internal += work
                    checked.append(self.names[index])
                    differences.append(difference.tolist())
                    if np.any(difference > 1e-10):
                        accepted = False
                        break
        command = proposed if accepted else baseline_command
        self.step += 1
        if not robust_certificate(x, prev, command, self.packed, .5):
            raise RuntimeError('Chosen command failed physical certificate')
        return command, dict(
            in_set=True, certified=True, intervention=float(not accepted),
            nominal_joint_action_admitted=accepted, bank_joint_action_admitted=accepted,
            bank=self.bank, bank_scenarios_checked=checked,
            bank_differences=differences,
            gate_ms=(time.perf_counter() - started) * 1000.,
            performance_forecast_transitions=transitions,
            performance_model_transitions=internal,
            performance_claim='Componentwise pathwise guarantee only for declared finite bank; no out-of-bank risk certificate')
