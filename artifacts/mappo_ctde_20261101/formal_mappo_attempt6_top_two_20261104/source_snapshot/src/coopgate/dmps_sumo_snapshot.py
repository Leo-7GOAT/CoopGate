"""Actual SUMO snapshot bridge for upstream DMPS interface validation.

The planner forecasts the cloned true simulator here, including actuator state
and exogenous tape. This privileged diagnostic is NOT a fair deployable DMPS
baseline or a learned DMPS implementation. No CoopGate gate is applied inside
the bridge. Only the declared amplitude/slew actuator map clips commands.
"""
from copy import deepcopy
from dataclasses import dataclass
from types import FunctionType
import uuid

import numpy as np
import traci

from .disturbance_v9 import disturbance_tape
from .matched_shift_v21 import MatchedShiftedMergeEnv
from .merge_v4 import MergeEnv, admissible, backup


OPTIONS = ['--save-state.rng', 'true', '--save-state.precision', '17']


class _Connection:
    def __init__(self, original):
        self.original = original

    def load(self, args):
        return self.original.load([*args, *OPTIONS])

    def __getattr__(self, name):
        return getattr(self.original, name)


class _TraCI:
    def start(self, command, **kwargs):
        return traci.start([*command, *OPTIONS], **kwargs)

    def getConnection(self, label):
        return _Connection(traci.getConnection(label))


class _ConfiguredReset(MergeEnv):
    def reset(self, seed=0, scenario='nominal'):
        # Same original reset body with private dependency binding. Do not patch
        # module globals used by live frozen-study workers.
        original = MergeEnv.reset
        function = FunctionType(original.__code__, {**original.__globals__, 'traci': _TraCI()},
                                argdefs=original.__defaults__)
        return function(self, seed, scenario)


class _ConfiguredMatched(MatchedShiftedMergeEnv, _ConfiguredReset):
    # DisturbedMergeEnv.super().reset resolves to _ConfiguredReset via this MRO.
    pass


class _CommandGate:
    def __init__(self):
        self.command = np.zeros(3)

    def select(self, state, previous, ignored):
        return self.command.copy(), dict(in_set=False, certified=False, intervention=0., gate_ms=0.,
            proof_scope='No CoopGate certificate; upstream DMPS snapshot diagnostic')


class _Space:
    def __init__(self, shape, rng=None):
        self.shape, self.rng = shape, rng

    def sample(self):
        if self.rng is None:
            raise TypeError('Observation space sampling is not supported')
        return self.rng.uniform(-1, 1, self.shape)


@dataclass
class Snapshot:
    owner: str
    generation: int
    native: bytes
    host: dict
    sampler_state: dict
    config: dict
    simulator_time: float


class DMPSSumoSnapshotEnv:
    """Old Gym four-value interface, with full simulator + host state snapshots."""
    _EXCLUDED = frozenset(('conn', 'tmp', 'd', 'c'))

    def __init__(self, seed, scenario='nominal', plant='nominal', disturbance='clean'):
        self.seed, self.scenario, self.plant, self.disturbance = seed, scenario, plant, disturbance
        self.raw = _ConfiguredMatched(plant=plant)
        self.rng = np.random.default_rng(seed)
        self.action_space = _Space((3,), self.rng)
        self.observation_space = _Space((10,))
        self.num_steps = round(self.raw.c.horizon/self.raw.c.decision_dt)
        self.owner, self.generation = uuid.uuid4().hex, 0
        self.snapshots_saved = self.snapshots_loaded = self.simulator_steps = 0
        self.recovery_tail_seconds = 10.

    def reset(self):
        self.generation += 1
        self.raw.reset_disturbed(self.seed, self.scenario, self.disturbance, .5)
        self.raw.gate = _CommandGate()
        old = self.raw.tape
        tape = disturbance_tape(self.seed, self.disturbance, self.raw.innovation_radius,
                               round((self.raw.c.horizon+self.recovery_tail_seconds)/self.raw.c.dt))
        np.testing.assert_array_equal(tape[:len(old)], old)
        self.raw.tape = tape
        self.rng.bit_generator.state = np.random.default_rng(self.seed).bit_generator.state
        return self.obs()

    def obs(self):
        raw = self.raw
        return np.append(np.column_stack((raw.x[:, 0]/150, raw.x[:, 1]/20, raw.prev/5)).ravel(),
                         raw.t/raw.c.horizon).astype(np.float32)

    def action_to_command(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1+1e-12):
            raise ValueError('Finite normalized three-vehicle action in [-1,1] required')
        return admissible(-1.25+3.75*np.clip(a, -1, 1), self.raw.prev, self.raw.c)

    def backup(self):
        return np.clip((backup(self.raw.prev, self.raw.c)+1.25)/3.75, -1, 1)

    def safe(self):
        return bool(not self.raw.collision and self.raw.violations == 0 and not self.raw.abort_reason
                    and self.raw.t < self.raw.c.horizon+self.recovery_tail_seconds)

    def stable(self):
        # Stationarity under continued braking and any external innovation in
        # +/-radius for this known plant; not a robust unknown-plant certificate.
        force_upper = np.asarray(self.raw.plant['gain'])*self.raw.actuator+self.raw.innovation_radius
        return bool(self.safe() and np.all(self.raw.x[:, 1] <= 1e-10)
                    and np.all(force_upper <= 0) and np.all(self.raw.prev <= 0))

    def step(self, action):
        raw = self.raw
        if not self.safe():
            return self.obs(), 0., True, dict(unsafe=True, recovery_timeout=True)
        command = self.action_to_command(action)
        active, previous = raw.x[:, 0] < raw.c.goal, raw.prev.copy()
        raw.gate.command = command
        _, _, terminated, truncated, info = raw.step([2]*3)
        self.simulator_steps += 1
        reward = -float(np.mean(active*(raw.c.decision_dt+.05*np.abs(command-previous))))
        return self.obs(), reward, bool(terminated or truncated or not self.safe()), info

    def get_state(self):
        path = self.raw.d/'dmps-snapshot.xml'
        self.raw.conn.simulation.saveState(str(path))
        self.snapshots_saved += 1
        return Snapshot(self.owner, self.generation, path.read_bytes(),
            deepcopy({k: v for k, v in self.raw.__dict__.items() if k not in self._EXCLUDED}),
            deepcopy(self.rng.bit_generator.state), self.raw.c.dict(), self.raw.conn.simulation.getTime())

    def set_state(self, state):
        if not isinstance(state, Snapshot) or (state.owner, state.generation) != (self.owner, self.generation):
            raise ValueError('Snapshot belongs to another simulator or reset generation')
        if state.config != self.raw.c.dict():
            raise ValueError('Physical configuration changed after snapshot')
        path = self.raw.d/'dmps-snapshot.xml'
        path.write_bytes(state.native)
        self.raw.conn.simulation.loadState(str(path))
        for name in list(self.raw.__dict__):
            if name not in self._EXCLUDED:
                del self.raw.__dict__[name]
        self.raw.__dict__.update(deepcopy(state.host))
        self.rng.bit_generator.state = deepcopy(state.sampler_state)
        self.snapshots_loaded += 1
        np.testing.assert_allclose(self.raw.read(), self.raw.x, atol=1e-8, rtol=0)
        if self.raw.conn.simulation.getTime() != state.simulator_time:
            raise ValueError('SUMO clock was not restored')

    def close(self):
        self.raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
