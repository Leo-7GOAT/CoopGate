"""Nominal SUMO shadow rollout initialized only from observed initial kinematics.

The reset deliberately ignores external scenario/seed labels and fixes the
engine seed. This is an initial-time predictor for the declared three-route
merge task, not an arbitrary simulator snapshot/hidden-state clone.
"""
import numpy as np

from .disturbance_v9 import DisturbedMergeEnv
from .fast_v8 import parameters
from .merge_v4 import BRANCH, MergeConfig
from .robust_v9 import robust_certificate, validate


class ObservableInitialMergeEnv(DisturbedMergeEnv):
    def __init__(self, initial, **kwargs):
        initial = np.asarray(initial, dtype=float)
        validate(initial, np.zeros(3), MergeConfig(), .5)
        if np.any(initial[:, 0] >= MergeConfig().zone_start):
            raise ValueError('Initial shadow forecasts require all vehicles before conflict entry')
        self.observed_initial = initial.copy()
        self.observed_initial.flags.writeable = False
        super().__init__(**kwargs)

    def reset_disturbed(self, seed, scenario, mode, radius):
        if mode != 'clean' or radius != .5:
            raise ValueError('A nominal shadow predictor cannot read actual disturbance modes')
        # Engine seed is constant and only concerns construction. No actual
        # episode ID, future innovation or scenario label enters the forecast.
        super().reset_disturbed(19367, 'dense', 'clean', .5)
        target = self.observed_initial
        for i, ident in enumerate(self.ids):
            route = 'a' if BRANCH[i] == 0 else 'b'
            length = self.conn.lane.getLength(route + '_0')
            self.conn.vehicle.moveTo(ident, route + '_0', float(length + target[i, 0] - target[i, 1] * self.c.dt))
            self.conn.vehicle.setSpeed(ident, float(target[i, 1]))
        self.conn.simulationStep()
        self.offsets = []
        for i, ident in enumerate(self.ids):
            route = 'a' if BRANCH[i] == 0 else 'b'
            position = self.conn.vehicle.getLanePosition(ident) - self.conn.lane.getLength(route + '_0')
            self.offsets.append(position - self.conn.vehicle.getDistance(ident))
        self.x = self.read()
        np.testing.assert_allclose(self.x, target, atol=1e-8, rtol=0)
        if not robust_certificate(self.x, np.zeros(3), np.full(3, -2.5), parameters(self.c), .5):
            raise ValueError('Observed initial is outside the declared recoverable set')
        self.initial = self.x.copy()
        self.scenario = 'observed_state_only_nominal_forecast'
        return self.obs()
