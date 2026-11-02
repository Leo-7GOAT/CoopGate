"""State-reset curriculum; PPO still trains only on freshly sampled rollouts.

This changes the training initial-state law. It is not off-policy action replay,
expert imitation, held-out-state reuse, or a change to the evaluation scenario.
"""
import numpy as np

from .mappo_ctde import PPOVector
from .robust_v9 import robust_certificate


class ResetCurriculumVector(PPOVector):
    def __init__(self, path, n=64, seed=1, stage='single', probability=.5):
        if stage != 'single' or probability != .5:
            raise ValueError('Development D fixes single stage and 50% reset probability')
        with np.load(path, allow_pickle=False) as data:
            self.pool_x=np.array(data['state'], dtype=np.float64, copy=True)
            self.pool_previous=np.array(data['previous'], dtype=np.float64, copy=True)
            self.pool_steps=np.array(data['step'], dtype=int, copy=True)
        if (self.pool_x.shape != (len(self.pool_steps),3,2)
                or self.pool_previous.shape != (len(self.pool_steps),3)
                or not np.isfinite(self.pool_x).all() or not np.isfinite(self.pool_previous).all()
                or np.any(self.pool_steps<1) or np.any(self.pool_steps>=50)):
            raise ValueError('Invalid finite reset pool')
        self.probability=probability
        self.curriculum_resets=0
        super().__init__(n,seed,stage)

    def reset(self, ids):
        super().reset(ids)
        for i in ids:
            if self.rng.random() >= self.probability:
                continue
            index=int(self.rng.integers(len(self.pool_steps)))
            x=self.pool_x[index]; previous=self.pool_previous[index]
            backup=np.maximum(previous-2.5,-5.)
            if not robust_certificate(x,previous,backup,self.packed,.5):
                raise RuntimeError('Stored curriculum state outside common recoverable set')
            self.x[i]=x; self.previous[i]=previous; self.steps[i]=self.pool_steps[index]
            self.curriculum_resets+=1
            # time/variation remain zero: metrics are costs of this suffix,
            # not costs of an original-start evaluation episode.
