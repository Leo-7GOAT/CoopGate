"""Offline common-action adoption, without changing earlier declared events."""
from numbers import Integral

import numpy as np

from .joint_planner_v15 import JointRolloutGate


def changed_active_commands(state, command, baseline, goal=60., tolerance=1e-9):
    state, command, baseline = (np.asarray(x,dtype=np.float64) for x in (state,command,baseline))
    if (state.shape!=(3,2) or command.shape!=(3,) or baseline.shape!=(3,) or
            not all(np.isfinite(x).all() for x in (state,command,baseline))):
        raise ValueError('Three finite vehicle states and commands required')
    if not np.isfinite(goal) or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Finite goal and nonnegative comparison tolerance required')
    return (state[:,0]<goal) & (np.abs(command-baseline)>tolerance)


class BaselineReplayCache:
    """Exact memoization within one initial, keyed by all baseline inputs.

    JointRolloutGate is deterministic in measured state, previous command,
    decision step and fixed configuration. Returned arrays are copied, so one
    caller cannot alter later cache hits. This cache is never used for control.
    """
    def __init__(self, baseline=None):
        self.baseline = JointRolloutGate() if baseline is None else baseline
        self.cache = {}
        self.evaluations = self.hits = 0

    def command(self, state, previous, step):
        state, previous = (np.ascontiguousarray(v,dtype=np.float64) for v in (state,previous))
        if (state.shape!=(3,2) or previous.shape!=(3,) or
                not np.isfinite(state).all() or not np.isfinite(previous).all() or
                isinstance(step,(bool,np.bool_)) or not isinstance(step,Integral) or step<0):
            raise ValueError('Finite measured state, previous command and nonnegative integer step required')
        key=(int(step),state.tobytes(),previous.tobytes())
        if key not in self.cache:
            self.baseline.step=int(step)
            command,_=self.baseline.select(state.copy(),previous.copy())
            command=np.asarray(command,dtype=np.float64)
            if command.shape!=(3,) or not np.isfinite(command).all():
                raise ValueError('Invalid reconstructed baseline command')
            self.cache[key]=command.copy()
            self.evaluations+=1
        else:
            self.hits+=1
        return self.cache[key].copy()


def audit_trace(initial, trace, cache):
    state,previous=np.asarray(initial,dtype=np.float64).copy(),np.zeros(3)
    records=[]
    for step,item in enumerate(trace):
        base=cache.command(state,previous,step)
        command=np.asarray(item['command'],dtype=np.float64)
        changed=changed_active_commands(state,command,base)
        records.append(dict(step=step,baseline_command=base.tolist(),
                            changed_active_vehicles=changed.tolist(),adopted=bool(changed.any())))
        state,previous=np.asarray(item['x'],dtype=np.float64),command
    return records
