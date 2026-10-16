"""Actual SUMO execution with complete per-episode accounting."""
import time
import numpy as np
from coopgate.confidence_v12 import conflict_group


def run_episode(env,seed,regime,mode,factory,trace=False):
    env.reset_disturbed(seed,regime,mode,.5);initial=env.x.copy()
    started=time.perf_counter();env.gate=factory(initial);setup_ms=(time.perf_counter()-started)*1000
    while True:
        _,_,done,truncated,_=env.step([4]*3)
        if done or truncated:break
    commands=np.array([np.zeros(3).tolist()]+[t['command'] for t in env.trace])
    changes=np.abs(np.diff(commands,axis=0));result=env.metrics()
    result.update(group=conflict_group(initial),variation=float(changes.sum()),
        ego_variation=float(changes[:,1].sum()) if len(changes) else 0.,
        ego_finish_s=float(np.nan_to_num(env.finish[1],nan=env.c.horizon)),
        controller_ms=[t['gate_ms'] for t in env.trace],setup_ms=setup_ms,
        steps=len(env.trace),joint_acceptance=float(np.mean([t.get('joint_proposal_accepted',False) for t in env.trace])) if env.trace else 0.)
    if hasattr(env.gate,'accept'):
        result.update(accepted=env.gate.accept,gate_reason=env.gate.reason,predicted_delta_cost=env.gate.predicted_delta,
            upper_delta_cost=env.gate.upper_delta,admission_ms=env.gate.admission_ms)
    return result,env.trace if trace else None,initial
