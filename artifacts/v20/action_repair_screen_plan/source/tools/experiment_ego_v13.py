"""SUMO accounting that stops each vehicle's variation at its own arrival."""
import time
import numpy as np
from coopgate.confidence_v12 import conflict_group


def run_episode(env,seed,regime,mode,factory,trace=False):
    env.reset_disturbed(seed,regime,mode,.5);initial=env.x.copy()
    start=time.perf_counter();env.gate=factory(initial);setup_ms=(time.perf_counter()-start)*1000
    while True:
        _,_,done,truncated,_=env.step([4]*3)
        if done or truncated:break
    result=env.metrics();previous=np.zeros(3);state=initial.copy();variation=np.zeros(3)
    for item in env.trace:
        command=np.array(item['command']);variation+=(state[:,0]<env.c.goal)*np.abs(command-previous)
        previous=command;state=np.array(item['x'])
    finish=np.nan_to_num(env.finish,nan=env.c.horizon)
    result.update(group=conflict_group(initial),ego_time=float(finish[1]),ego_variation=float(variation[1]),
        ego_cost=float(finish[1]+.05*variation[1]),other_time=float(finish[[0,2]].mean()),
        per_vehicle_finish=finish.tolist(),per_vehicle_variation=variation.tolist(),
        setup_ms=setup_ms,controller_ms=[t['gate_ms'] for t in env.trace],
        learned_tail_fraction=float(np.mean([t.get('learned_tail_improved',False) for t in env.trace])))
    if hasattr(env.gate,'accept'):
        result.update(accepted=env.gate.accept,gate_reason=env.gate.reason,predicted=env.gate.predicted.tolist(),
            upper=env.gate.upper.tolist(),fault_fallback=env.gate.fault_fallback,admission_ms=env.gate.admission_ms)
    return result,env.trace if trace else None,initial
