"""Nominal whole-profile performance prediction, with no future error tape."""
import numpy as np
from .merge_v4 import MergeConfig
from .fast_v8 import parameters,advance_one
from .robust_v9 import robust_certificate,validate


def predict_profile(initial,factory,config=None):
    c=config or MergeConfig();packed=parameters(c)
    state=np.array(initial,dtype=float,copy=True)
    if state.shape!=(3,2) or not np.isfinite(state).all():raise ValueError('Finite joint initial state required')
    previous=np.zeros(3);arrival=np.zeros(3);variation=np.zeros(3);steps=0;internal=0
    controller=factory(state.copy())
    for step in range(round(c.horizon/c.decision_dt)):
        active=state[:,0]<c.goal
        if not active.any():break
        validate(state,previous,c,.5)
        command,info=controller.select(state.copy(),previous.copy())
        command=np.asarray(command,dtype=float)
        if command.shape!=(3,) or not np.isfinite(command).all():raise ValueError('Prediction controller failed')
        if np.any(command < -c.brake-1e-9) or np.any(command > c.accel+1e-9):raise ValueError('Prediction acceleration limit')
        if np.any(np.abs(command-previous)>c.jerk*c.decision_dt+1e-9):raise ValueError('Prediction slew limit')
        if not robust_certificate(state,previous,command,packed,.5):raise ValueError('Prediction physical certificate failed')
        arrival+=active*c.decision_dt;variation+=active*np.abs(command-previous)
        if not advance_one(state,command,packed):raise RuntimeError('Nominal forecast left the physical contract')
        previous=command;steps+=1;internal+=int(info.get('performance_model_transitions',0))
    result=np.column_stack((arrival+.05*variation,arrival))
    return result,dict(completed=bool(np.all(state[:,0]>=c.goal)),forecast_transitions=steps,
                      baseline_planning_internal_transitions=internal)
