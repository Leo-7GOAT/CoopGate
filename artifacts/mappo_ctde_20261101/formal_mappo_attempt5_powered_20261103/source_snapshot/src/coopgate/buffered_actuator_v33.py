"""Exact buffered physical checks inside the V32 actuator forecaster.

The baseline rollout, actuator recurrence and validated forecast wrapper have
identical syntax trees to their frozen originals. Only local physical-check
and rule-tail dependencies use the already verified reusable-buffer kernel.
The gate inherits every decision rule and overrides only its predictor.
"""
import numpy as np
from numba import njit
from .actuator_prediction_roundoff_v31 import plant_arrays
from .arrival_phase_v20 import crossing_time
from .buffered_baseline_v25 import project_joint,robust_certificate,team_rule_tail
from .fast_v8 import advance_one,transition_values
from .marl_v12 import LEVELS
from .merge_v4 import MergeConfig
from .online_actuator_gate_v32 import ReferenceActuatorGate
from .robust_v9 import validate
from .single_v12 import constant_proposal



@njit(cache=True,fastmath=False)
def joint_rollout(state,previous,step,c,sweeps=2):
    chosen,_,ok=project_joint(state,previous,constant_proposal(previous,c),c,.5)
    if not ok:raise RuntimeError('No certified baseline command')
    best,transitions=team_rule_tail(state,previous,step,c);base_value=best
    if np.all(state[:,0]>=c[11]):return chosen,best,base_value,transitions
    # Ramp first is a fixed ordering, not chosen from test outcomes.
    for _ in range(sweeps):
        improved=False
        for i in (1,0,2):
            if state[i,0]>=c[11]:continue
            center=chosen.copy()
            for level in LEVELS:
                if abs(level-previous[i])>c[6]*c[1]+1e-8:continue
                proposal=center.copy();proposal[i]=level
                for j in range(3):
                    if state[j,0]>=c[11]:proposal[j]=min(c[5],previous[j]+c[6]*c[1])
                command,_,valid=project_joint(state,previous,proposal,c,.5)
                if not valid:raise RuntimeError('No certificate for candidate')
                if np.array_equal(command,chosen):continue
                stage=0.
                for j in range(3):
                    if state[j,0]<c[11]:stage+=(c[1]+.05*abs(command[j]-previous[j]))/3.
                if stage>=best:continue
                following=state.copy()
                if not advance_one(following,command,c):raise RuntimeError('Unsafe planning transition')
                tail,count=team_rule_tail(following,command,step+1,c,best-stage)
                transitions+=1+count
                if stage+tail<best-1e-10:
                    chosen=command.copy();best=stage+tail;improved=True
        if not improved:break
    return chosen,best,base_value,transitions


@njit(cache=True,fastmath=False)
def forecast_kernel(state,previous,memory,step,tape,gain,drag,retention,packed,first_command,use_first,record_trace=False):
    """Same complete dynamics as V31, with no observed-trajectory corrections."""
    x,prev,m = state.copy(),previous.copy(),memory.copy()
    finish,variation,crossing = np.zeros(3),np.zeros(3),np.zeros(3)
    for i in range(3):
        if x[i,0] < packed[11]:
            finish[i] = packed[2]-step*packed[1]
            crossing[i] = np.nan
    substeps,steps,work,error,violations = round(packed[1]/packed[0]),0,0,0,0
    capacity = round(packed[2]/packed[1])-step if record_trace else 0
    states = np.empty((capacity,3,2))
    commands = np.empty((capacity,3))
    memories = np.empty((capacity,substeps,3))
    errors = np.empty((capacity,substeps,3))
    maximum,exceeded = 0.,0
    for k in range(step,round(packed[2]/packed[1])):
        active = x[:,0] < packed[11]
        if not np.any(active):
            break
        backup = np.maximum(prev-packed[6]*packed[1],-packed[4])
        if not robust_certificate(x,prev,backup,packed,.5):
            error = 1
            break
        if use_first and k == step:
            command = first_command.copy()
        else:
            try:
                command,_,_,count = joint_rollout(x,prev,k,packed)
                work += count
            except Exception:
                error = 2
                break
        if not robust_certificate(x,prev,command,packed,.5):
            error = 3
            break
        for i in range(3):
            if active[i]:
                variation[i] += abs(command[i]-prev[i])
        for sub in range(substeps):
            before = x[:,0].copy()
            sub_error = 0.
            for i in range(3):
                m[i] = retention[i]*m[i]+(1.-retention[i])*command[i]
                applied = gain[i]*m[i]-drag[i]*x[i,1]**2+tape[k*substeps+sub,i]
                if record_trace:
                    memories[steps,sub,i] = m[i]
                    errors[steps,sub,i] = applied-command[i]
                sub_error = max(sub_error,abs(applied-command[i]))
                x[i,1] = min(packed[3],max(0.,x[i,1]+packed[0]*applied))
                x[i,0] += packed[0]*x[i,1]
                if np.isnan(crossing[i]):
                    crossing[i] = crossing_time(before[i],x[i,0],packed[11],
                        (k-step)*packed[1]+sub*packed[0],packed[0])
            maximum = max(maximum,sub_error)
            exceeded += int(sub_error > .5+1e-9)
            violations += int(not transition_values(before[0],before[1],before[2],x[0,0],x[1,0],x[2,0],packed))
        for i in range(3):
            if active[i] and x[i,0] >= packed[11]:
                finish[i] = (k-step+1)*packed[1]
        prev = command.copy()
        if record_trace:
            states[steps] = x
            commands[steps] = command
        steps += 1
    values = np.zeros((3,2))
    for i in range(3):
        values[i,0],values[i,1] = finish[i]+.05*variation[i],finish[i]
    complete = bool(np.all(x[:,0] >= packed[11]) and error == 0 and violations == 0)
    return values,crossing,complete,error,steps,work,maximum,exceeded,x,m,violations,states[:steps],commands[:steps],memories[:steps],errors[:steps]


def forecast(state,previous,memory,step,tape,plant,first_command=None,config=None,record_trace=False):
    c = config or MergeConfig()
    x,prev,packed = validate(state,previous,c,.5)
    m = np.asarray(memory,dtype=float)
    if (m.shape != (3,) or not np.isfinite(m).all()
            or np.any(m < -c.brake-8*np.spacing(c.brake))
            or np.any(m > c.accel+8*np.spacing(c.brake))):
        raise ValueError('Finite bounded actuator memory required')
    if type(step) is not int or not 0 <= step <= round(c.horizon/c.decision_dt):
        raise ValueError('Original decision clock required')
    path = np.asarray(tape,dtype=float)
    if path.shape != (round(c.horizon/c.dt),3) or not np.isfinite(path).all():
        raise ValueError('Complete finite external path required')
    first = np.zeros(3) if first_command is None else np.asarray(first_command,dtype=float)
    if (first.shape != (3,) or not np.isfinite(first).all() or
            (first_command is not None and (np.any(first < -c.brake) or np.any(first > c.accel)
             or np.any(np.abs(first-prev) > c.jerk*c.decision_dt+1e-10)))):
        raise ValueError('Finite physically bounded first command required')
    gain,drag,retention = plant_arrays(plant,c.dt)
    result = forecast_kernel(x,prev,m,step,path,gain,drag,retention,packed,first,first_command is not None,record_trace)
    values,crossing,complete,error,steps,work,maximum,exceeded,final,memory,violations,states,commands,memories,errors = result
    return dict(values=values.tolist(),crossing=[float(v) if np.isfinite(v) else None for v in crossing],
        complete=bool(complete),error_code=int(error),error_step=step+int(steps) if error else None,
        steps=int(steps),internal_transitions=int(work),
        max_actual_error=float(maximum),error_contract_exceeded_substeps=int(exceeded),
        final_state=final.tolist(),final_actuator=memory.tolist(),route_violations=int(violations),
        trace=[dict(t=(step+i+1)*c.decision_dt,x=states[i].tolist(),command=commands[i].tolist(),
            actuator_states=memories[i].tolist(),disturbance=errors[i].tolist()) for i in range(steps)] if record_trace else [])


class BufferedActuatorGate(ReferenceActuatorGate):
    def predict(self,x,prev,command,path):
        gain,drag,retention = self.arrays
        first = np.zeros(3) if command is None else command
        return forecast_kernel(x,prev,self.memory,self.step,self.tapes[path],gain,drag,retention,
            self.packed,first,command is not None)
