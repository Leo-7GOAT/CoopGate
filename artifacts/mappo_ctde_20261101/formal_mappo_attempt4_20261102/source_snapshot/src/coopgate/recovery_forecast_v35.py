"""Full continuation of the actuator-aware recovery baseline.

The predictor carries both a reference actuator state for physical prediction
and a command-history interval for the baseline's physical stopping check.
It never replaces that interval by the reference point. This module is not
loaded by either existing prospective study.
"""
import numpy as np
from numba import njit

from .actuator_prediction_roundoff_v31 import plant_arrays
from .actuator_recovery_v34 import memory_step, stopping_certificate, validate_inputs
from .arrival_phase_v20 import crossing_time
from .buffered_baseline_v25 import deduplicated_joint_rollout
from .fast_v8 import parameters, transition_values
from .merge_v4 import MergeConfig


@njit(cache=True, fastmath=False)
def recovery_command(state, previous, step, memory_lower, memory_upper, bounds, packed):
    """Same command as V34, with explicit failure and recovery indicators."""
    proposal_available = True
    proposal = np.zeros(3)
    planning_transitions = 0
    certificate_substeps = 0
    try:
        proposal, _, _, planning_transitions = deduplicated_joint_rollout(state,previous,step,packed)
    except Exception:
        proposal_available = False
    if proposal_available:
        result = stopping_certificate(state,previous,proposal,memory_lower,memory_upper,bounds,packed)
        certificate_substeps += result[2]
        if result[0]:
            return proposal,True,False,False,planning_transitions,certificate_substeps
    backup = np.maximum(previous-packed[6]*packed[1],-packed[4])
    result = stopping_certificate(state,previous,backup,memory_lower,memory_upper,bounds,packed)
    certificate_substeps += result[2]
    return backup,bool(result[0]),True,not proposal_available,planning_transitions,certificate_substeps


@njit(cache=True, fastmath=False)
def forecast_kernel(state,previous,memory,observer_lower,observer_upper,step,tape,
                    gain,drag,retention,bounds,packed,first_command,use_first,record_trace=False):
    x,prev,point = state.copy(),previous.copy(),memory.copy()
    mlo,mhi = observer_lower.copy(),observer_upper.copy()
    finish,variation,crossing = np.zeros(3),np.zeros(3),np.zeros(3)
    for i in range(3):
        if x[i,0] < packed[11]:
            finish[i] = packed[2]-step*packed[1]
            crossing[i] = np.nan
    substeps = round(packed[1]/packed[0])
    capacity = round(packed[2]/packed[1])-step if record_trace else 0
    states = np.empty((capacity,3,2))
    commands = np.empty((capacity,3))
    memories = np.empty((capacity,substeps,3))
    errors = np.empty((capacity,substeps,3))
    observer_los,observer_his = np.empty((capacity,3)),np.empty((capacity,3))
    recovered = np.zeros(capacity,np.bool_)
    steps = work = checked_substeps = error = violations = exceeded = 0
    planning_work_complete = True
    maximum = 0.
    for k in range(step,round(packed[2]/packed[1])):
        active = x[:,0] < packed[11]
        if not np.any(active):
            break
        recovery = False
        if use_first and k == step:
            command = first_command.copy()
        else:
            command,valid,recovery,proposal_failed,count,checks = recovery_command(x,prev,k,mlo,mhi,bounds,packed)
            planning_work_complete = planning_work_complete and not proposal_failed
            work += count
            checked_substeps += checks
            if not valid:
                error = 2
                break
        check = stopping_certificate(x,prev,command,mlo,mhi,bounds,packed)
        checked_substeps += check[2]
        if not check[0]:
            error = 3
            break
        if record_trace:
            observer_los[steps],observer_his[steps] = mlo,mhi
            recovered[steps] = recovery
        for i in range(3):
            if active[i]:
                variation[i] += abs(command[i]-prev[i])
        for sub in range(substeps):
            before = x[:,0].copy()
            mlo,mhi = memory_step(mlo,mhi,command,bounds)
            current_error = 0.
            for i in range(3):
                point[i] = retention[i]*point[i]+(1.-retention[i])*command[i]
                applied = gain[i]*point[i]-drag[i]*x[i,1]**2+tape[k*substeps+sub,i]
                if record_trace:
                    memories[steps,sub,i] = point[i]
                    errors[steps,sub,i] = applied-command[i]
                current_error = max(current_error,abs(applied-command[i]))
                x[i,1] = min(packed[3],max(0.,x[i,1]+packed[0]*applied))
                x[i,0] += packed[0]*x[i,1]
                if np.isnan(crossing[i]):
                    crossing[i] = crossing_time(before[i],x[i,0],packed[11],
                        (k-step)*packed[1]+sub*packed[0],packed[0])
            maximum = max(maximum,current_error)
            exceeded += int(current_error > .5+1e-9)
            violations += int(not transition_values(before[0],before[1],before[2],x[0,0],x[1,0],x[2,0],packed))
        for i in range(3):
            if active[i] and x[i,0] >= packed[11]:
                finish[i] = (k-step+1)*packed[1]
        prev = command.copy()
        if record_trace:
            states[steps],commands[steps] = x,command
        steps += 1
    values = np.zeros((3,2))
    for i in range(3):
        values[i,0],values[i,1] = finish[i]+.05*variation[i],finish[i]
    complete = bool(np.all(x[:,0] >= packed[11]) and error == 0 and violations == 0)
    return (values,crossing,complete,error,steps,work,checked_substeps,maximum,exceeded,x,point,
            mlo,mhi,violations,states[:steps],commands[:steps],memories[:steps],errors[:steps],
            observer_los[:steps],observer_his[:steps],recovered[:steps],planning_work_complete)


def forecast(state,previous,memory,observer_lower,observer_upper,step,tape,plant,bounds,
             first_command=None,record_trace=False):
    c = MergeConfig()
    x,prev,lo,hi,b = validate_inputs(state,previous,observer_lower,observer_upper,bounds,c)
    point = np.asarray(memory,dtype=float)
    if (point.shape != (3,) or not np.isfinite(point).all()
            or np.any(point < lo-1e-12) or np.any(point > hi+1e-12)):
        raise ValueError('Predicted actuator state must be inside its command-history interval')
    if type(step) is not int or not 0 <= step <= round(c.horizon/c.decision_dt):
        raise ValueError('Original global decision clock required')
    if type(record_trace) is not bool:
        raise ValueError('Explicit trace flag required')
    path = np.ascontiguousarray(tape,dtype=float)
    if path.shape != (250,3) or not np.isfinite(path).all() or np.any(np.abs(path) > b[:,6]+1e-12):
        raise ValueError('Complete path within the declared innovation bound required')
    gain,drag,retention = plant_arrays(plant,c.dt)
    for value,left,right in ((gain,0,1),(drag,2,3),(retention,4,5)):
        if np.any(value < b[:,left]-1e-15) or np.any(value > b[:,right]+1e-15):
            raise ValueError('Prediction coefficients must belong to the declared actuator class')
    first = np.zeros(3) if first_command is None else np.asarray(first_command,dtype=float)
    if first.shape != (3,) or not np.isfinite(first).all():
        raise ValueError('Finite joint first command required')
    result = forecast_kernel(x,prev,point,lo,hi,step,path,gain,drag,retention,b,parameters(c),
                             first,first_command is not None,record_trace)
    return dict(values=result[0].tolist(),crossing=[float(v) if np.isfinite(v) else None for v in result[1]],
        complete=bool(result[2]),error_code=int(result[3]),steps=int(result[4]),
        completed_planning_model_transitions=int(result[5]),planning_work_complete=bool(result[21]),
        recovery_certificate_substeps=int(result[6]),
        max_actual_error=float(result[7]),legacy_error_exceeded_substeps=int(result[8]),
        final_state=result[9].tolist(),final_actuator=result[10].tolist(),
        final_observer_lower=result[11].tolist(),final_observer_upper=result[12].tolist(),
        route_violations=int(result[13]),
        trace=[dict(t=(step+k+1)*c.decision_dt,x=result[14][k].tolist(),command=result[15][k].tolist(),
            actuator_states=result[16][k].tolist(),disturbance=result[17][k].tolist(),
            actuator_memory_lower=result[18][k].tolist(),actuator_memory_upper=result[19][k].tolist(),
            recovery_applied=bool(result[20][k])) for k in range(result[4])] if record_trace else [])
