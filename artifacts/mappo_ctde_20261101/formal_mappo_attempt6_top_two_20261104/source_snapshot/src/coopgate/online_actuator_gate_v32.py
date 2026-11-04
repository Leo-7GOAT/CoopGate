"""Online finite-bank admission with a configured reference actuator model.

Reference coefficients are fixed before episodes. Only current kinematics and
executed command history reach the selector; no true randomized coefficients,
realized actuator state, episode seed, or future innovation is an input.
The original physical contract remains conditional on its +/-0.5 error bound.
"""
import time

import numpy as np
from numba import njit

from .actuator_prediction_roundoff_v31 import plant_arrays
from .arrival_phase_v20 import crossing_time
from .cooperative_v15 import JointPolicyGate
from .fast_v8 import parameters, transition_values
from .joint_planner_v15 import JointRolloutGate, joint_rollout
from .merge_v4 import MergeConfig
from .repair_portfolio_v20 import expanded_compositions
from .robust_v9 import robust_certificate, validate
from .temporal_portfolio_v20 import BANK_STREAM, temporal_stress_bank


REFERENCE_PARAMETERS = {
    'nominal': dict(gain=[1.]*3,drag=[0.]*3,tau=[0.]*3),
    'gain_drag': dict(gain=[1.]*3,drag=[.000375]*3,tau=[0.]*3),
    'lag_mild': dict(gain=[1.]*3,drag=[0.]*3,tau=[.04]*3),
    'lag_strong': dict(gain=[1.]*3,drag=[0.]*3,tau=[.4]*3),
}


def reference_parameters(profile):
    if profile not in REFERENCE_PARAMETERS:
        raise ValueError('A preconfigured reference actuator profile is required')
    return {key:list(value) for key,value in REFERENCE_PARAMETERS[profile].items()}


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


class ReferenceActuatorGate:
    """Shared joint candidates and fixed paths, with a reference-model ablation.

    A finite-bank margin is not a probability. The baseline, original physical
    check, and their mismatch limitations are retained explicitly.
    """
    def __init__(self,model,profile,model_kind='reference',innovation_radius=.1,phase_tolerance=.05):
        if model_kind not in ('reference','nominal'):
            raise ValueError('Expected reference or nominal model')
        if innovation_radius != .1 or phase_tolerance != .05:
            raise ValueError('The first online development protocol fixes radius .1 and phase .05')
        self.configured = reference_parameters(profile)
        self.plant = self.configured if model_kind == 'reference' else reference_parameters('nominal')
        self.profile,self.model_kind = profile,model_kind
        self.c = MergeConfig()
        self.packed = parameters(self.c)
        self.arrays = plant_arrays(self.plant,self.c.dt)
        self.primary,self.baseline = JointPolicyGate(model),JointRolloutGate(self.c)
        self.names,tapes = temporal_stress_bank('sampled')
        self.tapes = np.ascontiguousarray(tapes*(innovation_radius/.5))
        self.memory,self.step = np.zeros(3),0
        self.phase_tolerance = phase_tolerance

    def predict(self,x,prev,command,path):
        gain,drag,retention = self.arrays
        first = np.zeros(3) if command is None else command
        return forecast_kernel(x,prev,self.memory,self.step,self.tapes[path],gain,drag,retention,
            self.packed,first,command is not None)

    def select(self,state,previous,unused=None):
        started = time.perf_counter()
        x,prev,_ = validate(state,previous,self.c,.5)
        if self.step >= round(self.c.horizon/self.c.decision_dt):
            raise ValueError('Original finite horizon exhausted')
        if self.step == 0:
            if np.any(prev):
                raise ValueError('Reference memory requires the declared reset command')
        else:
            retention = self.arrays[2]
            # Use the actually executed previous command, never a stored proposal.
            for _ in range(round(self.c.decision_dt/self.c.dt)):
                self.memory = retention*self.memory+(1.-retention)*prev
        memory_at_decision = self.memory.copy()
        base,base_info = self.baseline.select(x,prev)
        self.primary.step = self.step
        _,actor_info = self.primary.select(x,prev)
        candidates,original_count,generation_errors = expanded_compositions(
            x,prev,np.asarray(actor_info['proposal']),base,self.packed,True)
        baseline_values,candidate_values = {},{}
        calls,transitions,internal = 0,0,0
        def predict(command,path):
            nonlocal calls,transitions,internal
            result = self.predict(x,prev,command,path)
            calls += 1
            transitions += result[4]
            internal += result[5]
            return result
        baseline_values[0] = predict(None,0)
        nominal,errors = [],[]
        for index,(code,weight,command) in enumerate(candidates):
            result = predict(command,0)
            candidate_values[index] = result
            if not result[2] or not baseline_values[0][2]:
                errors.append(dict(candidate=index,path=0,baseline_complete=bool(baseline_values[0][2]),candidate_complete=bool(result[2])))
                continue
            difference = result[0]-baseline_values[0][0]
            if np.all(difference <= 1e-10) and difference[:,0].mean() < -1e-9:
                nominal.append((float(difference[:,0].mean()),-weight,code,index))
        nominal.sort()
        chosen,selected,weight,margin = base,0,None,None
        evidence,winner,phases = [],[],[]
        for _,_,code,index in nominal:
            candidate = candidates[index][2]
            diffs,dphases,passed = [],[],True
            failure = None
            for path in range(len(self.tapes)):
                if path not in baseline_values:
                    baseline_values[path] = predict(None,path)
                before = baseline_values[path]
                after = candidate_values[index] if path == 0 else predict(candidate,path)
                if not before[2] or not after[2]:
                    passed = False
                    failure = dict(path=path,baseline_complete=bool(before[2]),candidate_complete=bool(after[2]))
                    break
                delta,dphase = after[0]-before[0],after[1]-before[1]
                diffs.append(delta.tolist())
                dphases.append(dphase.tolist())
                if np.any(delta > 1e-10) or np.any(dphase > self.phase_tolerance+1e-10):
                    passed = False
                    break
            evidence.append(dict(candidate=index,composition=code,checked_paths=len(diffs),
                passed=passed,failure=failure,differences=diffs,phase_differences=dphases))
            if passed:
                if len(diffs) != 33:
                    raise RuntimeError('Incomplete bank cannot authorize learning')
                chosen,selected,weight = candidate.copy(),code,candidates[index][1]
                winner,phases = diffs,dphases
                margin = min(-float(np.max(diffs)),self.phase_tolerance-float(np.max(dphases)))
                break
        if not robust_certificate(x,prev,chosen,self.packed,.5):
            raise RuntimeError('Selected command violates the original physical contract')
        self.step += 1
        admitted = bool(np.any(np.abs(chosen[x[:,0] < self.c.goal]-base[x[:,0] < self.c.goal]) > 1e-9))
        return chosen,dict(in_set=True,certified=True,intervention=float(not admitted),
            gate_ms=1000*(time.perf_counter()-started),baseline_command=base.tolist(),
            nominal_joint_action_admitted=admitted,composition=selected,repair_weight=weight,
            actor_proposal=actor_info['proposal'],candidate_count=len(candidates),original_candidate_count=original_count,
            candidate_evidence=evidence,selected_bank_differences=winner,selected_phase_differences=phases,
            nominal_forecast_errors=errors,generation_errors=generation_errors,
            reference_profile=self.profile,prediction_model=self.model_kind,reference_parameters=self.plant,
            reconstructed_actuator=memory_at_decision.tolist(),forecast_bank_stream=BANK_STREAM,
            forecast_bank_size=33,external_bank_radius=.1,phase_tolerance_seconds=self.phase_tolerance,
            forecast_calls=calls,performance_forecast_transitions=transitions,performance_model_transitions=internal,
            finite_bank_performance_margin=margin,margin_is_probability=False,
            nominal_reference_baseline_complete=bool(baseline_values[0][2]),
            future_realization_used=False,true_randomized_plant_used=False,new_risk_certificate=False,
            physical_claim='Original conditional error contract only; actuator mismatch has not been newly certified')
