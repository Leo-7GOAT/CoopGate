"""Candidate rejection by an optimistic, outward-rounded arrival bound.

Independent prototype: no frozen experiment imports this module. Baseline
forecasts, accepted candidates and physical certificates remain unchanged.
The bound ignores nonnegative drag and traffic constraints and maximizes all
future slew-admissible commands. It can reject, never authorize, a candidate.
"""
import time

import numpy as np
from numba import njit

from .actuator_recovery_v34 import memory_step, stopping_certificate
from .arrival_phase_v20 import crossing_time
from .fast_v8 import transition_values
from .memoized_recovery_v37 import (MemoizedRecoveryGate, CAPACITY, make_cache,
                                    recovery_command)


@njit(cache=True,fastmath=False)
def up(value):
    return np.nextafter(value,np.inf)


@njit(cache=True,fastmath=False)
def optimistic_values(state,previous,memory,step,root_step,tape,gain,retention,packed,finish,variation):
    """Lower bounds for the full root-relative cost/arrival of each vehicle.

    Speeds/positions/memory are propagated upward. Removing drag only raises
    attainable speed. Prefix variation cannot decrease. Time is rounded to
    the SAME decision boundary as the reference objective, not interpolated.
    Inputs describe one fixed reference-model forecast prefix, not true plant.
    """
    result=np.empty((3,2));substeps=round(packed[1]/packed[0]);work=0
    for i in range(3):
        earliest=finish[i]
        if state[i,0]<packed[11]:
            position,speed,point,command=state[i,0],state[i,1],memory[i],previous[i]
            earliest=packed[2]-root_step*packed[1]
            arrived=False
            for k in range(step,round(packed[2]/packed[1])):
                command=min(packed[5],up(command+packed[6]*packed[1]))
                for sub in range(substeps):
                    point=up(up(retention[i]*point)+up((1.-retention[i])*command))
                    acceleration=up(up(gain[i]*point)+tape[k*substeps+sub,i])
                    speed=min(packed[3],max(0.,up(speed+up(packed[0]*acceleration))))
                    position=up(position+up(packed[0]*speed));work+=1
                    if position>=packed[11]:
                        earliest=(k-root_step+1)*packed[1];arrived=True;break
                if arrived:break
        result[i,0]=earliest+.05*variation[i];result[i,1]=earliest
    return result,work


@njit(cache=True,fastmath=False)
def pruned_forecast(state,previous,memory,observer_lower,observer_upper,step,tape,
                    gain,drag,retention,bounds,packed,first_command,use_first,cache,stats,cache_capacity,
                    reference_limits,enable_pruning=True):
    """Original recurrence with a rejection-only check before future planning.

    Error4 denotes a proved baseline-relative rejection, not physical failure.
    First22 return entries preserve the existing forecast interface. Extra
    entries contain bound work and a replayable rejection-prefix proof.
    """
    x,prev,point=state.copy(),previous.copy(),memory.copy()
    mlo,mhi=observer_lower.copy(),observer_upper.copy()
    finish,variation,crossing=np.zeros(3),np.zeros(3),np.zeros(3)
    for i in range(3):
        if x[i,0]<packed[11]:
            finish[i]=packed[2]-step*packed[1];crossing[i]=np.nan
    substeps=round(packed[1]/packed[0])
    states=np.empty((0,3,2));commands=np.empty((0,3))
    memories=np.empty((0,substeps,3));errors=np.empty((0,substeps,3))
    observer_los,observer_his=np.empty((0,3)),np.empty((0,3));recovered=np.zeros(0,np.bool_)
    steps=work=checked_substeps=error=violations=exceeded=bound_calls=bound_work=0
    planning_work_complete=True;maximum=0.;lower=np.zeros((3,2))
    for k in range(step,round(packed[2]/packed[1])):
        active=x[:,0]<packed[11]
        if not np.any(active):break
        if enable_pruning and use_first and k>step:
            lower,count=optimistic_values(x,prev,point,k,step,tape,gain,retention,packed,finish,variation)
            bound_calls+=1;bound_work+=count
            # Conservative slack only reduces pruning; it never relaxes the
            # original acceptance comparisons or physical certificate.
            if np.any(lower>reference_limits+1e-8):
                error=4;break
        if use_first and k==step:
            command=first_command.copy()
        else:
            command,valid,recovery,proposal_failed,count,checks=recovery_command(
                x,prev,k,mlo,mhi,bounds,packed,cache,stats,cache_capacity)
            planning_work_complete=planning_work_complete and not proposal_failed
            work+=count;checked_substeps+=checks
            if not valid:error=2;break
        check=stopping_certificate(x,prev,command,mlo,mhi,bounds,packed)
        checked_substeps+=check[2]
        if not check[0]:error=3;break
        for i in range(3):
            if active[i]:variation[i]+=abs(command[i]-prev[i])
        for sub in range(substeps):
            before=x[:,0].copy();mlo,mhi=memory_step(mlo,mhi,command,bounds);current_error=0.
            for i in range(3):
                point[i]=retention[i]*point[i]+(1.-retention[i])*command[i]
                applied=gain[i]*point[i]-drag[i]*x[i,1]**2+tape[k*substeps+sub,i]
                current_error=max(current_error,abs(applied-command[i]))
                x[i,1]=min(packed[3],max(0.,x[i,1]+packed[0]*applied))
                x[i,0]+=packed[0]*x[i,1]
                if np.isnan(crossing[i]):
                    crossing[i]=crossing_time(before[i],x[i,0],packed[11],
                        (k-step)*packed[1]+sub*packed[0],packed[0])
            maximum=max(maximum,current_error);exceeded+=int(current_error>.5+1e-9)
            violations+=int(not transition_values(before[0],before[1],before[2],x[0,0],x[1,0],x[2,0],packed))
        for i in range(3):
            if active[i] and x[i,0]>=packed[11]:finish[i]=(k-step+1)*packed[1]
        prev=command.copy();steps+=1
    values=np.zeros((3,2))
    for i in range(3):values[i,0],values[i,1]=finish[i]+.05*variation[i],finish[i]
    complete=bool(np.all(x[:,0]>=packed[11]) and error==0 and violations==0)
    return (values,crossing,complete,error,steps,work,checked_substeps,maximum,exceeded,x,point,
            mlo,mhi,violations,states,commands,memories,errors,observer_los,observer_his,recovered,
            planning_work_complete,bound_calls,bound_work,lower,prev,finish,variation)


class AdmissibleRecoveryGate(MemoizedRecoveryGate):
    def predict(self,state,previous,mlo,mhi,command,path):
        if command is None:
            result=super().predict(state,previous,mlo,mhi,command,path)
            if result[2]:self._reference_limits[path]=result[0].copy()
            return result
        if path not in self._reference_limits:
            return super().predict(state,previous,mlo,mhi,command,path)
        if getattr(self,'_cache_step',None)!=self.step:
            self._plan_cache,self._cache_stats=make_cache();self._cache_step=self.step
        result=pruned_forecast(state,previous,self.memory,mlo,mhi,self.step,self.tapes[path],
            self.gain,self.drag,self.retention,self.bounds,self.packed,command,True,
            self._plan_cache,self._cache_stats,CAPACITY,self._reference_limits[path])
        self._bound_calls+=int(result[22]);self._bound_work+=int(result[23])
        if result[3]==4:
            self._rejections.append(dict(path=path,forecast_prefix_steps=int(result[4]),
                lower_values=result[24].tolist(),reference_limits=self._reference_limits[path].tolist(),
                prefix_state=result[9].tolist(),prefix_memory=result[10].tolist(),
                prefix_previous=result[25].tolist(),prefix_finish=result[26].tolist(),
                prefix_variation=result[27].tolist(),root_step=self.step))
        return result[:22]

    def select(self,state,previous,unused=None):
        started=time.perf_counter();self._reference_limits={};self._bound_calls=self._bound_work=0
        self._rejections=[]
        command,info=super().select(state,previous,unused)
        info.update(optimistic_bound_calls=self._bound_calls,optimistic_bound_physical_substeps=self._bound_work,
            optimistic_rejected_forecasts=len(self._rejections),optimistic_rejection_proofs=self._rejections,
            pruning_role='rejection only; accepted forecasts and physical checks remain complete',
            gate_ms=1000*(time.perf_counter()-started))
        return command,info
