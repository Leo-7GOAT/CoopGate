"""Reuse only the immediately preceding identical prediction certificate.

The certified command, state, previous command, memory interval, bounds and
configuration cannot change between the original duplicate calls. First forced
commands still get a full check. All logical output counters remain unchanged;
actual prediction-certificate work is reported separately. Experimental only:
no frozen scientific execution imports this module.
"""
import numpy as np
from numba import njit
from .actuator_recovery_v34 import memory_step, stopping_certificate
from .arrival_phase_v20 import crossing_time
from .fast_v8 import transition_values
from .memoized_recovery_v37 import (CAPACITY, MemoizedRecoveryGate, make_cache, cached_proposal)

PROOF_FIELDS = ('evaluations', 'reuses', 'logical_substeps', 'computed_substeps')


@njit(cache=True, fastmath=False)
def recovery_command(state,previous,step,memory_lower,memory_upper,bounds,packed,cache,stats,capacity,proof_stats):
    proposal,available,count = cached_proposal(state,previous,step,packed,cache,stats,capacity)
    checks = 0
    if available:
        proof = stopping_certificate(state,previous,proposal,memory_lower,memory_upper,bounds,packed)
        checks += proof[2]
        proof_stats[0] += 1
        proof_stats[2] += proof[2]
        proof_stats[3] += proof[2]
        if proof[0]:
            return proposal,True,False,False,count,checks,proof[2]
    backup = np.maximum(previous-packed[6]*packed[1],-packed[4])
    proof = stopping_certificate(state,previous,backup,memory_lower,memory_upper,bounds,packed)
    checks += proof[2]
    proof_stats[0] += 1
    proof_stats[2] += proof[2]
    proof_stats[3] += proof[2]
    return backup,bool(proof[0]),True,not available,count,checks,proof[2]


@njit(cache=True, fastmath=False)
def forecast_kernel(state,previous,memory,observer_lower,observer_upper,step,tape,
                    gain,drag,retention,bounds,packed,first_command,use_first,cache,stats,cache_capacity,proof_stats,record_trace=False):
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
            command,valid,recovery,proposal_failed,count,checks,reused_steps = recovery_command(x,prev,k,mlo,mhi,bounds,packed,cache,stats,cache_capacity,proof_stats)
            planning_work_complete = planning_work_complete and not proposal_failed
            work += count
            checked_substeps += checks
            if not valid:
                error = 2
                break
        if use_first and k == step:
            check = stopping_certificate(x,prev,command,mlo,mhi,bounds,packed)
            checked_substeps += check[2]
            proof_stats[0] += 1
            proof_stats[2] += check[2]
            proof_stats[3] += check[2]
            if not check[0]:
                error = 3
                break
        else:
            # recovery_command just certified these exact, unchanged arguments.
            # Retain the original logical work counter; avoid only its duplicate.
            checked_substeps += reused_steps
            proof_stats[1] += 1
            proof_stats[2] += reused_steps
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


class CertificateReuseGate(MemoizedRecoveryGate):
    def predict(self,state,previous,mlo,mhi,command,path):
        if getattr(self,'_cache_step',None) != self.step:
            self._plan_cache,self._cache_stats = make_cache()
            self._proof_stats = np.zeros(len(PROOF_FIELDS),dtype=np.int64)
            self._cache_step = self.step
        first = np.zeros(3) if command is None else command
        return forecast_kernel(state,previous,self.memory,mlo,mhi,self.step,self.tapes[path],
            self.gain,self.drag,self.retention,self.bounds,self.packed,first,command is not None,
            self._plan_cache,self._cache_stats,CAPACITY,self._proof_stats)

    def select(self,state,previous,unused=None):
        command,info = super().select(state,previous,unused)
        current = getattr(self,'_cache_step',None) == self.step-1
        counts = self._proof_stats.tolist() if current else [0]*len(PROOF_FIELDS)
        if (counts[2] != info['recovery_certificate_substeps'] or counts[3] > counts[2]
                or any(x < 0 for x in counts)):
            raise RuntimeError('Prediction certificate work accounting differs')
        info['prediction_certificate_reuse'] = dict(zip(PROOF_FIELDS,map(int,counts)))
        return command,info
