"""Exact within-decision memoization of the frozen nominal planning proposal.

Only the pure nominal proposal is cached. Every actuator-memory recovery check,
full path, candidate order, threshold, terminal condition and prediction update
remains unchanged. The key preserves all float64 input bits, clock and config.
Cache lifetime is one online decision; capacity exhaustion computes normally.
Reference work counters stay comparable, with actual cache work separately
labeled. No active scientific study imports this experimental module.
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict

from .actuator_recovery_v34 import memory_step, stopping_certificate
from .arrival_phase_v20 import crossing_time
from .buffered_baseline_v25 import deduplicated_joint_rollout
from .fast_v8 import transition_values
from .recovery_joint_gate_v35 import RecoveryJointGate

KEY = types.UniTuple(types.uint64,22)
VALUE = types.Tuple((types.float64,types.float64,types.float64,types.int64))
CAPACITY = 16384
CACHE_FIELDS = ('requests','hits','misses','successful_misses','failed_misses',
                'logical_successful_transitions','computed_successful_transitions','capacity_skips')


def make_cache():
    return Dict.empty(key_type=KEY,value_type=VALUE),np.zeros(len(CACHE_FIELDS),dtype=np.int64)


@njit(cache=True,fastmath=False)
def proposal_key(state,previous,step,packed):
    # These internal inputs are float64 arrays validated by RecoveryJointGate.
    # Integer bit keys distinguish +0/-0 and do not merge nearby floats.
    x,p,c = state.view(np.uint64),previous.view(np.uint64),packed.view(np.uint64)
    return (x[0,0],x[0,1],x[1,0],x[1,1],x[2,0],x[2,1],p[0],p[1],p[2],np.uint64(step),
            c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11])


@njit(cache=True,fastmath=False)
def cached_proposal(state,previous,step,packed,cache,stats,capacity):
    key = proposal_key(state,previous,step,packed)
    stats[0] += 1
    if key in cache:
        saved = cache[key]
        stats[1] += 1
        stats[5] += saved[3]
        return np.array([saved[0],saved[1],saved[2]]),True,saved[3]
    stats[2] += 1
    try:
        command,_,_,count = deduplicated_joint_rollout(state,previous,step,packed)
    except Exception:
        # Never persist an exception. A transient resource failure must not
        # turn into a cached refusal on later requests.
        stats[4] += 1
        return np.zeros(3),False,0
    stats[3] += 1
    stats[5] += count
    stats[6] += count
    if len(cache) < capacity:
        cache[key] = (command[0],command[1],command[2],count)
    else:
        stats[7] += 1
    return command,True,count


@njit(cache=True, fastmath=False)
def recovery_command(state, previous, step, memory_lower, memory_upper, bounds, packed, cache, stats, cache_capacity):
    """Same command as V34, with explicit failure and recovery indicators."""
    proposal,proposal_available,planning_transitions = cached_proposal(state,previous,step,packed,cache,stats,cache_capacity)
    certificate_substeps = 0
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
                    gain,drag,retention,bounds,packed,first_command,use_first,cache,stats,cache_capacity,record_trace=False):
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
            command,valid,recovery,proposal_failed,count,checks = recovery_command(x,prev,k,mlo,mhi,bounds,packed,cache,stats,cache_capacity)
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


class MemoizedRecoveryGate(RecoveryJointGate):
    def predict(self,state,previous,mlo,mhi,command,path):
        if getattr(self,'_cache_step',None) != self.step:
            self._plan_cache,self._cache_stats = make_cache()
            self._cache_step = self.step
        first = np.zeros(3) if command is None else command
        return forecast_kernel(state,previous,self.memory,mlo,mhi,self.step,self.tapes[path],
            self.gain,self.drag,self.retention,self.bounds,self.packed,first,command is not None,
            self._plan_cache,self._cache_stats,CAPACITY)

    def select(self,state,previous,unused=None):
        command,info = super().select(state,previous,unused)
        current = getattr(self,'_cache_step',None) == self.step-1
        values = self._cache_stats.tolist() if current else [0]*len(CACHE_FIELDS)
        if (values[0] != values[1]+values[2] or values[2] != values[3]+values[4]
                or values[5] != info['completed_planning_model_transitions']
                or values[6] > values[5]):
            raise RuntimeError('Planning cache work accounting differs from reference calls')
        info.update(proposal_cache=dict(zip(CACHE_FIELDS,map(int,values))),
            proposal_cache_entries=len(self._plan_cache) if current else 0,
            proposal_cache_capacity=CAPACITY,
            planning_counter_scope='Original completed_planning_model_transitions retains logical reference work; proposal_cache separately records work actually recomputed. Transitions inside failed planning calls remain uncounted, as flagged by planning_work_complete.')
        return command,info
