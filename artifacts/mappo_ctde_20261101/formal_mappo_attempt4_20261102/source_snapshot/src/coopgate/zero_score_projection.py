"""Exact zero-score projection shortcut inside the nominal planner.

The first zero-score vertex has minimum score and the same lexicographic tie
index as the frozen stable sort. Certify it before constructing/sorting all 27
vertices; on rejection preserve the full original order. Squared-score zero,
not approximate command equality, handles signed zero and underflow ties.
All copied downstream function bodies are unchanged, only their local projection
and planning bindings differ. This experimental module is outside frozen studies.
"""
import numpy as np
from numba import njit
from .buffered_baseline_v25 import robust_certificate
from .fast_v8 import advance_one, transition_values
from .marl_v12 import LEVELS
from .single_v12 import constant_proposal
from .actuator_recovery_v34 import memory_step, stopping_certificate
from .arrival_phase_v20 import crossing_time
from .memoized_recovery_v37 import (CAPACITY, MemoizedRecoveryGate, make_cache, proposal_key)

@njit(cache=True, fastmath=False)
def project_joint(x, previous, proposed, c, radius):
    """Nearest proposed/hold/backup joint vertex; no rollout optimizer."""
    first_zero = -1
    if np.all(np.isfinite(proposed)):
        command = np.empty(3)
        first_zero = 0
        for i in range(3):
            backup = max(previous[i]-c[6]*c[1],-c[4])
            # Test the actual squared float64 score, not an epsilon or equality
            # of commands: distinct subnormal differences can square to zero.
            if (backup-proposed[i])**2 == 0.:
                choice,value = 0,backup
            elif (previous[i]-proposed[i])**2 == 0.:
                choice,value = 1,previous[i]
            else:
                choice,value = 2,proposed[i]
            first_zero = 3*first_zero+choice
            command[i] = value
        if robust_certificate(x,previous,command,c,radius):
            return command.copy(),first_zero,True
    commands = np.empty((27,3)); scores = np.empty(27)
    for mode in range(27):
        code = mode; score = 0.
        for i in range(2,-1,-1):
            option = code % 3; code //= 3
            backup = max(previous[i]-c[6]*c[1],-c[4])
            v = backup if option == 0 else (previous[i] if option == 1 else proposed[i])
            commands[mode,i] = v
            score += (v-proposed[i])**2
        scores[mode] = score
    order = np.argsort(scores,kind='mergesort')
    for m in order:
        if m == first_zero:
            continue  # This exact minimum-score vertex already failed.
        if robust_certificate(x,previous,commands[m],c,radius):
            return commands[m].copy(), int(m), True
    return commands[0].copy(), 0, False

@njit(cache=True, fastmath=False)
def team_rule_tail(state,previous,step,c,cutoff=1e12):
    x=state.copy();prev=previous.copy();value=0.;transitions=0
    for k in range(step,round(c[2]/c[1])):
        active=x[:,0]<c[11]
        if not np.any(active):break
        command,_,ok=project_joint(x,prev,constant_proposal(prev,c),c,.5)
        if not ok:raise RuntimeError('Rule continuation outside physical contract')
        for i in range(3):
            if active[i]:value+=(c[1]+.05*abs(command[i]-prev[i]))/3.
        if value>=cutoff:return value,transitions
        if not advance_one(x,command,c):raise RuntimeError('Unsafe rule continuation')
        transitions+=1;prev=command
    return value,transitions

@njit(cache=True, fastmath=False)
def deduplicated_joint_rollout(state, previous, step, c, sweeps=2):
    chosen, _, ok = project_joint(state, previous, constant_proposal(previous, c), c, .5)
    if not ok:
        raise RuntimeError('No certified baseline command')
    best, transitions = team_rule_tail(state, previous, step, c)
    base_value = best
    if np.all(state[:, 0] >= c[11]):
        return chosen, best, base_value, transitions
    seen = np.empty((1 + max(0, sweeps) * 3 * len(LEVELS), 3))
    n_seen = 0
    for _ in range(sweeps):
        improved = False
        for i in (1, 0, 2):
            if state[i, 0] >= c[11]:
                continue
            center = chosen.copy()
            for level in LEVELS:
                if abs(level - previous[i]) > c[6] * c[1] + 1e-8:
                    continue
                proposal = center.copy()
                proposal[i] = level
                for j in range(3):
                    if state[j, 0] >= c[11]:
                        proposal[j] = min(c[5], previous[j] + c[6] * c[1])
                command, _, valid = project_joint(state, previous, proposal, c, .5)
                if not valid:
                    raise RuntimeError('No certificate for candidate')
                if np.array_equal(command, chosen):
                    continue
                duplicate = False
                for k in range(n_seen):
                    if np.array_equal(command, seen[k]):
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen[n_seen] = command
                n_seen += 1
                stage = 0.
                for j in range(3):
                    if state[j, 0] < c[11]:
                        stage += (c[1] + .05 * abs(command[j] - previous[j])) / 3.
                if stage >= best:
                    continue
                following = state.copy()
                if not advance_one(following, command, c):
                    raise RuntimeError('Unsafe planning transition')
                tail, count = team_rule_tail(following, command, step + 1, c, best - stage)
                transitions += 1 + count
                if stage + tail < best - 1e-10:
                    chosen = command.copy()
                    best = stage + tail
                    improved = True
        if not improved:
            break
    return chosen, best, base_value, transitions

@njit(cache=True, fastmath=False)
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


class FastProjectionGate(MemoizedRecoveryGate):
    def predict(self,state,previous,mlo,mhi,command,path):
        if getattr(self,'_cache_step',None) != self.step:
            self._plan_cache,self._cache_stats = make_cache()
            self._cache_step = self.step
        first = np.zeros(3) if command is None else command
        return forecast_kernel(state,previous,self.memory,mlo,mhi,self.step,self.tapes[path],
            self.gain,self.drag,self.retention,self.bounds,self.packed,first,command is not None,
            self._plan_cache,self._cache_stats,CAPACITY)
