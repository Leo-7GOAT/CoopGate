# input buffered_baseline_v25.py sha256=e25980a011405f9159384f8fa965798be545a2d4333dfead80018d64148b9ea1
# input robust_v9.py sha256=c579b0c57516c9954e0c1c0d06da7324804a01e40402e8aff39fa98f5d80bd6b
# input zero_score_projection.py sha256=8889e24de5dc0c6419e246b65e4ae965d2ffd8697901428b54462de56b686743
# input memoized_recovery_v37.py sha256=3ba963fa2f2cbf6735784c73eae8cbfb595206010ecdcbc4591b46021b69c5df
"""Generated exact scalar nominal certificate; independent runtime prototype.

See tools/build_scalar_nominal_recovery.py. No approximate math, path removal,
new MAPPO weights or altered physical baseline. Forecast semantics identical.
"""
import numpy as np
from numba import njit
from .fast_v8 import clip,advance_one,transition_values
from .robust_v9 import occupancy
from .marl_v12 import LEVELS
from .single_v12 import constant_proposal
from .actuator_recovery_v34 import memory_step,stopping_certificate
from .arrival_phase_v20 import crossing_time
from .memoized_recovery_v37 import proposal_key,make_cache,CAPACITY
from .parameter_scenario_gate import ParameterScenarioGate

@njit(cache=True, fastmath=False, inline='always')
def scalar_transition(lo0_0, lo1_0, lo2_0, hi0_0, hi1_0, hi2_0, qlo0_0, qlo1_0, qlo2_0, qhi0_0, qhi1_0, qhi2_0, c):
    if min(lo0_0 - hi2_0, qlo0_0 - qhi2_0) < c[7] - 1e-08:
        return False
    a, b, present_i = occupancy(lo0_0, hi0_0, qlo0_0, qhi0_0, c)
    e, f, present_j = occupancy(lo1_0, hi1_0, qlo1_0, qhi1_0, c)
    if present_i and present_j and (max(a, e) <= min(b, f)):
        return False
    if qhi0_0 >= c[8] and qhi1_0 >= c[8]:
        ti = 0.0 if hi0_0 >= c[8] else (c[8] - hi0_0) / max(qhi0_0 - hi0_0, 1e-300)
        tj = 0.0 if hi1_0 >= c[8] else (c[8] - hi1_0) / max(qhi1_0 - hi1_0, 1e-300)
        t = max(ti, tj)
        low_start = lo0_0 + t * (qlo0_0 - lo0_0) - hi1_0 - t * (qhi1_0 - hi1_0)
        high_start = hi0_0 + t * (qhi0_0 - hi0_0) - lo1_0 - t * (qlo1_0 - lo1_0)
        ahead = min(low_start, qlo0_0 - qhi1_0) >= c[7] - 1e-08
        behind = max(high_start, qhi0_0 - qlo1_0) <= -c[7] + 1e-08
        if not (ahead or behind):
            return False
    a, b, present_i = occupancy(lo1_0, hi1_0, qlo1_0, qhi1_0, c)
    e, f, present_j = occupancy(lo2_0, hi2_0, qlo2_0, qhi2_0, c)
    if present_i and present_j and (max(a, e) <= min(b, f)):
        return False
    if qhi1_0 >= c[8] and qhi2_0 >= c[8]:
        ti = 0.0 if hi1_0 >= c[8] else (c[8] - hi1_0) / max(qhi1_0 - hi1_0, 1e-300)
        tj = 0.0 if hi2_0 >= c[8] else (c[8] - hi2_0) / max(qhi2_0 - hi2_0, 1e-300)
        t = max(ti, tj)
        low_start = lo1_0 + t * (qlo1_0 - lo1_0) - hi2_0 - t * (qhi2_0 - hi2_0)
        high_start = hi1_0 + t * (qhi1_0 - hi1_0) - lo2_0 - t * (qlo2_0 - lo2_0)
        ahead = min(low_start, qlo1_0 - qhi2_0) >= c[7] - 1e-08
        behind = max(high_start, qhi1_0 - qlo2_0) <= -c[7] + 1e-08
        if not (ahead or behind):
            return False
    return True

@njit(cache=True,fastmath=False)
def robust_certificate(state,previous,command,c,radius):
    bounded=clip(command[0],max(-c[4],previous[0]-c[6]*c[1]),min(c[5],previous[0]+c[6]*c[1]))
    if abs(command[0]-bounded)>=1e-9:return False
    lo0_0=hi0_0=state[0,0]
    lo0_1=hi0_1=state[0,1]
    tail0=command[0]
    bounded=clip(command[1],max(-c[4],previous[1]-c[6]*c[1]),min(c[5],previous[1]+c[6]*c[1]))
    if abs(command[1]-bounded)>=1e-9:return False
    lo1_0=hi1_0=state[1,0]
    lo1_1=hi1_1=state[1,1]
    tail1=command[1]
    bounded=clip(command[2],max(-c[4],previous[2]-c[6]*c[1]),min(c[5],previous[2]+c[6]*c[1]))
    if abs(command[2]-bounded)>=1e-9:return False
    lo2_0=hi2_0=state[2,0]
    lo2_1=hi2_1=state[2,1]
    tail2=command[2]
    for stage in range(1+int(c[10])):
        if stage>0:
            tail0=max(tail0-c[6]*c[1],-c[4])
            tail1=max(tail1-c[6]*c[1],-c[4])
            tail2=max(tail2-c[6]*c[1],-c[4])
        for _ in range(round(c[1]/c[0])):
            qlo0_1=clip(lo0_1+(tail0-radius)*c[0],0.,c[3])
            qhi0_1=clip(hi0_1+(tail0+radius)*c[0],0.,c[3])
            qlo0_0=lo0_0+qlo0_1*c[0]
            qhi0_0=hi0_0+qhi0_1*c[0]
            qlo1_1=clip(lo1_1+(tail1-radius)*c[0],0.,c[3])
            qhi1_1=clip(hi1_1+(tail1+radius)*c[0],0.,c[3])
            qlo1_0=lo1_0+qlo1_1*c[0]
            qhi1_0=hi1_0+qhi1_1*c[0]
            qlo2_1=clip(lo2_1+(tail2-radius)*c[0],0.,c[3])
            qhi2_1=clip(hi2_1+(tail2+radius)*c[0],0.,c[3])
            qlo2_0=lo2_0+qlo2_1*c[0]
            qhi2_0=hi2_0+qhi2_1*c[0]
            if not scalar_transition(lo0_0,lo1_0,lo2_0,hi0_0,hi1_0,hi2_0,qlo0_0,qlo1_0,qlo2_0,qhi0_0,qhi1_0,qhi2_0,c):return False
            lo0_0=qlo0_0
            lo0_1=qlo0_1
            lo1_0=qlo1_0
            lo1_1=qlo1_1
            lo2_0=qlo2_0
            lo2_1=qlo2_1
            hi0_0=qhi0_0
            hi0_1=qhi0_1
            hi1_0=qhi1_0
            hi1_1=qhi1_1
            hi2_0=qhi2_0
            hi2_1=qhi2_1
        if stage>0 and hi0_1==0. and hi1_1==0. and hi2_1==0. and tail0<=-radius and tail1<=-radius and tail2<=-radius:return True
    return False


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


@njit(cache=True,fastmath=False)
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

class ScalarNominalParameterGate(ParameterScenarioGate):
    def predict(self,state,previous,mlo,mhi,command,path):
        if getattr(self,'_cache_step',None)!=self.step:
            self._plan_cache,self._cache_stats=make_cache();self._cache_step=self.step
        if path<self.original_paths:
            gain,drag,rho,memory=self.gain,self.drag,self.retention,self.memory
        else:
            index=path-self.original_paths
            gain,drag,rho=self.parameter_scenarios[index]
            memory=self.parameter_memories[index]
            if np.any(memory<mlo-1e-12) or np.any(memory>mhi+1e-12):
                raise ValueError('Command-history scenario memory escaped the declared interval')
        return forecast_kernel(state,previous,memory,mlo,mhi,self.step,self.tapes[path],gain,drag,rho,
            self.bounds,self.packed,np.zeros(3) if command is None else command,command is not None,
            self._plan_cache,self._cache_stats,CAPACITY)
