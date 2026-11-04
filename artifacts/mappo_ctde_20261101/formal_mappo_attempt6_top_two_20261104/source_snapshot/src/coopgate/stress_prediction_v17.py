"""Finite stress-bank forecasts for a whole joint policy and its comparator.

The bank is fixed independently of the test seed and realized error tape.
It is not exhaustive for arbitrary bounded errors. Conformal residuals must
still be calibrated on independent scenes; this module emits no certificate.
"""
from itertools import product
import numpy as np
from numba import njit,prange
from .joint_planner_v15 import joint_rollout
from .marl_v12 import LEVELS,action_mask,batch_step
from .ego_v13 import ego_observations
from .merge_v4 import MergeConfig
from .fast_v8 import parameters,transition_values
from .robust_v9 import validate,robust_certificate


def fixed_stress_bank(radius=.5):
    if radius!=.5:raise ValueError('This controller contract fixes radius 0.5')
    tapes=[np.zeros((250,3))];names=['clean']
    for kind in ('constant_bias','fixed_sign_pulses'):
        for signs in product((-1.,1.),repeat=3):
            tape=np.zeros((250,3));error=radius*np.array(signs)
            if kind=='constant_bias':tape[:]=error
            else:
                for start in range(10,250,20):tape[start:start+10]=error
            names.append(kind+':' + ','.join(str(int(s)) for s in signs));tapes.append(tape)
    return tuple(names),np.array(tapes)


@njit(cache=True,parallel=True,fastmath=False)
def baseline_bank(initials,tapes,packed):
    n=len(initials);s=len(tapes);values=np.zeros((n*s,3,2));counts=np.zeros(n*s,np.int64)
    internal=np.zeros(n*s,np.int64);complete=np.zeros(n*s,np.bool_);safe=np.ones(n*s,np.bool_)
    for idx in prange(n*s):
        state=initials[idx//s].copy();previous=np.zeros(3);time=np.zeros(3);variation=np.zeros(3)
        for step in range(50):
            active=state[:,0]<packed[11]
            if not np.any(active):break
            command,_,_,work=joint_rollout(state,previous,step,packed);internal[idx]+=work
            for i in range(3):
                if active[i]:time[i]+=.5;variation[i]+=abs(command[i]-previous[i])
            for substep in range(5):
                p0,p1,p2=state[0,0],state[1,0],state[2,0]
                for i in range(3):
                    state[i,1]=min(20.,max(0.,state[i,1]+(command[i]+tapes[idx%s,5*step+substep,i])*.1))
                    state[i,0]+=state[i,1]*.1
                safe[idx]&=transition_values(p0,p1,p2,state[0,0],state[1,0],state[2,0],packed)
            previous=command;counts[idx]+=1
        for i in range(3):values[idx,i,0]=time[i]+.05*variation[i];values[idx,i,1]=time[i]
        complete[idx]=np.all(state[:,0]>=packed[11])
    return values.reshape((n,s,3,2)),counts,internal,complete,safe


def learned_bank(initials,tapes,model):
    c=MergeConfig();packed=parameters(c);n=len(initials);s=len(tapes)
    state=np.repeat(np.asarray(initials,dtype=float),s,axis=0);previous=np.zeros((len(state),3))
    times=np.zeros_like(previous);variation=np.zeros_like(previous);done=np.zeros(len(state),bool);transitions=0
    for step in range(50):
        ids=np.flatnonzero(~done)
        if not len(ids):break
        x,prev=state[ids],previous[ids]
        actions=model.proposal(ego_observations(x,prev,step),action_mask(prev))
        following,command,valid=batch_step(x,prev,LEVELS[actions],tapes[ids%s,step*5:(step+1)*5],packed,.5)
        if not valid.all():raise RuntimeError('Stress prediction left the physical contract')
        active=x[:,:,0]<60;times[ids]+=active*.5;variation[ids]+=active*np.abs(command-prev)
        state[ids]=following;previous[ids]=command;done[ids]=np.all(following[:,:,0]>=60,axis=1);transitions+=len(ids)
    values=np.stack((times+.05*variation,times),axis=-1).reshape((n,s,3,2))
    return values,transitions,done


def predict_stress(initials,model):
    initials=np.asarray(initials,dtype=float)
    if initials.ndim!=3 or initials.shape[1:]!=(3,2) or not np.isfinite(initials).all():raise ValueError('Finite joint initials required')
    names,tapes=fixed_stress_bank();packed=parameters(MergeConfig())
    for initial in initials:
        validate(initial,np.zeros(3),MergeConfig(),.5)
        if not robust_certificate(initial,np.zeros(3),np.full(3,-2.5),packed,.5):raise ValueError('Initial outside recoverable contract')
    base,bcount,internal,complete,safe=baseline_bank(initials,tapes,packed)
    if not safe.all():raise RuntimeError('Baseline stress forecast violated physical contract')
    learned,lcount,lcomplete=learned_bank(initials,tapes,model)
    difference=learned-base
    return difference.max(axis=1),dict(scenario_names=list(names),baseline=base,learned=learned,
        baseline_transitions=int(bcount.sum()),learned_transitions=int(lcount),
        baseline_internal_transitions=int(internal.sum()),all_baselines_complete=bool(complete.all()),
        all_learned_complete=bool(lcomplete.all()))
