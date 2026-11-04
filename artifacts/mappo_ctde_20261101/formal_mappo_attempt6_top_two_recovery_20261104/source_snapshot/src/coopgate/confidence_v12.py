"""Calibrated episode-level performance admission + per-cycle safety checks.

The conformal statement is marginal harmful-selection risk under exchangeable
episodes within a fixed initial-state stratum, NOT conditional safety probability.
The learned policy still acts at every control step after admission.
"""
import math,time
import numpy as np
from .merge_v4 import MergeConfig
from .fast_v8 import parameters
from .marl_v12 import MARLGate,observations,action_mask,batch_step,LEVELS
from .single_v12 import SingleRolloutGate,single_rollout
from .fast_v8 import advance_one
from numba import njit

VARIATION_WEIGHT=.05


@njit(cache=True,fastmath=False)
def rollout_baseline_prediction(initial,c):
    x=initial.copy();prev=np.zeros(3);cost=0.;variation=0.
    for step in range(round(c[2]/c[1])):
        active=np.sum(x[:,0]<c[11])
        if active==0:break
        command,_,_,_=single_rollout(x,prev,step,c,.5,VARIATION_WEIGHT)
        cost+=active*c[1]/3.;variation+=np.abs(command-prev).sum()
        if not advance_one(x,command,c):raise RuntimeError('Baseline prediction unsafe')
        prev=command
    return cost,variation


def conflict_group(initial,config=None):
    c=config or MergeConfig();x=np.asarray(initial)
    eta=np.maximum(0,c.zone_start-x[:,0])/np.maximum(x[:,1],1.)
    separation=float(np.min(np.abs(eta[[0,2]]-eta[1])))
    return 'high_conflict' if separation<=1.25 else 'low_conflict'


def in_training_domain(initial):
    x=np.asarray(initial)
    return (x.shape==(3,2) and np.isfinite(x).all() and np.all((x[:,1]>=8-1e-6)&(x[:,1]<=17+1e-6))
            and np.all((x[:2,0]>=-110-1e-6)&(x[:2,0]<=-70+1e-6))
            and 30-1e-6<=x[0,0]-x[2,0]<=45+1e-6)


def predict_costs(initials,model=None,config=None,baseline='single_rollout'):
    """Nominal full-episode model rollout; future error tape is never observed."""
    c=config or MergeConfig();packed=parameters(c);x=np.asarray(initials,dtype=np.float64).copy()
    if x.ndim!=3 or x.shape[1:]!=(3,2):raise ValueError('Expected batch of (3,2) states')
    if model is None and baseline=='single_rollout':
        scores=np.array([rollout_baseline_prediction(s,packed) for s in x])
        return scores[:,0],scores[:,1]
    prev=np.zeros((len(x),3));cost=np.zeros(len(x));tv=np.zeros(len(x));done=np.zeros(len(x),bool)
    for step in range(round(c.horizon/c.decision_dt)):
        mask=action_mask(prev,c);actions=np.where(mask,np.arange(len(LEVELS)),-1).max(-1)
        if model is not None:
            obs=observations(x,prev,step,c)
            a=model.proposal(obs[:,1],mask[:,1])
            if np.asarray(a).shape!=(len(x),) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>=len(LEVELS)):
                raise ValueError('Invalid learned proposal')
            actions[:,1]=a
        nxt,command,valid=batch_step(x,prev,LEVELS[actions],np.zeros((len(x),5,3)),packed,.5)
        if not valid.all():raise RuntimeError('Nominal prediction outside certificate')
        cost+=(~done)*(x[:,:,0]<c.goal).sum(-1)*c.decision_dt/3.
        tv+=(~done)*np.abs(command-prev).sum(-1)
        done|=np.all(nxt[:,:,0]>=c.goal,axis=1);x=nxt;prev=command
        if done.all():break
    return cost,tv


def conformal_upper(residuals,alpha=.1):
    r=np.asarray(residuals,dtype=float)
    if r.ndim!=1 or not len(r) or not np.isfinite(r).all() or not 0<alpha<1:raise ValueError('Finite nonempty calibration required')
    rank=math.ceil((len(r)+1)*(1-alpha))
    return float(np.sort(r)[rank-1]) if rank<=len(r) else math.inf


class EpisodeConfidenceGate:
    def __init__(self,model,initial,calibration,config=None,radius=.5):
        self.c=config or MergeConfig();self.model=model;self.calibration=calibration
        self.baseline=calibration.get('baseline','single_rollout')
        self.base=(SingleRolloutGate(self.c,radius,VARIATION_WEIGHT) if self.baseline=='single_rollout'
                   else MARLGate(None,self.c,radius,mode='constant'))
        self.learned=MARLGate(model,self.c,radius,mode='single')
        self.group=conflict_group(initial,self.c);self.reason='calibrated_gain';self.accept=False
        self.predicted_delta=None;self.upper_delta=None;self.faults=0
        start=time.perf_counter()
        if not in_training_domain(initial):self.reason='outside_calibration_domain'
        elif model is None:self.reason='missing_policy'
        else:
            try:
                bt,bv=predict_costs(np.asarray(initial)[None],None,self.c,self.baseline)
                lt,lv=predict_costs(np.asarray(initial)[None],model,self.c)
                baseline=float(bt[0]+VARIATION_WEIGHT*bv[0]);learned=float(lt[0]+VARIATION_WEIGHT*lv[0])
                self.predicted_delta=learned-baseline
                self.upper_delta=self.predicted_delta+calibration['groups'][self.group]['q']
                self.accept=bool(self.upper_delta<=-calibration.get('minimum_gain_s',.05))
                if not self.accept:self.reason='insufficient_gain_confidence'
            except (RuntimeError,ValueError,IndexError,TypeError):self.reason='prediction_failure'
        self.admission_ms=(time.perf_counter()-start)*1000.

    def select(self,state,previous,unused=None):
        if self.accept:
            try:
                command,info=self.learned.select(state,previous)
                self.base.step=self.learned.step
            except (RuntimeError,ValueError,IndexError,TypeError):
                self.accept=False;self.reason='policy_fault';self.faults+=1
                command,info=self.base.select(state,previous)
        else:command,info=self.base.select(state,previous)
        info.update(performance_mode='learned' if self.accept else 'baseline',
            admission_reason=self.reason,upper_delta_cost=self.upper_delta,
            predicted_delta_cost=self.predicted_delta,conflict_group=self.group,policy_faults=self.faults)
        return command,info
