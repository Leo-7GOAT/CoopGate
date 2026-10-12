"""Backup-relative switching with actuator memory and stopping reserve.

The predictor shares the longitudinal QP with the executor, but lateral and
background forecasts remain approximations. No unconditional safety claim.
"""
from dataclasses import dataclass
from itertools import product
import time
import numpy as np

from .controllers import Intent, SpeedMPC, backup_intents
from .shield import GateResult


def stopping_advance(v_rear, v_front, delay=.5, accel=2.5, brake_rear=5., brake_front=5.):
    """Exact max relative travel for a specified pair of piecewise profiles.

    Front brakes immediately; rear accelerates for delay then brakes to rest.
    Maxima occur at phase endpoints or zero-relative-speed stationary points.
    Assumes straight fixed lanes, constant rates, and nonnegative speeds.
    """
    if min(v_rear,v_front,delay,accel) < 0 or min(brake_rear,brake_front) <= 0:
        raise ValueError('Invalid stopping profile')
    rear_peak=v_rear+accel*delay
    rear_end=delay+rear_peak/brake_rear
    front_end=v_front/brake_front
    times=sorted(set((0.,delay,rear_end,front_end,max(rear_end,front_end))))

    def rear(t):
        if t <= delay:
            return v_rear*t+.5*accel*t*t, v_rear+accel*t
        s=min(t-delay,rear_peak/brake_rear)
        return v_rear*delay+.5*accel*delay*delay+rear_peak*s-.5*brake_rear*s*s, max(0.,rear_peak-brake_rear*s)

    def front(t):
        s=min(t,front_end)
        return v_front*s-.5*brake_front*s*s, max(0.,v_front-brake_front*s)

    points=list(times)
    for left,right in zip(times[:-1],times[1:]):
        if right-left < 1e-12:
            continue
        r0=rear(left)[1]-front(left)[1]
        r1=rear(right)[1]-front(right)[1]
        if r0*r1 < 0:
            points.append(left+(right-left)*r0/(r0-r1))
    return max(0.,max(rear(t)[0]-front(t)[0] for t in points))


def stopping_reserve(states,cfg):
    reserve=100.
    # This only certifies controlled-following pairs under the stated profiles.
    # An uncontrolled car approaching from behind is outside this certificate.
    for i,s in enumerate(states[:cfg.n_agents]):
        if not s[4]:
            continue
        for j,o in enumerate(states):
            if i==j or not o[4] or o[0]<=s[0] or abs(s[1]-o[1])>=cfg.vehicle_width+.6:
                continue
            required=stopping_advance(s[2],o[2],cfg.stopping_delay,cfg.max_accel,
                                      cfg.max_brake,9. if j<cfg.n_agents else 9.)
            reserve=min(reserve,o[0]-s[0]-cfg.vehicle_length-cfg.safety_gap-required)
    return float(reserve)


def potential(states,cfg):
    active=(states[:cfg.n_agents,0]<cfg.goal_x) & (states[:cfg.n_agents,4]>0)
    return float(np.minimum(states[:cfg.n_agents,0],cfg.goal_x).sum()
                 +1.5*np.sum(states[:cfg.n_agents,2]*active)
                 -5*np.sum((states[:cfg.n_agents,1]<1.6)*active))


@dataclass
class Forecast:
    margin: float
    reserve: float  # post-decision braking reserve, not an invariant terminal set
    value: float
    next_states: np.ndarray


class ClosedLoopPredictor:
    def __init__(self,cfg):
        self.cfg=cfg
        self.controllers=[SpeedMPC(cfg) for _ in range(cfg.n_agents)]

    def rollout(self,states,intents,context,background_accel):
        c=self.cfg
        x=np.array(states,dtype=float,copy=True)
        previous=np.array(context.get('previous_accel',x[:3,3]),copy=True)
        targets=np.array(context.get('lane_targets',(x[:3,1]>1.6).astype(int)),copy=True)
        remaining=np.array(context.get('lane_remaining',np.zeros(3)),copy=True)
        margin=100.
        reserve=stopping_reserve(x,c)
        first_state=x.copy()
        # Remove warm-start dependence between counterfactual candidates.
        for m in self.controllers:
            m.solver.warm_start(x=np.zeros(c.mpc_horizon),y=np.zeros(3*c.mpc_horizon))
        for k in range(c.paired_horizon):
            candidate=intents if k==0 else backup_intents(x,c)
            accelerations=[]
            for i in range(3):
                accelerations.append(self.controllers[i].control(x[i,2],candidate[i].speed,previous[i]))
                if remaining[i]<=1e-6:
                    targets[i]=candidate[i].lane
                    if abs(x[i,1]-targets[i]*c.lane_width)>.15:
                        remaining[i]=c.lane_change_seconds
            previous=np.array(accelerations)
            for sub in range(round(c.decision_dt/c.sim_dt)):
                # Check occupancy at simulator resolution, including start state.
                for i in range(3):
                    if not x[i,4]:
                        continue
                    for j in range(len(x)):
                        if i!=j and x[j,4] and abs(x[i,1]-x[j,1])<c.vehicle_width+.6:
                            gap=abs(x[i,0]-x[j,0])-c.vehicle_length-c.safety_gap
                            margin=min(margin,gap-c.prediction_error*(k*c.decision_dt+sub*c.sim_dt))
                    if x[i,0]>c.merge_end-3 and x[i,1]<c.lane_width-1.:
                        margin=min(margin,c.merge_end-3-x[i,0])
                for i in range(len(x)):
                    if not x[i,4]:
                        continue
                    a=accelerations[i] if i<3 else background_accel
                    x[i,2]=np.clip(x[i,2]+a*c.sim_dt,0,c.max_speed)
                    x[i,0]+=x[i,2]*c.sim_dt  # SUMO's semi-implicit Euler default
                    if i<3 and remaining[i]>0:
                        x[i,1]+=(targets[i]*c.lane_width-x[i,1])*min(1,c.sim_dt/remaining[i])
                        remaining[i]=max(0,remaining[i]-c.sim_dt)
                    if x[i,0]>=c.merge_end:
                        x[i,1]=c.lane_width
            if k==0:
                reserve=stopping_reserve(x,c)
                first_state=x.copy()
        # Include the final integration endpoint in the finite-horizon check.
        for i in range(3):
            if not x[i,4]:
                continue
            for j in range(len(x)):
                if i!=j and x[j,4] and abs(x[i,1]-x[j,1])<c.vehicle_width+.6:
                    margin=min(margin,abs(x[i,0]-x[j,0])-c.vehicle_length-c.safety_gap-c.prediction_error*c.paired_horizon*c.decision_dt)
        return Forecast(float(margin),reserve,potential(x,c),first_state)


class PairedSupervisor:
    def __init__(self,cfg,method='paired'):
        if not cfg.commit_lane_changes:
            raise ValueError('Paired gating requires commit_lane_changes=True so forecast and executor share commitments.')
        self.cfg,self.method=cfg,method
        self.predictor=ClosedLoopPredictor(cfg)
        self.previous=np.zeros(3,bool)

    def select(self,states,learned,context=None):
        start=time.perf_counter()
        c=self.cfg
        context=context or {}
        backup=backup_intents(states,c)
        learned=list(learned)
        remaining=context.get('lane_remaining',np.zeros(3))
        targets=context.get('lane_targets',(states[:3,1]>1.6).astype(int))
        if c.commit_lane_changes:
            for i in range(3):
                if remaining[i]>1e-6:
                    backup[i]=Intent(backup[i].speed,int(targets[i]))
                    learned[i]=Intent(learned[i].speed,int(targets[i]))
        hypotheses=(-c.background_brake,0.,1.5)
        baseline=[self.predictor.rollout(states,backup,context,a) for a in hypotheses]
        checks=3
        best=np.zeros(3,bool)
        chosen=backup
        best_value=0.
        chosen_margin=min(r.margin for r in baseline)
        chosen_terminal=min(r.reserve for r in baseline)
        chosen_gain=0.
        failed=chosen_margin<c.paired_margin or chosen_terminal<0
        budget=False
        evaluated=0
        # Deterministic anytime search, all-RL first followed by fewer RL agents.
        candidates=sorted(product((False,True),repeat=3),key=lambda b:(-sum(b),b))
        for bits in candidates:
            if not any(bits):
                continue
            if (time.perf_counter()-start)*1000>c.gate_budget_ms:
                budget=True
                break
            candidate=[learned[i] if bits[i] else backup[i] for i in range(3)]
            forecasts=[self.predictor.rollout(states,candidate,context,a) for a in hypotheses]
            checks+=3; evaluated+=1
            margin=min(r.margin for r in forecasts)
            terminal=min(r.reserve for r in forecasts)
            gain=min(r.value-base.value for r,base in zip(forecasts,baseline))
            mask=np.array(bits)
            if margin<c.paired_margin or (self.method!='paired_no_reserve' and terminal<0):
                continue
            if self.method!='paired_no_gain' and gain<c.paired_gain:
                continue
            value=(sum(bits) if self.method=='paired_no_gain' else gain)-c.switch_penalty*np.count_nonzero(mask!=self.previous)
            if value>best_value:
                best_value=value;best=mask;chosen=candidate
                chosen_margin=margin;chosen_terminal=terminal;chosen_gain=gain
        elapsed=(time.perf_counter()-start)*1000
        self.previous=best.copy()
        diagnostics={'predicted_gain':float(chosen_gain),'postdecision_braking_reserve':float(chosen_terminal),
                     'backup_unverified':bool(failed and not best.any()),
                     'budget_exceeded':bool(budget or elapsed>c.gate_budget_ms),
                     'evaluated_modes':evaluated,
                     'certificate':'finite-model-only'}
        # Failed baseline verification does NOT imply emergency braking is safe.
        # Preserve rule backup, explicitly log unverified recovery.
        score=float(1/(1+np.exp(-np.clip(chosen_margin/2,-50,50))))
        return GateResult(chosen,best,score,float(chosen_margin),checks,False,elapsed,diagnostics)
