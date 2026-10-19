"""Joint braking witnesses for a fixed-order, single-lane discrete plant.

Conditional safety: exact semi-implicit integration, full state, no lane entry,
bounded lead deceleration, and all controlled cars obey the joint controller.
This module does not certify the V2 merge or a physical vehicle.
"""
from dataclasses import dataclass
from itertools import product
import time
import numpy as np
from .config import Config
from .controllers import SpeedMPC, backup_intents


@dataclass
class RecoveryConfig(Config):
    desired_speed: float = 20.
    max_speed: float = 20.
    episode_seconds: float = 25.
    continuous_network: bool = True
    commit_lane_changes: bool = True
    leader_brake: float = 9.
    witness_steps: int = 24
    witness_buffer: float = .05


def backup_acceleration(previous,cfg):
    return np.maximum(np.asarray(previous)-cfg.max_jerk*cfg.decision_dt,-cfg.max_brake)


def admissible_acceleration(proposal,previous,cfg):
    slew=cfg.max_jerk*cfg.decision_dt
    return np.clip(np.asarray(proposal),np.maximum(-cfg.max_brake,np.asarray(previous)-slew),
                   np.minimum(cfg.max_accel,np.asarray(previous)+slew))


def margins(x,cfg):
    # Front-to-rear order: exogenous lead (3), ego0, ego1, ego2.
    return np.min(x[:,[3,0,1],0]-x[:,[0,1,2],0]-cfg.vehicle_length-cfg.safety_gap,axis=1)


def pair_margins(x,cfg):
    return x[:,[3,0,1],0]-x[:,[0,1,2],0]-cfg.vehicle_length-cfg.safety_gap


def integrate(x,commands,lead_acceleration,cfg,per_pair=False):
    """Mutate a batch for one decision, checking every piecewise-linear segment."""
    measure=pair_margins if per_pair else margins
    minimum=measure(x,cfg)
    a=np.concatenate((commands,np.full((len(x),1),lead_acceleration)),axis=1)
    for _ in range(round(cfg.decision_dt/cfg.sim_dt)):
        x[:,:,2]=np.clip(x[:,:,2]+a*cfg.sim_dt,0,cfg.max_speed)
        x[:,:,0]+=x[:,:,2]*cfg.sim_dt
        x[:,:,3]=a
        minimum=np.minimum(minimum,measure(x,cfg))
    return minimum


def witnesses(states,previous,cfg,proposals=None,forget_memory=False,independent_front=False,per_pair=False):
    """Candidate once then shared braking to the invariant all-stopped set.

    With proposals=None, test membership in K_B by executing B immediately.
    Returns minimum clearance beyond the required net gap, plus terminal flag.
    Vectorization checks all proposed joint actions with identical numerics.
    """
    if proposals is None:
        proposals=np.asarray(backup_acceleration(previous,cfg))[None,:]
    proposals=np.atleast_2d(proposals)
    x=np.repeat(np.asarray(states,dtype=float)[None,:,:],len(proposals),axis=0)
    minimum=integrate(x,proposals,-cfg.leader_brake,cfg,per_pair)
    prev=proposals.copy()
    if forget_memory:prev[:]=0
    if independent_front:
        # Freeze independent front profiles after the common candidate prefix.
        # Each controlled front is assumed to brake instantly at leader_brake.
        fronts=x[:,[3,0,1],:].copy()
    for _ in range(cfg.witness_steps):
        command=backup_acceleration(prev,cfg)
        if independent_front:
            for _ in range(round(cfg.decision_dt/cfg.sim_dt)):
                fronts[:,:,2]=np.maximum(0,fronts[:,:,2]-cfg.leader_brake*cfg.sim_dt)
                fronts[:,:,0]+=fronts[:,:,2]*cfg.sim_dt
                x[:,:3,2]=np.clip(x[:,:3,2]+command*cfg.sim_dt,0,cfg.max_speed)
                x[:,:3,0]+=x[:,:3,2]*cfg.sim_dt
                gaps=fronts[:,:,0]-x[:,:3,0]-cfg.vehicle_length-cfg.safety_gap
                minimum=np.minimum(minimum,gaps if per_pair else np.min(gaps,axis=1))
        else:
            minimum=np.minimum(minimum,integrate(x,command,-cfg.leader_brake,cfg,per_pair))
        prev=command
        if np.all(x[:,:3,2]<=1e-10):break
    terminal=np.all(x[:,:3,2]<=1e-10,axis=1)&np.all(prev<=0,axis=1)
    return minimum,terminal


class RecoverySupervisor:
    methods=('raw_mpc','raw_rl','joint_mpc','joint_rl','individual','independent_front','no_memory')

    def __init__(self,cfg,method):
        if method not in self.methods:raise ValueError(method)
        self.cfg,self.method=cfg,method
        self.mpcs=[SpeedMPC(cfg) for _ in range(3)]

    def nominal(self,states,previous):
        intents=backup_intents(states,self.cfg)
        return np.array([m.control(states[i,2],intents[i].speed,previous[i]) for i,m in enumerate(self.mpcs)])

    def select(self,states,previous,proposal):
        start=time.perf_counter();c=self.cfg
        rl=admissible_acceleration(proposal,previous,c)
        mpc=self.nominal(states,previous)
        backup=backup_acceleration(previous,c)
        kb,terminal=witnesses(states,previous,c)
        in_set=bool(terminal[0] and kb[0]>=c.witness_buffer)
        preferred=mpc if self.method in ('raw_mpc','joint_mpc') else rl
        chosen=preferred.copy();checked=0;mask=np.ones(3,dtype=bool)
        if self.method not in ('raw_mpc','raw_rl'):
            bits=np.array(list(product((False,True),repeat=3)),dtype=bool)
            # Full shared witness is the ultimate fallback, never an unchecked rule.
            candidates=np.where(bits,preferred,backup)
            values,ends=witnesses(states,previous,c,candidates,
                forget_memory=self.method=='no_memory',independent_front=self.method=='independent_front')
            safe=ends&(values>=c.witness_buffer);checked=len(bits)
            if self.method=='individual':
                # Deliberately flawed independent certificate composition, for ablation.
                pair_values,_=witnesses(states,previous,c,preferred[None,:],per_pair=True)
                mask=pair_values[0]>=c.witness_buffer
                chosen=np.where(mask,preferred,backup)
            elif np.any(safe):
                # Least squared intervention; tie-break by more progress.
                cost=np.sum((candidates-preferred)**2,axis=1)-1e-8*np.sum(candidates,axis=1)
                index=int(np.argmin(np.where(safe,cost,np.inf)))
                chosen=candidates[index];mask=bits[index]
            else:
                chosen=backup;mask=np.zeros(3,dtype=bool)
        verified,ends=witnesses(states,previous,c,chosen[None,:])
        # Monitor the real contract even for intentionally flawed ablations.
        certified=bool(ends[0] and verified[0]>=c.witness_buffer)
        elapsed=(time.perf_counter()-start)*1000
        changed=np.abs(chosen-preferred)>1e-5
        return chosen,{'in_recoverable_set':in_set,'set_margin':float(kb[0]),
            'chosen_certified':certified,'chosen_margin':float(verified[0]),
            'intervention_fraction':float(np.mean(changed)),
            'rl_retained_fraction':float(np.mean(np.abs(chosen-rl)<1e-5)) if self.method not in ('raw_mpc','joint_mpc') else 0.,
            'mpc':mpc.tolist(),'proposal':rl.tolist(),'backup':backup.tolist(),
            'checked_modes':checked,'gate_ms':elapsed,'scope':'single-lane exact-discrete-model'}
