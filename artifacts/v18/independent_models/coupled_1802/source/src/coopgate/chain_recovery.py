"""Exact chain factorization of the scoped joint switching problem.

Each car has two first-step choices and then its own actuator-aware braking
tail. Adjacent-pair constraints therefore factor into binary edge tables.
This is an application of standard chain dynamic programming, not a claim
that generic multi-agent shielding or dynamic programming is new.
"""
from itertools import product
import time
import numpy as np
from .recovery import backup_acceleration,admissible_acceleration,RecoverySupervisor


def build_chain(states,previous,preferred,cfg):
    states=np.asarray(states);n=len(previous)
    if states.shape!=(n+1,5):raise ValueError('Controlled cars front-to-rear, exogenous leader last')
    command=np.stack((backup_acceleration(previous,cfg),preferred),axis=1)
    choices=command.copy()
    positions=np.repeat(states[:n,0,None],2,axis=1);velocities=np.repeat(states[:n,2,None],2,axis=1)
    lead_p=float(states[-1,0]);lead_v=float(states[-1,2]);history=[positions.copy()];lead_history=[lead_p]
    for k in range(cfg.witness_steps+1):
        if k:command=backup_acceleration(command,cfg)
        for _ in range(round(cfg.decision_dt/cfg.sim_dt)):
            velocities=np.clip(velocities+command*cfg.sim_dt,0,cfg.max_speed)
            positions+=velocities*cfg.sim_dt
            lead_v=max(0,lead_v-cfg.leader_brake*cfg.sim_dt);lead_p+=lead_v*cfg.sim_dt
            history.append(positions.copy());lead_history.append(lead_p)
        if np.all(velocities<=1e-10):break
    history=np.asarray(history).transpose(1,2,0);lead_history=np.asarray(lead_history)
    edge=np.empty((n,2,2))
    for i in range(n):
        if i==0:
            edge[i]=np.min(lead_history[None,:]-history[0],axis=1)[None,:]-cfg.vehicle_length-cfg.safety_gap
        else:
            edge[i]=np.min(history[i-1,:,None,:]-history[i,None,:,:],axis=2)-cfg.vehicle_length-cfg.safety_gap
    terminal=(velocities<=1e-10)&(command<=0)
    feasible=(edge>=cfg.witness_buffer)&terminal[:,None,:]
    cost=(choices-preferred[:,None])**2-1e-8*choices
    return choices,edge,feasible,cost


def chain_modes(feasible,cost):
    # Two DP states per vehicle; back-pointers avoid quadratic prefix copying.
    values=np.where(feasible[0,0],cost[0],np.inf);ranks=[0,1]
    parents=np.zeros((len(cost),2),dtype=int)
    for i in range(1,len(cost)):
        new=[];keys=[]
        for b in (0,1):
            options=[(float(values[a]+cost[i,b]),ranks[a],a) for a in (0,1) if feasible[i,a,b]]
            value,rank,parent=min(options) if options else (float('inf'),0,0)
            new.append(value);parents[i,b]=parent;keys.append((rank,b))
        ranks=[int(keys[b]>keys[1-b]) for b in (0,1)];values=np.asarray(new)
    last=min((0,1),key=lambda b:(values[b],ranks[b]))
    if not np.isfinite(values[last]):return None
    path=np.empty(len(cost),dtype=int);path[-1]=last
    for i in range(len(cost)-1,0,-1):path[i-1]=parents[i,path[i]]
    return path


def brute_modes(feasible,cost):
    best=(float('inf'),None)
    for path in product((0,1),repeat=len(cost)):
        if all(feasible[i,path[i-1] if i else 0,path[i]] for i in range(len(cost))):
            value=float(sum(cost[i,b] for i,b in enumerate(path)))
            if value<best[0]:best=(value,path)
    return None if best[1] is None else np.asarray(best[1])


def solve_chain(states,previous,preferred,cfg):
    choices,edge,feasible,cost=build_chain(states,previous,preferred,cfg)
    mode=chain_modes(feasible,cost)
    base={'set_margin':float(np.min(edge[:,0,0])),'backup_certified':bool(np.all(feasible[:,0,0]))}
    if mode is None:return choices[:,0],{**base,'certified':False,'margin':float(np.min(edge[:,0,0]))}
    margin=min(edge[i,mode[i-1] if i else 0,b] for i,b in enumerate(mode))
    return choices[np.arange(len(mode)),mode],{**base,'certified':True,'margin':float(margin),'modes':mode.tolist(),
        'edge_entries':2+4*(len(mode)-1),'scope':'fixed-order pairwise-separable backup trajectories'}


class ChainSupervisor(RecoverySupervisor):
    def select(self,states,previous,proposal):
        start=time.perf_counter();c=self.cfg
        preferred=admissible_acceleration(proposal,previous,c)
        nominal=self.nominal(states,previous)
        command,certificate=solve_chain(states,previous,preferred,c)
        fraction=float(np.mean(np.abs(command-preferred)>1e-5))
        return command,{'in_recoverable_set':certificate['backup_certified'],'set_margin':certificate['set_margin'],
            'chosen_certified':certificate['certified'],'chosen_margin':certificate['margin'],
            'intervention_fraction':fraction,'rl_retained_fraction':1-fraction,
            'mpc':nominal.tolist(),'proposal':preferred.tolist(),'backup':backup_acceleration(previous,c).tolist(),
            'checked_modes':0,'checked_pair_entries':2+4*(len(previous)-1),
            'gate_ms':(time.perf_counter()-start)*1000,'scope':'fixed-order pairwise-separable backup trajectories'}
