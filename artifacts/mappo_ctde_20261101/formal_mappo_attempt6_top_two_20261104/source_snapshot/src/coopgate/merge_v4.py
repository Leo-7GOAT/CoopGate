"""Controlled two-to-one SUMO merge with a route-coordinate backup contract.

No background traffic or free lane changes. The proof concerns route separation
and conflict-zone occupancy, not arbitrary vehicle geometry or uncertain plants.
"""
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
import subprocess, tempfile, uuid, time
import numpy as np
import sumolib, traci

ACTIONS=np.array([-5.,-2.5,0.,1.25,2.5])
BRANCH=np.array([0,1,0])


@dataclass
class MergeConfig:
    dt: float=.1
    decision_dt: float=.5
    horizon: float=25.
    vmax: float=20.
    brake: float=5.
    accel: float=2.5
    jerk: float=5.
    separation: float=6.8
    zone_start: float=-25.
    zone_end: float=6.8
    tail_steps: int=24
    goal: float=60.
    def dict(self):return asdict(self)


def admissible(a,prev,c):
    return np.clip(a,np.maximum(-c.brake,prev-c.jerk*c.decision_dt),np.minimum(c.accel,prev+c.jerk*c.decision_dt))


def backup(prev,c):return np.maximum(prev-c.jerk*c.decision_dt,-c.brake)


def transition_safe(p,q,c):
    """Exact affine-substep interval checks, vectorized over joint candidates."""
    safe=np.ones(len(p),dtype=bool)
    delta=q-p
    for i,j in ((0,1),(0,2),(1,2)):
        if BRANCH[i]==BRANCH[j]:
            # This benchmark initializes ego0 ahead of ego2; no overtaking.
            safe &= np.minimum(p[:,i]-p[:,j],q[:,i]-q[:,j])>=c.separation-1e-8
            continue
        # Each vehicle occupies a closed interval of normalized substep time.
        intervals=[]
        for k in (i,j):
            moving=delta[:,k]>1e-12
            enter=np.where(moving,(c.zone_start-p[:,k])/np.maximum(delta[:,k],1e-12),-np.inf)
            leave=np.where(moving,(c.zone_end-p[:,k])/np.maximum(delta[:,k],1e-12),np.inf)
            present=(q[:,k]>=c.zone_start)&(p[:,k]<=c.zone_end)
            intervals.append((np.maximum(0,enter),np.minimum(1,leave),present))
        a,b=intervals
        overlap=a[2]&b[2]&(np.maximum(a[0],b[0])<=np.minimum(a[1],b[1]))
        safe &= ~overlap
        # Once both cars reach the guarded merge approach, require separation.
        start=np.maximum(a[0],b[0]);active=(q[:,i]>=c.zone_start)&(q[:,j]>=c.zone_start)
        start=np.clip(start,0,1)
        d0=(p[:,i]-p[:,j])+start*(delta[:,i]-delta[:,j]);d1=q[:,i]-q[:,j]
        gap=np.where(d0*d1<=0,0,np.minimum(np.abs(d0),np.abs(d1)))
        safe &= (~active)|(gap>=c.separation-1e-8)
    return safe


def advance(x,commands,c):
    safe=np.ones(len(x),dtype=bool)
    for _ in range(round(c.decision_dt/c.dt)):
        old=x[:,:,0].copy();x[:,:,1]=np.clip(x[:,:,1]+commands*c.dt,0,c.vmax)
        x[:,:,0]+=x[:,:,1]*c.dt
        safe &= transition_safe(old,x[:,:,0],c)
    return safe


def certify(states,previous,proposals,c):
    proposals=np.atleast_2d(proposals)
    x=np.repeat(states[None,:,:],len(proposals),axis=0)
    safe=advance(x,proposals,c);prev=proposals.copy()
    for _ in range(c.tail_steps):
        prev=backup(prev,c);safe &= advance(x,prev,c)
        if np.all(x[:,:,1]<=1e-10):break
    return safe & np.all(x[:,:,1]<=1e-10,axis=1) & np.all(prev<=0,axis=1)


class MergeGate:
    methods=('joint_rl','constant','eta','search','raw_rl','no_memory','individual')
    def __init__(self,c,method):
        if method not in self.methods:raise ValueError(method)
        self.c,self.method=c,method
        self.bits=np.array(list(product((0,1),repeat=3)),dtype=bool)
        self.grid=np.array(list(product(ACTIONS,repeat=3)))

    def eta(self,x,prev):
        c=self.c;p=x[:,0];v=x[:,1]
        pending=np.flatnonzero(p<c.zone_end)
        proposal=np.full(3,c.accel)
        if len(pending):
            inside=pending[p[pending]>=c.zone_start]
            first=int(inside[0] if len(inside) else pending[np.argmin(np.maximum(0,c.zone_start-p[pending])/np.maximum(v[pending],1))])
            for i in pending:
                if i!=first:
                    distance=max(0,c.zone_start-2-p[i]);target=min(c.vmax,np.sqrt(2*c.brake*distance)*.65)
                    proposal[i]=np.clip((target-v[i])*1.5,-c.brake,c.accel)
        return admissible(proposal,prev,c)

    def select(self,x,prev,proposal):
        start=time.perf_counter();c=self.c;b=backup(prev,c)
        proposal=admissible(proposal,prev,c)
        if self.method=='constant':proposal=admissible(np.full(3,c.accel),prev,c)
        if self.method=='eta':proposal=self.eta(x,prev)
        # Same post-goal cruise lifecycle for every method; completed vehicles
        # must not be rewarded for parking immediately beyond the goal line.
        cleared=x[:,0]>=c.goal
        cruise=admissible(np.full(3,c.accel),prev,c)
        proposal=np.where(cleared,cruise,proposal)
        in_set=bool(certify(x,prev,b,c)[0])
        if self.method=='search':
            candidates=admissible(self.grid,prev,c)
            candidates[:,cleared]=cruise[cleared]
            valid=certify(x,prev,candidates,c)
            # Fixed 2-second open-loop progress surrogate, same certificate.
            forecast=np.repeat(x[None,:,:],len(candidates),axis=0)
            for _ in range(4):advance(forecast,candidates,c)
            score=np.minimum(forecast[:,:,0],c.goal).sum(axis=1)-.03*np.sum((candidates-prev)**2,axis=1)
            chosen=candidates[np.argmax(np.where(valid,score,-np.inf))] if np.any(valid) else b
        elif self.method=='raw_rl':chosen=proposal
        else:
            candidates=np.where(self.bits,proposal,b)
            if self.method=='no_memory':
                # Only the tail loses memory; the first command stays feasible.
                f=np.repeat(x[None,:,:],len(candidates),axis=0)
                valid=advance(f,candidates,c);tail=np.zeros_like(candidates)
                for _ in range(c.tail_steps):tail=backup(tail,c);valid &= advance(f,tail,c)
                valid &= np.all(f[:,:,1]<=1e-10,axis=1)
            else:valid=certify(x,prev,candidates,c)
            if self.method=='individual':
                # Independently authorize each proposal against everyone else's B.
                isolated=np.repeat(b[None,:],3,axis=0);isolated[np.arange(3),np.arange(3)]=proposal
                accepts=certify(x,prev,isolated,c)
                chosen=np.where(accepts,proposal,b)
            else:
                cost=np.sum((candidates-proposal)**2,axis=1)-1e-8*candidates.sum(axis=1)
                chosen=candidates[np.argmin(np.where(valid,cost,np.inf))] if np.any(valid) else b
        verified=bool(certify(x,prev,chosen,c)[0])
        return chosen,{'in_set':in_set,'certified':verified,'intervention':float(np.mean(np.abs(chosen-proposal)>1e-5)),
                      'gate_ms':(time.perf_counter()-start)*1000}


class MergeEnv:
    def __init__(self,cfg=None,method='joint_rl',gui=False):
        self.c=cfg or MergeConfig();self.method=method;self.gui=gui
        self.tmp=tempfile.TemporaryDirectory(prefix='coopgate-v4-');d=Path(self.tmp.name);self.d=d;self.conn=None
        (d/'nodes.xml').write_text('<nodes><node id="a" x="0" y="-50"/><node id="b" x="0" y="50"/><node id="m" x="300" y="0"/><node id="e" x="1500" y="0"/></nodes>')
        (d/'edges.xml').write_text('<edges><edge id="a" from="a" to="m" numLanes="1" speed="20"/><edge id="b" from="b" to="m" numLanes="1" speed="20"/><edge id="out" from="m" to="e" numLanes="1" speed="20"/></edges>')
        subprocess.run([sumolib.checkBinary('netconvert'),'-n',str(d/'nodes.xml'),'-e',str(d/'edges.xml'),'-o',str(d/'net.xml'),'--no-internal-links','true'],capture_output=True,check=True)
        (d/'routes.xml').write_text('<routes><vType id="car" length="4.8" width="1.8" accel="2.5" decel="5" emergencyDecel="9" sigma="0" minGap="0" maxSpeed="20" speedFactor="1" speedDev="0"/><route id="a" edges="a out"/><route id="b" edges="b out"/></routes>')
        self.ids=['ego0','ego1','ego2']

    def reset(self,seed=0,scenario='nominal'):
        self.seed=int(seed);self.scenario=scenario;c=self.c;rng=np.random.default_rng(seed)
        args=['-n',str(self.d/'net.xml'),'-r',str(self.d/'routes.xml'),'--step-length',str(c.dt),
              '--seed',str(seed),'--insertion-checks','collision','--collision.action','warn','--collision.mingap-factor','0',
              '--time-to-teleport','-1','--no-step-log','true','--no-warnings','true','--duration-log.disable','true','--start','true','--quit-on-end','true']
        if self.conn is None:
            label='v4-'+uuid.uuid4().hex;traci.start([sumolib.checkBinary('sumo-gui' if self.gui else 'sumo'),*args],label=label,verbose=False,stdout=subprocess.DEVNULL);self.conn=traci.getConnection(label)
        else:self.conn.load(args)
        v=rng.uniform(8,14,3) if scenario=='nominal' else rng.uniform(12,17,3)
        p=-rng.uniform(70,110,3) if scenario=='nominal' else -rng.uniform(70,85,3)
        p[2]=p[0]-rng.uniform(30,45)
        x=np.column_stack((p,v));prev=np.zeros(3)
        # Validate generated initial states instead of silently rejecting seeds.
        if not certify(x,prev,backup(prev,c),c)[0]:raise RuntimeError('Generated initial state outside contract')
        self.offsets=[]
        for i,ident in enumerate(self.ids):
            route='a' if BRANCH[i]==0 else 'b';length=self.conn.lane.getLength(route+'_0')
            self.conn.vehicle.add(ident,route,'car',depart='now',departPos=str(length+p[i]),departSpeed=str(v[i]))
        self.conn.simulationStep()
        for i,ident in enumerate(self.ids):
            if ident not in self.conn.vehicle.getIDList():raise RuntimeError('Insertion failed')
            self.conn.vehicle.setSpeedMode(ident,96);self.conn.vehicle.setLaneChangeMode(ident,0)
            route='a' if BRANCH[i]==0 else 'b'
            pos=self.conn.vehicle.getLanePosition(ident)-self.conn.lane.getLength(route+'_0')
            self.offsets.append(pos-self.conn.vehicle.getDistance(ident))
        self.x=self.read();self.initial=self.x.copy();self.prev=prev;self.gate=MergeGate(c,self.method)
        if not certify(self.x,prev,backup(prev,c),c)[0]:raise RuntimeError('SUMO initial state outside contract')
        self.t=0.;self.trace=[];self.collision=False;self.violations=0;self.outside=0;self.uncertified=0;self.error=0.
        self.finish=np.full(3,np.nan);self.interventions=[];self.jerks=[]
        if self.gui:self.conn.gui.setBoundary('View #0',160,-70,460,70)
        return self.obs()

    def read(self):return np.array([[self.conn.vehicle.getDistance(v)+self.offsets[i],self.conn.vehicle.getSpeed(v)] for i,v in enumerate(self.ids)])

    def obs(self):
        features=np.column_stack((self.x[:,0]/150,self.x[:,1]/20,self.prev/5,BRANCH))
        return np.array([features[[i]+[j for j in range(3) if j!=i]].flatten() for i in range(3)],dtype=np.float32)

    def step(self,actions):
        c=self.c;old=self.x.copy();previous=self.prev.copy()
        command,info=self.gate.select(self.x,self.prev,ACTIONS[np.asarray(actions,dtype=int)])
        predicted=self.x[None,:,:].copy();advance(predicted,command[None,:],c)
        for _ in range(round(c.decision_dt/c.dt)):
            before=self.x.copy()
            for i,ident in enumerate(self.ids):self.conn.vehicle.setSpeed(ident,float(np.clip(self.x[i,1]+command[i]*c.dt,0,c.vmax)))
            self.conn.simulationStep();self.x=self.read()
            self.collision |= bool(self.conn.simulation.getCollidingVehiclesIDList())
            self.violations+=not bool(transition_safe(before[:,0][None,:],self.x[:,0][None,:],c)[0])
        self.t+=c.decision_dt;self.prev=command
        self.error=max(self.error,float(np.max(np.abs(predicted[0]-self.x))))
        self.outside+=not info['in_set'];self.uncertified+=not info['certified'];self.interventions.append(info['intervention'])
        jerk=float(np.mean(np.abs(command-previous))/c.decision_dt);self.jerks.append(jerk)
        newly=(self.x[:,0]>=c.goal)&np.isnan(self.finish);self.finish[newly]=self.t
        progress=float(np.sum(np.minimum(self.x[:,0],c.goal)-np.minimum(old[:,0],c.goal)))
        reward=progress/30+.5*newly.sum()-.04*jerk-.05*info['intervention']-.05*np.isnan(self.finish).sum()-30*self.collision
        terminal=self.collision or bool(np.all(np.isfinite(self.finish)));truncated=self.t>=c.horizon and not terminal
        self.trace.append({'t':self.t,'x':self.x.tolist(),'command':command.tolist(),**info})
        return self.obs(),float(reward),terminal,truncated,self.metrics()

    def metrics(self):
        return {'seed':self.seed,'scenario':self.scenario,'method':self.method,'collision':self.collision,
            'completed':bool(np.all(np.isfinite(self.finish))) and not self.collision,'elapsed':self.t,
            'mean_capped_finish_s':float(np.nan_to_num(self.finish,nan=self.c.horizon).mean()) if not self.collision else self.c.horizon,
            'contract_violations':self.violations,'outside_steps':self.outside,'uncertified_steps':self.uncertified,
            'max_prediction_error':self.error,'mean_jerk':float(np.mean(self.jerks)) if self.jerks else 0.,
            'intervention':float(np.mean(self.interventions)) if self.interventions else 0.}
    def close(self):
        if self.conn is not None:self.conn.close();self.conn=None
        self.tmp.cleanup()
    def __enter__(self):return self
    def __exit__(self,*a):self.close()
