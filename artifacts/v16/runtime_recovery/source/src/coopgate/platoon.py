"""Real SUMO benchmark matching the scoped V3 longitudinal contract."""
from pathlib import Path
import subprocess,tempfile,uuid
import numpy as np
import sumolib,traci
from .recovery import RecoveryConfig,RecoverySupervisor,backup_acceleration,integrate,witnesses

ACCEL_ACTIONS=np.array([-5.,-2.5,0.,1.25,2.5])
PLATOON_OBS=12


def initial_state(seed,cfg):
    rng=np.random.default_rng(seed)
    x=np.zeros((4,5));x[:,1]=cfg.lane_width;x[:,4]=1
    x[:,2]=rng.uniform(12,19,4)
    # Determine required adjacent gaps from relative stopping trajectories.
    minimum,_=witnesses(x,np.zeros(3),cfg,per_pair=True)
    gaps=-minimum[0]+rng.uniform(2,10,3)
    x[3,0]=200
    for i,front in enumerate((3,0,1)):x[i,0]=x[front,0]-gaps[i]
    return x


class PlatoonEnv:
    def __init__(self,cfg=None,method='joint_rl',gui=False):
        self.cfg=cfg or RecoveryConfig();self.method=method;self.gui=gui
        self.tmp=tempfile.TemporaryDirectory(prefix='coopgate-v3-')
        d=Path(self.tmp.name)
        (d/'nodes.xml').write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="1200" y="0"/></nodes>')
        (d/'edges.xml').write_text('<edges><edge id="road" from="a" to="b" numLanes="1" speed="20"/></edges>')
        subprocess.run([sumolib.checkBinary('netconvert'),'-n',str(d/'nodes.xml'),'-e',str(d/'edges.xml'),'-o',str(d/'road.net.xml')],capture_output=True,check=True)
        (d/'types.xml').write_text('<routes><vType id="car" length="4.8" width="1.8" accel="2.5" decel="5" emergencyDecel="9" sigma="0" speedFactor="1" speedDev="0" minGap="0" maxSpeed="20"/><route id="r" edges="road"/></routes>')
        self.directory=d;self.connection=None;self.ids=['ego0','ego1','ego2','lead']

    def reset(self,seed=0,scenario='cruise'):
        self.seed=int(seed);self.scenario=scenario;c=self.cfg
        d=self.directory
        args=['-n',str(d/'road.net.xml'),'-r',str(d/'types.xml'),'--step-length',str(c.sim_dt),
              '--insertion-checks','collision',
              '--seed',str(seed),'--collision.action','warn','--collision.mingap-factor','0',
              '--time-to-teleport','-1','--no-step-log','true','--no-warnings','true','--duration-log.disable','true',
              '--start','true','--quit-on-end','true']
        if self.connection is None:
            label='coopgate-v3-'+uuid.uuid4().hex
            traci.start([sumolib.checkBinary('sumo-gui' if self.gui else 'sumo'),*args],label=label,verbose=False,stdout=subprocess.DEVNULL)
            self.connection=traci.getConnection(label)
        else:self.connection.load(args)
        x=initial_state(seed,c)
        for i,ident in enumerate(self.ids):
            self.connection.vehicle.add(ident,'r','car',depart='now',departPos=str(x[i,0]),departSpeed=str(x[i,2]))
        self.connection.simulationStep()
        for ident in self.ids:
            if ident not in self.connection.vehicle.getIDList():raise RuntimeError('Failed to insert '+ident)
            self.connection.vehicle.setSpeedMode(ident,96);self.connection.vehicle.setLaneChangeMode(ident,0)
        self.states=self.read();self.initial=self.states.copy();self.previous=np.zeros(3)
        self.supervisor=RecoverySupervisor(c,self.method)
        self.elapsed=0.;self.trace=[];self.collided=False;self.min_gap=float('inf')
        self.mismatches=[];self.outside=0;self.uncertified=0;self.interventions=[];self.retained=[];self.latencies=[]
        self.jerks=[];self.gap_violation_steps=0
        rng=np.random.default_rng(seed+710000)
        self.onset=float(rng.uniform(3,7));self.duration=float(rng.uniform(1.5,3))
        margin,terminal=witnesses(self.states,self.previous,c)
        if not terminal[0] or margin[0]<c.witness_buffer:raise RuntimeError('Initial state outside recovery contract')
        if self.gui:self.connection.gui.setBoundary('View #0',100,-30,800,30)
        return self.observation()

    def read(self):
        return np.array([[self.connection.vehicle.getPosition(v)[0],self.cfg.lane_width,
            self.connection.vehicle.getSpeed(v),self.connection.vehicle.getAcceleration(v),1.] for v in self.ids])

    def observation(self):
        observations=[]
        for i in range(3):
            features=[self.states[i,2]/20,self.previous[i]/5,i/2]
            for j in range(4):
                if i==j:continue
                features.extend([(self.states[j,0]-self.states[i,0])/100,self.states[j,2]/20,self.states[j,3]/9])
            observations.append(features)
        return np.asarray(observations,dtype=np.float32)

    def leader_acceleration(self):
        if self.scenario=='cruise':return 1.
        if self.onset<=self.elapsed<self.onset+self.duration:
            return -3. if self.scenario=='moderate' else -9.
        if self.scenario=='double_brake' and self.onset+7<=self.elapsed<self.onset+7+self.duration:return -9.
        return 2.

    def step(self,actions):
        c=self.cfg;before=self.states.copy();old_command=self.previous.copy()
        command,info=self.supervisor.select(before,old_command,ACCEL_ACTIONS[np.asarray(actions,dtype=int)])
        lead=self.leader_acceleration()
        predicted=before[None,:,:].copy();integrate(predicted,command[None,:],lead,c)
        for _ in range(round(c.decision_dt/c.sim_dt)):
            # Explicit sampled acceleration adapter: setAcceleration can exceed the
            # speed cap in SUMO. This bridge realizes exactly the declared plant.
            for i,ident in enumerate(self.ids):
                acceleration=float(command[i]) if i<3 else lead
                v=self.connection.vehicle.getSpeed(ident)
                self.connection.vehicle.setSpeed(ident,float(np.clip(v+acceleration*c.sim_dt,0,c.max_speed)))
            self.connection.simulationStep()
            self.collided |= bool(self.connection.simulation.getCollidingVehiclesIDList())
            s=self.read();g=s[[3,0,1],0]-s[[0,1,2],0]-c.vehicle_length
            self.min_gap=min(self.min_gap,float(np.min(g)))
            self.gap_violation_steps+=int(np.min(g)<c.safety_gap-1e-6)
        self.states=self.read();self.elapsed+=c.decision_dt;self.previous=command.copy()
        error=float(np.max(np.abs(predicted[0,:,[0,2]]-self.states[:,[0,2]].T)))
        self.mismatches.append(error);self.outside+=not info['in_recoverable_set'];self.uncertified+=not info['chosen_certified']
        self.interventions.append(info['intervention_fraction']);self.retained.append(info['rl_retained_fraction']);self.latencies.append(info['gate_ms'])
        jerk=float(np.mean(np.abs(command-old_command))/c.decision_dt);self.jerks.append(jerk)
        progress=float(np.mean(self.states[:3,0]-before[:3,0]))
        reward=progress/(c.desired_speed*c.decision_dt)-.01*np.mean((command/5)**2)-.02*(jerk/c.max_jerk)**2-.1*info['intervention_fraction']-50*self.collided
        self.trace.append({'t':self.elapsed,'states':self.states.tolist(),'commands':command.tolist(),'lead_acceleration':lead,'diagnostics':info,'prediction_error':error})
        done=self.collided or self.elapsed>=c.episode_seconds
        return self.observation(),float(reward),self.collided,done and not self.collided,self.metrics()

    def metrics(self):
        progress=float(np.mean(self.states[:3,0]-self.initial[:3,0]))
        return {'seed':self.seed,'scenario':self.scenario,'method':self.method,'collision':self.collided,
            'completed_horizon':self.elapsed>=self.cfg.episode_seconds and not self.collided,'elapsed_s':self.elapsed,
            'mean_progress_m':progress,'collision_penalized_progress_m':0. if self.collided else progress,
            'min_gap_m':self.min_gap,'gap_violation_substeps':self.gap_violation_steps,'outside_set_steps':self.outside,
            'uncertified_action_steps':self.uncertified,'intervention_fraction':float(np.mean(self.interventions)) if self.interventions else 0.,
            'rl_retained_fraction':float(np.mean(self.retained)) if self.retained else 0.,
            'max_prediction_error':max(self.mismatches,default=0.),'mean_abs_jerk':float(np.mean(self.jerks)) if self.jerks else 0.,
            'gate_mean_ms':float(np.mean(self.latencies)) if self.latencies else 0.,'gate_p95_ms':float(np.percentile(self.latencies,95)) if self.latencies else 0.}

    def close(self):
        if self.connection is not None:self.connection.close(wait=True);self.connection=None
        self.tmp.cleanup()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
