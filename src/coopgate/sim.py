"""Real Eclipse SUMO backend, controlled over TraCI.

No hidden simulator collision avoidance is credited to the learned/gated policy:
controlled speedMode=96 and laneChangeMode=0 disable those interventions.
Background vehicles retain SUMO's car-following and lane-change models.
"""
from pathlib import Path
import subprocess
import tempfile
import uuid

import numpy as np
import sumolib
import traci

from .config import Config, OBS_DIM
from .controllers import SpeedMPC, decode_action
from .shield import Supervisor


def build_network(directory):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d/'nodes.xml').write_text('''<nodes>
<node id="start" x="0" y="0" type="priority"/>
<node id="merge" x="220" y="0" type="priority"/>
<node id="end" x="600" y="0" type="priority"/>
</nodes>''')
    (d/'edges.xml').write_text('''<edges>
<edge id="approach" from="start" to="merge" numLanes="2" speed="23" width="3.2"/>
<edge id="exit" from="merge" to="end" numLanes="1" speed="23" width="3.2"/>
</edges>''')
    (d/'connections.xml').write_text('''<connections>
<connection from="approach" to="exit" fromLane="0" toLane="0"/>
<connection from="approach" to="exit" fromLane="1" toLane="0"/>
</connections>''')
    subprocess.run([sumolib.checkBinary('netconvert'), '-n', str(d/'nodes.xml'),
                    '-e', str(d/'edges.xml'), '-x', str(d/'connections.xml'),
                    '-o', str(d/'road.net.xml'), '--no-turnarounds', 'true',
                    '--no-internal-links', 'true'], check=True, capture_output=True)
    (d/'types.rou.xml').write_text('''<routes>
<vType id="controlled" length="4.8" width="1.8" accel="2.5" decel="5" emergencyDecel="9" sigma="0" minGap="0" maxSpeed="23"/>
<vType id="human" length="4.8" width="1.8" accel="2" decel="4.5" sigma="0.3" minGap="2.5" maxSpeed="23"/>
<route id="route" edges="approach exit"/>
</routes>''')
    return d


class SumoEnv:
    def __init__(self, cfg=None, method='mappo', gui=False, workdir=None):
        self.cfg = cfg or Config()
        if self.cfg.n_agents != 3:
            raise ValueError('The bundled scenario currently supports exactly three controlled vehicles.')
        self.method, self.gui = method, gui
        self.tmp = tempfile.TemporaryDirectory(prefix='coopgate-') if workdir is None else None
        self.directory = build_network(workdir or self.tmp.name)
        self.connection = None
        self.label = 'coopgate-'+uuid.uuid4().hex
        self.ids = ['ego0', 'ego1', 'ego2', 'human0', 'human1']
        self.mpcs = [SpeedMPC(self.cfg) for _ in range(self.cfg.n_agents)]

    def reset(self, seed=0, scenario='nominal'):
        c = self.cfg
        self.seed, self.scenario = int(seed), scenario
        self.rng = np.random.default_rng(seed)
        args = ['-n', str(self.directory/'road.net.xml'), '-r', str(self.directory/'types.rou.xml'),
                '--step-length', str(c.sim_dt), '--seed', str(seed),
                '--lanechange.duration', str(c.lane_change_seconds), '--collision.action', 'warn',
                '--collision.mingap-factor', '0', '--time-to-teleport', '-1',
                '--no-step-log', 'true', '--no-warnings', 'true', '--duration-log.disable', 'true',
                '--start', 'true', '--quit-on-end', 'true']
        if self.connection is None:
            traci.start([sumolib.checkBinary('sumo-gui' if self.gui else 'sumo'), *args],
                        label=self.label, verbose=False, stdout=subprocess.DEVNULL)
            self.connection = traci.getConnection(self.label)
        else:
            self.connection.load(args)
        self.gate = Supervisor(c, self.method)
        self.elapsed = 0.0
        self.steps = 0
        self.previous_accel = np.zeros(c.n_agents)
        self.last_modes = np.zeros(c.n_agents, bool)
        self.finished = np.zeros(c.n_agents, bool)
        self.merged = np.array([False, True, True])
        for mpc in self.mpcs:
            mpc.failures = 0
        self.collided_ids = set()
        self.gate_ms, self.checks, self.jerk = [], [], []
        self.switches = self.accepted = self.decisions = self.emergencies = 0
        self.minimum_gap = float('inf')
        self.trace = []
        spacing = 0.7 if scenario == 'dense' else 1.0
        positions = [65, 65+29*spacing, 65-29*spacing, 142, 5]
        lanes = [0, 1, 1, 1, 0]
        for i, ident in enumerate(self.ids):
            v = float(self.rng.uniform(10, 16))
            pos = max(0, positions[i]+self.rng.uniform(-3, 3))
            self.connection.vehicle.add(ident, 'route', 'controlled' if i < 3 else 'human',
                                        depart='now', departLane=str(lanes[i]),
                                        departPos=str(pos), departSpeed=str(v))
        self.connection.simulationStep()
        for i, ident in enumerate(self.ids[:3]):
            if ident not in self.connection.vehicle.getIDList():
                raise RuntimeError('Scenario insertion failed: '+ident)
            self.connection.vehicle.setSpeedMode(ident, 96)
            self.connection.vehicle.setLaneChangeMode(ident, 0)
            self.connection.vehicle.setColor(ident, [(40,155,240,255),(20,190,150,255),(230,170,40,255)][i])
        if self.gui:
            self.connection.gui.setBoundary('View #0', 0, -30, 450, 30)
        self.states = self._read_states()
        return self.observation()

    def _read_states(self):
        states = np.zeros((len(self.ids), 5), dtype=np.float64)
        present = set(self.connection.vehicle.getIDList())
        for i, ident in enumerate(self.ids):
            if ident not in present:
                if hasattr(self, 'states'):
                    states[i, :4] = self.states[i, :4]
                continue
            x, y = self.connection.vehicle.getPosition(ident)
            road = self.connection.vehicle.getRoadID(ident)
            # Logical centers are 0 (merging lane), 3.2 (continuing lane).
            logical_y = y + 4.8 if road == 'approach' else self.cfg.lane_width
            states[i] = [x, logical_y, self.connection.vehicle.getSpeed(ident),
                         self.connection.vehicle.getAcceleration(ident), 1]
        return states

    def observation(self):
        obs = np.zeros((3, OBS_DIM), dtype=np.float32)
        for i in range(3):
            s = self.states[i]
            obs[i, :5] = [s[0]/600, s[1]/3.2, s[2]/23, s[3]/5, float(self.last_modes[i])]
            others = sorted((j for j in range(5) if j != i), key=lambda j: abs(self.states[j,0]-s[0]))
            for k, j in enumerate(others):
                o = self.states[j]
                obs[i, 5+5*k:10+5*k] = [np.clip((o[0]-s[0])/100,-3,3), (o[1]-s[1])/3.2,
                                        (o[2]-s[2])/23, o[4], float(j<3)]
        return obs

    def step(self, actions):
        c = self.cfg
        actions = np.asarray(actions)
        if actions.shape != (3,) or np.any(actions < 0) or np.any(actions >= 9):
            raise ValueError('Expected three discrete actions in [0, 8].')
        old = self.states.copy()
        learned = [decode_action(actions[i], int(old[i,1] > 1.6), old[i,0], c) for i in range(3)]
        result = self.gate.select(old, learned)
        self.gate_ms.append(result.elapsed_ms)
        self.checks.append(result.checks)
        active = ~self.finished
        self.switches += int(np.count_nonzero((result.modes != self.last_modes) & active)) if self.steps else 0
        self.accepted += int(np.sum(result.modes & active))
        self.decisions += int(np.sum(active))
        self.emergencies += int(result.emergency)
        self.last_modes = result.modes.copy()
        for i, ident in enumerate(self.ids[:3]):
            if not old[i,4]:
                continue
            intent = result.intents[i]
            acc = self.mpcs[i].control(old[i,2], intent.speed, self.previous_accel[i], result.emergency)
            self.jerk.append(abs(acc-self.previous_accel[i])/c.decision_dt)
            self.previous_accel[i] = acc
            self.connection.vehicle.setAcceleration(ident, acc, c.decision_dt)
            if old[i,0] < c.merge_end-3:
                self.connection.vehicle.changeLane(ident, intent.lane, c.decision_dt)
        if self.scenario == 'braking' and 2 <= self.elapsed < 4:
            if 'human0' in self.connection.vehicle.getIDList():
                self.connection.vehicle.slowDown('human0', 3.0, 1.0)
        for _ in range(round(c.decision_dt/c.sim_dt)):
            self.connection.simulationStep()
            self.collided_ids.update(self.connection.simulation.getCollidingVehiclesIDList())
        self.elapsed += c.decision_dt
        self.steps += 1
        self.states = self._read_states()
        progress = np.clip(self.states[:3,0]-old[:3,0], 0, 15)
        # Do not treat an unexplained disappearance as successful completion.
        self.finished |= self.states[:3,0] >= c.goal_x
        collision = any(k in self.collided_ids for k in self.ids[:3])
        for i in range(3):
            for j in range(i+1, len(self.ids)):
                if self.states[i,4] and self.states[j,4] and abs(self.states[i,1]-self.states[j,1]) < 2.1:
                    gap = abs(self.states[i,0]-self.states[j,0])-c.vehicle_length
                    self.minimum_gap = min(self.minimum_gap, gap)
        new_merged = (self.states[:3,1] >= 1.6) & ~self.merged
        merge_bonus = float(np.sum(new_merged))
        self.merged |= new_merged
        reward = float(progress.mean()/10 - 0.015*np.square(self.previous_accel).mean() + 0.4*merge_bonus)
        reward -= 20.0*collision
        success = bool(np.all(self.finished))
        if success:
            reward += 10
        terminated = collision or success
        truncated = self.elapsed >= c.episode_seconds and not terminated
        self.trace.append({'t':self.elapsed,'states':self.states.tolist(), 'modes':result.modes.astype(int).tolist(),
                           'score':result.score,'margin':result.margin,'emergency':result.emergency,
                           'reward':reward,'actions':actions.tolist()})
        info = self.metrics()
        return self.observation(), reward, terminated, truncated, info

    def metrics(self):
        return {'seed':self.seed,'scenario':self.scenario,'method':self.method,
                'collision':any(k in self.collided_ids for k in self.ids[:3]),
                'success':bool(np.all(self.finished)), 'elapsed_s':self.elapsed,
                'completion_fraction':float(self.finished.mean()),
                'progress_m':float(np.minimum(self.states[:3,0],self.cfg.goal_x).mean()),
                'rl_fraction':self.accepted/max(1,self.decisions),'switches':self.switches,
                'emergency_steps':self.emergencies,
                'gate_mean_ms':float(np.mean(self.gate_ms)) if self.gate_ms else 0,
                'gate_p95_ms':float(np.percentile(self.gate_ms,95)) if self.gate_ms else 0,
                'checks_per_step':float(np.mean(self.checks)) if self.checks else 0,
                'mean_abs_jerk':float(np.mean(self.jerk)) if self.jerk else 0,
                'min_gap_m':self.minimum_gap if np.isfinite(self.minimum_gap) else None,
                'mpc_failures':sum(m.failures for m in self.mpcs)}

    def close(self):
        if self.connection is not None:
            self.connection.close(wait=True)
            self.connection = None
        if self.tmp:
            self.tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
