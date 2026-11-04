"""Explicit full native reload and TraCI control-mode restoration for DMPS probes.

The quick-load counterexample and original implementation remain immutable.
This still exposes privileged true dynamics/tapes and does not train DMPS.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .dmps_sumo_snapshot import DMPSSumoSnapshotEnv, Snapshot


@dataclass
class ControlledSnapshot(Snapshot):
    speed_modes: tuple
    lane_change_modes: tuple


class DMPSReloadEnv(DMPSSumoSnapshotEnv):
    def get_state(self):
        original=super().get_state()
        return ControlledSnapshot(**vars(original),
            speed_modes=tuple(self.raw.conn.vehicle.getSpeedMode(v) for v in self.raw.ids),
            lane_change_modes=tuple(self.raw.conn.vehicle.getLaneChangeMode(v) for v in self.raw.ids))

    def reload_args(self,state,path):
        c=self.raw.c
        # Preserve original input files, actuator integration, collision policy,
        # and seed. _Connection.load appends the two save-state options.
        return ['-n',str(self.raw.d/'net.xml'),'-r',str(self.raw.d/'routes.xml'),
            '--step-length',str(c.dt),'--seed',str(self.seed),'--insertion-checks','collision',
            '--collision.action','warn','--collision.mingap-factor','0','--time-to-teleport','-1',
            '--no-step-log','true','--no-warnings','true','--duration-log.disable','true',
            '--start','true','--quit-on-end','true','--load-state',str(path),
            '--begin',str(state.simulator_time)]

    def set_state(self,state):
        if not isinstance(state,ControlledSnapshot) or (state.owner,state.generation)!=(self.owner,self.generation):
            raise ValueError('Controlled snapshot belongs to another simulator or reset generation')
        if state.config!=self.raw.c.dict():raise ValueError('Physical configuration changed after snapshot')
        ids=state.host['ids']
        if len(state.speed_modes)!=len(ids) or len(state.lane_change_modes)!=len(ids):
            raise ValueError('Incomplete TraCI control modes')
        path=self.raw.d/'dmps-reload.xml';path.write_bytes(state.native)
        self.raw.conn.load(self.reload_args(state,path))
        for name in list(self.raw.__dict__):
            if name not in self._EXCLUDED:del self.raw.__dict__[name]
        self.raw.__dict__.update(deepcopy(state.host))
        self.rng.bit_generator.state=deepcopy(state.sampler_state)
        for ident,speed_mode,lane_mode in zip(ids,state.speed_modes,state.lane_change_modes):
            self.raw.conn.vehicle.setSpeedMode(ident,speed_mode)
            self.raw.conn.vehicle.setLaneChangeMode(ident,lane_mode)
            if (self.raw.conn.vehicle.getSpeedMode(ident)!=speed_mode or
                    self.raw.conn.vehicle.getLaneChangeMode(ident)!=lane_mode):
                raise ValueError('TraCI mode restoration failed')
        self.snapshots_loaded+=1
        np.testing.assert_allclose(self.raw.read(),self.raw.x,atol=1e-8,rtol=0)
        if self.raw.conn.simulation.getTime()!=state.simulator_time:
            raise ValueError('Full reload did not restore SUMO clock')
