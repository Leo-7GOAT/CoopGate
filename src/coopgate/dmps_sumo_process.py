"""Isolate each snapshot restore in a fresh native SUMO process.

This engineering reference prioritizes truthful branches, not online speed.
Only the bridge-owned SUMO connection is closed. Other simulation workers and
all Python processes remain untouched.
"""
import subprocess
import uuid

import sumolib

from .dmps_sumo_snapshot import _TraCI
from .dmps_sumo_reload import DMPSReloadEnv
from .dmps_sumo_clock import AbsoluteStepConnection


class ProcessConnection:
    def __init__(self, original, dt):
        self.dt=dt
        self.original=AbsoluteStepConnection(original,dt)
        self.native_restarts=0

    def load(self,args):
        self.original.close()
        label='dmps-snapshot-'+uuid.uuid4().hex
        api=_TraCI()
        api.start([sumolib.checkBinary('sumo'),*args],label=label,verbose=False,stdout=subprocess.DEVNULL)
        self.original=AbsoluteStepConnection(api.getConnection(label),self.dt)
        self.native_restarts+=1

    def __getattr__(self,name):
        return getattr(self.original,name)


class DMPSProcessEnv(DMPSReloadEnv):
    def reset(self):
        observation=super().reset()
        if not isinstance(self.raw.conn,ProcessConnection):
            self.raw.conn=ProcessConnection(self.raw.conn,self.raw.c.dt)
        return observation
