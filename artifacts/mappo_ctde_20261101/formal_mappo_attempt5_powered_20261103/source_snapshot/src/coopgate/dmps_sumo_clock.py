"""Explicit absolute TraCI step targets after full state reload."""
import numpy as np

from .dmps_sumo_reload import DMPSReloadEnv


class AbsoluteStepConnection:
    def __init__(self, original, dt):
        self.original,self.dt=original,dt

    def simulationStep(self, step=0.):
        if step != 0.:
            return self.original.simulationStep(step)
        current=self.original.simulation.getTime()
        target=(round(current/self.dt)+1)*self.dt
        result=self.original.simulationStep(target)
        if not np.isclose(self.original.simulation.getTime(),target,atol=1e-10,rtol=0):
            raise RuntimeError('Native simulation did not advance exactly one physical substep')
        return result

    def __getattr__(self,name):
        return getattr(self.original,name)


class DMPSClockEnv(DMPSReloadEnv):
    def reset(self):
        observation=super().reset()
        if not isinstance(self.raw.conn,AbsoluteStepConnection):
            self.raw.conn=AbsoluteStepConnection(self.raw.conn,self.raw.c.dt)
        return observation
