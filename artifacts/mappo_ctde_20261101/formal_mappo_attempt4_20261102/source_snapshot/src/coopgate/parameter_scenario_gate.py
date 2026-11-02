"""Finite actuator-parameter scenarios reconstructed from executed commands.

Original33 midpoint/noise paths remain. Eight extra clean paths use declared
class endpoints. This is finite scenario evidence,not full-box robustness or
a calibrated probability. No realized coefficients/memory/tape is consumed.
"""
from itertools import product
import time
import numpy as np

from .actuator_recovery_v34 import class_bounds
from .memoized_recovery_v37 import forecast_kernel,make_cache,CAPACITY
from .scenario_confidence_gate import ScenarioConfidenceGate


def parameter_bank(profile):
    bounds=class_bounds(profile);result=[]
    if profile=='nominal':return result
    for bits in product((0,1),repeat=3):
        # Low gain/high drag/slow response versus the opposite paired corner.
        # For gain_drag these8 corners are NOT the full64 gain/drag box vertices.
        gain=np.array([bounds[i,bits[i]] for i in range(3)])
        drag=np.array([bounds[i,3-bits[i]] for i in range(3)])
        rho=np.array([bounds[i,5-bits[i]] for i in range(3)])
        result.append((gain,drag,rho))
    return result


class ParameterScenarioGate(ScenarioConfidenceGate):
    def __init__(self,model,profile,max_failed_paths=0):
        if max_failed_paths!=0:raise ValueError('This independent prototype registers strict budget0 only')
        super().__init__(model,profile,0)
        self.original_paths=len(self.tapes);self.parameter_scenarios=parameter_bank(profile)
        self.parameter_memories=np.zeros((len(self.parameter_scenarios),3))
        self.names=tuple(self.names)+tuple('class_parameter_corner:'+str(i) for i in range(len(self.parameter_scenarios)))
        if self.parameter_scenarios:
            self.tapes=np.concatenate((self.tapes,np.zeros((len(self.parameter_scenarios),250,3))))

    def predict(self,state,previous,mlo,mhi,command,path):
        if path<self.original_paths:return super().predict(state,previous,mlo,mhi,command,path)
        index=path-self.original_paths;gain,drag,rho=self.parameter_scenarios[index]
        memory=self.parameter_memories[index]
        if np.any(memory<mlo-1e-12) or np.any(memory>mhi+1e-12):
            raise ValueError('Command-history scenario memory escaped the declared interval')
        if getattr(self,'_cache_step',None)!=self.step:
            self._plan_cache,self._cache_stats=make_cache();self._cache_step=self.step
        return forecast_kernel(state,previous,memory,mlo,mhi,self.step,self.tapes[path],gain,drag,rho,
            self.bounds,self.packed,np.zeros(3) if command is None else command,command is not None,
            self._plan_cache,self._cache_stats,CAPACITY)

    def select(self,state,previous,unused=None):
        started=time.perf_counter();previous=np.asarray(previous,dtype=float)
        if previous.shape!=(3,) or not np.isfinite(previous).all():raise ValueError('Finite executed previous command required')
        if self.step:
            for index,(_,_,rho) in enumerate(self.parameter_scenarios):
                for _ in range(5):self.parameter_memories[index]=rho*self.parameter_memories[index]+(1-rho)*previous
        command,info=super().select(state,previous,unused)
        info.update(rule='parameter_scenario_confidence',original_midpoint_paths=self.original_paths,
            parameter_scenario_count=len(self.parameter_scenarios),
            reconstructed_parameter_memories=self.parameter_memories.tolist(),
            parameter_scenarios=[dict(gain=g.tolist(),drag=d.tolist(),retention=r.tolist()) for g,d,r in self.parameter_scenarios],
            full_parameter_box_certified=False,gate_ms=1000*(time.perf_counter()-started))
        return command,info
