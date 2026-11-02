"""Generate a static scalar implementation of the ORIGINAL nominal certificate.

Unroll the fixed three-car indexing, preserving expression order, float64,
geometric predicates, stop test and every downstream planner/forecast body.
Only prediction-internal nominal planning is rebound; physical baseline and
the actual command's actuator-memory certificate remain unchanged.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path

from coopgate import buffered_baseline_v25 as original
from coopgate import robust_v9 as geometry
from coopgate import zero_score_projection as projection
from coopgate import memoized_recovery_v37 as memo

ROOT=Path(__file__).resolve().parents[1]


class Scalarize(ast.NodeTransformer):
    def __init__(self,indices=None):self.indices=indices or {}

    def visit_Name(self,node):
        if node.id in self.indices:return ast.copy_location(ast.Constant(self.indices[node.id]),node)
        return node

    def visit_Subscript(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id in ('lo','hi','qlo','qhi'):
            if not isinstance(node.slice,ast.Tuple):raise ValueError('Unexpected array access')
            i,j=node.slice.elts
            if not isinstance(i,ast.Constant) or not isinstance(j,ast.Constant):raise ValueError('Unresolved index')
            return ast.copy_location(ast.Name(f'{node.value.id}{i.value}_{j.value}',node.ctx),node)
        return node

    def visit_For(self,node):
        if isinstance(node.target,ast.Tuple) and [x.id for x in node.target.elts]==['i','j']:
            result=[]
            for i,j in ((0,1),(1,2)):
                for statement in node.body:result.append(Scalarize(dict(i=i,j=j)).visit(copy.deepcopy(statement)))
            return result
        return self.generic_visit(node)


def source(function):return inspect.getsource(function.py_func)


def build():
    fn=ast.parse(source(geometry.interval_transition)).body[0]
    fn.name='scalar_transition'
    names=[f'{a}{i}_0' for a in ('lo','hi','qlo','qhi') for i in range(3)]
    fn.args.args=[ast.arg(arg=n) for n in names+['c']]
    fn.decorator_list=ast.parse('@njit(cache=True,fastmath=False,inline="always")\ndef f():pass').body[0].decorator_list
    fn=Scalarize().visit(fn)
    transition=ast.unparse(ast.fix_missing_locations(fn))+'\n\n'
    certificate=['@njit(cache=True,fastmath=False)',
        'def robust_certificate(state,previous,command,c,radius):']
    for i in range(3):
        certificate += [f'    bounded=clip(command[{i}],max(-c[4],previous[{i}]-c[6]*c[1]),min(c[5],previous[{i}]+c[6]*c[1]))',
            f'    if abs(command[{i}]-bounded)>=1e-9:return False',
            f'    lo{i}_0=hi{i}_0=state[{i},0]',f'    lo{i}_1=hi{i}_1=state[{i},1]',f'    tail{i}=command[{i}]']
    certificate += ['    for stage in range(1+int(c[10])):', '        if stage>0:']
    certificate += [f'            tail{i}=max(tail{i}-c[6]*c[1],-c[4])' for i in range(3)]
    certificate += ['        for _ in range(round(c[1]/c[0])):']
    for i in range(3):
        certificate += [f'            qlo{i}_1=clip(lo{i}_1+(tail{i}-radius)*c[0],0.,c[3])',
            f'            qhi{i}_1=clip(hi{i}_1+(tail{i}+radius)*c[0],0.,c[3])',
            f'            qlo{i}_0=lo{i}_0+qlo{i}_1*c[0]',f'            qhi{i}_0=hi{i}_0+qhi{i}_1*c[0]']
    certificate += ['            if not scalar_transition('+','.join(names)+',c):return False']
    for a in ('lo','hi'):
        for i in range(3):
            for j in (0,1):certificate += [f'            {a}{i}_{j}=q{a}{i}_{j}']
    certificate += ['        if stage>0 and '+' and '.join([f'hi{i}_1==0.' for i in range(3)]+[f'tail{i}<=-radius' for i in range(3)])+':return True','    return False','']
    header='''"""Generated exact scalar nominal certificate; independent runtime prototype.

See tools/build_scalar_nominal_recovery.py. No approximate math, path removal,
new MAPPO weights or altered physical baseline. Forecast semantics identical.
"""
import numpy as np
from numba import njit
from .fast_v8 import clip,advance_one,transition_values
from .robust_v9 import occupancy
from .marl_v12 import LEVELS
from .single_v12 import constant_proposal
from .actuator_recovery_v34 import memory_step,stopping_certificate
from .arrival_phase_v20 import crossing_time
from .memoized_recovery_v37 import proposal_key,make_cache,CAPACITY
from .parameter_scenario_gate import ParameterScenarioGate

'''
    blocks=[source(projection.project_joint)]+[source(getattr(original,n)) for n in ('team_rule_tail','deduplicated_joint_rollout')]
    blocks += [source(getattr(memo,n)) for n in ('cached_proposal','recovery_command','forecast_kernel')]
    footer='''
class ScalarNominalParameterGate(ParameterScenarioGate):
    def predict(self,state,previous,mlo,mhi,command,path):
        if getattr(self,'_cache_step',None)!=self.step:
            self._plan_cache,self._cache_stats=make_cache();self._cache_step=self.step
        if path<self.original_paths:
            gain,drag,rho,memory=self.gain,self.drag,self.retention,self.memory
        else:
            index=path-self.original_paths
            gain,drag,rho=self.parameter_scenarios[index]
            memory=self.parameter_memories[index]
            if np.any(memory<mlo-1e-12) or np.any(memory>mhi+1e-12):
                raise ValueError('Command-history scenario memory escaped the declared interval')
        return forecast_kernel(state,previous,memory,mlo,mhi,self.step,self.tapes[path],gain,drag,rho,
            self.bounds,self.packed,np.zeros(3) if command is None else command,command is not None,
            self._plan_cache,self._cache_stats,CAPACITY)
'''
    inputs=[Path(inspect.getsourcefile(m)) for m in (original,geometry,projection,memo)]
    manifest=''.join('# input '+p.name+' sha256='+hashlib.sha256(p.read_bytes()).hexdigest()+'\n' for p in inputs)
    result=manifest+header+transition+'\n'.join(certificate)+'\n\n'+'\n\n'.join(blocks)+footer
    ast.parse(result)
    target=ROOT/'src/coopgate/scalar_nominal_recovery.py'
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(result)
    print(str(target),flush=True)


if __name__=='__main__':build()
