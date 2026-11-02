"""Generate rejection-only min-confidence pruning from the frozen naive gate."""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
from coopgate.gain_margin_gate import ScalarGainMarginGate

ROOT=Path(__file__).resolve().parents[1]


def build():
    text=textwrap.dedent(inspect.getsource(ScalarGainMarginGate._select));fn=ast.parse(text).body[0]
    fn.body[0:0]=ast.parse('self.gain_margin_prefix_rejections=[]').body
    paths=[n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='path']
    if len(paths)!=1:raise ValueError('Original scenario loop changed')
    paths[0].body.extend(ast.parse('''
if self.gain_floor>0:
    denominator=float(before[0][:,0].mean())
    prefix_score=float(-delta[:,0].mean()/denominator) if np.isfinite(denominator) and denominator>0 else None
    if prefix_score is None or not np.isfinite(prefix_score) or prefix_score<self.gain_floor:
        self.gain_margin_prefix_rejections.append(dict(candidate=index,path=path,
            baseline_team_cost=denominator,score_upper_bound=prefix_score,floor=self.gain_floor))
        passed=False
        break
''').body)
    mappings=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict'
              and any(k.arg=='actual_learning_used' for k in n.keywords)]
    if len(mappings)!=1:raise ValueError('Final evidence mapping changed')
    mappings[0].keywords.extend([ast.keyword(arg='gain_margin_prefix_rejections',value=ast.parse('self.gain_margin_prefix_rejections',mode='eval').body),
        ast.keyword(arg='confidence_pruning_kind',value=ast.Constant('rejection_only_min_over_paths_upper_bound'))])
    cls=ast.ClassDef(name='PrunedGainMarginGate',bases=[ast.Name(id='ScalarGainMarginGate',ctx=ast.Load())],keywords=[],body=[fn],decorator_list=[])
    header='''"""Reject candidates as soon as a computed path upper-bounds min confidence below floor.

All admitted candidates retain complete original evidence;no extra admission,
new actor,altered nominal ranking or weakened physical certificate.
"""
import time
import numpy as np
from .gain_margin_gate import ScalarGainMarginGate,relative_gain_score
from .actuator_recovery_v34 import validate_inputs,stopping_certificate
from .ego_v13 import ego_observations
from .marl_v12 import LEVELS,action_mask
from .physical_switch_v27 import recovery_search
from .recovery_joint_gate_v35 import actuator_certificate_flags,compositions
from .temporal_portfolio_v20 import BANK_STREAM

'''
    target=ROOT/'src/coopgate/pruned_gain_margin_gate.py'
    result='# naive _select sha256='+hashlib.sha256(text.encode()).hexdigest()+'\n'+header+ast.unparse(ast.fix_missing_locations(cls))+'\n'
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(result)
    print(str(target),flush=True)


if __name__=='__main__':build()
