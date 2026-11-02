"""Static adapters of the original inferential rules and adverse-outcome audit.

No frozen old study/source is edited. Building code is not an alpha allocation.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
from coopgate import recovery_risk_v39 as stats
from coopgate import joint_risk_data_v42 as data
import run_joint_risk_v42 as runtime
import freeze_fair_comparison_v39 as fair_helpers
import run_fair_runtime_v41 as row_helpers
import audit_buffered_scope_v25 as call_helpers

ROOT=Path(__file__).resolve().parents[1]


def save(name,tree,header=''):
    path=ROOT/'tools'/name
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(header+ast.unparse(ast.fix_missing_locations(tree))+'\n')
    print(str(path),flush=True)


def build():
    common=ROOT/'tools/mappo_formal_common.py'
    with common.open('x',encoding='utf-8',newline='\n') as f:
        f.write('"""Unchanged pure helpers extracted from the original formal implementation."""\nimport hashlib\nimport json\nimport time\nimport numpy as np\nfrom coopgate.shifted_sumo_v19 import shifted_initial\n\n')
        f.write('\n\n'.join(inspect.getsource(x) for x in (fair_helpers.canonical_sha256,fair_helpers.array_sha256,
            fair_helpers.planned_initial,row_helpers.json_ready,call_helpers.TimedGate,call_helpers.check_calls)))
    tree=ast.parse(Path(stats.__file__).read_text());tree.body[0]=ast.Expr(ast.Constant('Prospective genuine-MAPPO attempt3 adapter. Original statistical function bodies unchanged;no error allocation by import.'))
    insert=0
    for i,node in enumerate(tree.body):
        if isinstance(node,ast.ImportFrom) and node.level:
            node.module='coopgate.'+node.module;node.level=0
        if isinstance(node,(ast.Import,ast.ImportFrom)):insert=i+1
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ATTEMPT' for t in node.targets):node.value=ast.Constant(3)
    tree.body[insert:insert]=ast.parse("MODELS=(4703,4704,4705)\nMEMBERS=tuple(f'{m}:{target}' for m in MODELS for target in ('mixture','strong_lag'))").body
    save('mappo_risk_attempt3.py',tree,'# original statistics sha256='+hashlib.sha256(Path(stats.__file__).read_bytes()).hexdigest()+'\n')
    tree=ast.parse(Path(data.__file__).read_text())
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module=='recovery_risk_v39':node.module='mappo_risk_attempt3';node.level=0
    class Rule(ast.NodeTransformer):
        def visit_Constant(self,node):
            return ast.copy_location(ast.Constant('parameter_scenario_confidence'),node) if node.value=='bank_performance' else node
    tree=Rule().visit(tree)
    save('mappo_joint_data.py',tree,'# original paired-data sha256='+hashlib.sha256(Path(data.__file__).read_bytes()).hexdigest()+'\n')
    functions=[]
    for function in (runtime.execute_episode,runtime.audit_record):
        fn=ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        class Wiring(ast.NodeTransformer):
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name) and t.id=='rule' for t in node.targets):
                    node.value=ast.parse('truthful_rule(method,case)',mode='eval').body;return node
                return self.generic_visit(node)
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name) and node.func.id in ('MemoizedRecoveryGate','RecoveryJointGate'):
                    name='controller_factory' if node.func.id=='MemoizedRecoveryGate' else 'reference_factory'
                    return ast.copy_location(ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=[ast.Name(id=n,ctx=ast.Load()) for n in ('method','model','case')],keywords=[]),node)
                return self.generic_visit(node)
            def visit_ExceptHandler(self,node):
                node.type=ast.Name(id='Exception',ctx=ast.Load())
                return self.generic_visit(node)
            def visit_Attribute(self,node):
                if isinstance(node.value,ast.Attribute) and isinstance(node.value.value,ast.Name) and node.value.value.id=='design' and node.value.attr=='shared':
                    return ast.copy_location(ast.Name(id=node.attr,ctx=node.ctx),node)
                return self.generic_visit(node)
        fn=Wiring().visit(fn)
        if function is runtime.execute_episode:
            fn.args.kwonlyargs.append(ast.arg(arg='controller_factory'));fn.args.kw_defaults.append(ast.Name(id='make_controller',ctx=ast.Load()))
            loop=next(n for n in ast.walk(fn) if isinstance(n,ast.While))
            step=loop.body[0]
            handler=ast.parse('''
try:pass
except Exception as exc:
    if wrapped.calls and wrapped.calls[-1]['error'] is not None:
        env.abort_reason=str(exc)
        break
    raise
''').body[0]
            handler.body=[step];loop.body[0]=handler
        if function is runtime.audit_record:
            fn.args.kwonlyargs.append(ast.arg(arg='reference_factory'));fn.args.kw_defaults.append(ast.Name(id='make_controller',ctx=ast.Load()))
            # Additional confidence arithmetic remains independently checked;
            # source controller replay establishes the denominator/model values.
            loop=[n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
                  and any(isinstance(v,ast.Name) and v.id=='entry' for v in n.target.elts)][0]
            loop.body[0:0]=ast.parse("if method in TARGET_METHODS:validate_prefix_evidence(entry)").body
        functions.append(fn)
    header='''"""Owned actual SUMO execution and independently reconstructed adverse outcomes.

Generated from the original fixed-N audit;collisions/violations are outcomes,
not reasons to drop samples. No model/plant seed reaches the online gate.
"""
import time
import numpy as np
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv,matched_plant_parameters,tape_fingerprint
from coopgate.memoized_recovery_v37 import MemoizedRecoveryGate,make_cache,CAPACITY
from coopgate.scalar_nominal_recovery import forecast_kernel
from coopgate.pruned_gain_margin_gate import PrunedGainMarginGate
from coopgate.pruned_gain_margin_audit import validate_prefix_evidence
from coopgate.episode_thinning_control import EpisodeThinnedNominal
from coopgate.merge_v4 import MergeConfig,transition_safe
from mappo_formal_common import TimedGate,check_calls,json_ready,array_sha256,canonical_sha256
from mappo_joint_data import episode_outcome

TARGET_METHODS=('4703','4704','4705')
CALIBRATION_METHODS=('baseline',)+TARGET_METHODS+tuple('nominal_'+s for s in TARGET_METHODS)
CONFIRMATION_METHODS=CALIBRATION_METHODS+tuple(kind+'_'+s for kind in ('thinned','mps','mamps') for s in TARGET_METHODS)
METHODS=CONFIRMATION_METHODS


class ScalarNominalOnlyGate(MemoizedRecoveryGate):
    def __init__(self,model,profile):super().__init__(model,profile,'nominal_performance')
    def predict(self,state,previous,mlo,mhi,command,path):
        if getattr(self,'_cache_step',None)!=self.step:
            self._plan_cache,self._cache_stats=make_cache();self._cache_step=self.step
        return forecast_kernel(state,previous,self.memory,mlo,mhi,self.step,self.tapes[path],
            self.gain,self.drag,self.retention,self.bounds,self.packed,
            np.zeros(3) if command is None else command,command is not None,
            self._plan_cache,self._cache_stats,CAPACITY)


def nominal_factory(model,profile,rule):
    if rule=='baseline':return MemoizedRecoveryGate(None,profile,'baseline')
    if rule=='nominal_performance':return ScalarNominalOnlyGate(model,profile)
    raise ValueError('Only nominal or identical baseline branches are registered')


def truthful_rule(method,case):
    if method not in METHODS:raise ValueError('Unknown fixed method')
    if method=='baseline':return 'baseline'
    if method in TARGET_METHODS:return 'parameter_scenario_confidence'
    kind,seed=method.split('_')
    if kind=='nominal':return 'nominal_performance'
    if kind=='thinned':return 'nominal_performance' if case['controller_coins'][seed] else 'baseline'
    return 'mps_shared' if kind=='mps' else 'mamps_shared'


def make_controller(method,model,case):
    profile=case['plant'];rule=truthful_rule(method,case)
    if method in TARGET_METHODS:return PrunedGainMarginGate(model,profile,.0025)
    if method.startswith('thinned_'):
        seed=method.split('_')[1]
        return EpisodeThinnedNominal(model,profile,case['controller_coins'][seed],factory=nominal_factory)
    if method.startswith('nominal_'):return ScalarNominalOnlyGate(model,profile)
    return MemoizedRecoveryGate(model if method!='baseline' else None,profile,rule)

'''
    save('mappo_formal_runtime.py',ast.Module(body=functions,type_ignores=[]),header)


if __name__=='__main__':build()
