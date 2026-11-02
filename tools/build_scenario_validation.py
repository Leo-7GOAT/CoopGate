"""Materialize separate score-gate physics reviewer and actual SUMO runner."""
import inspect
from pathlib import Path
import textwrap
from coopgate.recovery_gate_audit_v35 import review_joint_trace

ROOT=Path(__file__).resolve().parents[1]
VALIDATOR='''
def validate_scenario_evidence(item,used):
    budget=item['scenario_max_failed_paths']
    if type(budget) is not int or budget not in (0,1,3):raise ValueError('Unregistered scenario budget')
    if item['scenario_score_is_probability'] or item['confidence_threshold']!=(33-budget)/33:
        raise ValueError('Score probability/threshold identity changed')
    if not used:
        if item['scenario_pass_fraction'] is not None:raise ValueError('Unexecuted score must be undefined')
        return
    delta=np.asarray(item['selected_bank_differences']);phases=np.asarray(item['selected_phase_differences'])
    if delta.shape!=(33,3,2) or phases.shape!=(33,3) or not np.isfinite(delta).all() or not np.isfinite(phases).all():
        raise ValueError('Complete finite score evidence required')
    bad=np.any(delta>1e-10,axis=(1,2))|np.any(phases>.05+1e-10,axis=1)
    if bad[0] or delta[0,:,0].mean()>=-1e-9:raise ValueError('Nominal Pareto gain changed')
    score=(33-int(bad.sum()))/33
    if bad.sum()>budget or item['scenario_pass_fraction']!=score or score<item['confidence_threshold']:
        raise ValueError('Executed below independently reconstructed score threshold')
'''


def build():
    src=textwrap.dedent(inspect.getsource(review_joint_trace)).replace('def review_joint_trace(', 'def review_scenario_trace(')
    src=src.replace('gate = RecoveryJointGate(model, profile, rule) if replay else None',
        "gate = (ScenarioConfidenceGate(model,profile,row['scenario_max_failed_paths']) if rule=='scenario_confidence' else MemoizedRecoveryGate(None,profile,'baseline')) if replay else None")
    start=src.index("        if rule in ('nominal_performance','bank_performance') and used:")
    end=src.index('        if replay:',start)
    src=src[:start]+"        if rule=='scenario_confidence':validate_scenario_evidence(item,used)\n"+src[end:]
    header='''"""Independent actuator physics and explicit score reconstruction; no rule relabeling."""
import numpy as np
from .actuator_recovery_v34 import ActuatorRecoveryBaseline,class_bounds,stopping_certificate
from .disturbance_v9 import disturbance_tape
from .matched_shift_v21 import matched_plant_parameters,tape_fingerprint
from .merge_v4 import MergeConfig,transition_safe
from .scenario_confidence_gate import ScenarioConfidenceGate
from .memoized_recovery_v37 import MemoizedRecoveryGate
RULES=('baseline','scenario_confidence')
'''
    with (ROOT/'src/coopgate/scenario_gate_audit.py').open('x',encoding='utf-8',newline='\n') as f:f.write(header+VALIDATOR+'\n'+src)
    runner=(ROOT/'tools/evaluate_mappo_recovery.py').read_text()
    runner=runner.replace('from coopgate.recovery_gate_audit_v35 import review_joint_trace, relative_outcome, cell_summary',
        'from coopgate.recovery_gate_audit_v35 import relative_outcome, cell_summary\nfrom coopgate.scenario_gate_audit import review_scenario_trace\nfrom coopgate.scenario_confidence_gate import ScenarioConfidenceGate')
    runner=runner.replace('from coopgate.recovery_joint_gate_v35 import RULES',"RULES=('baseline',)+tuple(f'confidence_{seed}_{budget}' for seed in (4703,4704,4705) for budget in (0,1,3))")
    helpers='''
def load_models():
    loaded={seed:load_deployment_actor(ROOT/f'artifacts/mappo_ctde_20261101/development/joint_{seed}/policy.pt') for seed in (4703,4704,4705)}
    return {s:v[0] for s,v in loaded.items()},{str(s):v[1] for s,v in loaded.items()}

def factory_gate(method,models,profile):
    if method=='baseline':return make_recovery_gate(None,profile,'baseline')
    _,seed,budget=method.split('_')
    return ScenarioConfidenceGate(models[int(seed)],profile,int(budget))

'''
    runner=runner.replace('def sha(path):',helpers+'def sha(path):')
    runner=runner.replace('model,identity=load_deployment_actor(checkpoint)','models,identity=load_models()')
    runner=runner.replace("checkpoint=str(Path(checkpoint).resolve())","checkpoint=None")
    runner=runner.replace("make_recovery_gate(None if rule=='baseline' else model,profile,rule)","factory_gate(rule,models,profile)")
    runner=runner.replace('plant_name=profile,rule=rule,',"plant_name=profile,rule='baseline' if rule=='baseline' else 'scenario_confidence',method_id=rule,\n                    scenario_max_failed_paths=None if rule=='baseline' else int(rule.split('_')[-1]),")
    runner=runner.replace('review_joint_trace(row,trace,model)','review_scenario_trace(row,trace,None)')
    runner=runner.replace("model,identity=load_deployment_actor(plan['checkpoint'])",'models,identity=load_models()')
    runner=runner.replace("make_recovery_gate(None if row['rule']=='baseline' else model,row['plant_name'],row['rule'])", "factory_gate(row['method_id'],models,row['plant_name'])")
    runner=runner.replace("policy_sha256=identity['policy_sha256']",'models=identity')
    runner=runner.replace("if not args.checkpoint:parser.error('--checkpoint required for collection')",'pass # Fixed three-model family; no best-seed checkpoint argument')
    runner=runner.replace('96800001','96900001')
    runner=runner.replace("default='integration'","default='smoke'")
    runner=runner.replace("ROOT/'tools/audit_buffered_scope_v25.py',ROOT/'tools/experiment_ego_v13.py'", "ROOT/'tools/audit_buffered_scope_v25.py',ROOT/'tools/experiment_ego_v13.py',ROOT/'docs/scenario_score_sumo_20261102_protocol.md'")
    with (ROOT/'tools/evaluate_scenario_confidence.py').open('x',encoding='utf-8',newline='\n') as f:f.write(runner)


if __name__=='__main__':build()
