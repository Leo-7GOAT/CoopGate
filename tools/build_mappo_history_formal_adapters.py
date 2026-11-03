"""Prepare separate prospective history-gate adapters;allocate no error.

Preserve every statistical and paired-outcome function body. Old source and
all failed experiments remain immutable. Preparation alone is not registration.
"""
import ast,json,subprocess
from pathlib import Path
import mappo_coupled_formal_experiment as old
from mappo_completed_attempt4 import complete_evidence
from build_mappo_coupled_formal_adapters import function_bodies

BASE=old.BASE;ROOT=old.ROOT
NAMES=['mappo_risk_attempt4','mappo_coupled_joint_data','mappo_coupled_formal_runtime',
 'mappo_coupled_formal_design','mappo_coupled_formal_experiment','mappo_coupled_formal_finalize',
 'run_mappo_coupled_formal_validation','run_mappo_coupled_formal']
TESTS=['test_mappo_coupled_formal_adapters','test_mappo_coupled_formal_integrity','test_mappo_coupled_formal_failures']
RENAME={
 'mappo_coupled':'mappo_history','mappo_risk_attempt4':'mappo_risk_attempt5',
 'formal_mappo_attempt4_20261102':'formal_mappo_attempt5_20261103',
 'coupled_formal_adapter_validation_20261102':'history_formal_adapter_validation_20261103',
 'compact_coupled_fixed_candidate_20261102.json':'observed_history_fixed_candidate_20261103.json',
 'mappo_formal_attempt4_20261102_protocol.md':'mappo_formal_attempt5_20261103_protocol.md',
 'audit_mappo_attempt4_predecessor.py':'audit_mappo_attempt5_predecessor.py',
 'mappo_attempt4_predecessor_audit_20261102':'mappo_attempt5_predecessor_audit_20261103',
}


def substitute(text):
    for a,b in sorted(RENAME.items(),key=lambda pair:-len(pair[0])):text=text.replace(a,b)
    return text


def run():
    complete_evidence();development=BASE/'development/observed_history_validation_20261103'
    report=old.read(development/'report.json');owner=old.read(development/'owner_native_exit.json')
    if owner['actual_native_exit_code']!=0 or owner['report_sha256']!=old.sha(development/'report.json') or report['actual_sumo_executions']!=644:
        raise ValueError('Complete fixed whole development evidence required')
    outputs={};proof=[]
    for directory,names in (('tools',NAMES),('tests',TESTS)):
        for name in names:
            source=ROOT/directory/(name+'.py');original=source.read_text(encoding='utf-8');text=substitute(original)
            if name=='mappo_risk_attempt4':text=text.replace('ATTEMPT = 4','ATTEMPT = 5').replace('attempt4 adapter','attempt5 adapter')
            if name=='mappo_coupled_formal_runtime':
                text=text.replace('from coopgate.compact_coupled_gate import CompactCoupledGate','from coopgate.observed_history_gate import ObservedHistoryGate')
                text=text.replace('return CompactCoupledGate(model,profile,COUPLED_BANK)','return ObservedHistoryGate(model,profile,COUPLED_BANK)')
            if name=='mappo_coupled_formal_design':
                text=text.replace('862000001','875000001').replace('863000001','876000001').replace('866000001','878000001')
                text=text.replace('540101','550101').replace('540201','550201').replace('540301','550301').replace('mappo4:','mappo5:')
                text=text.replace('candidate["controller"]!="CompactCoupledGate"','candidate["controller"]!="ObservedHistoryGate"')
            if name=='mappo_coupled_formal_experiment':
                text=text.replace("allocation['attempt']!=4 or allocation['allocated_error']!=.05/20","allocation['attempt']!=5 or allocation['allocated_error']!=.05/30")
                text=text.replace("def build(validation=False):","def build(validation=False,source_commit='HEAD'):")
                text=text.replace("['git','show','HEAD:'+name]","['git','show',source_commit+':'+name]")
                text=text.replace("['git','rev-parse','HEAD'],cwd=ROOT","['git','rev-parse',source_commit+'^{commit}'],cwd=ROOT")
                text=text.replace("p.add_argument('--validation',action='store_true');a=p.parse_args()","p.add_argument('--validation',action='store_true');p.add_argument('--source-commit',default='HEAD');a=p.parse_args()")
                text=text.replace("if a.action=='build':build(a.validation)","if a.action=='build':build(a.validation,a.source_commit)")
            if name=='run_mappo_coupled_formal':
                text=text.replace('attempt4','attempt5').replace('attempt3_allocation_sha256','attempt4_allocation_sha256').replace('attempt3_owner_sha256','attempt4_owner_sha256')
                # The predecessor must stay the completed attempt4 directory.
                text=text.replace("old=BASE/'formal_mappo_attempt3_20261102'","old=BASE/'formal_mappo_attempt4_20261102'")
                text=text.replace('attempt=4,allocated_error=.05/20','attempt=5,allocated_error=.05/30').replace('attempt_reserved=4,alpha=.05/20','attempt_reserved=5,alpha=.05/30')
                text=text.replace('old_completed_attempt3_alpha_retained=.05/12,old_error_reassigned=False,','old_completed_attempt3_alpha_retained=.05/12,old_completed_attempt4_alpha_retained=.05/20,old_error_reassigned=False,')
            if name=='test_mappo_coupled_formal_adapters':
                text=text.replace('stats.ATTEMPT==4','stats.ATTEMPT==5').replace('.05/20','.05/30').replace('attempt4_inference','attempt5_inference')
            ast.parse(text)
            if name in ('mappo_risk_attempt4','mappo_coupled_joint_data') and function_bodies(original)!=function_bodies(text):
                raise ValueError('Statistical/paired-data functions changed')
            if name=='mappo_coupled_formal_runtime':
                a=function_bodies(original);b=function_bodies(text)
                if {k:v for k,v in a.items() if k!='make_controller'}!={k:v for k,v in b.items() if k!='make_controller'}:
                    raise ValueError('SUMO/physical/failure/cost accounting changed')
            target=ROOT/directory/(substitute(name)+'.py')
            if target.exists():raise FileExistsError('Preserve existing prospective adapter')
            outputs[target]=text;proof.append(dict(source=source.relative_to(ROOT).as_posix(),source_sha256=old.sha(source),target=target.relative_to(ROOT).as_posix()))
    target=BASE/'observed_history_fixed_candidate_20261103.json'
    if target.exists():raise FileExistsError('Preserve candidate')
    candidate=old.design.validate_candidate().copy()
    candidate.update(controller='ObservedHistoryGate',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        completed_development_sha256=old.sha(development/'report.json'),development_data_includes_all_attempt4_outcomes=True,
        development_known_harmful_model_initials_before=18,development_known_harmful_model_initials_after=4,
        strict_crossing_phase_tolerance_s=0.,future_fitted_bias_is_hypothesis=True,
        formal_attempt5_error_allocated=False,formal_attempt5_samples=0,
        unresolved=['Fresh independent full-family risk/adoption/gain validation','Four known harmful model-initials retained',
          'Observed-history fit is not future disturbance coverage','Callbacks exceed500ms;no hard-real-time claim',
          'Generic raw MAPPO and consistent central-critic superiority remain unproved'])
    for n in ('src/coopgate/observed_history_gate.py','src/coopgate/observed_lag_bias_fit.py'):
        candidate['controller_source_sha256'][n]=old.sha(ROOT/n)
    for path,text in outputs.items():
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    old.write(target,candidate)
    for row in proof:row['target_sha256']=old.sha(ROOT/row['target'])
    old.write(BASE/'history_formal_adapter_generation_20261103.json',dict(adapters=proof,
        statistics_and_paired_data_functions_identical=True,original_sumo_physics_cost_failure_replay_unchanged=True,
        generator_sha256=old.sha(__file__),alpha_allocated=False,new_sumo_runs=0))
    print(json.dumps(dict(adapters=len(proof),candidate=target.name,new_error_allocated=False)),flush=True)


if __name__=='__main__':run()
