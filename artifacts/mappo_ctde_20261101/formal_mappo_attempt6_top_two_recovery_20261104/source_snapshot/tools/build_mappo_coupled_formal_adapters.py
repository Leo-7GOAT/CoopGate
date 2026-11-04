"""Create exclusive attempt4 adapters from completed immutable attempt3 sources.

Statistical function bodies and paired harm/cost accounting stay identical.
Only prospective attempt/streams,candidate,paths and explicit binding change.
Old source files,studies,allocations and results are never overwritten.
"""
import ast
import json
from pathlib import Path
import subprocess
import mappo_formal_experiment as e
from plot_mappo_formal_results import complete_evidence

RENAME={
 'mappo_formal_design':'mappo_coupled_formal_design',
 'mappo_formal_runtime':'mappo_coupled_formal_runtime',
 'mappo_formal_experiment':'mappo_coupled_formal_experiment',
 'mappo_formal_finalize':'mappo_coupled_formal_finalize',
 'mappo_joint_data':'mappo_coupled_joint_data','mappo_risk_attempt3':'mappo_risk_attempt4',
 'build_mappo_formal_adapters':'build_mappo_coupled_formal_adapters',
 'run_mappo_formal_validation':'run_mappo_coupled_formal_validation',
 'run_mappo_formal':'run_mappo_coupled_formal',
 'test_mappo_formal_integrity':'test_mappo_coupled_formal_integrity',
 'test_mappo_formal_adapters':'test_mappo_coupled_formal_adapters',
 'test_mappo_formal_failures':'test_mappo_coupled_formal_failures',
 'gain_margin_fixed_candidate_20261102.json':'compact_coupled_fixed_candidate_20261102.json',
 'formal_mappo_attempt3_20261102':'formal_mappo_attempt4_20261102',
 'formal_adapter_validation_20261102':'coupled_formal_adapter_validation_20261102',
 'mappo_formal_attempt3_20261102_protocol.md':'mappo_formal_attempt4_20261102_protocol.md',
}
NAMES=['mappo_risk_attempt3','mappo_joint_data','mappo_formal_runtime','mappo_formal_design',
 'mappo_formal_experiment','mappo_formal_finalize','run_mappo_formal_validation','run_mappo_formal']
TESTS=['test_mappo_formal_adapters','test_mappo_formal_integrity','test_mappo_formal_failures']

REGISTER='''def register():
    plan=experiment.verify_plan(OUT,False)
    if plan['validation_only'] or plan['main_sumo_runs']!=16384 or plan['total_sumo_runs']!=37888:
        raise ValueError('Unchanged full-size study required')
    experiment.design.validate_candidate()
    proof_path=BASE/'mappo_attempt4_predecessor_audit_20261102/verification.json'
    proof=experiment.read(proof_path)
    owner_path=proof_path.parent/'owner_native_exit.json';owner=experiment.read(owner_path)
    if (not proof['complete'] or proof['attempt4_or_later_entries'] or proof['parse_errors']
        or proof['removed_prior_allocation_candidates'] or proof['new_error_allocated']
        or owner['actual_native_exit_code']!=0 or owner['verification_sha256']!=experiment.sha(proof_path)):
        raise ValueError('Complete actual-terminal unused-attempt4 audit required')
    old=BASE/'formal_mappo_attempt3_20261102'
    if (proof['attempt3_allocation_sha256']!=experiment.sha(old/'allocation.json')
        or proof['attempt3_owner_sha256']!=experiment.sha(old/'orchestrator_native_receipt.json')):
        raise ValueError('Completed predecessor changed')
    validation=experiment.VALIDATION;v=experiment.read(validation/'verification.json')
    native=experiment.read(validation/'orchestrator_native_receipt.json')
    if (v['actual_sumo_executions']!=64 or v['full_independent_reviews']!=64
        or native['actual_native_exit_code']!=0
        or native['verification_sha256']!=experiment.sha(validation/'verification.json')
        or any(event['actual_native_exit_code']!=0 for event in v['stages'])
        or v['source_verification_sha256']!=experiment.sha(validation/'validation/shard0/verification.json')
        or experiment.verify_plan(validation,True)['source_sha256']!=plan['source_sha256']):
        raise ValueError('Exact current-source actual-runtime validation required')
    allocation=dict(attempt=4,allocated_error=.05/20,total_attempt_error=.05,predecessor_verified=True,
        predecessor_audit_sha256=experiment.sha(proof_path),predecessor_native_sha256=experiment.sha(owner_path),
        old_withdrawn_alpha_retained=.025,old_completed_attempt2_alpha_retained=.05/6,
        old_completed_attempt3_alpha_retained=.05/12,old_error_reassigned=False,
        old_certificate_transferred=False,historical_cross_family_fwer_claim=False,
        fixed_models=[4703,4704,4705],common_floor=.0025,
        plan_sha256=experiment.sha(OUT/'plan.json'),validation_sha256=experiment.sha(validation/'verification.json'),
        reserved_utc=datetime.now(timezone.utc).isoformat())
    experiment.write(OUT/'allocation.json',allocation)
    experiment.write(OUT/'binding.json',dict(plan_sha256=experiment.sha(OUT/'plan.json'),
        allocation_sha256=experiment.sha(OUT/'allocation.json'),predecessor_audit_sha256=experiment.sha(proof_path),
        validation_sha256=experiment.sha(validation/'verification.json'),source_commit=plan['source_commit'],
        formal_runs_at_binding=0,controller_or_threshold_selection=False))
    print(json.dumps(dict(attempt_reserved=4,alpha=.05/20,formal_runs=0)),flush=True)
'''


def substitute(text):
    for old,new in sorted(RENAME.items(),key=lambda x:-len(x[0])):text=text.replace(old,new)
    return text


def replace_function(text,name,replacement):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement+'\n'+''.join(lines[node.end_lineno:])


def function_bodies(text):
    return {n.name:ast.dump(ast.Module(body=n.body,type_ignores=[])) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}


def run():
    complete_evidence();e.bound(e.FORMAL,False)
    development=e.BASE/'development/compact_coupled_validation_20261102'
    report=e.read(development/'report.json');owner=e.read(development/'owner_native_exit.json')
    if report['actual_sumo_executions']!=665 or owner['actual_native_exit_code']!=0 or owner['report_sha256']!=e.sha(development/'report.json'):
        raise ValueError('Complete unchanged coupled development required')
    outputs={};proof=[]
    for directory,names in (('tools',NAMES),('tests',TESTS)):
        for name in names:
            original=(e.ROOT/directory/(name+'.py')).read_text(encoding='utf-8');text=substitute(original)
            if name=='mappo_risk_attempt3':
                text=text.replace('ATTEMPT = 3','ATTEMPT = 4').replace('attempt3 adapter','attempt4 adapter')
            if name=='mappo_formal_runtime':
                text=text.replace('from coopgate.pruned_gain_margin_gate import PrunedGainMarginGate',
                    'from pathlib import Path\nfrom coopgate.compact_coupled_gate import CompactCoupledGate\nCOUPLED_BANK=Path(__file__).resolve().parents[1]/"artifacts/mappo_ctde_20261101/development/compact_coupled_bank_20261102.json"')
                text=text.replace('return PrunedGainMarginGate(model,profile,.0025)','return CompactCoupledGate(model,profile,COUPLED_BANK)')
            if name=='mappo_formal_design':
                text=text.replace('852000001','862000001').replace('853000001','863000001').replace('854000001','866000001')
                text=text.replace('520101','540101').replace('520201','540201').replace('520301','540301').replace('mappo3:','mappo4:')
                text=text.replace('    return candidate',
                    '    bank=BASE/"development/compact_coupled_bank_20261102.json"\n'
                    '    if candidate["controller"]!="CompactCoupledGate" or hashlib.sha256(bank.read_bytes()).hexdigest()!=candidate["coupled_bank_sha256"]:raise ValueError("Coupled bank changed")\n'
                    '    return candidate')
            if name=='mappo_formal_experiment':
                text=text.replace("allocation['attempt']!=3 or allocation['allocated_error']!=.05/12",
                                  "allocation['attempt']!=4 or allocation['allocated_error']!=.05/20")
                text=text.replace("    return sources\n","    sources += [ROOT/'tools/audit_mappo_attempt4_predecessor.py']\n    return sources\n")
            if name=='run_mappo_formal':
                text=replace_function(text,'register',REGISTER)
                text=text.replace('Register explicit unspent attempt3','Register explicit unspent attempt4')
            if name=='test_mappo_formal_adapters':
                text=text.replace('stats.ATTEMPT==3','stats.ATTEMPT==4').replace('.05/12','.05/20').replace('attempt3_inference','attempt4_inference')
            ast.parse(text)
            if name in ('mappo_risk_attempt3','mappo_joint_data'):
                if function_bodies(original)!=function_bodies(text):raise ValueError('Statistical or paired-data function bodies changed')
            if name=='mappo_formal_runtime':
                before=function_bodies(original);after=function_bodies(text)
                if {k:v for k,v in before.items() if k!='make_controller'}!={k:v for k,v in after.items() if k!='make_controller'}:
                    raise ValueError('SUMO/cost/physical replay accounting changed')
            target=e.ROOT/directory/(RENAME[name]+'.py')
            if target.exists():raise FileExistsError('Never overwrite generated adapter '+str(target))
            outputs[target]=text;proof.append(dict(source=str(Path(directory)/(name+'.py')),source_sha256=e.sha(e.ROOT/directory/(name+'.py')),
                target=str(target.relative_to(e.ROOT))))
    candidatepath=e.BASE/'compact_coupled_fixed_candidate_20261102.json'
    if candidatepath.exists():raise FileExistsError('Retain existing candidate')
    for path,text in outputs.items():
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    candidate=e.read(e.BASE/'gain_margin_fixed_candidate_20261102.json')
    candidate.update(controller='CompactCoupledGate',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=e.ROOT,text=True).strip(),
        coupled_bank_sha256=e.sha(e.BASE/'development/compact_coupled_bank_20261102.json'),
        completed_development_sha256=e.sha(development/'report.json'),
        development_data_includes_all_attempt3_outcomes=True,
        formal_attempt4_error_allocated=False,formal_attempt4_samples=0,
        unresolved=['Fresh independent full-family formal risk/adoption/gain and matching validation',
            'Two known adverse initials retained,not fitted away',
            'Callbacks still exceed500ms;no hard-real-time claim',
            'Raw MAPPO/consistent centralized-critic superiority remains unproved'])
    for name in ('src/coopgate/compact_coupled_gate.py','src/coopgate/coupled_corner_witness.py'):
        candidate['controller_source_sha256'][name]=e.sha(e.ROOT/name)
    e.write(candidatepath,candidate)
    for item in proof:item['target_sha256']=e.sha(e.ROOT/item['target'])
    e.write(e.BASE/'coupled_formal_adapter_generation_20261102.json',dict(adapters=proof,
        original_statistics_and_paired_data_function_bodies_identical=True,
        original_sumo_cost_physics_replay_unchanged=True,alpha_allocated=False,new_sumo_runs=0,
        generator_sha256=e.sha(__file__)))
    print(json.dumps(dict(adapters=len(outputs),statistics_body_identity=True,candidate=str(candidatepath),alpha_allocated=False)))


if __name__=='__main__':run()
