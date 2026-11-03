"""New unallocated sample-size proposal;never edit frozen3072/1024 sources.

Same gate/actors/bound/function bodies. The blind development power rule chose
4096/1024. No alpha reservation or formal SUMO by this preparation command.
"""
import ast,json
from pathlib import Path
import mappo_history_formal_experiment as old
from build_mappo_coupled_formal_adapters import function_bodies

NAMES=['mappo_risk_attempt5','mappo_history_joint_data','mappo_history_formal_runtime',
 'mappo_history_formal_design','mappo_history_formal_experiment','mappo_history_formal_finalize',
 'run_mappo_history_formal_validation','run_mappo_history_formal']
TESTS=['test_mappo_history_formal_adapters','test_mappo_history_formal_integrity','test_mappo_history_formal_failures']
RENAME={'mappo_history':'mappo_powered_history','mappo_risk_attempt5':'mappo_powered_risk_attempt5',
 'formal_mappo_attempt5_20261103':'formal_mappo_attempt5_powered_20261103',
 'history_formal_adapter_validation_20261103':'powered_history_formal_adapter_validation_20261103',
 'mappo_formal_attempt5_20261103_protocol.md':'mappo_formal_attempt5_powered_20261103_protocol.md'}


def replace(text):
    for a,b in sorted(RENAME.items(),key=lambda x:-len(x[0])):text=text.replace(a,b)
    return text


def run():
    power=old.BASE/'history_power_design_20261103';report=old.read(power/'report.json');owner=old.read(power/'owner_native_exit.json')
    selected=report['selected_design']
    if owner['actual_native_exit_code']!=0 or owner['report_sha256']!=old.sha(power/'report.json') or not selected or (selected['calibration_initials'],selected['confirmation_initials'])!=(4096,1024):
        raise ValueError('Exact completed blind sizing rule must select4096/1024')
    if (old.FORMAL/'allocation.json').exists() or (old.FORMAL/'execution').exists():raise ValueError('Original draft was already allocated/run;do not reuse a slot')
    old.verify_plan(old.FORMAL,False);outputs={};proof=[]
    for directory,names in (('tools',NAMES),('tests',TESTS)):
        for name in names:
            source=old.ROOT/directory/(name+'.py');original=source.read_text(encoding='utf-8');text=replace(original)
            if name=='mappo_risk_attempt5':
                text=text.replace('MODELS = (4703, 4704, 4705)','CALIBRATION_N = 4096  # Fixed before allocation by blind power rule;original bounds unchanged.\nMODELS = (4703, 4704, 4705)')
            if name=='mappo_history_formal_design':
                text=text.replace('calibration=3072','calibration=4096').replace('{p:768 for p in PLANTS}','{p:1024 for p in PLANTS}').replace('878000001','880000001')
            if name in ('mappo_history_formal_experiment','mappo_history_formal_finalize','run_mappo_history_formal'):
                text=text.replace('16384','20480').replace('37888','45056').replace('independent_initials=4096','independent_initials=5120')
                text=text.replace('calibration_initials=3072','calibration_initials=4096').replace("n=3072;lookup","n=4096;lookup")
            if name=='mappo_history_formal_experiment':
                text=text.replace("source_sha256=sources,\n", "source_sha256=sources,power_design_sha256=sha(BASE/'history_power_design_20261103/report.json'),\n")
                text=text.replace("    candidate=design.validate_candidate()\n    current=", "    if plan['power_design_sha256']!=sha(BASE/'history_power_design_20261103/report.json'):raise ValueError('Fixed sizing report changed')\n    candidate=design.validate_candidate()\n    current=")
            if name=='test_mappo_history_formal_adapters':
                text=text.replace('3072','4096').replace('768','1024').replace('==4096','==5120').replace('==1024 and len(confirm)==1024','==4096 and len(confirm)==1024')
                # Explicit total identity replacement must not alter cal-size assertions.
                text=text.replace('len(cal)==5120','len(cal)==4096').replace('37888','45056').replace('16384','20480')
            ast.parse(text)
            if name in ('mappo_risk_attempt5','mappo_history_joint_data','mappo_history_formal_runtime') and function_bodies(original)!=function_bodies(text):
                raise ValueError('Statistics,paired data,controller,SUMO/physical/cost functions changed')
            target=old.ROOT/directory/(replace(name)+'.py')
            if target.exists():raise FileExistsError('Preserve every existing adapter')
            outputs[target]=text;proof.append(dict(source=source.relative_to(old.ROOT).as_posix(),source_sha256=old.sha(source),target=target.relative_to(old.ROOT).as_posix()))
    for path,text in outputs.items():
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    for row in proof:row['target_sha256']=old.sha(old.ROOT/row['target'])
    old.write(old.BASE/'powered_history_adapter_generation_20261103.json',dict(adapters=proof,
        original_statistical_and_paired_function_bodies_identical=True,online_controller_and_sumo_physics_identical=True,
        algorithm_candidate_unchanged=True,candidate_sha256=old.sha(old.BASE/'observed_history_fixed_candidate_20261103.json'),
        selected_calibration_initials=4096,selected_confirmation_initials=1024,selected_actual_runs=45056,
        power_design_sha256=old.sha(power/'report.json'),generator_sha256=old.sha(__file__),
        prior_unallocated_proposal_preserved=True,new_error_allocated=False,new_formal_sumo_runs=0))
    print(json.dumps(dict(adapters=len(proof),calibration=4096,confirmation=1024,total_runs=45056,algorithm_unchanged=True,new_error_allocated=False)),flush=True)


if __name__=='__main__':run()
