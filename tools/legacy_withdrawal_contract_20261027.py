"""Authorized prerequisite withdrawal; preserve original joint scientific contract."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/legacy_withdrawal_20261027'
CAL = ROOT/'artifacts/recovery_20261027/calibration_recovery'
PROPOSAL = ROOT/'artifacts/proposals/legacy_attempt_withdrawal_20261027/proposal.json'
JOINT = ROOT/'artifacts/v42/joint_risk/plan.json'
PLAN = OUT/'amendment.json'
INVENTORY = OUT/'retained_calibration/inventory.json'
ROUTE = 'tools/execute_joint_risk_after_withdrawal_20261027.py'
SOURCES = ('tools/legacy_withdrawal_contract_20261027.py', ROUTE,
    'tests/test_legacy_withdrawal_20261027.py',
    'operations/withdraw_legacy_processes_20261027.ps1',
    'operations/continue_after_withdrawal_20261027.ps1',
    'operations/continue_mainline_20261027_v2.ps1',
    'docs/legacy_withdrawal_20261027.md')
OPERATIONAL = ('.git/mainline-20261027-after-ood-v2.jsonl',
    '.git/mainline-20261027-calibration_after_ood.stdout.log',
    '.git/mainline-20261027-calibration_after_ood.stderr.log',
    '.git/durable_mainline_continuation_20261027_v2.json')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def exclusive(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=True,indent=2);stream.write('\n')


def no_legacy_processes():
    code='[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',code],
        capture_output=True,text=True,encoding='utf-8',check=True)
    processes=json.loads(result.stdout)
    markers=('tools/recover_prospective_20261027.py',
             'operations/continue_mainline_20261027_v2.ps1')
    live=[p['ProcessId'] for p in processes if any(
        tag in (p.get('CommandLine') or '').replace('\\','/') for tag in markers)]
    if live:raise ValueError('Withdrawn execution is still live: '+str(live))


def original_files():
    paths=sorted(p for p in CAL.rglob('*') if p.is_file())
    paths += [ROOT/name for name in OPERATIONAL]
    return {p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths}


def validate_decision(proposal, joint, authorization, stopped):
    if (not authorization['human_approval_received'] or authorization['objective_changed']
            or authorization['partial_effects_read'] or authorization['old_alpha_reassigned']
            or authorization['old_alpha'] != .025 or authorization['new_alpha'] != .05/6
            or authorization['new_sumo_runs'] != 16384):
        raise ValueError('Approval, separate error allocation or unchanged scope missing')
    if (not stopped['all_original_tree_processes_absent'] or stopped['legacy_completed']
            or stopped['legacy_certificate_obtained'] or stopped['data_deleted']
            or stopped['unknown_historical_exits_rewritten']
            or not stopped['automatic_legacy_continuation_disabled']):
        raise ValueError('Withdrawal is not an honestly terminated incomplete attempt')
    if (canonical(joint['stages']) != proposal['joint_study']['case_schedule_sha256']
            or canonical(joint['models']) != proposal['joint_study']['models_sha256']
            or canonical(joint['contract']) != proposal['joint_study']['statistical_contract_sha256']):
        raise ValueError('Original schedule, model family or statistical contract changed')
    c=joint['contract']
    if (c['alpha'] != .05/6 or c['expected_sumo_runs'] != 16384
            or c['models'] != [1802,1812,1822] or not c['previous_attempt_terminal_required']):
        raise ValueError('Keep original contract; amend only its execution prerequisite explicitly')


def decision_inputs():
    proposal,joint,authorization,stopped=map(read,[PROPOSAL,JOINT,OUT/'authorization.json',OUT/'process_stop.json'])
    validate_decision(proposal,joint,authorization,stopped)
    legacy=read(CAL/'frozen/plan.json')
    if (authorization['proposal_sha256']!=digest(PROPOSAL)
            or stopped['authorization_sha256']!=digest(OUT/'authorization.json')
            or digest(JOINT)!=proposal['joint_study']['original_joint_plan_sha256']
            or digest(CAL/'frozen/plan.json')!=proposal['old_attempt']['legacy_plan_sha256']
            or digest(CAL/'recovery/plan.json')!=proposal['old_attempt']['legacy_recovery_sha256']
            or legacy['allocated_error']!=.025):
        raise ValueError('Approval chain or either original attempt identity changed')
    return proposal,joint,authorization,stopped


def archive():
    no_legacy_processes()
    proposal,joint,authorization,stopped=decision_inputs()
    target=INVENTORY.parent
    target.mkdir(exist_ok=False)
    files=original_files()
    parts=[];batch=[];size=0
    for name,item in files.items():
        if batch and size+item['bytes'] > 60*1024*1024:
            parts.append(batch);batch=[];size=0
        batch.append(name);size+=item['bytes']
    if batch:parts.append(batch)
    archives=[]
    for index,names in enumerate(parts):
        path=target/f'part-{index:03d}.zip'
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in names:z.write(ROOT/name,name)
        with zipfile.ZipFile(path) as z:
            if z.namelist()!=names or any(hashlib.sha256(z.read(n)).hexdigest()!=files[n]['sha256'] for n in names):
                raise ValueError('Archive does not preserve every original byte')
        archives.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,
                             sha256=digest(path),files=names))
        print(json.dumps(dict(archive_part=index+1,total_parts=len(parts))),flush=True)
    if original_files()!=files:raise ValueError('Original bytes changed during snapshot')
    progress=[read(p) for p in sorted((CAL/'calibration/execution').glob('collect_*.progress.json'))]
    exclusive(INVENTORY,dict(utc=datetime.now(timezone.utc).isoformat(),files=files,archives=archives,
        source=CAL.relative_to(ROOT).as_posix(),logical_bytes=sum(v['bytes'] for v in files.values()),
        completed_initials=sum(v['completed_initials'] for v in progress),required_initials=4096,
        every_archived_entry_sha256_verified=True,source_bytes_unchanged_after_archive=True,
        partial_efficacy_outcomes_read=False,legacy_completion_claimed=False,data_deleted=False))


def verify_inventory(*, full=False):
    inventory=read(INVENTORY)
    if (not inventory['every_archived_entry_sha256_verified']
            or not inventory['source_bytes_unchanged_after_archive']
            or inventory['partial_efficacy_outcomes_read'] or inventory['legacy_completion_claimed']
            or inventory['data_deleted']):raise ValueError('Full data preservation proof missing')
    for item in inventory['archives']:
        path=ROOT/item['path']
        if not path.resolve().is_relative_to(OUT.resolve()) or digest(path)!=item['sha256']:
            raise ValueError('Retained archive changed')
    if full and original_files()!=inventory['files']:
        raise ValueError('Withdrawn original data bytes changed')
    return inventory


def receipt_paths():
    return [PROPOSAL,PROPOSAL.with_name('proposal.md'),OUT/'authorization.json',
        OUT/'process_stop.json',OUT/'processes_before.json',INVENTORY,
        ROOT/'artifacts/recovery_20261027/ood_finalization_receipt.json']


def freeze():
    import execute_joint_risk_v42 as original
    if PLAN.exists() or original.BINDING.exists() or any((original.OUT/s/'execution').exists() for s in original.design.STAGES):
        raise FileExistsError('Amend before new binding and any affected simulation')
    no_legacy_processes()
    decision_inputs()
    inventory=verify_inventory(full=True)
    original.plan_inputs()
    ood=read(ROOT/'artifacts/recovery_20261027/ood_finalization_receipt.json')
    if (ood['actual_sumo_runs']!=22464 or ood['independent_initials']!=192
            or not ood['all_original_episodes_independently_replayed']
            or not ood['native_summary_exit']['ended'] or ood['native_summary_exit']['exit_code']!=0
            or ood['owned_summary_exit']['detail']['exit_code']!=0
            or ood['report_sha256']!=digest(ROOT/'artifacts/recovery_20261026/ood_recovery/analysis/report.json')):
        raise ValueError('Full unchanged OOD replay must already have actual zero exits')
    for path in receipt_paths()+[ROOT/n for n in SOURCES]:original.io.committed(path)
    exclusive(PLAN,dict(schema='coopgate.authorized-prerequisite-withdrawal.20261027',
        utc=datetime.now(timezone.utc).isoformat(),joint_plan_sha256=digest(JOINT),
        explicit_override={'original_field':'contract.previous_attempt_terminal_required',
            'original_value':True,'replacement':'authorized resource withdrawal without completion or certificate'},
        only_execution_prerequisite_changed=True,old_alpha=.025,new_alpha=.05/6,
        old_alpha_reassigned=False,combined_cross_family_fwer_claim=False,
        retained_initials=inventory['completed_initials'],legacy_status='withdrawn incomplete; no certificate',
        required_new_runs=16384,required_new_initials=4096,
        inputs={p.relative_to(ROOT).as_posix():digest(p) for p in receipt_paths()},
        source_sha256={n:digest(ROOT/n) for n in SOURCES},
        statistical_contract_sha256=canonical(read(JOINT)['contract']),
        joint_effects_read=False,original_plan_edited=False))


def inputs(*, full=False):
    import execute_joint_risk_v42 as original
    original.io.committed(PLAN)
    plan=read(PLAN)
    if (plan['joint_plan_sha256']!=digest(JOINT) or not plan['only_execution_prerequisite_changed']
            or plan['old_alpha']!=.025 or plan['new_alpha']!=.05/6 or plan['old_alpha_reassigned']
            or plan['required_new_runs']!=16384 or plan['required_new_initials']!=4096
            or plan['combined_cross_family_fwer_claim'] or plan['joint_effects_read']
            or plan['original_plan_edited']):raise ValueError('Amendment scope changed')
    for name,expected in {**plan['inputs'],**plan['source_sha256']}.items():
        original.io.committed(ROOT/name)
        if digest(ROOT/name)!=expected:raise ValueError('Amendment input changed: '+name)
    decision_inputs()
    if full:
        no_legacy_processes();verify_inventory(full=True)
    return plan


def prerequisite():
    plan=inputs(full=True)
    return {**plan['inputs'],PLAN.relative_to(ROOT).as_posix():digest(PLAN)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('archive','freeze','verify'))
    args=parser.parse_args()
    if args.command=='verify':inputs(full=True)
    else:globals()[args.command]()
