"""Second explicit full-sample recovery, preserving every earlier byte and budget."""
import argparse
import json
from types import FunctionType
import joint_recovery_20261028 as base

ROOT=base.ROOT
OUT=ROOT/'artifacts/joint_risk_recovery_20261031'
PLAN=OUT/'recovery.json'
SCRIPT='tools/joint_recovery_20261031.py'
SOURCES=(*base.SOURCES,SCRIPT,'tests/test_joint_recovery_20261031.py',
    'operations/continue_joint_recovery_20261031.ps1',
    'operations/launch_joint_recovery_task_20261031.ps1',
    'docs/joint_recovery_20261031.md',base.PLAN.relative_to(ROOT).as_posix(),
    'artifacts/joint_risk_recovery_20261031/system_interruption.json')


def assert_stopped():
    code='[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'
    result=base.subprocess.run(['powershell.exe','-NoProfile','-Command',code],capture_output=True,check=True)
    for p in json.loads(result.stdout.decode('utf-8-sig')):
        command=(p.get('CommandLine') or '').replace('\\','/')
        if any(n in command for n in ('tools/joint_recovery_20261028.py','operations/continue_joint_recovery_20261028.ps1')):
            raise ValueError('Prior recovery is still live; never restart a live study')
    execution=base.OUT/'calibration/execution'
    if any((execution/n).exists() for n in ('exit.json','collection_complete.json','blocks_manifest.json')):
        raise ValueError('Terminal evidence exists; inspect before recovery')


def file_map():
    paths=sorted(p for p in (base.OUT/'calibration').rglob('*') if p.is_file())
    paths += [base.original.PLAN,base.original.BINDING,base.original.OUT/'parallel_replay_binding.json',base.PLAN]
    paths += list((base.OUT/'boundary_proof').glob('*.json'))
    paths += [ROOT/n for n in ('.git/mainline-joint-recovery-20261028.jsonl',
        '.git/mainline-joint-recovery-20261028.stdout.log','.git/scheduled_joint_recovery_launch_20261028.json')]
    return {p.relative_to(ROOT).as_posix():base.original.io.digest(p) for p in paths}


def routed_engine():
    namespace={**vars(base),'PRIOR':base.OUT,'OUT':OUT,'PLAN':PLAN,'SNAP':OUT/'interruption',
        'SCRIPT':SCRIPT,'SOURCES':SOURCES}
    for name,function in vars(base).items():
        if isinstance(function,FunctionType) and function.__module__==base.__name__:
            namespace[name]=base.clone(function,namespace)
    namespace.update(assert_original_stopped=assert_stopped,file_map=file_map)
    return namespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for n in ('snapshot','freeze','verify','finalize'):commands.add_parser(n)
    for n in ('watch','worker'):commands.add_parser(n).add_argument('stage',choices=base.original.design.STAGES)
    args=parser.parse_args();engine=routed_engine()
    if args.command in ('snapshot','freeze'):
        base.inputs()
        return engine[args.command]()
    engine['inputs'](full=args.command in ('verify','finalize'))
    if args.command=='verify':return
    namespace=engine['routed_functions']()
    if args.command=='finalize':
        engine['finalize'](namespace)
        base.original.io.exclusive_json(OUT/'analysis/lineage_accounting.json',dict(
            earlier_engineering_duplicate_records=base.original.io.read(base.PLAN)['original_partial_engineering_runs'],
            current_boundary_duplicate_records=base.original.io.read(PLAN)['original_partial_engineering_runs'],
            final_unique_records=16384,interrupted_in_flight_started_episode_count='unknown',
            previous_recovery_plan_sha256=base.original.io.digest(base.PLAN),statistical_attempt=2))
        return
    namespace[args.command](args.stage)


if __name__=='__main__':main()
