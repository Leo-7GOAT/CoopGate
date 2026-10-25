"""Additive container-format compatibility for three frozen episode consumers.

Only the private observations dependency yields JSON-style list keys instead of
the store's tuple keys. Payloads, ordering, seals, replay and inference stay intact.
No imported module globals or frozen source function bodies are modified.
"""
import argparse
from pathlib import Path
from types import FunctionType

import design_cspi_factorial_comparison as files

ROOT=files.ROOT
OUT=ROOT/'artifacts/execution_key_compatibility'
PLAN=OUT/'plan.json'
TARGET_FILES={'factorial':'tools/execute_candidate_factorial.py',
    'joint-risk':'tools/execute_joint_risk_v42.py',
    'cspi-export':'tools/export_cspi_factorial_comparison.py',
    'cspi-analysis':'tools/run_cspi_factorial_comparison.py'}
SOURCES=('tools/finalize_episode_keys.py','tests/test_episode_key_compatibility.py',
    'docs/episode_key_compatibility.md','src/coopgate/episode_store_v21.py',
    'tools/design_cspi_factorial_comparison.py',*TARGET_FILES.values())
PARENT_PLANS=('artifacts/v42/joint_risk/plan.json',
    'artifacts/candidate_factorial/complete_fair_cases/plan.json',
    'artifacts/literature_reproduction/cspi_factorial_prospective/plan.json')


def list_observations(*args,**kwargs):
    from coopgate.episode_store_v21 import observations
    for key,body,trace in observations(*args,**kwargs):
        yield list(key),body,trace


def compatible_function(function):
    if 'observations' not in function.__globals__:raise ValueError('A frozen observations consumer is required')
    adapted=FunctionType(function.__code__,{**function.__globals__,'observations':list_observations},
        function.__name__,function.__defaults__,function.__closure__)
    adapted.__kwdefaults__=function.__kwdefaults__
    assert adapted.__code__ is function.__code__
    return adapted


def freeze():
    if OUT.exists():raise FileExistsError('Preserve first key compatibility contract')
    forbidden=('artifacts/v42/joint_risk/calibration/execution','artifacts/v42/joint_risk/confirmation/execution',
        'artifacts/candidate_factorial/complete_fair_cases/execution',
        'artifacts/literature_reproduction/cspi_factorial_prospective/data')
    if any((ROOT/name).exists() for name in forbidden):raise ValueError('Fix and freeze format compatibility before formal collection')
    for name in (*SOURCES,*PARENT_PLANS):files.committed(ROOT/name)
    OUT.mkdir(parents=True,exist_ok=False)
    files.write(PLAN,dict(source_sha256={name:files.digest(ROOT/name) for name in SOURCES},
        parent_plan_sha256={name:files.digest(ROOT/name) for name in PARENT_PLANS},
        target_files=TARGET_FILES,only_change='Tuple-to-list case-key container at three finalization/export consumers',
        frozen_function_code_unchanged=True,module_globals_mutated=False,
        payloads_order_and_existing_seal_validation_unchanged=True,
        collection_controllers_thresholds_and_statistics_changed=False,
        frozen_before_any_affected_formal_collection=True,formal_efficacy_outcomes_read=False))
    print('compatibility_plan_sha256='+files.digest(PLAN),flush=True)


def inputs():
    files.committed(PLAN);plan=files.read(PLAN)
    assert plan['target_files']==TARGET_FILES and plan['only_change']=='Tuple-to-list case-key container at three finalization/export consumers'
    for name,expected in {**plan['source_sha256'],**plan['parent_plan_sha256']}.items():
        files.committed(ROOT/name)
        if files.digest(ROOT/name)!=expected:raise ValueError('Key compatibility contract changed: '+name)
    return plan


def receipt(path,kind):
    value=files.read(path)
    if value['compatibility_plan_sha256']!=files.digest(PLAN) or value['target']!=kind:
        raise ValueError('Required prior compatible result missing')
    for name,expected in value['output_sha256'].items():
        if files.digest(ROOT/name)!=expected:raise ValueError('Prior compatible result changed')
    return value


def run(kind,source=None):
    plan=inputs();before=files.digest(PLAN)
    if kind=='factorial':
        import execute_candidate_factorial as target
        function=target.summarize;result_root=target.OUT/'analysis'
        outputs=[result_root/name for name in ('report.json','replay.json','verification.json')]
    elif kind=='joint-risk':
        import execute_joint_risk_v42 as target
        function=target.finalize;result_root=target.OUT/'analysis'
        outputs=[result_root/name for name in ('certificate.json','replay.json','verification.json')]
    elif kind=='cspi-export':
        receipt(files.PARENT/'analysis/key_representation_compatibility.json','factorial')
        files.committed(files.PARENT/'analysis/key_representation_compatibility.json')
        import export_cspi_factorial_comparison as target
        function=target.export;result_root=files.OUT
        outputs=[files.OUT/'data/complete.json']
    elif kind=='cspi-analysis':
        if source is None:raise ValueError('Pinned original CSPI source required')
        receipt(files.OUT/'key_export_compatibility.json','cspi-export')
        files.committed(files.OUT/'key_export_compatibility.json')
        import run_cspi_factorial_comparison as target
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):target.run(source)
        assert inputs()==plan and files.digest(PLAN)==before
        result_root=files.OUT/'analysis';function=target.run
        outputs=[result_root/'complete.json']
    else:raise ValueError('Unknown frozen consumer')
    if kind!='cspi-analysis':
        original_observations=function.__globals__['observations']
        compatible_function(function)() # Original native/quiet/full-replay gates all remain active.
        assert function.__globals__['observations'] is original_observations
    assert inputs()==plan and files.digest(PLAN)==before
    result=dict(compatibility_plan_sha256=before,target=kind,target_source_sha256=files.digest(ROOT/TARGET_FILES[kind]),
        frozen_function_body_unchanged=True,private_key_container_adapter_used=kind!='cspi-analysis',
        module_globals_mutated=False,output_sha256={p.relative_to(ROOT).as_posix():files.digest(p) for p in outputs})
    destination=result_root/('key_export_compatibility.json' if kind=='cspi-export' else 'key_representation_compatibility.json')
    files.write(destination,result)
    print('compatible consumer complete: '+kind,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze',*TARGET_FILES))
    parser.add_argument('--source',type=Path);args=parser.parse_args()
    if args.command=='freeze':freeze()
    else:run(args.command,args.source)
