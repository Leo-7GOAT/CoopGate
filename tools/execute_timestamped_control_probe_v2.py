"""Additive model-manifest bootstrap fix; original 117-run bodies stay frozen."""
import argparse
from copy import deepcopy
import subprocess

import execute_timestamped_control_probe as base

ROOT,io=base.ROOT,base.io
OUT=base.OUT.parent/'timestamped_control_probe_v2'
PLAN=OUT/'protocol.json'
VARIANTS=base.VARIANTS
private=base.private
metadata=base.metadata
native_functions=base.native_functions
compare_raw=base.compare_raw
SOURCES=('tools/execute_timestamped_control_probe_v2.py','tools/verify_timestamped_control_probe_v2.py',
    'tools/watch_timestamped_control_probe_v2.py','tests/test_timestamped_control_manifest_v2.py',
    'docs/timestamped_control_bootstrap_v2.md')


def no_case_records(directory):
    # Git does not preserve an empty directory. An absent empty-prefix folder
    # and the observed empty folder have the same archived zero-record meaning.
    return not directory.exists() or (directory.is_dir() and not list(directory.iterdir()))


def frozen_models():
    io.committed(io.PLAN)
    models=io.read(io.PLAN)['models']
    assert tuple(models)==('1802','1812','1822')
    for item in models.values():assert io.digest(ROOT/item['path'])==item['sha256']
    return deepcopy(models)


def parent_inputs():
    result=base.parent_inputs()
    failure=io.read(base.OUT/'bootstrap_failure.json')
    receipt=io.read(base.OUT/'observed_execution_exit.json')
    assert failure['actual_sumo_runs']==0 and failure['actual_case_records']==0
    assert receipt['exit_code']==1 and receipt['actual_sumo_runs']==0
    assert no_case_records(base.OUT/'blocks') and not (base.OUT/'complete.json').exists()
    paths=[base.PLAN,base.OUT/'native_start.json',base.OUT/'observed_execution_exit.json',
           base.OUT/'bootstrap_failure.json',io.PLAN]
    for path in paths:io.committed(path)
    result.update({p.relative_to(ROOT).as_posix():io.digest(p) for p in paths})
    return result


def freeze():
    if OUT.exists():raise FileExistsError('Preserve the first explicit bootstrap revision')
    original=base.inputs();parents=parent_inputs();models=frozen_models()
    inventory=subprocess.check_output(['git','ls-files','--','tools/*.py','src/coopgate/*.py'],cwd=ROOT,text=True).splitlines()
    parent_sources=io.read(base.parent.OUT/'report.json')['source_sha256']
    names=sorted(set(inventory)|set(SOURCES)|set(parent_sources))
    for name in names:io.committed(ROOT/name)
    p=deepcopy(original)
    p.update(parent_sha256=parents,models=models,
        source_sha256={**p['source_sha256'],**{n:io.digest(ROOT/n) for n in names}},
        source_set_policy='Fixed tracked package/tool Python plus explicit probe and parent audit sources; later unrelated additions excluded.',
        bootstrap_revision=dict(parent_protocol_sha256=io.digest(base.PLAN),parent_actual_sumo_runs=0,
            correction='Attach the pinned formal structural model manifest to the unchanged development-plan view.',
            controller_and_auditor_bodies_unchanged=True,all_117_cases_and_order_unchanged=True,
            parent_prefix_preserved=True,no_implicit_resume=True),utc=io.utc())
    OUT.mkdir(parents=True,exist_ok=False);io.exclusive_json(PLAN,p)
    print('protocol_v2_sha256='+io.digest(PLAN),flush=True)


def inputs():
    p=private(base.inputs,PLAN=PLAN,parent_inputs=parent_inputs)()
    assert p['models']==frozen_models()
    assert p['bootstrap_revision']==dict(parent_protocol_sha256=io.digest(base.PLAN),parent_actual_sumo_runs=0,
        correction='Attach the pinned formal structural model manifest to the unchanged development-plan view.',
        controller_and_auditor_bodies_unchanged=True,all_117_cases_and_order_unchanged=True,
        parent_prefix_preserved=True,no_implicit_resume=True)
    return p


def source_case(index):
    p,group=base.source_case(index)
    assert 'models' not in p
    return {**p,'models':io.read(PLAN)['models']},group


def configured_run():
    return private(base.run,OUT=OUT,PLAN=PLAN,inputs=inputs,source_case=source_case)


def run():configured_run()()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run'))
    globals()[parser.parse_args().command]()
