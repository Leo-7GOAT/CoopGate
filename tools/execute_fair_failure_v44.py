"""Additive, pre-outcome failure-accounting revision of the unstarted fair study.

Reuse the frozen collector, supervisor and full replay in private function-global
maps. No source or imported module is patched, and no historical output is used
as a new trial. Freeze and bind this execution before any formal collection.
"""
import argparse
from copy import deepcopy
import json
from types import FunctionType

from coopgate.fair_failure_analysis_v44 import analysis_contract, summarize_matched_records
import execute_fair_comparison_v41 as original
import execute_joint_risk_v42 as isolation

io, design, ROOT = original.io, original.design, original.ROOT
OUT = ROOT/'artifacts/v44/fair_failure_preserving_execution'
PLAN, BINDING = OUT/'plan.json', OUT/'binding.json'
SCRIPT = 'tools/execute_fair_failure_v44.py'
TEST = 'tests/test_execute_fair_failure_v44.py'
REVISION_FILES = (SCRIPT, TEST, 'src/coopgate/fair_failure_analysis_v44.py',
    'tests/test_fair_failure_analysis_v44.py', 'docs/v44_fair_failure_accounting.md')


def require_original_unstarted():
    if any((original.OUT/name).exists() for name in ('execution', 'blocks', 'analysis')):
        raise ValueError('Original formal study has execution evidence; do not revise after outcomes')
    for item in io.native_snapshot():
        command = (item.get('CommandLine') or '').replace('\\', '/')
        if original.SCRIPT in command and any(' '+mode in command for mode in ('watch','worker','summarize')):
            raise ValueError('Original formal study is running')


def revision():
    return dict(contract=analysis_contract(),
        original_plan_sha256=io.digest(original.PLAN),
        original_binding_sha256=io.digest(original.BINDING),
        source_sha256={name: io.digest(ROOT/name) for name in REVISION_FILES},
        original_formal_collection_started=False, supplementary_sample_count=0)


def freeze():
    if OUT.exists():
        raise FileExistsError('Preserve the first revised fair plan')
    require_original_unstarted()
    previous, _ = original.bound_inputs()
    for name in REVISION_FILES:
        io.committed(ROOT/name)
    plan = deepcopy(previous)
    plan['analysis_revision'] = revision()
    OUT.mkdir(parents=True, exist_ok=False)
    io.exclusive_json(PLAN, plan)
    print(json.dumps(dict(plan_sha256=io.digest(PLAN), actual_sumo_runs_started=0)), flush=True)


def plan_inputs():
    io.committed(PLAN)
    require_original_unstarted()
    previous, _ = original.bound_inputs()
    plan = io.read(PLAN)
    inherited = {k:v for k,v in plan.items() if k != 'analysis_revision'}
    if inherited != previous or plan.get('analysis_revision') != revision():
        raise ValueError('Changed analysis revision or inherited controller/schedule')
    return plan


def source_paths():
    return sorted(set(original.source_paths()+[ROOT/name for name in REVISION_FILES]
        +list((ROOT/'src/coopgate').glob('*.py'))+list((ROOT/'tools').glob('*.py'))))


def bind():
    if BINDING.exists():
        raise FileExistsError('Preserve the first revised binding')
    plan = plan_inputs()
    paths = source_paths()
    for path in paths:
        io.committed(path)
    binding = dict(plan_sha256=io.digest(PLAN), source_sha256={
        p.relative_to(ROOT).as_posix(): io.digest(p) for p in paths},
        runtime=io.runtime_identity(), expected_runs=5120, expected_blocks=160,
        evidence_contract=plan['evidence_execution'], analysis_contract=analysis_contract(),
        source_set_policy='All package/tool Python at bind plus original and revised tests/docs',
        failure_policy='Retain all records; reject ranking with method or paired baseline failure',
        original_execution_superseded=True, actual_sumo_runs_started=0,
        resumable=False, utc=io.utc())
    io.exclusive_json(BINDING, binding)
    print(json.dumps(dict(binding_sha256=io.digest(BINDING), launched=False)), flush=True)


def bound_inputs():
    plan = plan_inputs()
    io.committed(BINDING)
    binding = io.read(BINDING)
    if (binding['plan_sha256'] != io.digest(PLAN) or binding['runtime'] != io.runtime_identity()
            or binding['analysis_contract'] != analysis_contract()
            or binding['expected_runs'] != 5120 or binding['expected_blocks'] != 160
            or binding['evidence_contract'] != plan['evidence_execution']
            or not set(REVISION_FILES) <= set(binding['source_sha256'])):
        raise ValueError('Changed revised fair execution binding')
    original.verify_sources(binding)
    return plan, binding


def final_analysis(episodes, spec):
    # Repeat source/model, native-exit, receipt and block checks after the long
    # replay, before the inherited writer creates any analysis output.
    bound_inputs()
    execution = OUT/'execution'
    receipt = io.read(execution/'exit.json')
    done = io.read(execution/'collection_complete.json')
    manifest = io.read(execution/'blocks_manifest.json')
    original.validate_terminal(receipt, done, manifest, io.digest(PLAN), io.digest(BINDING))
    for name, expected in receipt['hashes'].items():
        if io.digest(execution/name) != expected:
            raise ValueError('Terminal evidence changed during replay')
    io.check_acknowledgment(execution, 'worker', receipt['launcher_pid'], receipt['native_pid'])
    for item in io.native_snapshot():
        if item['ProcessId'] == receipt['native_pid'] and item['CreationDate'] == receipt['native_creation_date']:
            raise ValueError('Original worker remains live')
    for entry in manifest['entries']:
        if original.block_hashes(OUT/'blocks'/f"{entry['block_index']:05d}", entry['block_index']) != entry:
            raise ValueError('Block changed during replay')
    return summarize_matched_records(episodes, spec)


def execution_function(name, finalizer=None):
    """Bind only paths, validation and final analysis; preserve worker bytecode."""
    namespace = dict(vars(original))
    namespace.update(OUT=OUT, PLAN=PLAN, BINDING=BINDING, SCRIPT=SCRIPT, TEST=TEST,
        bound_inputs=bound_inputs, summarize_matched_records=finalizer or final_analysis)
    function = getattr(original, name)
    return FunctionType(function.__code__, namespace, function.__name__,
                        function.__defaults__, function.__closure__)


def watch():
    isolation.require_quiet(io.native_snapshot())
    return execution_function('watch')()


def worker():
    isolation.require_quiet(io.native_snapshot())
    return execution_function('worker')()


def summarize():
    isolation.require_quiet(io.native_snapshot())
    # Preserve entry identities across the long replay; a rewritten exit and
    # freshly self-consistent manifest must not replace the data just reviewed.
    files = ('exit.json', 'spawn.json', 'launch.json', 'worker.native.json',
             'worker.acknowledged.json', 'collection_complete.json', 'blocks_manifest.json')
    before = {name: io.digest(OUT/'execution'/name) for name in files}
    def finalize(episodes, spec):
        if any(io.digest(OUT/'execution'/name) != value for name, value in before.items()):
            raise ValueError('Terminal identities changed during full replay')
        return final_analysis(episodes, spec)
    return execution_function('summarize', finalizer=finalize)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','bind','watch','worker','summarize'))
    globals()[parser.parse_args().command]()
