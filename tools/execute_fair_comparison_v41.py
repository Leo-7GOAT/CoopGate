"""Freeze, collect and independently finalize the complete fair comparison.

No partial efficacy analysis; no overwrite or implicit resume. Design and
execution binding must both be committed before the native worker can start.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import observations
from coopgate.fair_comparison_v39 import CaseIdentity, summarize_matched_records
from coopgate.fair_runtime_v41 import EvidenceFactory
import execute_ood_validation_v39 as io
import freeze_fair_comparison_v39 as design
import run_fair_runtime_v41 as runtime


ROOT = Path(__file__).resolve().parents[1]
PLAN = design.OUT/'plan.json'
OUT = ROOT/'artifacts/v41/fair_comparison_execution'
BINDING = OUT/'binding.json'
SCRIPT = 'tools/execute_fair_comparison_v41.py'
TEST = 'tests/test_execute_fair_comparison_v41.py'


def source_paths():
    # Conservative, fixed inventory at bind. Later unrelated added files do not
    # modify it. Every actually imported repository tool/package is included.
    return sorted([*ROOT.joinpath('src/coopgate').glob('*.py'),
                   *ROOT.joinpath('tools').glob('*.py'), ROOT/TEST,
                   ROOT/'tests/test_fair_runtime_v41.py'])


def bind():
    if OUT.exists():
        raise FileExistsError('Preserve the first fair execution binding')
    io.committed(PLAN)
    plan = io.read(PLAN)
    design.plan_check(plan)
    paths = source_paths()
    for path in paths:
        io.committed(path)
    binding = dict(plan_sha256=io.digest(PLAN), source_sha256={
        path.relative_to(ROOT).as_posix(): io.digest(path) for path in paths},
        runtime=io.runtime_identity(), expected_runs=plan['actual_sumo_runs'],
        expected_blocks=plan['blocks'], utc=io.utc(),
        source_set_policy='All package and tool Python files at bind; later unrelated additions excluded',
        warmup='Each model and actuator class, development seed 139881001 nominal observation; no SUMO',
        evidence_contract=plan['evidence_execution'],
        failure_policy='Durably save completed/aborted run before audit; infrastructure failures stop study',
        resumable=False, actual_sumo_runs_started=0)
    OUT.mkdir(exist_ok=False, parents=True)
    io.exclusive_json(BINDING, binding)
    print(json.dumps(dict(binding_sha256=io.digest(BINDING),
                          expected_runs=binding['expected_runs'], launched=False)), flush=True)


def bound_inputs():
    io.committed(PLAN)
    io.committed(BINDING)
    plan, binding = io.read(PLAN), io.read(BINDING)
    design.plan_check(plan)
    if (binding['plan_sha256'] != io.digest(PLAN)
            or binding['expected_runs'] != 5120 or binding['expected_blocks'] != 160
            or plan['actual_sumo_runs'] != binding['expected_runs']
            or plan['blocks'] != binding['expected_blocks']
            or binding['runtime'] != io.runtime_identity()
            or binding['evidence_contract'] != plan['evidence_execution']):
        raise ValueError('Changed fair execution design/runtime')
    required = {SCRIPT, TEST, 'tools/run_fair_runtime_v41.py', 'src/coopgate/fair_runtime_v41.py',
                'tools/freeze_fair_comparison_v39.py', 'tools/execute_ood_validation_v39.py'}
    if not required <= set(binding['source_sha256']):
        raise ValueError('Missing fair execution dependency')
    verify_sources(binding)
    return plan, binding


def verify_sources(binding):
    for name, expected in binding['source_sha256'].items():
        if io.digest(ROOT/name) != expected:
            raise ValueError('Bound source changed: '+name)


def load_inputs(plan):
    models = {name: load_coupled(ROOT/item['path'])[0] for name, item in plan['models'].items()}
    calibrations = io.read(ROOT/plan['calibration']['path'])
    return models, calibrations


def warm(plan, models, calibrations):
    """Compile all paths on a separate fixed development input, before timing."""
    spec = design.protocol_from_dict(plan['protocol'])
    spec = replace(spec, budget=replace(spec.budget, decision_timeout_ms=1e9))
    original = CaseIdentity(**plan['cases'][0])
    x = design.planned_initial(139881001, 'nominal')
    for model_id, model in models.items():
        for profile in design.PLANTS:
            case = replace(original, policy_id=model_id, dynamics=profile,
                           policy_sha256=plan['models'][model_id]['sha256'])
            calibration = calibrations[plan['calibration']['historical_model_key_map'][model_id]]
            evidence, record = EvidenceFactory(case, spec, model, calibration).build(x, [])
            if evidence is None or record['error'] is not None or evidence.ledger.timed_out:
                raise RuntimeError('Development warmup failed before data collection')


def block_hashes(directory, index):
    return dict(block_index=index, **{name: io.digest(directory/name) for name in
        ('complete.json', 'observations.sqlite', 'observations.sqlite.seal.json')})


def worker():
    execution = OUT/'execution'
    native = execution/'worker.native.json'
    io.exclusive_json(native, dict(native_pid=os.getpid(), parent_pid=os.getppid(), utc=io.utc()))
    acknowledgment = execution/'worker.acknowledged.json'
    deadline = time.monotonic()+180
    while not acknowledgment.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('No parent acknowledgment; no simulation started')
        time.sleep(.2)
    ack = io.read(acknowledgment)
    if (ack['native_pid'] != os.getpid() or ack['parent_pid'] != os.getppid()
            or ack['native_start_sha256'] != io.digest(native)):
        raise ValueError('Native acknowledgment differs')
    plan, binding = bound_inputs()
    torch.set_num_threads(1)
    models, calibrations = load_inputs(plan)
    warm(plan, models, calibrations)
    blocks = OUT/'blocks'
    blocks.mkdir(exist_ok=False)
    entries, completed = [], 0
    plan_sha, binding_sha = io.digest(PLAN), io.digest(BINDING)
    for group in plan['groups']:
        verify_sources(binding)
        design.validate_dependencies(plan)
        if io.digest(PLAN) != plan_sha or io.digest(BINDING) != binding_sha:
            raise ValueError('Plan or binding changed during collection')
        directory = blocks/f"{group['block_index']:05d}"
        receipt = runtime.collect_block(plan, group, directory, models, calibrations, plan_sha, binding_sha)
        if receipt['actual_sumo_runs'] != group['actual_sumo_runs']:
            raise ValueError('Incomplete matched block')
        entries.append(block_hashes(directory, group['block_index']))
        completed += receipt['actual_sumo_runs']
        progress = dict(native_pid=os.getpid(), completed_runs=completed, expected_runs=5120,
                        completed_blocks=len(entries), expected_blocks=160, utc=io.utc())
        io.atomic_bytes(execution/'progress.json', json.dumps(progress).encode())
        print(json.dumps(progress), flush=True)
    if completed != 5120 or len(entries) != 160:
        raise ValueError('Incomplete formal schedule')
    verify_sources(binding)
    design.validate_dependencies(plan)
    if io.digest(PLAN) != plan_sha or io.digest(BINDING) != binding_sha:
        raise ValueError('Plan or binding changed during final collection block')
    io.exclusive_json(execution/'blocks_manifest.json', dict(plan_sha256=plan_sha,
        binding_sha256=binding_sha, entries=entries, actual_sumo_runs=completed))
    io.exclusive_json(execution/'collection_complete.json', dict(native_pid=os.getpid(),
        plan_sha256=plan_sha, binding_sha256=binding_sha, actual_sumo_runs=completed,
        blocks=len(entries), blocks_manifest_sha256=io.digest(execution/'blocks_manifest.json'), utc=io.utc()))


def watch():
    plan, _ = bound_inputs()
    execution = OUT/'execution'
    execution.mkdir(exist_ok=False)
    with (execution/'stdout.log').open('x') as stdout, (execution/'stderr.log').open('x') as stderr:
        command = [sys.executable, '-X', 'utf8', SCRIPT, 'worker']
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        io.exclusive_json(execution/'spawn.json', dict(watcher_pid=os.getpid(), launcher_pid=process.pid,
            command=command, plan_sha256=io.digest(PLAN), binding_sha256=io.digest(BINDING), utc=io.utc()))
        identity = None
        try:
            identity = io.acknowledge_child(process, execution, 'worker')
            if SCRIPT not in identity['command_line'] or 'worker' not in identity['command_line']:
                raise ValueError('Observed worker has unexpected command line')
            io.exclusive_json(execution/'launch.json', identity)
        except Exception as exc:
            io.exclusive_json(execution/'observation_error.json', dict(type=type(exc).__name__, message=str(exc)))
            # Preserve the specific process handle; an observation failure is
            # not authority to relaunch, overwrite, or kill its work.
            code = process.wait()
            io.exclusive_json(execution/'exit.json', dict(exit_code=code, launcher_pid=process.pid,
                                                         acknowledgment_verified=False, utc=io.utc()))
            raise
        code = process.wait()
        hashes = {name: io.digest(execution/name) for name in ('spawn.json', 'launch.json',
            'worker.native.json', 'worker.acknowledged.json', 'collection_complete.json', 'blocks_manifest.json')
            if (execution/name).exists()}
        io.exclusive_json(execution/'exit.json', dict(**identity, exit_code=code,
            plan_sha256=io.digest(PLAN), binding_sha256=io.digest(BINDING), hashes=hashes, utc=io.utc()))
    if code:
        raise SystemExit(code)


def validate_terminal(receipt, done, manifest, plan_sha, binding_sha):
    if (receipt['exit_code'] != 0 or receipt.get('plan_sha256') != plan_sha
            or receipt.get('binding_sha256') != binding_sha
            or done['plan_sha256'] != plan_sha or done['binding_sha256'] != binding_sha
            or done['native_pid'] != receipt['native_pid'] or done['actual_sumo_runs'] != 5120
            or done['blocks'] != 160 or manifest['plan_sha256'] != plan_sha
            or manifest['binding_sha256'] != binding_sha or manifest['actual_sumo_runs'] != 5120
            or [entry['block_index'] for entry in manifest['entries']] != list(range(160))):
        raise ValueError('No verified complete zero-exit fair study')


def summarize():
    plan, binding = bound_inputs()
    execution = OUT/'execution'
    receipt = io.read(execution/'exit.json')
    done = io.read(execution/'collection_complete.json')
    manifest = io.read(execution/'blocks_manifest.json')
    plan_sha, binding_sha = io.digest(PLAN), io.digest(BINDING)
    validate_terminal(receipt, done, manifest, plan_sha, binding_sha)
    expected_files = {'spawn.json', 'launch.json', 'worker.native.json', 'worker.acknowledged.json',
                      'collection_complete.json', 'blocks_manifest.json'}
    if set(receipt['hashes']) != expected_files:
        raise ValueError('Incomplete exit hash chain')
    for name, expected in receipt['hashes'].items():
        if io.digest(execution/name) != expected:
            raise ValueError('Changed terminal receipt: '+name)
    if done['blocks_manifest_sha256'] != io.digest(execution/'blocks_manifest.json'):
        raise ValueError('Worker did not bind the complete block inventory')
    io.check_acknowledgment(execution, 'worker', receipt['launcher_pid'], receipt['native_pid'])
    for item in io.native_snapshot():
        if (item['ProcessId'] == receipt['native_pid']
                and item['CreationDate'] == receipt['native_creation_date']):
            raise ValueError('Original native worker remains live')
    for entry in manifest['entries']:
        if block_hashes(OUT/'blocks'/f"{entry['block_index']:05d}", entry['block_index']) != entry:
            raise ValueError('Block was modified after completion')
    torch.set_num_threads(1)
    models, calibrations = load_inputs(plan)
    cases = {v['case_id']: CaseIdentity(**v) for v in plan['cases']}
    spec = design.protocol_from_dict(plan['protocol'])
    episodes, reviews, builds, hits = [], [], 0, 0
    for group in plan['groups']:
        metadata = dict(plan_sha256=plan_sha, binding_sha256=binding_sha, block_index=group['block_index'])
        directory = OUT/'blocks'/f"{group['block_index']:05d}"
        factories, verified = {}, {}
        complete = io.read(directory/'complete.json')
        if any(complete[key] != value for key, value in metadata.items()):
            raise ValueError('Completed block identity differs')
        builds += complete['unique_packets_computed']
        hits += complete['cached_packet_uses']
        for (case_id, method), body, trace in observations(directory/'observations.sqlite', metadata,
                                                         runtime.block_schedule(group)):
            case = cases[case_id]
            if case_id not in factories:
                factories[case_id] = EvidenceFactory(case, spec, models[case.policy_id],
                    calibrations[plan['calibration']['historical_model_key_map'][case.policy_id]])
                verified[case_id] = {}
            if body['episode']['method'] != method or body['episode']['case'] != asdict_case(case):
                raise ValueError('Stored observation changed method/case')
            record = dict(**body, trace=trace)
            reviews.append(runtime.audit_episode(record, factories[case_id], verified[case_id]))
            episodes.append(runtime.decode_episode(body['episode']))
    if len(episodes) != 5120:
        raise ValueError('Missing paired episodes')
    report = summarize_matched_records(episodes, spec)
    report.update(actual_sumo_runs=len(episodes), unique_packets_computed=builds,
                  cached_packet_uses=hits, plan_sha256=plan_sha, binding_sha256=binding_sha,
                  standalone_controller_latency_benchmark=False)
    analysis = OUT/'analysis'
    analysis.mkdir(exist_ok=False)
    io.exclusive_json(analysis/'report.json', report)
    io.exclusive_json(analysis/'replay.json', dict(reviews=reviews, actual_sumo_runs=len(episodes)))
    io.exclusive_json(analysis/'verification.json', dict(plan_sha256=plan_sha, binding_sha256=binding_sha,
        exit_sha256=io.digest(execution/'exit.json'), report_sha256=io.digest(analysis/'report.json'),
        replay_sha256=io.digest(analysis/'replay.json'), actual_sumo_runs=5120))


def asdict_case(case):
    from dataclasses import asdict
    return asdict(case)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('bind', 'watch', 'worker', 'summarize'))
    globals()[parser.parse_args().command]()
