"""Immutable two-stage execution of the fixed joint recovery-risk study.

freeze does not read predecessor effects. bind requires its completed attempt.
No partial inference, overwrite, implicit resume, model selection or shutdown.
"""
import argparse
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import observations
from coopgate.joint_risk_data_v42 import certificate_from_complete_records
from coopgate.memoized_recovery_v37 import MemoizedRecoveryGate
from coopgate.merge_v4 import MergeConfig, backup, certify
from coopgate.fast_v8 import parameters
from coopgate.robust_v9 import robust_certificate
import design_joint_risk_v42 as design
import run_joint_risk_v42 as collect
import execute_ood_validation_v39 as io
import recover_prospective_v38 as predecessor
from select_completed_v21_scoped import project_processes

ROOT, OUT = design.ROOT, design.OUT
PLAN, BINDING = OUT/'plan.json', OUT/'binding.json'
SCRIPT = 'tools/execute_joint_risk_v42.py'
MEMO = ROOT/'artifacts/v40/memoized_recovery_finalization_20261024'
BLOCK_FILES = ('complete.json', 'observations.sqlite', 'observations.sqlite.seal.json')
TERMINAL_FILES = ('spawn.json', 'launch.json', 'worker.native.json',
                  'worker.acknowledged.json', 'collection_complete.json', 'blocks_manifest.json')


def reset_domain(plan):
    """Check simulator entry assumptions only; never reject/replace a draw."""
    c, checked = MergeConfig(), 0
    packed = parameters(c)
    failures = []
    for stage, cases in plan['stages'].items():
        for case in cases:
            scene, seed = case['scenario'], case['seed']
            initial = design.shared.planned_initial(seed, scene)
            bootstrap = design.shared.planned_initial(seed, scene if scene in ('nominal', 'dense') else 'dense')
            if (not certify(bootstrap, np.zeros(3), backup(np.zeros(3), c), c)[0]
                    or not robust_certificate(initial, np.zeros(3), np.full(3, -2.5), packed, .5)):
                failures.append([stage, case['index']])
            checked += 1
    return dict(checked_initials=checked, reset_domain_failures=failures,
                rejected_or_replaced_initials=0, outcome_free=True)


def freeze():
    if OUT.exists():
        raise FileExistsError('Preserve first joint-risk design')
    plan = design.build_plan()
    design.validate_plan(plan)
    preflight = reset_domain(plan)
    if preflight['reset_domain_failures']:
        raise ValueError('Reset assumptions fail; do not replace seeds: '+str(preflight['reset_domain_failures']))
    for name in plan['source_sha256']:
        io.committed(ROOT/name)
    plan.update(reset_domain=preflight, utc=io.utc())
    OUT.mkdir(parents=True, exist_ok=False)
    io.exclusive_json(PLAN, plan)
    print(json.dumps(dict(plan_sha256=io.digest(PLAN), formal_runs_started=0, **preflight)), flush=True)


def verify_previous_attempt():
    """Read effects only after both original stages have verified full exits."""
    end = predecessor.OUTPUT/'confirmation/analysis/acceptance.json'
    if not end.exists():
        raise ValueError('Previous statistical attempt has no complete confirmation analysis yet')
    predecessor.configure()
    recovery, plan = predecessor.inputs()
    if plan['attempt'] != 1 or recovery['original_attempt'] != 1:
        raise ValueError('Attempt-2 requires the unchanged attempt-1 predecessor')
    for stage in design.STAGES:
        predecessor.mechanics.complete_stage(plan, stage)
        predecessor.original.verify_manifest(predecessor.OUTPUT/stage/'analysis')
    calibration, cal_sha = predecessor.mechanics.calibration_for(plan, 'confirmation')
    comparison = io.read(predecessor.OUTPUT/'confirmation/analysis/comparison.json')
    accepted = io.read(end)
    if (comparison['calibration_sha256'] != cal_sha
            or comparison['recovery_plan_sha256'] != io.digest(predecessor.RECOVERY/'plan.json')
            or accepted != predecessor.original.final_acceptance(calibration, comparison, plan['family'])):
        raise ValueError('Predecessor final analysis identity or decision differs')
    paths = [predecessor.RECOVERY/'plan.json', predecessor.OUTPUT/'frozen/plan.json']
    for stage in design.STAGES:
        paths += [predecessor.OUTPUT/stage/'execution'/n for n in ('completed.json', 'supervisor_exit.json')]
        paths += list((predecessor.OUTPUT/stage/'analysis').glob('*.json'))
    for path in paths:
        io.committed(path)
    return {p.relative_to(ROOT).as_posix(): io.digest(p) for p in paths}


def verify_memoization():
    paths = [MEMO/'finalization_receipt.json', MEMO/'analysis/completion_verification.json',
             MEMO/'analysis/report.json']
    final, proof, report = map(io.read, paths)
    if (final['v39_worker_exit_code'] != 0 or not final['full_180_pair_coverage']
            or proof['exit_code'] != 0 or proof['actual_sumo_runs'] != 360
            or not proof['exact_full_coverage'] or report['pairs'] != 180
            or not report['all_original_non_timing_fields_exact']
            or not report['all_independent_physical_reviews_pass']
            or proof['report_sha256'] != io.digest(paths[2])
            or final['analysis_sha256']['report.json'] != io.digest(paths[2])
            or final['analysis_sha256']['completion_verification.json'] != io.digest(paths[1])):
        raise ValueError('Complete memoization equivalence proof required')
    for path in paths:
        io.committed(path)
    return {p.relative_to(ROOT).as_posix(): io.digest(p) for p in paths}


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if io.digest(ROOT/name) != expected:
            raise ValueError('Bound input changed: '+name)


def plan_inputs():
    io.committed(PLAN)
    plan = io.read(PLAN)
    design.validate_plan(plan)
    if plan['reset_domain'] != dict(checked_initials=4096, reset_domain_failures=[],
                                   rejected_or_replaced_initials=0, outcome_free=True):
        raise ValueError('Frozen reset-domain coverage changed')
    return plan


def bind():
    if BINDING.exists():
        raise FileExistsError('Preserve first joint-risk execution binding')
    plan = plan_inputs()
    prior = verify_previous_attempt()
    memo = verify_memoization()
    sources = set(plan['source_sha256'])
    sources.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'tools').glob('*.py'))
    for name in sources:
        io.committed(ROOT/name)
    binding = dict(plan_sha256=io.digest(PLAN), runtime=io.runtime_identity(),
        sources={n: io.digest(ROOT/n) for n in sorted(sources)}, predecessor=prior, memoization=memo,
        expected_runs={s: 4*design.COUNTS[s] for s in design.STAGES}, utc=io.utc(),
        source_policy='Fixed inventory at binding; unrelated later additions excluded',
        formal_runs_started=0, implicit_resume=False, threshold_or_model_selection=False)
    io.exclusive_json(BINDING, binding)
    print(json.dumps(dict(binding_sha256=io.digest(BINDING), formal_runs_started=0)), flush=True)


def bound_inputs():
    plan = plan_inputs()
    io.committed(BINDING)
    binding = io.read(BINDING)
    if (binding['plan_sha256'] != io.digest(PLAN) or binding['runtime'] != io.runtime_identity()
            or binding['expected_runs'] != {s: 4*design.COUNTS[s] for s in design.STAGES}
            or any(binding['sources'].get(n) != sha for n, sha in plan['source_sha256'].items())
            or not binding['predecessor'] or not binding['memoization']):
        raise ValueError('Execution binding changed or prerequisite proof missing')
    for key in ('sources', 'predecessor', 'memoization'):
        verify_hashes(binding[key])
    return plan, binding


def require_quiet(processes, own_pid=None, parent_pid=None):
    """Allow only this command's observed ancestor chain, never sibling jobs."""
    own = os.getpid() if own_pid is None else own_pid
    parent = os.getppid() if parent_pid is None else parent_pid
    by_id = {p['ProcessId']: p for p in processes}
    allowed = {own, parent}
    cursor = parent
    while cursor in by_id:
        following = by_id[cursor]['ParentProcessId']
        if following in allowed:
            break
        allowed.add(following)
        cursor = following
    relevant = {p['ProcessId'] for p in project_processes(processes)}
    commands = ['tools/'+p.name for p in (ROOT/'tools').glob('*.py')]
    relevant.update(p['ProcessId'] for p in processes if any(
        name in (p.get('CommandLine') or '').replace('\\', '/') for name in commands))
    while True:
        children = {p['ProcessId'] for p in processes if p['ParentProcessId'] in relevant}
        if children <= relevant:
            break
        relevant |= children
    if relevant-allowed:
        raise ValueError('Other project study processes remain live: '+str(sorted(relevant-allowed)))


def block_hashes(directory, index):
    return dict(index=index, files={name: io.digest(directory/name) for name in BLOCK_FILES})


def validate_terminal(receipt, complete, manifest, stage, plan_sha, binding_sha):
    n = design.COUNTS[stage]
    if (receipt['exit_code'] != 0 or complete['native_pid'] != receipt['native_pid']
            or any(v.get('stage') != stage or v.get('plan_sha256') != plan_sha
                   or v.get('binding_sha256') != binding_sha for v in (receipt, complete, manifest))
            or complete['actual_sumo_runs'] != 4*n or complete['clusters'] != n
            or manifest['actual_sumo_runs'] != 4*n
            or [v['index'] for v in manifest['entries']] != list(range(n))):
        raise ValueError('No complete zero-exit stage with the exact independent schedule')


def stage_terminal(stage, plan_sha, binding_sha):
    directory = OUT/stage
    execution = directory/'execution'
    receipt = io.read(execution/'exit.json')
    complete = io.read(execution/'collection_complete.json')
    manifest = io.read(execution/'blocks_manifest.json')
    validate_terminal(receipt, complete, manifest, stage, plan_sha, binding_sha)
    if set(receipt['hashes']) != set(TERMINAL_FILES):
        raise ValueError('Incomplete terminal receipt chain')
    for name, expected in receipt['hashes'].items():
        if io.digest(execution/name) != expected:
            raise ValueError('Changed terminal evidence: '+name)
    if complete['manifest_sha256'] != io.digest(execution/'blocks_manifest.json'):
        raise ValueError('Collection did not bind every block')
    io.check_acknowledgment(execution, 'worker', receipt['launcher_pid'], receipt['native_pid'])
    launch = io.read(execution/'launch.json')
    if any(receipt[k] != launch[k] for k in ('native_pid', 'launcher_pid', 'native_creation_date', 'command_line')):
        raise ValueError('Terminal process differs from launched process')
    if any(p['ProcessId'] == receipt['native_pid'] and p['CreationDate'] == receipt['native_creation_date']
           for p in io.native_snapshot()):
        raise ValueError('Original stage worker is still live')
    for entry in manifest['entries']:
        if block_hashes(directory/'blocks'/f"{entry['index']:05d}", entry['index']) != entry:
            raise ValueError('Completed block changed')
    return io.digest(execution/'exit.json')


def load_models(plan):
    return {name: load_coupled(ROOT/item['path'])[0] for name, item in plan['models'].items()}


def warm(models):
    state = design.shared.planned_initial(139_883_001, 'nominal')
    for plant in design.PLANTS:
        MemoizedRecoveryGate(None, plant, 'baseline').select(state, np.zeros(3))
        for model in models.values():
            MemoizedRecoveryGate(model, plant, 'bank_performance').select(state, np.zeros(3))


def worker(stage):
    execution = OUT/stage/'execution'
    startup = execution/'worker.native.json'
    io.exclusive_json(startup, dict(native_pid=os.getpid(), parent_pid=os.getppid(), stage=stage, utc=io.utc()))
    ack_path, deadline = execution/'worker.acknowledged.json', time.monotonic()+180
    while not ack_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('No native acknowledgment; no SUMO started')
        time.sleep(.2)
    ack = io.read(ack_path)
    if (ack['native_pid'] != os.getpid() or ack['parent_pid'] != os.getppid()
            or ack['native_start_sha256'] != io.digest(startup)):
        raise ValueError('Native acknowledgment mismatch')
    plan, binding = bound_inputs()
    require_quiet(io.native_snapshot())
    plan_sha, binding_sha = io.digest(PLAN), io.digest(BINDING)
    calibration_exit = (stage_terminal('calibration', plan_sha, binding_sha) if stage == 'confirmation' else None)
    torch.set_num_threads(1)
    models = load_models(plan)
    warm(models)
    blocks = OUT/stage/'blocks'
    blocks.mkdir(exist_ok=False)
    entries = []
    for case in plan['stages'][stage]:
        verify_hashes(binding['sources'])
        for item in plan['models'].values():
            if io.digest(ROOT/item['path']) != item['sha256']:
                raise ValueError('Fixed model bytes changed')
        if io.digest(PLAN) != plan_sha or io.digest(BINDING) != binding_sha:
            raise ValueError('Plan or binding changed during collection')
        directory = blocks/f"{case['index']:05d}"
        collect.collect_cluster(case, directory, models, plan_sha=plan_sha, binding_sha=binding_sha)
        entries.append(block_hashes(directory, case['index']))
        progress = dict(stage=stage, native_pid=os.getpid(), completed_clusters=len(entries),
                        expected_clusters=design.COUNTS[stage], completed_runs=4*len(entries), utc=io.utc())
        io.atomic_bytes(execution/'progress.json', json.dumps(progress).encode())
        print(json.dumps(progress), flush=True)
    bound_inputs()
    if io.digest(PLAN) != plan_sha or io.digest(BINDING) != binding_sha:
        raise ValueError('Plan or binding changed during final collection block')
    if stage == 'confirmation' and stage_terminal('calibration', plan_sha, binding_sha) != calibration_exit:
        raise ValueError('Calibration completion changed during confirmation')
    shared = dict(stage=stage, plan_sha256=plan_sha, binding_sha256=binding_sha, actual_sumo_runs=4*len(entries))
    io.exclusive_json(execution/'blocks_manifest.json', dict(**shared, entries=entries))
    io.exclusive_json(execution/'collection_complete.json', dict(**shared, native_pid=os.getpid(),
        clusters=len(entries), calibration_exit_sha256=calibration_exit,
        manifest_sha256=io.digest(execution/'blocks_manifest.json'), utc=io.utc()))


def watch(stage):
    bound_inputs()
    require_quiet(io.native_snapshot())
    if stage == 'confirmation':
        stage_terminal('calibration', io.digest(PLAN), io.digest(BINDING))
    execution = OUT/stage/'execution'
    execution.mkdir(parents=True, exist_ok=False)
    plan_sha, binding_sha = io.digest(PLAN), io.digest(BINDING)
    with (execution/'stdout.log').open('x') as stdout, (execution/'stderr.log').open('x') as stderr:
        command = [sys.executable, '-X', 'utf8', SCRIPT, 'worker', stage]
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        io.exclusive_json(execution/'spawn.json', dict(watcher_pid=os.getpid(), launcher_pid=process.pid,
            command=command, stage=stage, plan_sha256=plan_sha, binding_sha256=binding_sha, utc=io.utc()))
        try:
            identity = io.acknowledge_child(process, execution, 'worker')
            if SCRIPT not in identity['command_line'] or f'worker {stage}' not in identity['command_line']:
                raise ValueError('Unexpected native worker command')
            io.exclusive_json(execution/'launch.json', identity)
        except Exception as exc:
            io.exclusive_json(execution/'observation_error.json', dict(type=type(exc).__name__, message=str(exc)))
            code = process.wait()  # Retain the actual handle; never relaunch on observation timeout.
            io.exclusive_json(execution/'exit.json', dict(exit_code=code, launcher_pid=process.pid, utc=io.utc()))
            raise
        code = process.wait()
        hashes = {n: io.digest(execution/n) for n in TERMINAL_FILES if (execution/n).exists()}
        io.exclusive_json(execution/'exit.json', dict(**identity, stage=stage, exit_code=code,
            plan_sha256=plan_sha, binding_sha256=binding_sha, hashes=hashes, utc=io.utc()))
    if code:
        raise SystemExit(code)


def finalize():
    plan, binding = bound_inputs()
    require_quiet(io.native_snapshot())
    plan_sha, binding_sha = io.digest(PLAN), io.digest(BINDING)
    exits = {s: stage_terminal(s, plan_sha, binding_sha) for s in design.STAGES}
    confirmation = io.read(OUT/'confirmation/execution/collection_complete.json')
    if confirmation['calibration_exit_sha256'] != exits['calibration']:
        raise ValueError('Confirmation did not follow this completed calibration')
    torch.set_num_threads(1)
    models, records, reviews = load_models(plan), {}, []
    for stage, cases in plan['stages'].items():
        records[stage] = []
        for case in cases:
            directory = OUT/stage/'blocks'/f"{case['index']:05d}"
            metadata = dict(stage=stage, cluster_id=case['cluster_id'], plan_sha256=plan_sha, binding_sha256=binding_sha)
            complete = io.read(directory/'complete.json')
            if any(complete[k] != v for k, v in metadata.items()) or complete['actual_sumo_runs'] != 4:
                raise ValueError('Cluster completion identity differs')
            for key, body, trace in observations(directory/'observations.sqlite', metadata, collect.schedule(case)):
                if key != [body['cluster_id'], body['method']]:
                    raise ValueError('Stored identity differs from scheduled key')
                model = None if body['method'] == 'baseline' else models[body['method']]
                reviews.append(dict(cluster_id=case['cluster_id'], method=body['method'],
                    **collect.audit_record(case, dict(**body, trace=trace), model)))
                records[stage].append(body)
    # Recheck immutable inputs/receipts after the potentially lengthy replay.
    bound_inputs()
    if io.digest(PLAN) != plan_sha or io.digest(BINDING) != binding_sha:
        raise ValueError('Plan or binding changed during full replay')
    if {s: stage_terminal(s, plan_sha, binding_sha) for s in design.STAGES} != exits:
        raise ValueError('Stage evidence changed during full replay')
    result = certificate_from_complete_records(plan, records['calibration'], records['confirmation'])
    analysis = OUT/'analysis'
    analysis.mkdir(exist_ok=False)
    io.exclusive_json(analysis/'certificate.json', result)
    io.exclusive_json(analysis/'replay.json', dict(actual_sumo_runs=len(reviews), reviews=reviews))
    io.exclusive_json(analysis/'verification.json', dict(plan_sha256=plan_sha, binding_sha256=binding_sha,
        stage_exit_sha256=exits, actual_sumo_runs=len(reviews), independent_initials=4096,
        certificate_sha256=io.digest(analysis/'certificate.json'), replay_sha256=io.digest(analysis/'replay.json')))
    print(json.dumps(dict(actual_sumo_runs=len(reviews),
        joint_deployment_certified=result['joint_deployment_certified'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'bind', 'finalize'):
        commands.add_parser(name)
    for name in ('watch', 'worker'):
        commands.add_parser(name).add_argument('stage', choices=design.STAGES)
    args = parser.parse_args()
    globals()[args.command](*([args.stage] if hasattr(args, 'stage') else []))
