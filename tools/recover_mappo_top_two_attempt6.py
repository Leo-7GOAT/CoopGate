"""Explicit interrupted-attempt recovery; no new alpha, seeds or controller.

Reuse every checksum-valid committed observation, including unsealed prefixes.
Original files and unknown native exits remain unchanged. Private module scopes
redirect only output roots, native script paths and retained-record retrieval.
Registered collection, review, matching and inference function bodies are reused.
"""
from pathlib import Path
from types import ModuleType, FunctionType, SimpleNamespace
from copy import deepcopy
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mappo_top_two_formal_experiment as original
import mappo_top_two_formal_finalize as finalizer
import run_mappo_top_two_formal as launcher
from mappo_native_launch import command as native_command
from coopgate.episode_store_v21 import observations

ROOT = original.ROOT
AUDIT = original.BASE / 'attempt6_interruption_audit_20261104'
OUT = original.BASE / 'formal_mappo_attempt6_top_two_recovery_20261104'
PROTOCOL = ROOT / 'docs/mappo_attempt6_interruption_recovery_20261104.md'


def durable(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def prepare():
    plan = original.bound(original.FORMAL, False)
    audit = original.read(AUDIT / 'report.json')
    if (not audit['completed'] or not audit['exit_unknown_not_inferred']
            or audit['original_plan_sha256'] != original.sha(original.FORMAL / 'plan.json')
            or audit['retained_checksum_valid_committed_records'] != 642):
        raise ValueError('Exact original interruption audit required')
    OUT.mkdir(exist_ok=False)
    for name in ('plan.json', 'allocation.json', 'binding.json', 'source_snapshot.json'):
        shutil.copyfile(original.FORMAL / name, OUT / name)
    shutil.copytree(original.FORMAL / 'source_snapshot', OUT / 'source_snapshot')
    durable(OUT / 'recovery_binding.json', dict(
        original_plan_sha256=original.sha(original.FORMAL / 'plan.json'),
        original_allocation_sha256=original.sha(original.FORMAL / 'allocation.json'),
        audit_sha256=original.sha(AUDIT / 'report.json'),
        source_sha256=original.sha(__file__), protocol_sha256=original.sha(PROTOCOL),
        retained_actual_observations=642, new_actual_sumo_required=58750,
        effective_fixed_schedule_records=59392, independent_initials=7168,
        attempt=6, allocated_error=.05/42, new_error_allocated=False,
        original_native_exit_code=None, unknown_lost_inflight_executions=True,
        original_data_modified=False, controller_or_threshold_changed=False,
        sample_replacement_or_effect_based_selection=False))
    print(json.dumps(dict(prepared=True, retained=642, new_required=58750, new_error_allocated=False)), flush=True)


def guard():
    original.bound(original.FORMAL, False)
    binding = original.read(OUT / 'recovery_binding.json')
    for key, path in [('source_sha256', __file__), ('protocol_sha256', PROTOCOL),
                      ('audit_sha256', AUDIT / 'report.json'),
                      ('original_plan_sha256', original.FORMAL / 'plan.json'),
                      ('original_allocation_sha256', original.FORMAL / 'allocation.json')]:
        if binding[key] != original.sha(path):
            raise ValueError('Recovery binding changed: ' + str(path))
    if original.sha(OUT / 'plan.json') != binding['original_plan_sha256']:
        raise ValueError('Original schedule must remain byte-identical')
    audit = original.read(AUDIT / 'report.json')
    for item in audit['blocks']:
        path = ROOT / item['cache']
        if original.sha(path) != item['cache_sha256'] or original.sha(str(path)+'.seal.json') != item['seal_sha256']:
            raise ValueError('Retained committed observations changed')
    return binding, audit


def clone(module, **overrides):
    result = ModuleType(module.__name__ + '_private_recovery_scope')
    result.__dict__.update(module.__dict__)
    result.__dict__.update(overrides)
    for name, value in module.__dict__.items():
        if isinstance(value, FunctionType) and value.__globals__ is module.__dict__:
            copied = FunctionType(value.__code__, result.__dict__, value.__name__, value.__defaults__, value.__closure__)
            copied.__kwdefaults__ = value.__kwdefaults__
            result.__dict__[name] = copied
    return result


def redirect(script, *args):
    if Path(script).name in ('mappo_top_two_formal_experiment.py', 'mappo_top_two_formal_finalize.py'):
        return native_command(__file__, *args)
    raise ValueError('Unexpected native child script; do not launch original output root')


def context(shard=None):
    binding, audit = guard()
    retained = {}
    if shard is not None:
        for item in audit['blocks']:
            if item['shard'] != shard:
                continue
            for key, row, trace in observations(ROOT / item['cache'], item['metadata'], item['keys']):
                if key in retained:
                    raise ValueError('Duplicate retained observation')
                retained[key] = dict(row, trace=trace)
    counters = dict(retained=0, new=0)
    def execute(case, method, model, env):
        key = (case['cluster_id'], method)
        if key in retained:
            counters['retained'] += 1
            return deepcopy(retained[key])
        counters['new'] += 1
        return original.runtime.execute_episode(case, method, model, env)
    runtime = SimpleNamespace(**original.runtime.__dict__)
    runtime.execute_episode = execute
    experiment = clone(original, FORMAL=OUT, runtime=runtime)
    frozen_bound = experiment.bound
    def checked_bound(root, validation):
        guard()
        return frozen_bound(root, validation)
    experiment.bound = checked_bound
    return experiment, counters


def own():
    guard()
    receipt = original.read(OUT / 'public_recovery_registration.json')
    commit = receipt['registered_commit']
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT, text=True).split()[0]
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=ROOT, check=True)
    for name, digest in receipt['registered_files_sha256'].items():
        import hashlib
        raw = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != digest or original.sha(ROOT / name) != digest:
            raise ValueError('Public recovery registration bytes differ')
    if (OUT / 'owner_launch.json').exists() or (OUT / 'execution').exists():
        raise FileExistsError('Recovery already launched; do not duplicate')
    args = native_command(__file__, 'run')
    with (OUT / 'owner.log').open('xb') as log:
        child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        durable(OUT / 'owner_launch.json', dict(native_pid=child.pid, owner_pid=os.getpid(), args=args,
            registered_commit=commit, started_utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(recovery_native_pid=child.pid)), flush=True)
        code = child.wait()
    terminal = dict(native_pid=child.pid, actual_native_exit_code=code, observed_by_direct_native_wait=True,
        finished_utc=datetime.now(timezone.utc).isoformat(), original_owner_exit_code=None,
        original_interrupted_attempt_retained=True)
    if code == 0:
        terminal['completion_receipt_sha256'] = original.sha(OUT / 'completion_receipt.json')
    durable(OUT / 'orchestrator_native_receipt.json', terminal)
    print(json.dumps(terminal), flush=True)
    raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'own', 'run', 'collect', 'review', 'matching', 'finalize'))
    parser.add_argument('--stage'); parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(); return
    if args.action == 'own':
        own(); return
    experiment, counters = context(args.shard if args.action == 'collect' and args.stage == 'calibration' else None)
    if args.action == 'collect':
        experiment.collect(args.stage, args.shard)
        durable(OUT / args.stage / f'shard{args.shard}' / 'actual_recovery_collection_counts.json', counters)
    elif args.action == 'review':
        experiment.review(args.stage, args.shard)
    elif args.action in ('matching', 'finalize'):
        clone(finalizer, experiment=experiment).__dict__[args.action]()
    else:
        clone(launcher, experiment=experiment, OUT=OUT, command=redirect).run()
        counts = [original.read(OUT / stage / f'shard{s}/actual_recovery_collection_counts.json')
                  for stage in ('calibration', 'confirmation') for s in range(4)]
        if sum(q['retained'] for q in counts) != 642 or sum(q['new'] for q in counts) != 58750:
            raise ValueError('Exact full-schedule retained/new execution reconciliation failed')
        durable(OUT / 'recovery_delivery_accounting.json', dict(
            complete=True, retained_actual_records=642, new_actual_sumo_executions=58750,
            full_effective_schedule=59392, original_unknown_lost_inflight_executions=True,
            original_root_exit_code=None, new_error_allocated=False,
            old_allocation_sha256=original.sha(original.FORMAL / 'allocation.json')))


if __name__ == '__main__':
    main()
