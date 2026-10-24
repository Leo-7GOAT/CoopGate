"""Freeze and execute the full 640-case candidate-family/gate factorial.

No partial effects, overwrite, implicit resume, or alteration of parent studies.
The strict physical certificate audit retains a counterexample before stopping.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from coopgate.candidate_ablation_v43 import VARIANTS, paired_mechanism_summary
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import EpisodeStore, observations
from coopgate.joint_risk_data_v42 import episode_outcome
import execute_fair_failure_v44 as parent
import execute_joint_risk_v42 as isolation
import run_candidate_ablation_v43 as runtime

io, ROOT = parent.io, parent.ROOT
OUT = ROOT/'artifacts/candidate_factorial/complete_fair_cases'
PLAN = OUT/'plan.json'
SCRIPT = 'tools/execute_candidate_factorial.py'
TEST = 'tests/test_execute_candidate_factorial.py'
DOC = 'docs/candidate_factorial_execution.md'
BLOCK_FILES = ('observations.sqlite', 'observations.sqlite.seal.json', 'complete.json')
TERMINAL_FILES = ('spawn.json', 'launch.json', 'worker.native.json', 'worker.acknowledged.json',
                  'collection_complete.json', 'blocks_manifest.json')


def cases_from_parent(plan):
    cases = []
    for index, source in enumerate(plan['cases']):
        offset = index % 5
        variants = list(VARIANTS[offset:]+VARIANTS[:offset])
        if index % 2:
            variants.reverse()
        cases.append(dict(block_index=index, parent_case_id=source['case_id'],
            cluster_id=source['independent_cluster'], seed=source['initial_id'],
            scenario=source['scenario'], plant=source['dynamics'], disturbance=source['disturbance'],
            policy_id=source['policy_id'], policy_sha256=source['policy_sha256'],
            initial_sha256=source['initial_sha256'], dynamics_sha256=source['dynamics_sha256'],
            disturbance_sha256=source['disturbance_sha256'], variants=variants))
    if len(cases) != 640 or len({c['cluster_id'] for c in cases}) != 640:
        raise ValueError('Exactly the 640 original fair case identities are required')
    return cases


def contract():
    return dict(actual_sumo_runs=3200, independent_clusters=640,
        additional_independent_clusters_beyond_parent=0, blocks=640,
        variants=list(VARIANTS), cutoff_ms=None, equal_compute_claimed=False,
        effect_scope='Descriptive paired factorial; no significance or new risk certificate',
        failure_policy='Retain ordinary failures; all-scope gain unavailable when a relevant method/reference fails',
        certificate_counterexample_policy='Persist before audit; preserve prefix and stop on physical certificate violation',
        timing_scope='Measured callbacks, not hard realtime or independent latency benchmark',
        model_selection=False, resumable=False)


def verify_hashes(mapping):
    for name, sha in mapping.items():
        if io.digest(ROOT/name) != sha:
            raise ValueError('Frozen factorial input changed: '+name)


def freeze():
    if OUT.exists():
        raise FileExistsError('Preserve the first factorial plan')
    if any((parent.OUT/name).exists() for name in ('execution','blocks','analysis')):
        raise ValueError('Freeze factorial scope before parent formal outcomes exist')
    original, _ = parent.bound_inputs()
    paths = sorted(set([*ROOT.joinpath('src/coopgate').glob('*.py'),
                        *ROOT.joinpath('tools').glob('*.py'), ROOT/TEST, ROOT/DOC,
                        ROOT/'docs/v43_candidate_gate_factorial.md']))
    for path in paths:
        io.committed(path)
    plan = dict(contract=contract(), cases=cases_from_parent(original), models=original['models'],
                parent_plan_sha256=io.digest(parent.PLAN), parent_binding_sha256=io.digest(parent.BINDING),
                source_sha256={p.relative_to(ROOT).as_posix():io.digest(p) for p in paths},
                runtime=io.runtime_identity(), created_utc=io.utc(), actual_sumo_runs_started=0)
    OUT.mkdir(parents=True, exist_ok=False)
    io.exclusive_json(PLAN,plan)
    print(json.dumps(dict(plan_sha256=io.digest(PLAN), actual_sumo_runs=3200, launched=False)),flush=True)


def inputs():
    io.committed(PLAN)
    plan = io.read(PLAN)
    original, _ = parent.bound_inputs()
    if (plan['contract'] != contract() or plan['cases'] != cases_from_parent(original)
            or plan['models'] != original['models'] or plan['runtime'] != io.runtime_identity()
            or plan['parent_plan_sha256'] != io.digest(parent.PLAN)
            or plan['parent_binding_sha256'] != io.digest(parent.BINDING)
            or not {SCRIPT,TEST,DOC,'tools/run_candidate_ablation_v43.py',
                    'src/coopgate/candidate_ablation_v43.py'} <= set(plan['source_sha256'])):
        raise ValueError('Changed complete factorial design or parent identities')
    verify_hashes(plan['source_sha256'])
    for model in plan['models'].values():
        if io.digest(ROOT/model['path']) != model['sha256']:
            raise ValueError('Changed factorial policy')
    return plan


def schedule(case):
    if len(case['variants']) != 5 or set(case['variants']) != set(VARIANTS):
        raise ValueError('All five fixed cells required exactly once')
    return [[case['cluster_id'],v] for v in case['variants']]


def metadata(case, sha):
    return dict(plan_sha256=sha, block_index=case['block_index'], cluster_id=case['cluster_id'],
                parent_case_id=case['parent_case_id'], scope='complete-candidate-gate-factorial')


def collect(case, directory, model, sha):
    expected = schedule(case)
    directory.mkdir(exist_ok=False)
    meta = metadata(case,sha)
    store = EpisodeStore(directory/'observations.sqlite',meta)
    reviews = []
    try:
        with runtime.MatchedShiftedMergeEnv(plant=case['plant']) as env:
            for variant in case['variants']:
                record = runtime.execute_episode(case,variant,model,env)
                store.append([case['cluster_id'],variant],
                    {k:v for k,v in record.items() if k != 'trace'},record['trace'])
                reviews.append(dict(variant=variant,**runtime.audit_episode(case,record,model)))
        store.seal(expected)
    finally:
        store.close()
    receipt = dict(**meta,actual_sumo_runs=5,reviews=reviews,
        database_sha256=io.digest(directory/'observations.sqlite'),
        seal_sha256=io.digest(directory/'observations.sqlite.seal.json'))
    io.exclusive_json(directory/'complete.json',receipt)
    return receipt


def block_hashes(directory,index):
    return dict(block_index=index,files={name:io.digest(directory/name) for name in BLOCK_FILES})


def worker():
    isolation.require_quiet(io.native_snapshot())
    execution = OUT/'execution'
    native = execution/'worker.native.json'
    io.exclusive_json(native,dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=io.utc()))
    ack_path = execution/'worker.acknowledged.json'
    deadline = time.monotonic()+180
    while not ack_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('No parent acknowledgment; no simulations launched')
        time.sleep(.2)
    ack = io.read(ack_path)
    if (ack['native_pid'] != os.getpid() or ack['parent_pid'] != os.getppid()
            or ack['native_start_sha256'] != io.digest(native)):
        raise ValueError('Worker acknowledgment differs')
    plan, sha = inputs(), io.digest(PLAN)
    torch.set_num_threads(1)
    models = {m:load_coupled(ROOT/v['path'])[0] for m,v in plan['models'].items()}
    blocks = OUT/'blocks'
    blocks.mkdir(exist_ok=False)
    entries = []
    for case in plan['cases']:
        verify_hashes(plan['source_sha256'])
        if io.digest(PLAN) != sha:
            raise ValueError('Plan changed during collection')
        for item in plan['models'].values():
            if io.digest(ROOT/item['path']) != item['sha256']:
                raise ValueError('Model changed during collection')
        directory = blocks/f"{case['block_index']:05d}"
        collect(case,directory,models[case['policy_id']],sha)
        entries.append(block_hashes(directory,case['block_index']))
        progress = dict(native_pid=os.getpid(),completed_blocks=len(entries),expected_blocks=640,
                        completed_runs=5*len(entries),expected_runs=3200,utc=io.utc())
        io.atomic_bytes(execution/'progress.json',json.dumps(progress).encode())
        print(json.dumps(progress),flush=True)
    inputs()
    if io.digest(PLAN) != sha:
        raise ValueError('Plan changed in final collection block')
    io.exclusive_json(execution/'blocks_manifest.json',dict(plan_sha256=sha,entries=entries,actual_sumo_runs=3200))
    io.exclusive_json(execution/'collection_complete.json',dict(plan_sha256=sha,native_pid=os.getpid(),
        actual_sumo_runs=3200,blocks=640,blocks_manifest_sha256=io.digest(execution/'blocks_manifest.json'),utc=io.utc()))


def watch():
    isolation.require_quiet(io.native_snapshot())
    inputs()
    sha = io.digest(PLAN)
    execution = OUT/'execution'
    execution.mkdir(exist_ok=False)
    with (execution/'stdout.log').open('x') as stdout,(execution/'stderr.log').open('x') as stderr:
        command = [sys.executable,'-X','utf8',SCRIPT,'worker']
        process = subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        io.exclusive_json(execution/'spawn.json',dict(watcher_pid=os.getpid(),launcher_pid=process.pid,
                                                    command=command,plan_sha256=sha,utc=io.utc()))
        try:
            identity = io.acknowledge_child(process,execution,'worker')
            if SCRIPT not in identity['command_line'] or 'worker' not in identity['command_line']:
                raise ValueError('Unexpected factorial worker command')
            io.exclusive_json(execution/'launch.json',identity)
        except Exception as error:
            io.exclusive_json(execution/'observation_error.json',dict(type=type(error).__name__,message=str(error)))
            code = process.wait()
            io.exclusive_json(execution/'exit.json',dict(exit_code=code,launcher_pid=process.pid,
                                                        acknowledgment_verified=False,utc=io.utc()))
            raise
        code = process.wait()
        io.exclusive_json(execution/'exit.json',dict(**identity,exit_code=code,plan_sha256=sha,
            hashes={name:io.digest(execution/name) for name in TERMINAL_FILES if (execution/name).exists()},utc=io.utc()))
    if code:
        raise SystemExit(code)


def terminal(plan_sha):
    execution = OUT/'execution'
    receipt = io.read(execution/'exit.json')
    done, manifest = io.read(execution/'collection_complete.json'),io.read(execution/'blocks_manifest.json')
    if (receipt['exit_code'] != 0 or receipt.get('plan_sha256') != plan_sha
            or done['plan_sha256'] != plan_sha or manifest['plan_sha256'] != plan_sha
            or done['native_pid'] != receipt['native_pid'] or done['actual_sumo_runs'] != 3200
            or done['blocks'] != 640 or manifest['actual_sumo_runs'] != 3200
            or [e['block_index'] for e in manifest['entries']] != list(range(640))
            or set(receipt['hashes']) != set(TERMINAL_FILES)):
        raise ValueError('Complete zero-exit factorial identity required')
    for name,sha in receipt['hashes'].items():
        if io.digest(execution/name) != sha:
            raise ValueError('Terminal receipt changed: '+name)
    if done['blocks_manifest_sha256'] != io.digest(execution/'blocks_manifest.json'):
        raise ValueError('Complete block inventory changed')
    io.check_acknowledgment(execution,'worker',receipt['launcher_pid'],receipt['native_pid'])
    spawn = io.read(execution/'spawn.json')
    if spawn['plan_sha256'] != plan_sha or spawn['launcher_pid'] != receipt['launcher_pid']:
        raise ValueError('Spawn and exit identities differ')
    if io.read(execution/'launch.json') != {k:receipt[k] for k in io.read(execution/'launch.json')}:
        raise ValueError('Launch and exit identities differ')
    for item in io.native_snapshot():
        if item['ProcessId'] == receipt['native_pid'] and item['CreationDate'] == receipt['native_creation_date']:
            raise ValueError('Factorial native worker remains live')
    for entry in manifest['entries']:
        if block_hashes(OUT/'blocks'/f"{entry['block_index']:05d}",entry['block_index']) != entry:
            raise ValueError('Changed completed factorial block')
    return manifest


def analysis(cluster_ids,records):
    raw = paired_mechanism_summary(cluster_ids,records)
    failed = {v:False for v in VARIANTS}
    for r in records:
        failed[r['variant']] |= episode_outcome(r['row'])['failed']
    methods = {v:dict(row) for v,row in raw['method_means'].items()}
    for name,row in methods.items():
        if failed['baseline'] or failed[name]:
            row['gain_percentage_points'] = None
    dependencies = {
        'bank_minus_nominal_with_raw_candidates':('raw_bank','raw_nominal'),
        'bank_minus_nominal_with_projected_candidates':('projected_bank','projected_nominal'),
        'projection_minus_raw_with_nominal_gate':('projected_nominal','raw_nominal'),
        'projection_minus_raw_with_bank_gate':('projected_bank','raw_bank'),
        'interaction':VARIANTS[1:]}
    effects = {k:dict(v) for k,v in raw['paired_contrasts'].items()}
    for name,variants in dependencies.items():
        if failed['baseline'] or any(failed[v] for v in variants):
            effects[name]['gain_percentage_points'] = None
    return dict(raw_capped_cost_description=raw,all_scope_method_means=methods,
                all_scope_contrasts=effects,failed_pairs_removed=0,new_risk_certificate=False,
                caveat='Raw capped failed costs are retained diagnostics, not valid performance gains')


def summarize():
    isolation.require_quiet(io.native_snapshot())
    plan, sha = inputs(), io.digest(PLAN)
    manifest = terminal(sha)
    before = {n:io.digest(OUT/'execution'/n) for n in (*TERMINAL_FILES,'exit.json')}
    torch.set_num_threads(1)
    models = {m:load_coupled(ROOT/v['path'])[0] for m,v in plan['models'].items()}
    records, reviews = [], []
    for case in plan['cases']:
        directory = OUT/'blocks'/f"{case['block_index']:05d}"
        complete = io.read(directory/'complete.json')
        meta = metadata(case,sha)
        if (any(complete[k] != v for k,v in meta.items()) or complete['actual_sumo_runs'] != 5
                or len(complete['reviews']) != 5
                or complete['database_sha256'] != io.digest(directory/'observations.sqlite')
                or complete['seal_sha256'] != io.digest(directory/'observations.sqlite.seal.json')):
            raise ValueError('Completed case metadata changed')
        for index,(key,body,trace) in enumerate(observations(directory/'observations.sqlite',meta,schedule(case))):
            if key != [case['cluster_id'],body['variant']]:
                raise ValueError('Stored factorial variant differs')
            record = dict(**body,trace=trace)
            review = dict(variant=body['variant'],**runtime.audit_episode(case,record,models[case['policy_id']]))
            if review != complete['reviews'][index]:
                raise ValueError('Independent replay differs from collection audit')
            records.append(body)
            reviews.append(dict(cluster_id=case['cluster_id'],**review))
    if len(records) != 3200:
        raise ValueError('Incomplete replay')
    inputs()
    if (io.digest(PLAN) != sha or manifest != terminal(sha)
            or any(io.digest(OUT/'execution'/n) != h for n,h in before.items())):
        raise ValueError('Evidence changed during full replay')
    result = analysis([c['cluster_id'] for c in plan['cases']],records)
    result.update(plan_sha256=sha,actual_sumo_runs=3200,independent_clusters=640,
                  additional_independent_clusters_beyond_parent=0)
    target = OUT/'analysis'
    target.mkdir(exist_ok=False)
    io.exclusive_json(target/'report.json',result)
    io.exclusive_json(target/'replay.json',dict(reviews=reviews))
    io.exclusive_json(target/'verification.json',dict(plan_sha256=sha,terminal_hashes=before,
        report_sha256=io.digest(target/'report.json'),replay_sha256=io.digest(target/'replay.json'),actual_sumo_runs=3200))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','watch','worker','summarize'))
    globals()[parser.parse_args().command]()
