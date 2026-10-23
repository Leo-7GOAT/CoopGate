"""Complete the original paired scope without replacing any interrupted record."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import audit_buffered_scope_v25 as original
import recover_prospective_v24 as prefix
import snapshot_interruption_v29 as snapshot
from coopgate.episode_store_v21 import EpisodeStore, encode, observations, unpack

ROOT = original.ROOT
PRIOR = snapshot.BUFFER
OUT = ROOT/'artifacts/v29/buffered_scope_recovery_20261023'
RECOVERY = OUT/'recovery'
SCRIPT = 'tools/recover_buffered_v29.py'
SOURCES = (SCRIPT, 'tests/test_buffered_recovery_v29.py', 'tools/snapshot_interruption_v29.py',
           'tools/recover_prospective_v24.py')
BASE_INPUTS = original.inputs


def freeze():
    snapshot.assert_stopped()
    if OUT.exists():
        raise FileExistsError('An explicit separate protocol is required after another interruption')
    evidence = snapshot.verify_snapshot('buffered')
    plan = BASE_INPUTS()
    original.sources_committed([ROOT/name for name in SOURCES])
    _, _, entries = snapshot.checked_blocks('buffered')
    if entries != evidence['blocks']:
        raise ValueError('Interrupted paired prefixes changed')
    RECOVERY.mkdir(parents=True)
    shutil.copy2(PRIOR/'plan.json', OUT/'plan.json')
    blocks = [{k:v for k,v in entry.items() if k != 'inventory'} for entry in entries]
    cross = [plan['groups'][entry['initial_index']]['cases'][entry['records']//2]['case']
             for entry in entries if not entry['complete'] and entry['records']%2]
    recovery = dict(utc=original.utc(), source_plan_sha256=original.digest(PRIOR/'plan.json'),
        inventory_sha256=original.digest(snapshot.OUT/'buffered/inventory.json'),
        source_sha256={name:original.digest(ROOT/name) for name in SOURCES}, blocks=blocks,
        retained_roles=evidence['retained_roles'], missing_roles=plan['actual_sumo_runs']-evidence['retained_roles'],
        boundary_replays=evidence['partial_blocks'], cross_session_pair_cases=cross,
        original_exit_status='unknown', schedule_candidates_models_thresholds_unchanged=True,
        new_independent_initials=0, partial_effectiveness_outcomes_read=False,
        timing_scope='All original per-call times retained exactly. Split implementation pairs are identified; aggregate timing mixes execution epochs and is not a homogeneous-load or hard-real-time result.',
        continuation='Replay every last committed partial role before appending missing roles; recheck every complete/partial/new record and pair against the old reviewed archive; preserve original aborts.',
        repeated_interruption='No automatic restart, seed replacement or overwriting.')
    original.exclusive_json(RECOVERY/'plan.json', recovery)
    original.exclusive_json(RECOVERY/'sha256.json', {'plan.json':original.digest(RECOVERY/'plan.json')})
    print(json.dumps({k:recovery[k] for k in ('retained_roles','missing_roles','boundary_replays','cross_session_pair_cases')}), flush=True)


def inputs():
    plan = BASE_INPUTS()
    original.committed(RECOVERY/'plan.json')
    recovery = original.read(RECOVERY/'plan.json')
    evidence = snapshot.verify_snapshot('buffered')
    if (original.digest(OUT/'plan.json') != recovery['source_plan_sha256']
            or recovery['source_plan_sha256'] != evidence['source_plan_sha256']
            or recovery['inventory_sha256'] != original.digest(snapshot.OUT/'buffered/inventory.json')
            or recovery['retained_roles'] != evidence['retained_roles']
            or recovery['boundary_replays'] != evidence['partial_blocks']
            or recovery['retained_roles']+recovery['missing_roles'] != plan['actual_sumo_runs']):
        raise ValueError('Original complete scope or preserved boundary changed')
    for name, expected in recovery['source_sha256'].items():
        if original.digest(ROOT/name) != expected:
            raise ValueError('Frozen recovery source changed: '+name)
    original.sources_committed([ROOT/name for name in recovery['source_sha256']])
    return plan


def execute_record(group, entry, implementation, plan, models, envs):
    case = tuple(entry['case'])
    wrapped = []
    cls = original.PhaseRepairGate if implementation == 'reference' else original.BufferedPhaseGate
    model = models[case[-1].rsplit('_', 1)[1]]
    def factory(_):
        gate = original.TimedGate(cls(model, .05))
        wrapped.append(gate)
        return gate
    env = envs['phase'] if group['study'] == 'phase' else envs[case[4]]
    start = time.perf_counter()
    row, trace = original.original.execute(group['study'], case,
        plan['protocols'][group['study']]['spec'], factory, env)
    return dict(row=row, calls=wrapped[0].calls, episode_wall_seconds=time.perf_counter()-start), trace


def check_record(group, entry, record, trace, plan, old_rows):
    if original.digest(ROOT/entry['trace_path']) != entry['trace_sha256']:
        raise ValueError('Independently reviewed old trace changed')
    original.check_calls(record['row'], trace, record['calls'])
    original.original.compare_replay(old_rows[group['study']][tuple(entry['case'])], record['row'],
                                     original.read(ROOT/entry['trace_path']), trace)
    original.original.verify_execution(group['study'], record['row'], trace, plan['protocols'][group['study']]['spec'])


def check_pair(case, current):
    a, b = current['reference'], current['buffered']
    original.original.compare_replay(a[0]['row'], b[0]['row'], a[1], b[1])
    np.testing.assert_equal([c['error'] for c in a[0]['calls']], [c['error'] for c in b[0]['calls']])
    return dict(case=list(case), decisions=len(a[1]), select_calls=len(a[0]['calls']),
        aborted=a[0]['row']['aborted'], completed=a[0]['row']['completed'],
        all_non_timing_fields_exact=True, both_physical_reconstructions_pass=True)


def replay_boundary(entry, plan, models, envs, old_rows):
    group = plan['groups'][entry['initial_index']]
    source = PRIOR/'blocks'/f"{entry['initial_index']:05d}"/'observations.sqlite'
    _, raw = prefix.prefix_records(source, original.metadata(group), original.schedule(group), entry['database_sha256'])
    _, key, rb, tb, rh, th = raw[-1]
    case = tuple(json.loads(key))
    record, trace = unpack(rb, rh), unpack(tb, th)
    item = next(item for item in group['cases'] if tuple(item['case']) == case[:-1])
    new, steps = execute_record(group, item, case[-1], plan, models, envs)
    directory = OUT/'boundary'/f"{entry['initial_index']:05d}"
    directory.mkdir(parents=True, exist_ok=False)
    original.exclusive_json(directory/'observation.json', dict(case=list(case), record=new, trace=steps))
    check_record(group, item, new, steps, plan, old_rows)
    original.original.compare_replay(record['row'], new['row'], trace, steps)
    np.testing.assert_equal([c['error'] for c in record['calls']], [c['error'] for c in new['calls']])
    original.exclusive_json(directory/'verified.json', dict(case=list(case), exact_non_timing_match=True,
        source_database_sha256=entry['database_sha256'], original_times_preserved=True,
        actual_sumo_runs=1, new_independent_initials=0))
    original.exclusive_json(directory/'sha256.json', original.original.manifest(directory))


def complete_group(group, entry, plan, models, envs, old_rows):
    index = group['block_index']
    directory = OUT/'blocks'/f'{index:05d}'
    source = PRIOR/'blocks'/f'{index:05d}'
    expected = original.schedule(group)
    meta = original.metadata(group)
    prior_records, store = {}, None
    if entry is not None and entry['complete']:
        shutil.copytree(source, directory)
        if snapshot.files_at(source) != snapshot.files_at(directory):
            raise ValueError('A completed original paired block changed during copy')
        prior_records = {key:(row, trace) for key,row,trace in observations(directory/'observations.sqlite', meta, expected)}
    else:
        directory.mkdir(parents=True, exist_ok=False)
        if entry is None:
            store = EpisodeStore(directory/'observations.sqlite', meta)
        else:
            store, raw = prefix.copy_prefix(source/'observations.sqlite', directory/'observations.sqlite', meta,
                                            expected, entry['database_sha256'])
            prior_records = {tuple(json.loads(key)):(unpack(rb,rh),unpack(tb,th)) for _,key,rb,tb,rh,th in raw}
    reports, added = [], 0
    try:
        for i, item in enumerate(group['cases']):
            case = tuple(item['case'])
            current = {}
            order = original.IMPLEMENTATIONS if i%2 == 0 else original.IMPLEMENTATIONS[::-1]
            for implementation in order:
                key = (*case, implementation)
                if key in prior_records:
                    record, trace = prior_records[key]
                else:
                    if store is None:
                        raise ValueError('A sealed block cannot lose a role')
                    record, trace = execute_record(group, item, implementation, plan, models, envs)
                    store.append(key, record, trace)
                    added += 1
                check_record(group, item, record, trace, plan, old_rows)
                current[implementation] = (record, trace)
            reports.append(check_pair(case, current))
        if store is not None:
            store.seal(expected)
            original.exclusive_json(directory/'complete.json', dict(**meta, actual_sumo_runs=len(expected),
                paired_cases=reports, database_sha256=original.digest(directory/'observations.sqlite'),
                seal_sha256=original.digest(directory/'observations.sqlite.seal.json'),
                retained_sumo_roles=len(prior_records), new_sumo_roles=added,
                recovery_plan_sha256=original.digest(RECOVERY/'plan.json')))
        else:
            if original.read(directory/'complete.json')['paired_cases'] != reports:
                raise ValueError('Fresh complete-block review differs from preserved review')
        if entry is not None and not entry['complete']:
            before = prefix.inventory(source/'observations.sqlite')
            after = prefix.inventory(directory/'observations.sqlite')
            for name in ('cases','row_sha256','trace_sha256'):
                if after[name][:entry['records']] != before[name]:
                    raise ValueError('A retained partial record changed')
        return dict(block_index=index, retained_roles=len(prior_records), new_roles=added,
                    reviewed_roles=len(expected), paired_cases=len(reports), all_original_and_pair_checks_pass=True)
    finally:
        if store is not None:
            store.close()


def worker():
    execution = OUT/'execution'
    startup = execution/'worker.native.json'
    original.atomic_bytes(startup, encode(dict(native_pid=os.getpid(), parent_pid=os.getppid(), utc=original.utc())))
    acknowledgment = execution/'worker.acknowledged.json'
    deadline = time.monotonic()+180
    while not acknowledgment.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('No native acknowledgment; no simulation started')
        time.sleep(.2)
    proof = original.read(acknowledgment)
    if (proof['native_pid'] != os.getpid() or proof['parent_pid'] != os.getppid()
            or proof['native_start_sha256'] != original.digest(startup)):
        raise ValueError('Native acknowledgment changed')
    plan = inputs()
    recovery = original.read(RECOVERY/'plan.json')
    entries = {entry['initial_index']:entry for entry in recovery['blocks']}
    torch.set_num_threads(1)
    models = {s:original.load_coupled(ROOT/m['path'])[0] for s,m in plan['models'].items()}
    old_rows = {study:{original.original.identity(study,r):r for r in original.read(original.OLD/'results'/study/'episodes.json')}
                for study in plan['protocols']}
    warm = np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for model in models.values():
        original.PhaseRepairGate(model,.05).select(warm,np.zeros(3))
        original.BufferedPhaseGate(model,.05).select(warm,np.zeros(3))
    reviews, retained, new = [], 0, 0
    with ExitStack() as stack:
        envs = {'phase':stack.enter_context(original.DisturbedMergeEnv())}
        envs.update({p:stack.enter_context(original.MatchedShiftedMergeEnv(plant=p))
                     for p in plan['protocols']['transfer']['spec']['plants']})
        for entry in recovery['blocks']:
            if not entry['complete']:
                replay_boundary(entry, plan, models, envs, old_rows)
        (OUT/'blocks').mkdir(exist_ok=False)
        for group in plan['groups']:
            result = complete_group(group, entries.get(group['block_index']), plan, models, envs, old_rows)
            reviews.append(result)
            retained += result['retained_roles']
            new += result['new_roles']
            progress = dict(utc=original.utc(), native_pid=os.getpid(), completed_blocks=len(reviews),
                expected_blocks=plan['blocks'], completed_runs=retained+new, expected_runs=plan['actual_sumo_runs'],
                retained_roles=retained, new_roles=new, boundary_replays=recovery['boundary_replays'])
            original.atomic_bytes(execution/'progress.json', encode(progress))
            if len(reviews)%20 == 0 or result['new_roles']:
                print(json.dumps(progress), flush=True)
    if (retained != recovery['retained_roles'] or new != recovery['missing_roles']
            or retained+new != plan['actual_sumo_runs'] or len(reviews) != plan['blocks']):
        raise ValueError('The complete original paired scope was not preserved')
    original.exclusive_json(execution/'all_blocks_reviewed.json', dict(blocks=reviews,
        reviewed_roles=retained+new, paired_cases=sum(r['paired_cases'] for r in reviews)))
    original.exclusive_json(execution/'collection_complete.json', dict(plan_sha256=original.digest(OUT/'plan.json'),
        native_pid=os.getpid(), actual_sumo_runs=retained+new, blocks=len(reviews), utc=original.utc(),
        retained_roles=retained, new_roles=new, boundary_replays=recovery['boundary_replays'],
        recovery_plan_sha256=original.digest(RECOVERY/'plan.json'),
        fresh_review_sha256=original.digest(execution/'all_blocks_reviewed.json')))


def summarize():
    plan = inputs()
    recovery = original.read(RECOVERY/'plan.json')
    done = original.read(OUT/'execution/collection_complete.json')
    reviews = original.read(OUT/'execution/all_blocks_reviewed.json')
    if (done['retained_roles'] != recovery['retained_roles'] or done['new_roles'] != recovery['missing_roles']
            or done['recovery_plan_sha256'] != original.digest(RECOVERY/'plan.json')
            or done['fresh_review_sha256'] != original.digest(OUT/'execution/all_blocks_reviewed.json')
            or reviews['reviewed_roles'] != plan['actual_sumo_runs']
            or reviews['paired_cases'] != plan['paired_cases']
            or len(reviews['blocks']) != plan['blocks']
            or not all(r['all_original_and_pair_checks_pass'] for r in reviews['blocks'])):
        raise ValueError('Every old/new role requires a fresh full review')
    for entry in recovery['blocks']:
        if not entry['complete']:
            directory = OUT/'boundary'/f"{entry['initial_index']:05d}"
            prefix.original.verify_manifest(directory)
            check = original.read(directory/'verified.json')
            if not check['exact_non_timing_match'] or check['source_database_sha256'] != entry['database_sha256']:
                raise ValueError('Actual partial-boundary replay required')
    original.summarize()
    original.exclusive_json(OUT/'recovery_accounting.json', dict(
        source_report_sha256=original.digest(OUT/'analysis/report.json'),
        recovery_plan_sha256=original.digest(RECOVERY/'plan.json'),
        retained_roles=recovery['retained_roles'], new_roles=recovery['missing_roles'],
        boundary_replays=recovery['boundary_replays'], cross_session_pair_cases=recovery['cross_session_pair_cases'],
        timing_scope=recovery['timing_scope'], original_exit_status='unknown', new_independent_initials=0))


def configure():
    original.OUT, original.SCRIPT, original.inputs = OUT, SCRIPT, inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','worker','watch','summarize'))
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        configure()
        if args.action == 'watch':
            original.watch()
        else:
            globals()[args.action]()
