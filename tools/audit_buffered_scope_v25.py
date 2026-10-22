"""Full paired SUMO scope audit on all completed phase05 development cases.

The complete old grid is fixed before execution. Every failure is retained.
This is implementation equivalence and timing, not new statistical calibration.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from coopgate.buffered_phase_v25 import BufferedPhaseGate
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.phase_repair_v21 import PhaseRepairGate
from execute_prospective_v23 import (acknowledge_child, check_acknowledgment,
    exclusive_json, native_snapshot, runtime_identity, sources_committed, utc)
from present_completed_v21 import load_complete
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes
import recover_interrupted_v21 as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v25/buffered_complete_scope'
OLD = ROOT / 'artifacts/v21/interrupted_recovery_20261022'
SCRIPT = 'tools/audit_buffered_scope_v25.py'
TEST = 'tests/test_buffered_scope_v25.py'
IMPLEMENTATIONS = ('reference', 'buffered')
MODELS = ('1802', '1812', '1822')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class TimedGate:
    """Keep the gate's evidence unchanged; include unsuccessful select calls."""
    def __init__(self, gate):
        self.gate = gate
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.gate, name)

    def select(self, *args, **kwargs):
        started = time.perf_counter()
        error = None
        try:
            return self.gate.select(*args, **kwargs)
        except BaseException as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            self.calls.append(dict(wall_ms=1000*(time.perf_counter()-started), error=error))


def check_calls(row, trace, calls):
    successes = [c for c in calls if c['error'] is None]
    failures = [c for c in calls if c['error'] is not None]
    if (len(successes) != len(trace) or len(failures) != int(row['aborted'])
            or (failures and calls[-1] is not failures[-1])
            or any(not np.isfinite(c['wall_ms']) or c['wall_ms'] < 0 for c in calls)):
        raise ValueError('Incomplete successful/aborted select timing')
    if failures and failures[0]['error']['message'] != row['abort_reason']:
        raise ValueError('Terminal select error differs from recorded abort')


def expected_grid(study, spec):
    expected = []
    for shard in range(spec['shards']):
        expected.extend(case for case in original.MODULES[study].schedule(spec, shard)
                        if case[-1] in {f'phase05_{seed}' for seed in MODELS})
    return sorted(expected)


def validate_scope(study, rows, spec):
    actual = [original.identity(study, row) for row in rows]
    if sorted(actual) != expected_grid(study, spec) or len(set(actual)) != len(actual):
        raise ValueError('The complete original phase05 grid, including failures, is required')


def ordered_groups(records):
    groups = defaultdict(list)
    for entry in records:
        groups[entry['study'], entry['case'][0]].append(entry)
    result = []
    for (study, initial), entries in groups.items():
        result.append(dict(study=study, initial_index=initial,
            local_index=entries[0]['local_index'], scenario=entries[0]['scenario'],
            cases=sorted(entries, key=lambda e: e['case'])))
    # Coverage first by original within-scenario index, never by outcome/timing.
    result.sort(key=lambda g: (g['local_index'], g['study'], g['scenario'], g['initial_index']))
    for index, group in enumerate(result):
        group['block_index'] = index
    return result


def schedule(group):
    entries = []
    for i, entry in enumerate(group['cases']):
        order = IMPLEMENTATIONS if i % 2 == 0 else IMPLEMENTATIONS[::-1]
        entries.extend((*entry['case'], name) for name in order)
    return entries


def freeze():
    if OUT.exists():
        raise FileExistsError('Preserve the original full-scope protocol')
    summaries, protocols, _ = load_complete()
    earlier = ROOT / 'artifacts/v25/buffered_phase_sumo_audit'
    paired = read(earlier / 'results/report.json')
    if paired['actual_sumo_runs'] != 24 or not paired['all_24_trajectories_match_reviewed_archive']:
        raise ValueError('Complete the exact paired integration first')
    inputs, records = {}, []
    for path in [earlier/'plan.json', earlier/'results/report.json',
                 OLD/'execution/supervisor_exit.json', OLD/'execution/completed.json',
                 OLD/'execution/watcher_launch.json', OLD/'execution/review_exits.json']:
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    for study, protocol in protocols.items():
        directory = OLD / 'results' / study
        rows = read(directory / 'episodes.json')
        if digest(directory/'episodes.json') != summaries[study]['episodes_sha256']:
            raise ValueError('Completed data changed')
        selected = [r for r in rows if r['controller'] in {f'phase05_{s}' for s in MODELS}]
        validate_scope(study, selected, protocol['spec'])
        provenance = {tuple(p['case']): p for p in read(directory/'provenance.json')}
        for path in [directory/n for n in ('episodes.json', 'provenance.json', 'summary.json', 'sha256.json')]:
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        path = original.STUDIES[study] / 'protocol.json'
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        for row in selected:
            case = original.identity(study, row)
            source = provenance[case]
            if digest(ROOT/source['path']) != source['sha256']:
                raise ValueError('Archived complete trace changed')
            inputs[source['path']] = source['sha256']
            records.append(dict(study=study, case=list(case), scenario=row['scenario'],
                local_index=row['initial_index'], trace_path=source['path'], trace_sha256=source['sha256']))
    groups = ordered_groups(records)
    sources = dict(read(earlier/'plan.json')['source_sha256'])
    for name in (SCRIPT, TEST, 'tools/present_completed_v21.py'):
        sources[name] = digest(ROOT/name)
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen implementation changed: ' + name)
    sources_committed([ROOT/name for name in sources])
    if protocols['phase']['models'] != protocols['transfer']['models']:
        raise ValueError('The same three fixed models are required')
    plan = dict(purpose='Complete implementation-equivalence and timing audit on every completed phase05 development and transfer case',
        groups=groups, protocols=protocols, models=protocols['phase']['models'],
        source_sha256=sources, inputs_sha256=inputs, runtime=runtime_identity(),
        paired_cases=len(records), actual_sumo_runs=2*len(records), blocks=len(groups),
        old_independent_initials=dict(phase=128, transfer=160), new_independent_initials=0,
        active_calibration_outcomes_used=False, workers=1,
        selection='All 1536 phase and 1920 matched-transfer phase05 cases, all 3 fixed models, with every old failure; no outcome filtering',
        order='Alternate implementation order within each block; group ordering uses only original indices and scenario labels',
        warmup='Both implementations and all three fixed models on the same declared state before any timed episode',
        checks='Store row and full trace before assertions; compare every non-timing field including work and failure against paired implementation and independently reviewed archive; independently reconstruct every physical trace',
        failed_call_timing='External identical wrapper records every select call including the final exception; original gate_ms fields remain unchanged',
        timing_scope='Paired concurrent desktop measurements; full select calls and successful reported calls shown separately; no population quantile or hard-real-time claim',
        physical_scope='Original bounded-error assumptions unchanged; strong-lag violations and original aborts must reproduce, not be hidden or relabeled as safety',
        interruption='Exclusive creation and immutable seals; preserve any partial block and actual exits, never automatically restart',
        effects='No change to policy, candidate set, paths, prediction horizon, thresholds, reward, live calibration or publication acceptance')
    if plan['paired_cases'] != 3456 or plan['blocks'] != 288:
        raise ValueError('Unexpected complete study size')
    OUT.mkdir()
    exclusive_json(OUT/'plan.json', plan)
    print(json.dumps(dict(paired_cases=3456, actual_sumo_runs=6912, blocks=288,
        plan_sha256=digest(OUT/'plan.json'))), flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    for name, expected in {**plan['source_sha256'], **plan['inputs_sha256']}.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen full-scope source/input changed: ' + name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    if runtime_identity() != plan['runtime']:
        raise ValueError('Frozen runtime changed')
    for model in plan['models'].values():
        if digest(ROOT/model['path']) != model['sha256']:
            raise ValueError('Frozen policy changed')
    return plan


def metadata(group):
    return dict(plan_sha256=digest(OUT/'plan.json'), block_index=group['block_index'],
                study=group['study'], initial_index=group['initial_index'])


def worker():
    execution = OUT/'execution'
    startup = execution/'worker.native.json'
    atomic_bytes(startup, encode(dict(native_pid=os.getpid(), parent_pid=os.getppid(), utc=utc())))
    deadline = time.monotonic()+180
    ack = execution/'worker.acknowledged.json'
    while not ack.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('No parent acknowledgment; no simulation started')
        time.sleep(.2)
    proof = read(ack)
    if (proof['native_pid'] != os.getpid() or proof['parent_pid'] != os.getppid()
            or proof['native_start_sha256'] != digest(startup)):
        raise ValueError('Native process acknowledgment mismatch')
    plan = inputs()
    torch.set_num_threads(1)
    models = {s: load_coupled(ROOT/m['path'])[0] for s,m in plan['models'].items()}
    old_rows = {study: {original.identity(study,r):r for r in read(OLD/'results'/study/'episodes.json')}
                for study in plan['protocols']}
    warm = np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for model in models.values():
        PhaseRepairGate(model,.05).select(warm,np.zeros(3))
        BufferedPhaseGate(model,.05).select(warm,np.zeros(3))
    root = OUT/'blocks'
    root.mkdir(exist_ok=False)
    completed_runs = 0
    with ExitStack() as stack:
        additive = stack.enter_context(DisturbedMergeEnv())
        shifted = {p:stack.enter_context(MatchedShiftedMergeEnv(plant=p))
                   for p in plan['protocols']['transfer']['spec']['plants']}
        for group in plan['groups']:
            directory = root/f"{group['block_index']:05d}"
            directory.mkdir()
            store = EpisodeStore(directory/'observations.sqlite',metadata(group))
            study = group['study']
            spec = plan['protocols'][study]['spec']
            pair_reports = []
            try:
                for pair_index, entry in enumerate(group['cases']):
                    case = tuple(entry['case'])
                    old_row = old_rows[study][case]
                    if digest(ROOT/entry['trace_path']) != entry['trace_sha256']:
                        raise ValueError('Archived case changed during execution')
                    old_trace = read(ROOT/entry['trace_path'])
                    current = {}
                    order = IMPLEMENTATIONS if pair_index%2 == 0 else IMPLEMENTATIONS[::-1]
                    for implementation in order:
                        wrapped = []
                        cls = PhaseRepairGate if implementation=='reference' else BufferedPhaseGate
                        model = models[case[-1].rsplit('_',1)[1]]
                        def factory(state):
                            gate = TimedGate(cls(model,.05))
                            wrapped.append(gate)
                            return gate
                        env = additive if study=='phase' else shifted[case[4]]
                        start = time.perf_counter()
                        row, trace = original.execute(study,case,spec,factory,env)
                        duration = time.perf_counter()-start
                        record = dict(row=row, calls=wrapped[0].calls, episode_wall_seconds=duration)
                        store.append((*case,implementation), record, trace)
                        check_calls(row,trace,record['calls'])
                        original.compare_replay(old_row,row,old_trace,trace)
                        original.verify_execution(study,row,trace,spec)
                        current[implementation] = (record,trace)
                        completed_runs += 1
                    a,b = current['reference'],current['buffered']
                    original.compare_replay(a[0]['row'],b[0]['row'],a[1],b[1])
                    np.testing.assert_equal([c['error'] for c in a[0]['calls']], [c['error'] for c in b[0]['calls']])
                    pair_reports.append(dict(case=list(case), decisions=len(a[1]),
                        select_calls=len(a[0]['calls']), aborted=old_row['aborted'], completed=old_row['completed'],
                        all_non_timing_fields_exact=True, both_physical_reconstructions_pass=True))
                store.seal(schedule(group))
            finally:
                store.close()
            exclusive_json(directory/'complete.json',dict(**metadata(group), actual_sumo_runs=2*len(pair_reports),
                paired_cases=pair_reports, database_sha256=digest(directory/'observations.sqlite'),
                seal_sha256=digest(directory/'observations.sqlite.seal.json')))
            atomic_bytes(execution/'progress.json',encode(dict(utc=utc(),native_pid=os.getpid(),
                completed_blocks=group['block_index']+1,expected_blocks=plan['blocks'],
                completed_runs=completed_runs,expected_runs=plan['actual_sumo_runs'])))
            print(json.dumps(dict(block=group['block_index'],study=study,scenario=group['scenario'],
                completed_runs=completed_runs,total_runs=plan['actual_sumo_runs'])),flush=True)
    if completed_runs != plan['actual_sumo_runs']:
        raise ValueError('Full original scope not completed')
    exclusive_json(execution/'collection_complete.json',dict(plan_sha256=digest(OUT/'plan.json'),
        native_pid=os.getpid(),actual_sumo_runs=completed_runs,blocks=plan['blocks'],utc=utc()))


def watch():
    inputs()
    execution = OUT/'execution'
    execution.mkdir(exist_ok=False)
    with (execution/'worker.stdout.log').open('x') as stdout, (execution/'worker.stderr.log').open('x') as stderr:
        command = [sys.executable,'-X','utf8',SCRIPT,'worker']
        process = subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        exclusive_json(execution/'spawn.json',dict(watcher_pid=os.getpid(),launcher_pid=process.pid,
            command=command,plan_sha256=digest(OUT/'plan.json'),utc=utc()))
        try:
            native = acknowledge_child(process,execution,'worker')
            exclusive_json(execution/'launch.json',dict(**native,watcher_pid=os.getpid(),
                plan_sha256=digest(OUT/'plan.json'),utc=utc()))
        except Exception as error:
            exclusive_json(execution/'observation_error.json',dict(type=type(error).__name__,message=str(error),
                launcher_pid=process.pid,utc=utc()))
            # An observation failure does not restart or terminate a live handle.
            code = process.wait()
            exclusive_json(execution/'exit.json',dict(launcher_pid=process.pid,exit_code=code,
                plan_sha256=digest(OUT/'plan.json'),utc=utc()))
            raise
        code = process.wait()
        exclusive_json(execution/'exit.json',dict(**native,exit_code=code,
            plan_sha256=digest(OUT/'plan.json'),utc=utc()))
    if code:
        raise SystemExit(code)


def validate_completion(plan, plan_sha, receipt, launch, done):
    if (receipt['exit_code'] != 0 or receipt['launcher_pid'] != launch['launcher_pid']
            or receipt['native_pid'] != launch['native_pid']
            or receipt['plan_sha256'] != plan_sha or launch['plan_sha256'] != plan_sha
            or done['plan_sha256'] != plan_sha or done['native_pid'] != launch['native_pid']
            or done['actual_sumo_runs'] != plan['actual_sumo_runs'] or done['blocks'] != plan['blocks']):
        raise ValueError('Successful native exit and complete frozen scope required before reporting')


def summarize():
    plan = inputs()
    execution = OUT/'execution'
    receipt, launch = read(execution/'exit.json'), read(execution/'launch.json')
    done = read(execution/'collection_complete.json')
    validate_completion(plan,digest(OUT/'plan.json'),receipt,launch,done)
    check_acknowledgment(execution,'worker',launch['launcher_pid'],launch['native_pid'])
    cells = defaultdict(lambda:dict(reference=[],buffered=[],reported_reference=[],reported_buffered=[],
        episodes=0,aborted=0,adopted=0,over_error_bound=0))
    total_runs = 0
    for group in plan['groups']:
        directory = OUT/'blocks'/f"{group['block_index']:05d}"
        verified = read(directory/'complete.json')
        if (verified['database_sha256'] != digest(directory/'observations.sqlite')
                or verified['seal_sha256'] != digest(directory/'observations.sqlite.seal.json')
                or len(verified['paired_cases']) != len(group['cases'])
                or not all(p['all_non_timing_fields_exact'] and p['both_physical_reconstructions_pass'] for p in verified['paired_cases'])):
            raise ValueError('Complete verified paired block required')
        for case,record,trace in observations(directory/'observations.sqlite',metadata(group),schedule(group)):
            row = record['row']
            check_calls(row,trace,record['calls'])
            key = (group['study'],row['scenario'],row.get('plant_name','additive'),row['disturbance'],row['controller'])
            cell = cells[key]
            implementation = case[-1]
            cell[implementation].extend(c['wall_ms'] for c in record['calls'])
            cell['reported_'+implementation].extend(row['controller_ms'])
            if implementation=='reference':
                cell['episodes'] += 1
                cell['aborted'] += int(row['aborted'])
                cell['adopted'] += int(row['learned_action_steps'] > 0)
                cell['over_error_bound'] += int(row.get('error_contract_exceeded_substeps',0)>0)
            total_runs += 1
    if total_runs != plan['actual_sumo_runs']:
        raise ValueError('Incomplete full-grid observations')
    rows = []
    for key, cell in sorted(cells.items()):
        row = dict(zip(('study','scenario','plant','mode','controller'),key))
        row.update({k:cell[k] for k in ('episodes','aborted','adopted','over_error_bound')})
        for name in ('reference','buffered','reported_reference','reported_buffered'):
            values = cell[name]
            row[name] = dict(calls=len(values),sum_ms=sum(values),
                p50_p95_p99_max_ms=np.percentile(values,[50,95,99,100]).tolist() if values else None,
                over_500ms=sum(t>500 for t in values))
        row['all_calls_reduction_percent'] = 100*(1-sum(cell['buffered'])/sum(cell['reference']))
        rows.append(row)
    target = OUT/'analysis'
    target.mkdir(exist_ok=False)
    exclusive_json(target/'report.json',dict(plan_sha256=digest(OUT/'plan.json'),exit_sha256=digest(execution/'exit.json'),
        actual_sumo_runs=total_runs,paired_cases=plan['paired_cases'],cells=rows,
        new_independent_initials=0,active_calibration_outcomes_used=False,
        timing_scope=plan['timing_scope'],physical_scope=plan['physical_scope'],
        new_risk_certificate=False,hard_real_time_certified=False))
    exclusive_json(target/'sha256.json',{'report.json':digest(target/'report.json')})
    print(json.dumps(dict(completed=True,actual_sumo_runs=total_runs,cells=len(rows))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','worker','watch','summarize'))
    globals()[parser.parse_args().action]()
