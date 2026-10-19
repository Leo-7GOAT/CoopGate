"""Exhaustive archived-admission screen; no controller or outcome metric edits."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.arrival_phase_v20 import arrival_phase_suffix
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.disturbed_advantage_v19 import disturbed_action_difference
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v20/temporal_repair_extension'
PLAN = ROOT / 'artifacts/v20/arrival_phase_screen_plan'
OUTPUT = ROOT / 'artifacts/v20/arrival_phase_screen'


def occurrences(protocol):
    rows = json.loads((STUDY / 'episodes.json').read_text())
    result, trace_hashes = [], {}
    for row in rows:
        path = STUDY / f"shard_{row['initial_index']%protocol['spec']['shards']}/trace_{row['initial_index']}_{row['disturbance']}_{row['controller']}.json"
        trace_hashes[path.relative_to(STUDY).as_posix()] = digest(path)
        trace = json.loads(path.read_text())
        for step, item in enumerate(trace):
            if item['portfolio_joint_action_admitted']:
                result.append(dict(initial_index=row['initial_index'], seed=row['seed'], disturbance=row['disturbance'],
                    controller=row['controller'], step=step, state=trace[step-1]['x'] if step else row['initial'],
                    previous=trace[step-1]['command'] if step else [0.,0.,0.], command=item['command'],
                    original_bank_differences=item['selected_bank_differences']))
    return result, trace_hashes


def identity(item):
    return {k:item[k] for k in ('initial_index','disturbance','controller','step')}


def prepare():
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve the existing phase-screen plan')
    verify_manifest(STUDY)
    _, protocol = read_protocol(STUDY)
    records, traces = occurrences(protocol)
    summary = json.loads((STUDY / 'summary.json').read_text())
    if len(records) != summary['independent_review_counts']['admitted_decisions_verified']:
        raise ValueError('Not every archived admission is covered')
    sources = dict(protocol['source_sha256'])
    for name in ('src/coopgate/arrival_phase_v20.py','tests/test_arrival_phase_v20.py','tools/screen_arrival_phase_v20.py'):
        sources[name] = digest(ROOT / name)
    plan = dict(purpose='Test sub-period arrival ordering at every archived full-repair admission',
        occurrences=[identity(r) for r in records], source_sha256=sources, trace_sha256=traces,
        study_manifest_sha256=digest(STUDY / 'sha256.json'),
        selection='All admitted decisions in all 1536 repair runs, including every model/mode and all harmless outcomes',
        check='Original recorded 33-path cost/time differences must match; extra criterion is candidate interpolated arrival <= baseline arrival + 1e-10 for every active car and every bank path; incomplete arrivals reject.',
        physical_interpolation='Linear position interpolation within the existing 0.1 second semi-implicit update; decision period and reported outcome metrics stay 0.5 second.',
        diagnostic_truth='Actual realized future tape only labels a one-action-then-baseline branch after the bank-only screen.',
        new_sumo_runs=0, new_independent_initials=0, risk_certification=False, controller_changed=False,
        scope='Exhaustive local screen on archived states; rejection changes future states, so retained archived admissions are not new closed-loop adoption rates')
    PLAN.mkdir(parents=True)
    save(PLAN / 'plan.json', plan)
    for name in sources:
        target = PLAN / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    save(PLAN / 'sha256.json', {p.relative_to(PLAN).as_posix():digest(p) for p in PLAN.rglob('*')
                               if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(admitted_occurrences=len(records), plan_sha256=digest(PLAN / 'plan.json'))), flush=True)


def run():
    torch.set_num_threads(1)
    if OUTPUT.exists():
        raise FileExistsError('Preserve existing screen output')
    for directory in (PLAN,STUDY):
        verify_manifest(directory)
    plan = json.loads((PLAN / 'plan.json').read_text())
    require_committed([PLAN / 'plan.json',PLAN / 'sha256.json',*[ROOT/name for name in plan['source_sha256']]])
    for name, expected in plan['source_sha256'].items():
        if digest(ROOT/name)!=expected or digest(PLAN/'source'/name)!=expected:
            raise ValueError('Frozen screen source changed: '+name)
    if digest(STUDY/'sha256.json') != plan['study_manifest_sha256']:
        raise ValueError('Completed source study changed')
    _, protocol = read_protocol(STUDY)
    records, traces = occurrences(protocol)
    if traces != plan['trace_sha256'] or [identity(r) for r in records] != plan['occurrences']:
        raise ValueError('Screen coverage changed')
    _, tapes = temporal_stress_bank('sampled')
    config = MergeConfig()
    packed = parameters(config)
    OUTPUT.mkdir(parents=True)
    cache, actual_cache, output, started = {}, {}, [], time.perf_counter()
    counts, by_model = Counter(), defaultdict(Counter)
    for number, item in enumerate(records):
        state, previous, command = (np.asarray(item[k]) for k in ('state','previous','command'))
        step = item['step']
        key = (step, state.tobytes(), previous.tobytes(), command.tobytes())
        if key not in cache:
            banks, base_times, candidate_times, complete = [], [], [], True
            for tape in tapes:
                base, bt, bc = arrival_phase_suffix(state,previous,step,tape,packed,command,False)
                candidate, ct, cc = arrival_phase_suffix(state,previous,step,tape,packed,command,True)
                banks.append(candidate-base)
                base_times.append(bt)
                candidate_times.append(ct)
                complete = complete and bc and cc
            bank, bt, ct = np.asarray(banks),np.asarray(base_times),np.asarray(candidate_times)
            np.testing.assert_allclose(bank,item['original_bank_differences'],atol=1e-10,rtol=0)
            phase = ct-bt
            keep = bool(complete and np.all(phase<=1e-10))
            record = dict(state_index=len(cache), first_occurrence=identity(item),
                original_bank_differences=bank.tolist(), baseline_arrival_phase=bt.tolist(),
                candidate_arrival_phase=ct.tolist(), phase_difference=phase.tolist(),
                all_forecasts_complete=bool(complete), phase_check_passes=keep)
            # NaN is evidence of noncompletion, not an admissible JSON numeric claim.
            if not complete:
                def finite_or_none(values):
                    return [[float(v) if np.isfinite(v) else None for v in row] for row in values]
                for field, values in (('baseline_arrival_phase',bt),('candidate_arrival_phase',ct),('phase_difference',phase)):
                    record[field] = finite_or_none(values)
            save(OUTPUT/f"state_{record['state_index']}.json",record)
            cache[key] = record
        record = cache[key]
        np.testing.assert_allclose(record['original_bank_differences'],item['original_bank_differences'],atol=1e-10,rtol=0)
        actual_key = key + (item['seed'],item['disturbance'])
        if actual_key not in actual_cache:
            actual_tape = disturbance_tape(item['seed'],item['disturbance'],.5,round(config.horizon/config.dt))
            actual_cache[actual_key] = disturbed_action_difference(state,previous,step,command,actual_tape,packed)[0]
        realized = actual_cache[actual_key]
        positive = bool(np.any(realized>1e-9))
        keep = record['phase_check_passes']
        label = ('retained_' if keep else 'rejected_') + ('positive_residual' if positive else 'nonpositive_residual')
        counts[label] += 1
        by_model[item['controller']][label] += 1
        output.append(dict(**identity(item),state_index=record['state_index'],phase_check_passes=keep,
                           realized_branch_difference=realized.tolist(),positive_realized_branch_residual=positive))
        if number % 20 == 0:
            print(json.dumps(dict(processed_occurrences=number+1,total_occurrences=len(records),
                                  distinct_bank_forecasts=len(cache),elapsed_s=time.perf_counter()-started)),flush=True)
    report = dict(admitted_occurrences=len(records),distinct_admitted_forecasts=len(cache),
        distinct_realized_branch_forecasts=len(actual_cache),counts=dict(counts),
        by_model={k:dict(v) for k,v in by_model.items()}, all_original_bank_differences_match=True,
        new_sumo_runs=0,new_independent_initials=0,controller_changed=False,
        closed_loop_improvement_verified=False,risk_certified=False,online_future_noise=False,
        plan_sha256=digest(PLAN/'plan.json'),scope=plan['scope'])
    save(OUTPUT/'occurrences.json',output)
    save(OUTPUT/'audit.json',report)
    save(OUTPUT/'sha256.json',{p.relative_to(OUTPUT).as_posix():digest(p) for p in OUTPUT.rglob('*')
                              if p.is_file() and p.name!='sha256.json'})
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','run'))
    (prepare if parser.parse_args().command=='prepare' else run)()
