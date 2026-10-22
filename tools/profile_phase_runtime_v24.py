"""Attribute full phase-gate cost on already completed integration decisions.

This deliberately selected diagnostic is not a runtime distribution or another
effectiveness study. It never reads the active prospective observations. The
frozen gate and all candidates, paths, metrics and evidence remain unchanged.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
import time

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import observations
from coopgate.phase_repair_v21 import PhaseRepairGate

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/v21/prospective_execution_preflight_native_20261022'
OUT = ROOT / 'artifacts/v24/phase_runtime_attribution'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def committed(path):
    relative = path.relative_to(ROOT).as_posix()
    blob = subprocess.run(['git', 'show', 'HEAD:' + relative], cwd=ROOT,
                          check=True, capture_output=True).stdout
    if hashlib.sha256(blob).hexdigest() != digest(path):
        raise ValueError('Commit exact source bytes before running: ' + relative)


def freeze():
    database = OLD / 'preflight/blocks/00000/observations.sqlite'
    seal_file = Path(str(database) + '.seal.json')
    seal = json.loads(seal_file.read_text())
    old_plan = json.loads((OLD / 'frozen/plan.json').read_text())
    exit_receipt = OLD / 'preflight/execution/supervisor_exit.json'
    if json.loads(exit_receipt.read_text())['exit_code'] != 0:
        raise ValueError('Use only the successfully completed integration')
    pools = {seed: [] for seed in old_plan['models']}
    for case, row, trace in observations(database, seal['metadata'], seal['cases']):
        if case[-1] != 'collection' or not case[2].startswith('phase05_'):
            continue
        seed = case[2].rsplit('_', 1)[1]
        state, previous = row['initial'], [0., 0., 0.]
        for step, item in enumerate(trace):
            expected = {k: v for k, v in item.items()
                        if k not in ('gate_ms', 't', 'x', 'command', 'disturbance')}
            pools[seed].append(dict(model=seed, case=list(case), step=step,
                                   state=state, previous=previous,
                                   expected_command=item['command'], expected_info=expected,
                                   old_gate_ms=item['gate_ms']))
            state, previous = item['x'], item['command']
    selected = []
    for seed, items in pools.items():
        order = sorted(items, key=lambda r: (-r['old_gate_ms'], r['case'], r['step']))
        if len(order) < 2:
            raise ValueError('Every fixed model needs two recorded diagnostic calls')
        chosen = order[:2]
        adopted = sorted((r for r in items if r['expected_info']['portfolio_joint_action_admitted']),
                         key=lambda r: (r['case'], r['step']))
        if adopted and not any(r['case'] == adopted[0]['case'] and r['step'] == adopted[0]['step'] for r in chosen):
            chosen.append(adopted[0])
        selected.extend(chosen)
    sources = dict(old_plan['source_sha256'])
    sources[Path(__file__).relative_to(ROOT).as_posix()] = digest(__file__)
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Completed integration source changed: ' + name)
    plan = dict(purpose='Post-hoc function-cost attribution on completed old-initial decisions only',
                selection='Two slowest recorded calls per fixed model plus the first adopted call if distinct',
                not_a_latency_distribution=True, no_new_sumo_runs=True, new_independent_initials=0,
                active_calibration_outcomes_used=False, models=old_plan['models'], cases=selected,
                source_sha256=sources,
                inputs_sha256={p.relative_to(ROOT).as_posix(): digest(p)
                              for p in (database, seal_file, exit_receipt, OLD / 'frozen/plan.json')},
                replay_rule='For each case first warm and verify one complete call, then cProfile a second complete call and verify every non-timing field',
                timing_scope='Concurrent desktop load with cProfile overhead; no real-time or speedup claim',
                invariants='Original candidates, ranking, 33 paths, full horizons, physical checks, fallback, and full evidence must all match')
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / 'plan.json', plan)
    print(json.dumps(dict(cases=len(selected), plan_sha256=digest(OUT / 'plan.json'), new_independent_initials=0)), flush=True)


def verify(command, info, case):
    if not np.array_equal(command, np.asarray(case['expected_command'])):
        raise ValueError('Replay command differs from completed SUMO trace')
    actual = {k: v for k, v in info.items() if k != 'gate_ms'}
    if actual != case['expected_info']:
        bad = sorted(k for k in actual.keys() | case['expected_info'].keys()
                     if actual.get(k) != case['expected_info'].get(k))
        raise ValueError('Replay evidence differs: ' + ', '.join(bad))


def run():
    plan_path = OUT / 'plan.json'
    committed(plan_path)
    plan = json.loads(plan_path.read_text())
    for name, expected in {**plan['source_sha256'], **plan['inputs_sha256']}.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen input changed: ' + name)
    for name in plan['source_sha256']:
        committed(ROOT / name)
    target = OUT / 'results'
    target.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    models = {}
    for seed, meta in plan['models'].items():
        if digest(ROOT / meta['path']) != meta['sha256']:
            raise ValueError('Frozen policy changed')
        models[seed], _ = load_coupled(ROOT / meta['path'])
    records = []
    for index, case in enumerate(plan['cases']):
        def call():
            gate = PhaseRepairGate(models[case['model']], .05)
            gate.step = case['step']
            return gate.select(np.array(case['state']), np.array(case['previous']))
        warm_command, warm_info = call()
        verify(warm_command, warm_info, case)
        profiler = cProfile.Profile()
        started = time.perf_counter()
        command, info = profiler.runcall(call)
        full_ms = (time.perf_counter() - started) * 1000
        verify(command, info, case)
        profiler.dump_stats(target / f'case_{index}.prof')
        stats = pstats.Stats(profiler)
        calls = sorted([dict(file=Path(file).name, line=line, function=function,
                             primitive_calls=value[0], total_calls=value[1],
                             self_ms=1000 * value[2], cumulative_ms=1000 * value[3])
                        for (file, line, function), value in stats.stats.items()],
                       key=lambda r: -r['cumulative_ms'])
        record = dict(index=index, model=case['model'], case=case['case'], step=case['step'],
                      profiled_complete_factory_and_select_ms=full_ms,
                      profiled_select_ms=info['gate_ms'],
                      selected_composition=info['composition'], candidate_count=info['candidate_count'],
                      nominal_candidates=info['nominal_candidate_count'],
                      admitted=info['portfolio_joint_action_admitted'],
                      forecast_transitions=info['performance_forecast_transitions'],
                      model_transitions=info['performance_model_transitions'],
                      cache_entries=info['exact_cache_entries'],
                      all_non_timing_trace_fields_exact=True, calls=calls)
        save(target / f'case_{index}.json', record)
        records.append(record)
        print(json.dumps(dict(index=index, model=case['model'], complete_ms=full_ms, exact=True)), flush=True)
    report = dict(plan_sha256=digest(plan_path), cases=len(records),
                  complete_replayed_calls=2 * len(records), actual_new_sumo_runs=0,
                  independent_new_initials=0, all_commands_and_evidence_exact=True,
                  active_calibration_outcomes_used=False,
                  timing_scope=plan['timing_scope'], records=records)
    save(target / 'report.json', report)
    save(target / 'sha256.json', {p.name: digest(p) for p in target.iterdir() if p.is_file()})
    print(json.dumps(dict(completed=True, cases=len(records), new_sumo_runs=0,
                          profile_report_sha256=digest(target / 'report.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run'])
    args = parser.parse_args()
    {'freeze': freeze, 'run': run}[args.action]()
