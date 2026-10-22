"""Frozen paired SUMO audit of exact cross-call baseline-command reuse."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.episode_store_v21 import observations
from coopgate.persistent_phase_v24 import PersistentPhaseRepairGate
from coopgate.phase_repair_v21 import PhaseRepairGate
from profile_phase_runtime_v24 import committed, digest, save
from prospective_runtime_v21 import IndependentReview, execute

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/v21/prospective_execution_preflight_native_20261022'
OUT = ROOT / 'artifacts/v24/persistent_phase_sumo_audit'
TRACE_WORK = {'gate_ms', 'performance_model_transitions', 'exact_cache_entries'}
ROW_WORK = {'setup_ms', 'controller_ms', 'internal_transitions', 'decision_sha256'}


def source_observations():
    database = OLD / 'preflight/blocks/00000/observations.sqlite'
    seal = json.loads(Path(str(database) + '.seal.json').read_text())
    return {(case[1], case[2]): (row, trace)
            for case, row, trace in observations(database, seal['metadata'], seal['cases'])
            if case[-1] == 'collection' and case[2].startswith('phase05_')}


def comparable_trace(trace):
    return [{k: v for k, v in item.items()
             if k not in TRACE_WORK and not k.startswith('persistent_cache_')} for item in trace]


def comparable_row(row):
    return {k: v for k, v in row.items() if k not in ROW_WORK}


def freeze():
    old = json.loads((OLD / 'frozen/plan.json').read_text())
    assert json.loads((OLD / 'preflight/execution/supervisor_exit.json').read_text())['exit_code'] == 0
    archived = source_observations()
    assert len(archived) == 12
    cases = []
    for mode in ('clean', 'uniform', 'bias', 'pulse'):
        for seed in ('1802', '1812', '1822'):
            cases.append([0, mode, 'phase05_' + seed])
    sources = dict(old['source_sha256'])
    for relative in ('src/coopgate/persistent_phase_v24.py', 'tests/test_persistent_phase_v24.py',
                     'tools/audit_persistent_phase_v24.py', 'tools/profile_phase_runtime_v24.py'):
        sources[relative] = digest(ROOT / relative)
    inputs = (OLD / 'frozen/plan.json', OLD / 'preflight/execution/supervisor_exit.json',
              OLD / 'preflight/blocks/00000/observations.sqlite',
              OLD / 'preflight/blocks/00000/observations.sqlite.seal.json')
    plan = dict(purpose='Exact complete closed-loop equivalence and paired timing on all four modes and three fixed models of one completed old initial',
                source_sha256=sources, inputs_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in inputs},
                models=old['models'], spec=old['stages']['preflight'], cases=cases,
                implementations=['reference', 'persistent'], retained_cache_limit=8192,
                actual_sumo_runs=24, new_independent_initials=0,
                active_calibration_outcomes_used=False,
                order='Reference then persistent on even case indices, reversed on odd indices',
                warmup='Warm each implementation with each frozen model on the same declared state before timing',
                checks='Save every new row and trace before assertions; compare all non-work fields to paired original and archived reviewed SUMO; independently review all 24 physical/candidate traces and every distinct full admitted forecast',
                timing_scope='All select lifecycle costs including eviction; report setup separately; concurrent desktop load, one old initial, no population P95 or hard realtime claim',
                reference_trace_excluded_fields=sorted(TRACE_WORK),
                reference_row_excluded_fields=sorted(ROW_WORK),
                invariants='Same candidates, rankings, 33 paths, full horizons, commands, states, disturbances, adoption, failures and per-car outcomes',
                recovery='No automatic restart; preserve completed prefixes and investigate any real terminal failure')
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / 'plan.json', plan)
    print(json.dumps(dict(cases=12, actual_sumo_runs=24, plan_sha256=digest(OUT / 'plan.json'))), flush=True)


def run():
    path = OUT / 'plan.json'
    committed(path)
    plan = json.loads(path.read_text())
    for name, expected in {**plan['source_sha256'], **plan['inputs_sha256']}.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen input differs: ' + name)
    for relative in plan['source_sha256']:
        committed(ROOT / relative)
    torch.set_num_threads(1)
    target = OUT / 'results'
    target.mkdir(exist_ok=False)
    models = {}
    for seed, meta in plan['models'].items():
        assert digest(ROOT / meta['path']) == meta['sha256']
        models[seed], _ = load_coupled(ROOT / meta['path'])
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for model in models.values():
        PhaseRepairGate(model).select(warm, np.zeros(3))
        PersistentPhaseRepairGate(model).select(warm, np.zeros(3))
    archived, pairs, runs = source_observations(), [], []
    with DisturbedMergeEnv() as env:
        for index, case in enumerate(plan['cases']):
            model = models[case[2].rsplit('_', 1)[1]]
            order = plan['implementations'] if index % 2 == 0 else list(reversed(plan['implementations']))
            current = {}
            for implementation in order:
                if implementation == 'reference':
                    factory = lambda state, m=model: PhaseRepairGate(m)
                else:
                    factory = lambda state, m=model: PersistentPhaseRepairGate(m, max_cache_entries=plan['retained_cache_limit'])
                start = time.perf_counter()
                row, trace = execute(env, case, plan['spec'], factory)
                wall = time.perf_counter() - start
                record = dict(case=case, implementation=implementation, wall_seconds=wall, row=row, trace=trace)
                save(target / f'case_{index}_{implementation}.json', record)
                current[implementation] = record
                runs.append(record)
                old_row, old_trace = archived[case[1], case[2]]
                np.testing.assert_equal(comparable_row(row), comparable_row(old_row))
                np.testing.assert_equal(comparable_trace(trace), comparable_trace(old_trace))
                if implementation == 'persistent':
                    assert all(item['persistent_cache_entries_after'] <= plan['retained_cache_limit'] for item in trace)
                print(json.dumps(dict(case_index=index, implementation=implementation, wall_seconds=wall, archived_exact=True)), flush=True)
            ref, new = current['reference'], current['persistent']
            np.testing.assert_equal(comparable_row(ref['row']), comparable_row(new['row']))
            np.testing.assert_equal(comparable_trace(ref['trace']), comparable_trace(new['trace']))
            pairs.append(dict(case=case, decisions=len(ref['trace']),
                reference_select_ms=sum(ref['row']['controller_ms']),
                persistent_select_ms=sum(new['row']['controller_ms']),
                reference_setup_ms=ref['row']['setup_ms'], persistent_setup_ms=new['row']['setup_ms'],
                reference_model_transitions=ref['row']['internal_transitions'],
                persistent_model_transitions=new['row']['internal_transitions'],
                reference_adopted_steps=ref['row']['learned_action_steps'],
                persistent_adopted_steps=new['row']['learned_action_steps'],
                maximum_retained_entries=max(i['persistent_cache_entries_after'] for i in new['trace']),
                maximum_transient_entries=max(i['persistent_cache_entries_peak'] for i in new['trace']),
                all_execution_and_admission_fields_exact=True))
    save(target / 'collection_complete.json', dict(actual_sumo_runs=len(runs), paired_cases=len(pairs), plan_sha256=digest(path)))
    reviewer = IndependentReview(dict(spec=plan['spec'], models=plan['models']))
    for i, record in enumerate(runs):
        reviewer.check(record['row'], record['trace'])
        print(json.dumps(dict(reviewed_runs=i+1, total=len(runs))), flush=True)
    report = dict(plan_sha256=digest(path), actual_sumo_runs=len(runs), paired_cases=len(pairs),
                  new_independent_initials=0, active_calibration_outcomes_used=False,
                  all_24_trajectories_match_reviewed_archive=True, independent_review_counts=reviewer.counts,
                  pairs=pairs, timing_scope=plan['timing_scope'],
                  no_new_risk_certificate=True, hard_real_time_certified=False)
    save(target / 'report.json', report)
    save(target / 'sha256.json', {p.name: digest(p) for p in target.iterdir() if p.is_file()})
    print(json.dumps(dict(completed=True, actual_sumo_runs=len(runs), reviewed=reviewer.counts)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run'])
    args = parser.parse_args()
    {'freeze': freeze, 'run': run}[args.action]()
