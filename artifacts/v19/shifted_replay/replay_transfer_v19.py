"""Frozen direct-transfer replay coverage, including every physical failure."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from coopgate.episode_selection_v19 import physical_failure
from coopgate.shifted_sumo_v19 import ShiftedMergeEnv, audit_shifted_trace
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import factories, read_protocol

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/v19/shifted_study'
PLAN = ROOT / 'artifacts/v19/shifted_replay_plan'


def freeze():
    if (STUDY / 'summary.json').exists():
        raise ValueError('Freeze replay coverage before finalized transfer outcomes')
    _, protocol = read_protocol(STUDY)
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [Path(__file__),
               ROOT / 'tools/experiment_ego_v13.py', ROOT / 'tools/probe_disturbed_gate_v19.py',
               ROOT / 'tools/study_gate_v19.py', ROOT / 'tools/study_shifted_v19.py']
    record = dict(purpose='Mechanism replay of direct transfer; not additional independent risk samples',
                  study_protocol_sha256=digest(STUDY / 'protocol.json'),
                  coverage=['Initial index zero for every scenario/plant/controller',
                            'First actual learning admission in every scenario/plant/controller cell',
                            'Every physical failure or noncompletion, including baseline and zero-adoption cases'],
                  models=protocol['models'], source_sha256={p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources})
    PLAN.mkdir(parents=True, exist_ok=False)
    save(PLAN / 'plan.json', record)
    for source in record['source_sha256']:
        target = PLAN / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    save(PLAN / 'sha256.json', {p.relative_to(PLAN).as_posix(): digest(p)
         for p in PLAN.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps(dict(plan_sha256=digest(PLAN / 'plan.json'), outcomes_inspected=False)), flush=True)


def selected_cases(rows):
    selected, seen_adoption = [], set()
    for row in sorted(rows, key=lambda r: (r['scenario'], r['plant_name'], r['controller'], r['initial_index'])):
        cell = row['scenario'], row['plant_name'], row['controller']
        reasons = []
        if row['initial_index'] == 0:
            reasons.append('prespecified_first_initial')
        if row['learned_action_steps'] > 0 and cell not in seen_adoption:
            reasons.append('first_learning_admission')
            seen_adoption.add(cell)
        if physical_failure(row):
            reasons.append('physical_failure_or_noncompletion')
        if reasons:
            selected.append((row, reasons))
    return selected


def run(output):
    torch.set_num_threads(1)
    verify_manifest(PLAN)
    require_committed([PLAN / 'plan.json', PLAN / 'sha256.json'])
    plan = json.loads((PLAN / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        assert digest(PLAN / 'source' / path) == digest(ROOT / path) == expected, path
    _, protocol = read_protocol(STUDY)
    verify_manifest(STUDY)
    assert digest(STUDY / 'protocol.json') == plan['study_protocol_sha256']
    summary = json.loads((STUDY / 'summary.json').read_text())
    assert summary['episodes_sha256'] == digest(STUDY / 'episodes.json')
    original = json.loads((STUDY / 'episodes.json').read_text())
    assert len(original) == summary['actual_sumo_runs'] == protocol['spec']['expected_actual_sumo_runs']
    cases, fs = selected_cases(original), factories(protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'cases.json', [dict(scenario=r['scenario'], initial_index=r['initial_index'], plant=r['plant_name'],
                                    controller=r['controller'], reasons=reasons) for r, reasons in cases])
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values():
        factory(warm).select(warm, np.zeros(3))
    rows, pairs = [], []
    with ExitStack() as stack:
        environments = {p: stack.enter_context(ShiftedMergeEnv(plant=p)) for p in protocol['spec']['plants']}
        for index, (reference, reasons) in enumerate(cases):
            method, plant, scenario = reference['controller'], reference['plant_name'], reference['scenario']
            row, trace, initial = run_episode(environments[plant], reference['seed'], scenario,
                                             protocol['spec']['mode'], fs[method], True)
            row.update(initial=initial.tolist(), initial_index=reference['initial_index'],
                       global_initial_index=reference['global_initial_index'], controller=method, plant_name=plant)
            assert row['plant'] == reference['plant']
            for field in ('initial', 'per_vehicle_finish', 'per_vehicle_variation', 'completed', 'collision',
                          'aborted', 'contract_violations', 'max_actual_error', 'error_contract_exceeded_substeps'):
                np.testing.assert_array_equal(row[field], reference[field], err_msg=f'{index}/{field}')
            kind = protocol['spec']['methods'][method]['kind']
            used = sum(t.get('nominal_joint_action_admitted', kind == 'raw') for t in trace)
            assert used == reference['learned_action_steps']
            assert len(trace) == reference['control_steps']
            audit_shifted_trace(trace, row)
            for item in trace:
                if item.get('portfolio_joint_action_admitted'):
                    values = np.array(item['selected_bank_differences'])
                    assert values.shape == (17, 3, 2) and np.all(values <= 1e-10)
            saved = STUDY / f"shard_{reference['global_initial_index'] % protocol['spec']['shards']}" / f"trace_{scenario}_{reference['initial_index']}_{plant}_{method}.json"
            if saved.exists():
                archived = json.loads(saved.read_text())
                assert len(archived) == len(trace)
                for actual, previous in zip(trace, archived):
                    for field in ('command', 'x', 'disturbance', 'innovation', 'actuator_states'):
                        np.testing.assert_array_equal(actual[field], previous[field], err_msg=f'{index}/{field}')
            row['learned_action_steps'] = int(used)
            rows.append(row)
            pairs.append(dict(case_index=index, reasons=reasons, complete_trajectory_compared=saved.exists(),
                              decision_steps=len(trace), physical_failure_reproduced=physical_failure(row)))
            save(output / f'trace_{index}.json', trace)
            save(output / 'episodes.json', rows)
            print(json.dumps(dict(actual_replays=len(rows), planned_replays=len(cases))), flush=True)
    all_failures = sum(physical_failure(r) for r in original)
    reproduced = sum(physical_failure(r) for r in rows)
    assert all_failures == reproduced
    report = dict(actual_sumo_replays=len(rows), source_actual_sumo_runs=len(original),
                  all_source_failures=all_failures, physical_failures_reproduced=reproduced,
                  complete_trajectories_compared=sum(p['complete_trajectory_compared'] for p in pairs),
                  actuator_trajectories_reconstructed=len(rows), pairs=pairs,
                  all_selected_metrics_match=True, risk_certified=False,
                  count_scope='Replay coverage, not independent risk/efficacy samples; all baseline/controller failures retained',
                  plan_sha256=digest(PLAN / 'plan.json'), study_summary_sha256=digest(STUDY / 'summary.json'),
                  source_episodes_sha256=digest(STUDY / 'episodes.json'), episodes_sha256=digest(output / 'episodes.json'))
    save(output / 'audit.json', report)
    shutil.copy2(__file__, output / Path(__file__).name)
    save(output / 'sha256.json', {p.relative_to(output).as_posix(): digest(p)
         for p in output.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({key: value for key, value in report.items() if key != 'pairs'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze')
    p = sub.add_parser('run')
    p.add_argument('output')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    else:
        run(args.output)
