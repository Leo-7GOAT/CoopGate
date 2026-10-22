"""Exercise real selector-to-SUMO execution before statistical calibration.

The calibration records here are explicitly synthetic branch tests, not fitted
values, statistical certificates, final-test data or efficacy evidence.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from replay_comparison_v19 import ObservedForecasts, assert_outcomes, deployment_factory
from study_gate_v19 import factories
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]


def freeze(directory):
    family = json.loads((ROOT / 'artifacts/v19/risk_validation/protocol.json').read_text())
    selection = json.loads((ROOT / 'artifacts/v19/comparison_plan/plan.json').read_text())
    calibration = dict(
        calibration_origin='Synthetic branch exercise only; zero calibration data',
        names=selection['profiles'], models=selection['models'], calibration_n_initials=0,
        conformal=dict(q=0., infinite_q=False),
        ltt_selection={'1802': 'nominal_1802', '1812': 'baseline', '1822': 'portfolio0_1822'},
        mean_spi_selection={'1802': 'raw_1802', '1812': 'nominal_1812', '1822': 'baseline'})
    sources = list((ROOT / 'src/coopgate').glob('*.py')) + [Path(__file__),
               ROOT / 'tools/replay_comparison_v19.py', ROOT / 'tools/experiment_ego_v13.py',
               ROOT / 'tools/verify_development_v14.py', ROOT / 'tools/study_gate_v19.py',
               ROOT / 'tools/probe_disturbed_gate_v19.py']
    plan = dict(purpose='Actual deployment wiring audit with explicitly synthetic calibration; no statistical risk claim',
                cases=[dict(seed=61940001, scenario='nominal', mode='uniform', infinite_q=False),
                       dict(seed=61940002, scenario='dense', mode='bias', infinite_q=True)],
                family=dict(spec=dict(methods=family['spec']['methods']), models=family['models']),
                selection_plan=selection, synthetic_calibration=calibration,
                adapters=['conformal', 'ltt', 'mean_spi'], training_seeds=selection['seeds'],
                actual_deployment_runs_planned=18, actual_reference_runs_planned=18, actual_shadow_runs_planned=20,
                source_sha256={p.resolve().relative_to(ROOT).as_posix(): digest(p) for p in sources})
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'plan.json', plan)
    for source in plan['source_sha256']:
        target = output / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    print(json.dumps(dict(plan_sha256=digest(output / 'plan.json'), actual_sumo_runs_planned=56)), flush=True)


def run(directory):
    torch.set_num_threads(1)
    output = Path(directory)
    if (output / 'episodes.json').exists():
        raise FileExistsError('Preserve previous actual deployment executions')
    plan = json.loads((output / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        assert digest(output / 'source' / path) == digest(ROOT / path) == expected, path
    fs = factories(plan['family'])
    warm = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    for factory in fs.values():
        factory(warm).select(warm, np.zeros(3))
    forecasts = ObservedForecasts(fs, output)
    rows, decisions, pairs = [], [], []
    with DisturbedMergeEnv() as env:
        for case_index, case in enumerate(plan['cases']):
            calibration = copy.deepcopy(plan['synthetic_calibration'])
            calibration['conformal'] = dict(q=None if case['infinite_q'] else 0., infinite_q=case['infinite_q'])
            for adapter in plan['adapters']:
                for seed in plan['training_seeds']:
                    factory = deployment_factory(adapter, seed, plan['selection_plan'], calibration, fs, forecasts, decisions)
                    row, trace, observed = run_episode(env, case['seed'], case['scenario'], case['mode'], factory, True)
                    row.update(initial=observed.tolist(), case_index=case_index, adapter=adapter,
                               training_seed=seed, execution='deployment')
                    trace_audit(trace, row)
                    choice = decisions[-1]['profile']
                    if adapter == 'conformal' and case['infinite_q']:
                        assert choice == 'baseline'
                    reference, ref_trace, ref_initial = run_episode(env, case['seed'], case['scenario'], case['mode'], fs[choice], True)
                    reference.update(initial=ref_initial.tolist(), case_index=case_index, adapter=adapter,
                                     training_seed=seed, execution='direct_reference', profile=choice)
                    trace_audit(ref_trace, reference)
                    np.testing.assert_array_equal(observed, ref_initial)
                    assert_outcomes(row, reference)
                    assert len(trace) == len(ref_trace)
                    for actual, expected in zip(trace, ref_trace):
                        np.testing.assert_array_equal(actual['command'], expected['command'])
                        np.testing.assert_allclose(actual['x'], expected['x'], atol=1e-8, rtol=0)
                    pair_index = len(pairs)
                    save(output / f'trace_{pair_index}_deployment.json', trace)
                    save(output / f'trace_{pair_index}_reference.json', ref_trace)
                    rows.extend([row, reference])
                    pairs.append(dict(case_index=case_index, adapter=adapter, training_seed=seed,
                                      selected_profile=choice, paired_decision_steps=len(trace)))
                    save(output / 'episodes.json', rows)
                    save(output / 'decisions.json', decisions)
                    save(output / 'preparations.json', forecasts.preparations)
                    print(json.dumps(dict(completed_pairs=len(pairs), actual_deployment_and_reference_runs=len(rows),
                                          actual_shadow_runs=forecasts.actual_sumo_runs)), flush=True)
    assert len(rows) == 36 and len(pairs) == 18 and forecasts.actual_sumo_runs == 20
    report = dict(
        actual_sumo_runs=len(rows) + forecasts.actual_sumo_runs,
        actual_deployment_runs=18, actual_reference_runs=18, actual_shadow_runs=forecasts.actual_sumo_runs,
        paired_profiles=len(pairs), paired_decision_steps=sum(p['paired_decision_steps'] for p in pairs),
        pairs=pairs, all_commands_and_outcomes_match=True,
        synthetic_calibration=True, risk_certified=False, final_comparison_validated=False,
        scope='Execution wiring and forced statistical branches only; no calibrated risk/effectiveness result',
        episodes_sha256=digest(output / 'episodes.json'), plan_sha256=digest(output / 'plan.json'))
    save(output / 'audit.json', report)
    save(output / 'sha256.json', {p.relative_to(output).as_posix(): digest(p)
         for p in output.rglob('*') if p.is_file() and p.name != 'sha256.json'})
    print(json.dumps({key: value for key, value in report.items() if key != 'pairs'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run'))
    parser.add_argument('directory')
    args = parser.parse_args()
    (freeze if args.command == 'freeze' else run)(args.directory)
