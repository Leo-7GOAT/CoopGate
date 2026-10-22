"""Validate new data/storage adapters against completed V19 evidence only.

No live V21 outcomes, new initial states, or fresh risk calibration are used.
Historical calibration, final comparisons and selected complete profiles must
all be reproduced; this is an implementation audit, not new efficacy evidence.
"""
import json
from pathlib import Path

import numpy as np

from coopgate.comparison_v19 import conformal_episode_bound, select_fixed_replicates
from coopgate.episode_store_v21 import EpisodeStore, observations, file_digest
from coopgate.prospective_data_v21 import paired_observations, actual_mode_indices, evaluate_adapters
from coopgate.risk_comparators_v19 import conditional_ltt, paired_mean_improvement_bands
from probe_disturbed_gate_v19 import digest, save
from start_final_v19 import verify_manifest
from study_gate_v19 import read_protocol

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/v21/prospective_data_audit'
SOURCE_FILES = ('src/coopgate/episode_store_v21.py', 'src/coopgate/prospective_data_v21.py',
                'tools/audit_prospective_data_v21.py', 'tests/test_episode_store_v21.py',
                'tests/test_prospective_data_v21.py')


def read(path):
    return json.loads(Path(path).read_text())


def data_from(directory, names):
    verify_manifest(directory)
    _, protocol = read_protocol(directory)
    summary = read(directory / 'summary.json')
    if (summary['protocol_sha256'] != digest(directory / 'protocol.json')
            or summary['episodes_sha256'] != digest(directory / 'episodes.json')):
        raise ValueError('Completed historical study identity differs')
    rows = read(directory / 'episodes.json')
    spec = protocol['spec']
    data = paired_observations(rows, profiles=names, independent_initials=spec['n_initials'],
        seed_base=spec['seed_base'], methods=spec['methods'])
    if data['actual_controller_runs'] != spec['expected_actual_sumo_runs']:
        raise ValueError('Historical complete common-data budget differs')
    return data, rows, spec


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve previous audit results')
    plan_path = ROOT / 'artifacts/v19/comparison_plan'
    calibration_path = ROOT / 'artifacts/v19/comparison_calibration'
    comparison_path = ROOT / 'artifacts/v19/final_comparison'
    for path in (plan_path, calibration_path, comparison_path):
        verify_manifest(path)
    plan = read(plan_path / 'plan.json')
    names = plan['profiles']
    historical_risk = read(calibration_path / 'conditional_risk.json')
    historical_calibration = read(calibration_path / 'calibration.json')
    calibration, _, cal_spec = data_from(ROOT / 'artifacts/v19/risk_validation', names)
    n = calibration['independent_initials']
    ids = list(range(cal_spec['seed_base'], cal_spec['seed_base'] + n))
    modes = actual_mode_indices(n, cal_spec['seed_base'], cal_spec['risk_validation']['mode_selection_stream'])
    a, h, d, failed = (calibration[k] for k in ('adopted','harmful','differences','failed'))
    targets = dict(actual_random_mode=conditional_ltt(ids, names, a[np.arange(n),modes], h[np.arange(n),modes],
                        alpha=historical_risk['alpha'], delta=historical_risk['total_delta']/2),
                   worst_mode_bundle=conditional_ltt(ids, names, a.any(1), h.any(1),
                        alpha=historical_risk['alpha'], delta=historical_risk['total_delta']/2))
    np.testing.assert_equal(targets, historical_risk['targets'])
    gains = -d[np.arange(n),modes,:,:,0].mean(-1)
    mean = paired_mean_improvement_bands(ids, names, gains, delta=historical_risk['total_delta'])
    np.testing.assert_equal(mean, historical_risk['mean_improvement_comparator'])
    _, conformal = conformal_episode_bound(d, d[:,0], failed, names, plan['conformal']['alpha'])
    np.testing.assert_equal(conformal, historical_calibration['conformal'])
    ltt_eligible = [name for j,name in enumerate(names) if all(t['candidates'][j]['certified'] for t in targets.values())]
    mean_eligible = [r['candidate'] for r in mean['candidates'] if r['mean_improvement_selected']]
    means = {r['candidate']:r['mean_gain'] for r in mean['candidates']}
    for key, eligible in (('ltt_selection',ltt_eligible),('mean_spi_selection',mean_eligible)):
        np.testing.assert_equal(select_fixed_replicates(names,eligible,means,plan['seeds']), historical_calibration[key])
    del calibration, a, h, d, failed
    final_path = ROOT / 'artifacts/v19/final_confirmation'
    final, rows, spec = data_from(final_path, names)
    settings = plan['final_test']
    report = evaluate_adapters(final, historical_calibration, family='portfolio0', seeds=plan['seeds'],
        seed_base=spec['seed_base'], mode_stream=settings['mode_selection_stream'],
        bootstrap_seed=settings['bootstrap_seed'], bootstrap_samples=settings['bootstrap_samples'],
        conformal_alpha=plan['conformal']['alpha'])
    np.testing.assert_equal(report['results'], read(comparison_path / 'comparison.json')['results'])
    np.testing.assert_equal(report['selections'], read(comparison_path / 'selections.json'))
    # A complete already-executed 40-case initial tests lossless archival of
    # physical traces and metrics across every controller and disturbance mode.
    archived = [r for r in rows if r['initial_index']==spec['audit_initial_indices'][0]]
    if len(archived)!=40 or not all(r['trace_saved'] for r in archived):
        raise ValueError('Historical archive preflight requires one complete saved initial')
    OUTPUT.mkdir(parents=True)
    metadata = dict(role='historical_archive_roundtrip', original_protocol_sha256=digest(final_path/'protocol.json'),
                    original_episodes_sha256=digest(final_path/'episodes.json'), new_independent_initials=0)
    path = OUTPUT/'historical_initial.sqlite'
    store, cases, expected, original_bytes = EpisodeStore(path,metadata), [], [], 0
    for row in archived:
        case = row['initial_index'],row['disturbance'],row['controller']
        trace_path = final_path/f"shard_{row['initial_index']%spec['shards']}"/f'trace_{case[0]}_{case[1]}_{case[2]}.json'
        trace = read(trace_path)
        store.append(case,row,trace)
        cases.append(case)
        expected.append((case,row,trace))
        original_bytes += trace_path.stat().st_size + len(json.dumps(row).encode())
    store.seal(cases)
    actual = list(observations(path,metadata,cases))
    np.testing.assert_equal(actual, expected)
    archive = dict(original_cases=len(cases),lossless_row_and_trace_roundtrips=len(actual),
                   uncompressed_source_bytes=original_bytes,sqlite_bytes=path.stat().st_size,
                   database_sha256=file_digest(path))
    save(OUTPUT/'audit.json',dict(purpose='Historical execution/data-interface parity, not new risk certification',
        historical_calibration_controller_runs=cal_spec['expected_actual_sumo_runs'],
        historical_final_controller_runs=spec['expected_actual_sumo_runs'],
        calibration_independent_initials=n,final_independent_initials=spec['n_initials'],
        all_nine_profiles_both_ltt_targets_reproduced=True,mean_spi_bands_and_choices_reproduced=True,
        conformal_calibration_reproduced=True,all_six_final_methods_all_three_models_reproduced=True,
        all_final_complete_profile_choices_reproduced=True,archive=archive,
        newly_executed_sumo_runs=0,additional_independent_initials=0,new_risk_certified=False,
        live_v21_results_read=False,source_sha256={name:digest(ROOT/name) for name in SOURCE_FILES},
        historical_input_manifests={str(p.relative_to(ROOT)):digest(p/'sha256.json') for p in
            (plan_path,calibration_path,comparison_path,final_path,ROOT/'artifacts/v19/risk_validation')}))
    save(OUTPUT/'sha256.json',{p.name:digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name!='sha256.json'})
    print(json.dumps(dict(historical_runs_recounted=61440,adapter_methods=6,fixed_models=3,
        archive_roundtrips=len(actual),new_risk_certified=False)),flush=True)


if __name__=='__main__':
    main()
