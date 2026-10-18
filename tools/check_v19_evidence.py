"""Integrity/accounting of completed V19 development; never a risk certificate."""
import json
from pathlib import Path

from probe_disturbed_gate_v19 import digest

ROOT = Path(__file__).resolve().parents[1]


def verify():
    verified = 0
    for name in ('bank_probe', 'portfolio_probe', 'shifted_harness', 'bank_probe_analysis',
                 'portfolio_probe_analysis', 'paired_bank', 'paired_bank_analysis',
                 'composition_ablation', 'portfolio_audit', 'cached_gate_audit', 'actuator_bounds',
                 'bounded_gate_audit', 'nominal_predictor_audit', 'runtime_profile', 'deployment_runtime_audit',
                 'shifted_study', 'shifted_analysis', 'runtime_plot', 'certification_power', 'shifted_replay'):
        directory = ROOT / 'artifacts/v19' / name
        manifest = json.loads((directory / 'sha256.json').read_text())
        for path, expected in manifest.items():
            assert digest(directory / path) == expected, (name, path)
            verified += 1
        if (directory / 'summary.json').exists():
            result = json.loads((directory / 'summary.json').read_text())
            assert result['episodes_sha256'] == digest(directory / 'episodes.json')
            assert result['protocol_sha256'] == digest(directory / 'protocol.json')
            rows = json.loads((directory / 'episodes.json').read_text())
            assert len(rows) == result['actual_sumo_runs']
        elif (directory / 'audit.json').exists():
            result = json.loads((directory / 'audit.json').read_text())
            assert result['episodes_sha256'] == digest(directory / 'episodes.json')
            plan_path = (ROOT / 'artifacts/v19/shifted_replay_plan/plan.json'
                         if name == 'shifted_replay' else directory / 'plan.json')
            assert result['plan_sha256'] == digest(plan_path)
            if name == 'portfolio_audit':
                assert result['actual_sumo_replays'] == 12
                assert result['all_strict_harm_cases_reproduced'] == 1
                assert len(result['nonvacuous_identity_cases']) == 6
                assert all(r['admissions'] > 0 and r['maximum_identity_residual'] == 0
                           for r in result['nonvacuous_identity_cases'])
            elif name == 'cached_gate_audit':
                assert result['actual_sumo_replays'] == 58 and result['exact_paired_cases'] == 29
                assert result['exact_paired_decision_steps'] == 667
                assert result['internal_work_reduction'] > .20
                assert result['hard_real_time_certified'] is False
                assert result['timings']['cached']['p95_ms'] > 500
            elif name == 'bounded_gate_audit':
                assert result['actual_sumo_replays'] == 58 and result['exact_paired_cases'] == 29
                assert result['exact_paired_decision_steps'] == 667
                assert result['verified_nominal_prunes'] == 150 and result['verified_bank_prunes'] == 82
                assert .06 < result['controller_time_reduction'] < .07
                assert result['hard_real_time_certified'] is False
                assert result['timings']['bounded']['p95_ms'] > 500
                assert result['timings']['bounded']['p99_ms'] > result['timings']['cached']['p99_ms']
            elif name == 'nominal_predictor_audit':
                assert result['actual_sumo_runs'] == 120 and result['paired_profiles'] == 60
                assert result['paired_decision_steps'] == 1392
                assert result['uses_actual_episode_seed'] is False and result['uses_future_disturbance'] is False
                assert all(pair['initial_error'] == 0 for pair in result['pairs'])
                assert len(result['preparations']) == 6
            elif name == 'deployment_runtime_audit':
                assert result['actual_sumo_runs'] == 56
                assert result['actual_deployment_runs'] == result['actual_reference_runs'] == 18
                assert result['actual_shadow_runs'] == 20 and result['paired_decision_steps'] == 409
                assert result['all_commands_and_outcomes_match'] is True
                assert result['synthetic_calibration'] is True and result['final_comparison_validated'] is False
            elif name == 'shifted_replay':
                assert result['actual_sumo_replays'] == result['actuator_trajectories_reconstructed'] == 336
                assert result['source_actual_sumo_runs'] == 6400
                assert result['all_source_failures'] == result['physical_failures_reproduced'] == 68
                assert result['complete_trajectories_compared'] == 291
                assert result['all_selected_metrics_match'] is True
                assert result['source_episodes_sha256'] == digest(ROOT / 'artifacts/v19/shifted_study/episodes.json')
                assert result['study_summary_sha256'] == digest(ROOT / 'artifacts/v19/shifted_study/summary.json')
            else:
                raise AssertionError('Unrecognized audit: ' + name)
            if name in ('bounded_gate_audit', 'nominal_predictor_audit', 'deployment_runtime_audit'):
                plan = json.loads((directory / 'plan.json').read_text())
                for path, expected in plan['source_sha256'].items():
                    assert digest(directory / 'source' / path) == digest(ROOT / path) == expected, path
            assert result['risk_certified'] is False
        elif name == 'actuator_bounds':
            result = json.loads((directory / 'analysis.json').read_text())
            assert result['gain_drag']['acceleration_error_upper_bound'] == .5
            assert result['lag_mild']['acceleration_error_upper_bound'] < .5
            assert result['lag_strong']['acceleration_error_upper_bound'] > .5
            assert .054 < result['maximum_uniformly_covered_lag_tau_seconds'] < .055
        elif name == 'runtime_profile':
            result = json.loads((directory / 'profile.json').read_text())
            assert result['source_audit_sha256'] == digest(ROOT / 'artifacts/v19/bounded_gate_audit/audit.json')
            assert len(result['records']) == 6
            assert sorted(r['seed'] for r in result['records']) == ['1802', '1802', '1812', '1812', '1822', '1822']
        elif name == 'runtime_plot':
            result = json.loads((directory / 'metadata.json').read_text())
            assert result['source_audit_sha256'] == digest(ROOT / 'artifacts/v19/bounded_gate_audit/audit.json')
            assert result['paired_decisions'] == 667 and result['runtime_requirement_met'] is False
        else:
            result = json.loads((directory / 'analysis.json').read_text())
            assert result['risk_certified'] is False
    bank = json.loads((ROOT / 'artifacts/v19/bank_probe_analysis/analysis.json').read_text())
    portfolio = json.loads((ROOT / 'artifacts/v19/portfolio_probe_analysis/analysis.json').read_text())
    shifted = json.loads((ROOT / 'artifacts/v19/shifted_harness/summary.json').read_text())
    assert bank['actual_sumo_runs'] == 640
    assert bank['results']['bank']['adopted_policy_initial_pairs'] == 10
    assert bank['results']['bank']['harmful_policy_initial_pairs'] == 0
    assert portfolio['actual_sumo_runs'] == 1280
    assert portfolio['results']['portfolio0']['adopted_policy_initial_pairs'] == 46
    assert portfolio['results']['portfolio0']['harmful_policy_initial_pairs'] == 1
    assert portfolio['results']['portfolio0']['controller_p95_ms'] > 500.
    assert shifted['actual_sumo_runs'] == 64 and shifted['parity_pairs'] == 8
    assert shifted['plants']['lag_strong']['aborted'] == 1
    paired = json.loads((ROOT / 'artifacts/v19/composition_ablation/analysis.json').read_text())
    assert paired['baseline_parity_pairs'] == 128
    assert paired['extra_team_gain_ci95'][0] > 0
    assert paired['adoption_transitions'] == dict(bank_only=0, portfolio_only=7, both=39, neither=50)
    assert paired['families']['bank']['harmful_adopted_pairs'] == 1
    # A weight hash alone does not identify a closed-loop policy. Bind runtime
    # controller/evaluator code across the calibration and final splits too.
    risk_dir = ROOT / 'artifacts/v19/risk_validation'
    final_dir = ROOT / 'artifacts/v19/final_confirmation'
    risk = json.loads((risk_dir / 'protocol.json').read_text())
    final = json.loads((final_dir / 'protocol.json').read_text())
    power = json.loads((ROOT / 'artifacts/v19/certification_power/analysis.json').read_text())
    assert power['protocol_sha256'] == digest(risk_dir / 'protocol.json')
    assert power['calibration_outcomes_read'] is False and power['protocol_changed'] is False
    assert power['minimum_zero_harm_adoptions'] == 115
    assert power['candidates'] == 9 and power['risk_targets'] == 2
    assert risk['models'] == final['models']
    assert risk['spec']['methods'] == final['spec']['methods']
    assert risk['spec']['modes'] == final['spec']['modes']
    assert risk['spec']['sampling'] == final['spec']['sampling'] == 'iid_mixture'
    assert risk['spec']['data_role'] == 'risk_validation'
    assert final['spec']['data_role'] == 'final_confirmation'
    calibration_ids = set(range(risk['spec']['seed_base'], risk['spec']['seed_base'] + risk['spec']['n_initials']))
    final_ids = set(range(final['spec']['seed_base'], final['spec']['seed_base'] + final['spec']['n_initials']))
    assert not calibration_ids.intersection(final_ids)
    for path, expected in risk['source_sha256'].items():
        assert final['source_sha256'][path] == expected, path
        assert digest(risk_dir / 'source' / path) == digest(final_dir / 'source' / path) == digest(ROOT / path) == expected
    comparison_dir = ROOT / 'artifacts/v19/comparison_plan'
    comparison = json.loads((comparison_dir / 'plan.json').read_text())
    assert comparison['calibration_protocol_sha256'] == digest(risk_dir / 'protocol.json')
    assert final['spec']['comparison_plan_sha256'] == digest(comparison_dir / 'plan.json')
    for path, expected in comparison['source_sha256'].items():
        assert digest(comparison_dir / 'source' / path) == digest(ROOT / path) == expected
    replay_dir = ROOT / 'artifacts/v19/comparison_replay_plan'
    replay_plan = json.loads((replay_dir / 'plan.json').read_text())
    for path, expected in json.loads((replay_dir / 'sha256.json').read_text()).items():
        assert digest(replay_dir / path) == expected, path
        verified += 1
    assert replay_plan['comparison_plan_sha256'] == digest(comparison_dir / 'plan.json')
    assert replay_plan['final_protocol_sha256'] == digest(final_dir / 'protocol.json')
    for path, expected in replay_plan['source_sha256'].items():
        assert digest(replay_dir / 'source' / path) == digest(ROOT / path) == expected, path
    transfer_replay_dir = ROOT / 'artifacts/v19/shifted_replay_plan'
    transfer_replay = json.loads((transfer_replay_dir / 'plan.json').read_text())
    for path, expected in json.loads((transfer_replay_dir / 'sha256.json').read_text()).items():
        assert digest(transfer_replay_dir / path) == expected, path
        verified += 1
    assert transfer_replay['study_protocol_sha256'] == digest(ROOT / 'artifacts/v19/shifted_study/protocol.json')
    for path, expected in transfer_replay['source_sha256'].items():
        assert digest(transfer_replay_dir / 'source' / path) == digest(ROOT / path) == expected, path
    transfer_dir = ROOT / 'artifacts/v19/shifted_study'
    transfer_protocol = json.loads((transfer_dir / 'protocol.json').read_text())
    transfer_rows = json.loads((transfer_dir / 'episodes.json').read_text())
    transfer_analysis = json.loads((ROOT / 'artifacts/v19/shifted_analysis/analysis.json').read_text())
    assert transfer_protocol['models'] == risk['models']
    assert transfer_protocol['spec']['methods'] == risk['spec']['methods']
    assert len(transfer_rows) == transfer_analysis['actual_sumo_runs'] == 6400
    assert len(transfer_analysis['cells']) == 60
    assert transfer_analysis['weights_or_thresholds_recalibrated'] is False
    assert sum(not r['completed'] for r in transfer_rows) == 68
    assert sum(r['trace_saved'] for r in transfer_rows) == 855
    for row in transfer_rows:
        if row['plant_name'] != 'lag_strong':
            assert row['max_actual_error'] <= .5 + 1e-9 and row['error_contract_exceeded_substeps'] == 0
            if not row['controller'].startswith('raw_'):
                assert row['completed']
        else:
            assert row['error_contract_exceeded_substeps'] > 0
    assert sum(not r['completed'] for r in transfer_rows if r['plant_name'] == 'lag_strong') == 57
    result = dict(verified_assets=verified, completed_development_sumo_runs=2496,
                  additional_posthoc_sumo_replays=128,
                  additional_observable_initial_predictor_sumo_runs=120,
                  additional_synthetic_calibration_deployment_audit_runs=56,
                  completed_direct_transfer_sumo_runs=6400,
                  additional_direct_transfer_sumo_replays=336,
                  direct_transfer_noncompleted=68,
                  frozen_split_policies_identical=True,
                  final_replay_coverage_frozen=True,
                  conditional_risk_certified=False, real_time_requirement_met=False,
                  harness_strong_lag_aborts=1, scope='Manifest integrity and accounting; not a substitute for independent risk validation or replay')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    verify()
