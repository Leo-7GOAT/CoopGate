"""Integrity/accounting of completed V19 development; never a risk certificate."""
import json
from pathlib import Path

from probe_disturbed_gate_v19 import digest

ROOT = Path(__file__).resolve().parents[1]


def verify():
    verified = 0
    for name in ('bank_probe', 'portfolio_probe', 'shifted_harness', 'bank_probe_analysis',
                 'portfolio_probe_analysis', 'paired_bank', 'paired_bank_analysis',
                 'composition_ablation', 'portfolio_audit', 'cached_gate_audit', 'actuator_bounds'):
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
            assert result['plan_sha256'] == digest(directory / 'plan.json')
            if name == 'portfolio_audit':
                assert result['actual_sumo_replays'] == 12
                assert result['all_strict_harm_cases_reproduced'] == 1
                assert len(result['nonvacuous_identity_cases']) == 6
                assert all(r['admissions'] > 0 and r['maximum_identity_residual'] == 0
                           for r in result['nonvacuous_identity_cases'])
            else:
                assert result['actual_sumo_replays'] == 58 and result['exact_paired_cases'] == 29
                assert result['exact_paired_decision_steps'] == 667
                assert result['internal_work_reduction'] > .20
                assert result['hard_real_time_certified'] is False
                assert result['timings']['cached']['p95_ms'] > 500
            assert result['risk_certified'] is False
        elif name == 'actuator_bounds':
            result = json.loads((directory / 'analysis.json').read_text())
            assert result['gain_drag']['acceleration_error_upper_bound'] == .5
            assert result['lag_mild']['acceleration_error_upper_bound'] < .5
            assert result['lag_strong']['acceleration_error_upper_bound'] > .5
            assert .054 < result['maximum_uniformly_covered_lag_tau_seconds'] < .055
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
    result = dict(verified_assets=verified, completed_development_sumo_runs=2496,
                  additional_posthoc_sumo_replays=70,
                  frozen_split_policies_identical=True,
                  conditional_risk_certified=False, real_time_requirement_met=False,
                  strong_lag_aborts=1, scope='Manifest integrity and accounting; not a substitute for independent risk validation or replay')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    verify()
