"""Integrity/accounting of completed V19 development; never a risk certificate."""
import json
from pathlib import Path

from probe_disturbed_gate_v19 import digest

ROOT = Path(__file__).resolve().parents[1]


def verify():
    verified = 0
    for name in ('bank_probe', 'portfolio_probe', 'shifted_harness', 'bank_probe_analysis', 'portfolio_probe_analysis'):
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
    result = dict(verified_assets=verified, completed_development_sumo_runs=1984,
                  conditional_risk_certified=False, real_time_requirement_met=False,
                  strong_lag_aborts=1, scope='Manifest integrity and accounting; not a substitute for independent risk validation or replay')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    verify()
