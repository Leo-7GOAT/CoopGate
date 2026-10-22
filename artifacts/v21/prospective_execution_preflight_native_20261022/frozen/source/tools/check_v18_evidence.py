"""Check published evidence bytes and reported counts, including failed acceptance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check():
    assets = 0
    for name in ('independent_study', 'independent_analysis', 'timeout_replays'):
        folder = ROOT / 'artifacts/v18' / name
        manifest = json.loads((folder / 'sha256.json').read_text())
        assert set(manifest) == {p.relative_to(folder).as_posix() for p in folder.rglob('*')
                                 if p.is_file() and p != folder / 'sha256.json'}
        for path, digest in manifest.items():
            assert hashlib.sha256((folder / path).read_bytes()).hexdigest() == digest, path
            assets += 1
    study = ROOT / 'artifacts/v18/independent_study'
    report = json.loads((ROOT / 'artifacts/v18/independent_analysis/analysis.json').read_text())
    rows = json.loads((study / 'episodes.json').read_text())
    audit = json.loads((study / 'audit.json').read_text())
    digest = hashlib.sha256((study / 'episodes.json').read_bytes()).hexdigest()
    assert report['source_sha256']['episodes.json'] == audit['episodes_sha256'] == digest
    assert report['audit_sha256'] == hashlib.sha256((study / 'audit.json').read_bytes()).hexdigest()
    for name, r in report['results'].items():
        chosen = [row for row in rows if row['controller'] in r['members']]
        assert len(chosen) == r['runs'] and sum(a['completed'] for a in chosen) == r['completed']
        assert abs(sum(a['team_cost'] for a in chosen) / len(chosen) - r['groups']['all']['method_cost']) < 1e-10
    incomplete = [r for r in rows if not r['completed']]
    complete_condition = not incomplete and all(not r['collision'] and not r['aborted'] and not r['contract_violations'] for r in rows)
    assert complete_condition == report['checks']['all_actual_runs_complete_and_no_recorded_events']
    assert report['passed_all_mean_performance_checks'] == all(report['checks'].values())
    assert not report['disturbance_performance_risk_certified']
    failed = json.loads((ROOT / 'artifacts/v18/timeout_replays/results.json').read_text())
    assert {(r['seed'], r['disturbance']) for r in incomplete} == {(r['seed'], r['mode']) for r in failed['cases']}
    result = dict(verified_assets=assets, actual_sumo_runs=len(rows), noncompleted=len(incomplete),
                  checks_passed=sum(report['checks'].values()), checks_total=len(report['checks']),
                  all_mean_performance_checks_passed=report['passed_all_mean_performance_checks'],
                  disturbance_performance_risk_certified=False)
    print(json.dumps(result)); return result


if __name__ == '__main__': check()
