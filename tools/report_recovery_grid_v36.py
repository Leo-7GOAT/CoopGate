"""Derive descriptive paired comparisons from the completed, hashed V36 report.

This does not run experiments, tune a gate, or issue an inferential certificate.
Run only after the complete recovery and independent replay have succeeded.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v38/interrupted_block_grid_recovery/v36'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def failed(row):
    return bool(not row['completed'] or row['aborted'] or row['collision']
                or row['contract_violations'])


def summarize(rows):
    adopted = sum(r['adopted'] for r in rows)
    harms = sum(r['adopted'] and r['relative_harm'] for r in rows)
    all_complete = all(not failed(r['row']) and not failed(r['baseline']) for r in rows)
    before = sum(r['baseline']['team_cost'] for r in rows)
    after = sum(r['row']['team_cost'] for r in rows)
    return dict(runs=len(rows), completed=sum(not failed(r['row']) for r in rows),
                adopted=adopted, adoption_fraction=adopted / len(rows),
                harmful_and_adopted=harms, joint_harm_fraction=harms / len(rows),
                empirical_conditional_harm=harms / adopted if adopted else None,
                all_scope_cost_gain_percent=100 * (before-after) / before if all_complete else None)


def derive():
    path = OUT / 'analysis/report.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    proof, report = read(OUT / 'completion_verification.json'), read(path)
    assert digest == proof['report_sha256'] == read(OUT / 'analysis/sha256.json')['report']
    assert proof['recovery_exit_code'] == 0 and proof['independent_reviews_replayed']
    assert report['all_physical_and_full_gate_replays_match']
    assert report['actual_sumo_runs'] == len(report['cases']) == 8320
    assert report['new_independent_initials'] == 0 and not report['new_risk_certificate']
    expected = read(ROOT / 'artifacts/v36/recovery_gate_complete_grid/plan.json')
    schedule = {tuple(case) for group in expected['groups'] for case in group['cases']}
    actual = {tuple(case['case']) for case in report['cases']}
    assert len(actual) == 8320 and actual == schedule
    # Recompute outcome flags from realized, paired rows rather than treating
    # the report's aggregate cells as evidence of their own correctness.
    for case in report['cases']:
        row, baseline = case['row'], case['baseline']
        assert not failed(baseline)
        harm = failed(row) or any(a-b > 1e-9 for key in ('per_vehicle_cost', 'per_vehicle_finish')
                                  for a, b in zip(row[key], baseline[key]))
        assert harm == case['relative_harm']
        assert (row['learned_action_steps'] > 0) == case['adopted']
        if not case['adopted'] and not failed(row):
            for key in ('per_vehicle_cost', 'per_vehicle_finish'):
                assert row[key] == baseline[key]
    rules = sorted({r['row']['rule'] for r in report['cases']})
    aggregate = {rule: summarize([r for r in report['cases'] if r['row']['rule'] == rule])
                 for rule in rules}
    for rule, result in aggregate.items():
        cell = next(c for c in report['cells'] if c['rule'] == rule and c['scene'] is None
                    and c['plant'] is None and c['model'] == ('none' if rule == 'baseline' else None))
        for key in ('runs', 'completed', 'adopted', 'all_scope_cost_gain_percent'):
            assert result[key] == cell[key]
        assert result['harmful_and_adopted'] == cell['harmful_adopted']
    learned = {rule: {tuple(r['case'][:5]) + (r['case'][-1],): r for r in report['cases']
                      if r['row']['rule'] == rule}
               for rule in ('bank_performance', 'nominal_performance')}
    bank, nominal = learned['bank_performance'], learned['nominal_performance']
    assert bank.keys() == nominal.keys() and len(bank) == 1920
    contingency = {f'bank_{b}_nominal_{n}': 0 for b in ('harm', 'no_harm') for n in ('harm', 'no_harm')}
    for key in bank:
        b, n = bank[key], nominal[key]
        assert b['baseline'] == n['baseline']
        label = lambda x: 'harm' if x['adopted'] and x['relative_harm'] else 'no_harm'
        contingency[f'bank_{label(b)}_nominal_{label(n)}'] += 1
    strata = {}
    for field in ('scenario', 'plant_name', 'model'):
        strata[field] = {
            value: {rule: summarize([r for r in learned[rule].values() if r['row'][field] == value])
                    for rule in learned}
            for value in sorted({r['row'][field] for r in bank.values()})}
    harms = [{key: r[key] for key in ('case', 'per_vehicle_cost_difference',
                                     'per_vehicle_time_difference', 'controller_failed')}
             for r in bank.values() if r['relative_harm']]
    return dict(source_report_sha256=digest, actual_sumo_runs=8320,
                reused_independent_initials=160, new_independent_initials=0,
                new_risk_certificate=False, aggregate=aggregate, strata=strata,
                paired_harm_contingency=contingency, bank_harm_cases=harms,
                interpretation='Descriptive complete frozen grid. Model/plant repeats are not independent. '
                'Paired whole-policy outcomes do not isolate an action-level gate effect. '
                'Different adopted subsets must not be compared as a randomized treatment effect.')


if __name__ == '__main__':
    result = derive()
    target = OUT / 'analysis/interpretation.json'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(target), paired_harm_contingency=result['paired_harm_contingency'],
                          aggregate=result['aggregate']), ensure_ascii=False))
