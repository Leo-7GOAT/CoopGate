"""Verify the telescoping baseline identity over the entire completed transfer scope."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from coopgate.episode_store_v21 import observations
from coopgate.performance_identity_v31 import audit_identity
from coopgate.shifted_sumo_v19 import PLANTS
from execute_prospective_v23 import exclusive_json, sources_committed, utc
from profile_phase_runtime_v24 import committed, digest
import audit_interventions_v31 as intervention
import audit_actuator_prediction_v30 as baseline_audit
import replay_interventions_v31 as repeated

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v31/performance_identity'
SOURCES = ('src/coopgate/performance_identity_v31.py','tools/check_performance_identity_v31.py',
           'tests/test_performance_identity_v31.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze():
    repeated.configure()
    sources_committed([ROOT/name for name in SOURCES])
    OUT.mkdir(parents=True,exist_ok=False)
    exclusive_json(OUT/'plan.json',dict(utc=utc(),source_sha256={p:digest(ROOT/p) for p in SOURCES},
        intervention_plan_sha256=digest(intervention.OUT/'plan.json'),
        baseline_plan_sha256=digest(baseline_audit.OUT/'plan.json'),
        baseline_complete_report_sha256=digest(baseline_audit.OUT/'analysis/report.json'),
        source_episodes=1920,all_admissions=349,
        identity='For complete admissible continuations on the same realized external path, sum every single-action baseline advantage over the actual learning trajectory. Compare with its full original per-vehicle cost/time difference from the initial baseline.',
        failures='Retain every source episode and all incomplete/unsafe reasons. Do not interpret capped values as successful improvements.',
        source_scope='All old five-scene/four-plant/three-model episodes, including zero-adoption episodes and all failures.',
        tolerance=1e-8,new_sumo_runs=0,new_independent_initials=0,new_risk_certificate=False,
        active_intervention_effects_read=False,active_calibration_effects_read=False))
    print(json.dumps(dict(plan_sha256=digest(OUT/'plan.json'),source_episodes=1920,all_admissions=349)),flush=True)


def run():
    repeated.verify()
    committed(OUT/'plan.json')
    plan = read(OUT/'plan.json')
    for path,expected in plan['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen identity code changed')
    sources_committed([ROOT/name for name in SOURCES])
    ip,bp = intervention.inputs(),baseline_audit.inputs()
    if (digest(intervention.OUT/'plan.json') != plan['intervention_plan_sha256']
            or digest(baseline_audit.OUT/'plan.json') != plan['baseline_plan_sha256']
            or digest(baseline_audit.OUT/'analysis/report.json') != plan['baseline_complete_report_sha256']):
        raise ValueError('Frozen input study changed')
    # No reading of unsealed or partial outcome collections.
    receipt = read(intervention.OUT/'execution/observation.json')
    done = read(intervention.OUT/'execution/collection_complete.json')
    start = read(intervention.OUT/'execution/start.json')
    if (receipt['exit_code'] != 0 or receipt['native_pid'] != start['native_pid']
            or done['native_pid'] != start['native_pid'] or done['actual_sumo_runs'] != 698
            or done['admissions'] != 349 or any(r['plan_sha256'] != plan['intervention_plan_sha256'] for r in (receipt,done,start))):
        raise ValueError('All real interventions and an actual successful exit are required')
    report_path = intervention.OUT/'analysis/report.json'
    if digest(report_path) != read(intervention.OUT/'analysis/sha256.json')['report']:
        raise ValueError('Completed intervention analysis changed')
    evidence = read(report_path)
    if not evidence['all_forecasts_match'] or evidence['admissions'] != 349 or evidence['actual_sumo_runs'] != 698:
        raise ValueError('All predicted interventions must have complete actual SUMO verification')
    local = defaultdict(list)
    all_keys = []
    for entry in evidence['cases']:
        key = entry['case']
        all_keys.append(key)
        local[tuple(key[:-1])].append(dict(step=key[-1],pair=entry['actual_model_pair']))
    if all_keys != ip['admission_cases']:
        raise ValueError('Every original intervention must be retained once')
    baseline = {key:record['row'] for key,record,_ in observations(
        baseline_audit.OUT/'inputs.sqlite',bp['input_metadata'],bp['input_cases'])}
    cases = []
    for key,record,_ in observations(intervention.OUT/'inputs.sqlite',ip['input_metadata'],ip['input_cases']):
        result = audit_identity(record['row'],baseline[(*key[:-1],'baseline')],local[key])
        cases.append(dict(case=list(key),**result))
    if len(cases) != 1920 or sum(c['admissions'] for c in cases) != 349:
        raise ValueError('The entire original scope is required')
    by_plant = {}
    for plant in PLANTS:
        rows = [c for c in cases if c['case'][4] == plant]
        applicable = [c for c in rows if c['assessable']]
        by_plant[plant] = dict(source_episodes=len(rows),assessable=len(applicable),
            unavailable_reasons=dict(Counter(reason for c in rows for reason in c['reasons'])),
            matched=sum(c['identity_matches'] for c in applicable),
            max_absolute_error=max((c['max_absolute_error'] for c in applicable),default=None),
            multiple_admission_episodes=sum(c['admissions'] > 1 for c in applicable),
            locally_nonworsening=sum(c['all_local_nonworsening'] for c in applicable),
            globally_nonworsening=sum(c['episode_nonworsening'] for c in applicable))
    result = dict(source_episodes=1920,admissions=349,by_plant=by_plant,cases=cases,
        all_applicable_identities_match=all(c['identity_matches'] for c in cases if c['assessable']),
        unavailable_episodes=sum(not c['assessable'] for c in cases),
        new_sumo_runs=0,new_independent_initials=0,new_risk_certificate=False,
        plan_sha256=digest(OUT/'plan.json'),intervention_report_sha256=digest(report_path),
        interpretation='A complete, post-hoc telescoping identity on old realized paths. It connects local interventions to full-episode outcomes but supplies neither online future information nor probabilistic coverage.')
    analysis = OUT/'analysis'
    analysis.mkdir(exist_ok=False)
    exclusive_json(analysis/'report.json',result)
    exclusive_json(analysis/'sha256.json',dict(report=digest(analysis/'report.json')))
    print(json.dumps(dict(all_applicable_identities_match=result['all_applicable_identities_match'],
        unavailable_episodes=result['unavailable_episodes'],by_plant=by_plant)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'))
    args = parser.parse_args()
    globals()[args.action]()
