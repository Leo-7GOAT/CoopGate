"""Repeat the whole frozen intervention scope after a one-ULP validation defect.

The unsuccessful first execution and its 404 observations remain unchanged.
All 698 continuations are executed again; repetitions are not new initials.
"""
import argparse
import json
from pathlib import Path

from coopgate.actuator_prediction_roundoff_v31 import predict_suffix
from coopgate.episode_store_v21 import inventory
from execute_prospective_v23 import exclusive_json, utc
from profile_phase_runtime_v24 import committed, digest
import audit_interventions_v31 as original

ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT/'artifacts/v31/actuator_interventions'
OUT = ROOT/'artifacts/v31/actuator_interventions_roundoff'
ADDED = ('src/coopgate/actuator_prediction_roundoff_v31.py','tools/replay_interventions_v31.py',
         'tests/test_actuator_roundoff_v31.py')


def configure():
    original.OUT = OUT
    original.predict_suffix = predict_suffix
    original.SOURCES = tuple(dict.fromkeys((*original.SOURCES,*ADDED)))


def failure_snapshot():
    receipt = original.read(FIRST/'execution/observation.json')
    if receipt['exit_code'] != 1 or receipt['native_pid'] != 46388 or receipt['completed_sumo_runs'] != 404:
        raise ValueError('The original actual unsuccessful exit must be preserved')
    files = ('plan.json','inputs.sqlite','inputs.sqlite.seal.json','execution/observation.json',
             'offline/observations.sqlite','replays/observations.sqlite')
    retained = {stage:inventory(FIRST/stage/'observations.sqlite') for stage in ('offline','replays')}
    if len(retained['offline']['cases']) != 202 or len(retained['replays']['cases']) != 404:
        raise ValueError('Preserve every observation from the original attempt')
    return dict(source_directory=str(FIRST.relative_to(ROOT)),
        source_sha256={name:digest(FIRST/name) for name in files},
        original_actual_sumo_runs=404,original_admissions_completed=202,
        original_failure='One-ULP bound excess: actuator state 2.5000000000000004 was rejected.',
        failing_case=[99,'asymmetric_speed',3,63403004,'lag_mild','phase05_1812',5],
        new_full_execution_runs=698,total_execution_accounting=1102,new_independent_initials=0,
        change='Only validation accepts an eight-ULP endpoint slack. The stored actuator value and every dynamics equation remain unchanged. No old outputs are overwritten or silently resumed.',
        retained_record_identities={stage:data['cases'] for stage,data in retained.items()})


def freeze():
    snapshot = failure_snapshot()
    configure()
    original.freeze()
    for name in ('inputs.sqlite','inputs.sqlite.seal.json'):
        if digest(OUT/name) != digest(FIRST/name):
            raise ValueError('The repeated experiment must retain byte-identical complete inputs')
    if original.read(OUT/'plan.json')['admission_cases'] != original.read(FIRST/'plan.json')['admission_cases']:
        raise ValueError('The original all-admission schedule must not change')
    exclusive_json(OUT/'repeat.json',dict(utc=utc(),plan_sha256=digest(OUT/'plan.json'),**snapshot))


def verify():
    configure()
    committed(OUT/'repeat.json')
    protocol = original.read(OUT/'repeat.json')
    if protocol['plan_sha256'] != digest(OUT/'plan.json'):
        raise ValueError('Repeated execution plan changed')
    fresh = failure_snapshot()
    for key,value in fresh.items():
        if protocol[key] != value:
            raise ValueError('Original failed attempt or recovery rule changed: '+key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','summarize'))
    args = parser.parse_args()
    if args.action == 'freeze': freeze()
    else:
        verify()
        getattr(original,args.action)()
