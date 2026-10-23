"""Engineering identity of the first sealed full-grid block, not partial efficacy."""
import json
from pathlib import Path

from coopgate.episode_store_v21 import observations
from execute_prospective_v23 import exclusive_json, utc
from profile_phase_runtime_v24 import digest
import extend_recovery_gate_v36 as extension
import study_recovery_gate_v35 as preflight

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v36/launch_checks'


def clean(record,trace):
    return dict(row={k:v for k,v in record['row'].items() if k not in ('setup_ms','controller_ms')},
        calls=[c['error'] for c in record['calls']],
        trace=[{k:v for k,v in item.items() if k != 'gate_ms'} for item in trace])


def main():
    plan = extension.inputs()
    group = plan['groups'][0]
    source_plan = preflight.inputs()
    matches = [g for g in source_plan['groups'] if g['condition'] == group['condition']]
    if len(matches) != 1:
        raise ValueError('First full-grid block must have exactly one preflight counterpart')
    prior = matches[0]
    path = extension.OUT/'blocks'/f"{group['block_index']:05d}"/'observations.sqlite'
    source = preflight.OUT/'blocks'/f"{prior['block_index']:05d}"/'observations.sqlite'
    old = {key:(row,trace) for key,row,trace in observations(source,preflight.metadata(prior),prior['cases'])}
    rows,commands = [],0
    for key,record,trace in observations(path,extension.metadata(group),group['cases']):
        before,old_trace = old[key]
        if clean(record,trace) != clean(before,old_trace):
            raise ValueError('First sealed repeated execution differs beyond timing: '+str(key))
        commands += len(trace)
        rows.append(dict(case=list(key),commands=len(trace),all_non_timing_fields_exact=True))
    if len(rows) != 13:
        raise ValueError('All thirteen method/model combinations are required')
    OUT.mkdir(exist_ok=False)
    report = dict(utc=utc(),plan_sha256=digest(extension.OUT/'plan.json'),
        preflight_plan_sha256=digest(preflight.OUT/'plan.json'),checked_actual_runs=13,
        commands=commands,cases=rows,all_non_timing_fields_exact=True,
        database_sha256=digest(path),preflight_database_sha256=digest(source),
        script_sha256=digest(Path(__file__)),new_sumo_runs=0,new_independent_initials=0,
        partial_full_grid_effectiveness_analyzed=False,
        scope='Only implementation identity on the first complete sealed block, with the full earlier integration as reference. No efficacy aggregation, threshold change or new certification.')
    exclusive_json(OUT/'first_block.json',report)
    print(json.dumps(dict(checked_actual_runs=13,commands=commands,all_non_timing_fields_exact=True)),flush=True)


if __name__ == '__main__':
    main()
