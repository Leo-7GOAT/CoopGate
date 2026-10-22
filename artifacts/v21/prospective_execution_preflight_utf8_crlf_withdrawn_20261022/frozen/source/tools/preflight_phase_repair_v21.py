"""Actual SUMO integration checks on one already-used development scene."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from coopgate.disturbance_v9 import DisturbedMergeEnv
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest,save
from resume_frozen_risk_v19 import build_record
from start_final_v19 import require_committed,verify_manifest
from study_gate_v19 import read_protocol,regime_for
from study_phase_repair_v21 import MODELS,REFERENCE_KEYS,ROOT,factories,methods,phase_record_check
from study_repair_v20 import trace_metadata,verify_trace

OLD=ROOT/'artifacts/v20/repair_development'
OUTPUT=ROOT/'artifacts/v21/phase_preflight'


def run():
    torch.set_num_threads(1)
    if OUTPUT.exists():raise FileExistsError('Preserve the preflight result')
    verify_manifest(OLD)
    _,old=read_protocol(OLD)
    names=('src/coopgate/phase_repair_v21.py','tools/study_phase_repair_v21.py',
           'tools/preflight_phase_repair_v21.py','tests/test_phase_repair_v21.py','tests/test_phase_study_v21.py')
    require_committed([ROOT/name for name in names])
    OUTPUT.mkdir(parents=True)
    sources=dict(old['source_sha256'])
    for path in (ROOT/'src/coopgate').glob('*.py'):
        sources[path.relative_to(ROOT).as_posix()]=digest(path)
    for name in names:sources[name]=digest(ROOT/name)
    for name in sources:
        target=OUTPUT/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    index,mode=14,'pulse'
    seed=old['spec']['seed_base']+index
    spec=dict(methods=methods(),all_traces_saved=True,audit_initial_indices=[])
    protocol=dict(models=old['models'],spec=spec)
    original={(r['initial_index'],r['disturbance'],r['controller']):r for r in json.loads((OLD/'episodes.json').read_text())}
    plan=dict(seed=seed,scenario=regime_for(old['spec'],index),mode=mode,methods=spec['methods'],
        source_sha256=sources,old_manifest_sha256=digest(OLD/'sha256.json'),
        expected_actual_sumo_runs=13,selection='One previously diagnosed pulse-harm initial; integration only',
        additional_independent_initials=0,new_study_initials_used=False,risk_certified=False)
    save(OUTPUT/'plan.json',plan)
    fs=factories(protocol)
    rows,traces=[],{}
    with DisturbedMergeEnv() as env:
        for name in spec['methods']:
            row,trace,initial=run_episode(env,seed,plan['scenario'],mode,fs[name],True)
            row,adopted,checks,failures=build_record(row,trace,initial,0,name,spec['methods'][name],spec)
            row.update(trace_metadata(trace))
            save(OUTPUT/f'trace_{name}.json',trace)
            rows.append(row)
            traces[name]=trace
            save(OUTPUT/'episodes.json',rows)
            verify_trace(trace,row,spec['methods'][name],adopted,checks,failures)
            for item in trace:phase_record_check(item,spec['methods'][name])
            if name=='baseline' or name.startswith('repair_'):
                archived=original[index,mode,name]
                for key in REFERENCE_KEYS:
                    np.testing.assert_equal(row[key],archived[key],err_msg='Prior SUMO result differs: '+key)
            if not row['completed'] or row['collision'] or row['aborted'] or row['contract_violations']:
                raise ValueError('Preflight physical execution did not complete successfully; retain its record')
            print(json.dumps(dict(completed_sumo_runs=len(rows),expected_sumo_runs=13)),flush=True)
    lookup={r['controller']:r for r in rows}
    for model in MODELS:
        old_name,ref=f'repair_{model}',f'reference_{model}'
        for key in REFERENCE_KEYS:np.testing.assert_equal(lookup[old_name][key],lookup[ref][key])
        if len(traces[old_name])!=len(traces[ref]):raise ValueError('Reference lengths differ')
        for old_item,new_item in zip(traces[old_name],traces[ref]):
            for key in ('command','x','disturbance','selected_bank_differences','repair_weight','composition'):
                np.testing.assert_equal(old_item[key],new_item[key])
    save(OUTPUT/'audit.json',dict(actual_sumo_runs=13,completed_runs=13,reference_pairs=3,
        reproduced_prior_original_results=4,new_study_initials_used=False,additional_independent_initials=0,
        all_original_and_reference_checks_pass=True,all_admitted_phase_records_checked=True,
        risk_certified=False,population_efficacy_claim=False,plan_sha256=digest(OUTPUT/'plan.json')))
    save(OUTPUT/'sha256.json',{p.relative_to(OUTPUT).as_posix():digest(p) for p in OUTPUT.rglob('*') if p.is_file() and p.name!='sha256.json'})
    print(json.dumps(dict(preflight_passed=True,actual_sumo_runs=13)),flush=True)


if __name__=='__main__':run()
