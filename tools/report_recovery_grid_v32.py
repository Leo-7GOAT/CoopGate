"""Read-only descriptive verification of the completed V32 prediction study."""
import json
from pathlib import Path

import numpy as np

import extend_online_actuator_v32 as original
import recover_block_grids_v38 as recovery

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v38/interrupted_block_grid_recovery/v32'


def failed(row):
    return bool(not row['completed'] or row['aborted'] or row['collision'] or row['contract_violations'])


def aggregate(cases):
    complete=[r for r in cases if not r['controller_failed'] and not r['baseline_failed']]
    adopted=[r for r in cases if r['adopted']]
    comparable=[r for r in adopted if not r['baseline_failed']]
    harms=sum(bool(r['relative_harm']) for r in comparable)
    before=sum(r['baseline_team_cost'] for r in complete)
    gain=100*(before-sum(r['row']['team_cost'] for r in complete))/before if before else None
    times=[c['wall_ms'] for r in cases for c in r['calls']]
    return dict(runs=len(cases),completed=sum(not r['controller_failed'] for r in cases),
        baseline_complete=sum(not r['baseline_failed'] for r in cases),adopted=len(adopted),
        comparable_adopted=len(comparable),harmful_and_adopted=harms,
        empirical_conditional_harm=harms/len(comparable) if comparable else None,
        adopted_with_incomplete_baseline=len(adopted)-len(comparable),
        paired_complete=len(complete),all_scope_cost_gain_percent=gain if len(complete)==len(cases) else None,
        complete_pairs_descriptive_gain_percent=gain,
        out_of_physical_contract=sum(r['row']['error_contract_exceeded_substeps']>0 for r in cases),
        callback_count=len(times),callback_sum_ms=sum(times),callback_over_500ms=sum(t>500 for t in times),
        callback_p95_ms=float(np.quantile(times,.95)) if times else None,callback_max_ms=max(times) if times else None)


def derive():
    plan,source_plan=recovery.inputs('v32')
    receipt,_,_=recovery.validate_recovery_exit('v32',plan)
    report=recovery.read(OUT/'analysis/report.json');proof=recovery.read(OUT/'completion_verification.json')
    report_hash=recovery.digest(OUT/'analysis/report.json')
    assert report_hash==proof['report_sha256']==recovery.read(OUT/'analysis/sha256.json')['report']
    assert report['all_physical_and_input_reviews_pass'] and proof['independent_reviews_replayed']
    assert receipt['exit_code']==0 and report['actual_sumo_runs']==3840
    assert report['new_independent_initials']==0 and not report['new_risk_certificate']
    expected={tuple(c) for g in source_plan['groups'] for c in g['cases']}
    assert {tuple(c['case']) for c in report['cases']}==expected and len(report['cases'])==len(expected)==3840
    for group in source_plan['groups']:
        folder,_,_=recovery.complete_directory('v32',plan,group)
        recovery.verify_complete_block('v32',group,folder)
        assert recovery.complete_block_hashes(folder)==proof['block_files_sha256'][f"{group['block_index']:05d}"]
    baselines=original.baselines()
    baseline_cases={}
    for case in report['cases']:
        key=tuple(case['case']);row=case['row'];base=baselines[(*key[:-2],'baseline')]
        baseline_cases[key[:-2]]=base
        base_failed=failed(base);row_failed=failed(row)
        harm=None if base_failed else row_failed or any(a-b>1e-9 for field in ('per_vehicle_cost','per_vehicle_finish')
                                                       for a,b in zip(row[field],base[field]))
        assert case['baseline_failed']==base_failed and case['controller_failed']==row_failed
        assert case['relative_harm']==harm and case['baseline_team_cost']==base['team_cost']
        assert case['adopted']==(case['review']['actual_admitted_steps']>0)
        assert case['relative_performance_assessable']==(not base_failed and not row_failed)
        if case['relative_performance_assessable']:
            assert case['paired_cost_gain_percent']==100*(base['team_cost']-row['team_cost'])/base['team_cost']
        else:assert case['paired_cost_gain_percent'] is None
    assert len(baseline_cases)==640
    for cell in report['plant_cells']:
        computed=aggregate([r for r in report['cases'] if r['case'][4]==cell['plant'] and r['case'][-1]==cell['kind']])
        for name,value in cell.items():
            if name not in ('plant','kind'):assert computed[name]==value,(name,computed[name],value)
    results={kind:aggregate([r for r in report['cases'] if r['case'][-1]==kind]) for kind in original.KINDS}
    return dict(source_report_sha256=report_hash,all_640_block_hashes_verified=True,
        original_baseline_runs=640,baseline_completed=sum(not failed(r) for r in baseline_cases.values()),
        aggregate=results,plant_cells=report['plant_cells'],new_independent_initials=0,
        reused_independent_initials=160,new_risk_certificate=False,
        interpretation='Old baseline and predictor variants; no cross-version ranking against V36. '
        'Failed pairs remain in denominators and prohibit all-scope gain. '
        'Neither this grid nor model/plant repetitions create fresh risk evidence.')


if __name__=='__main__':
    result=derive()
    recovery.exclusive_json(OUT/'analysis/interpretation.json',result)
    print(json.dumps(dict(aggregate=result['aggregate'],baseline_completed=result['baseline_completed'])))
