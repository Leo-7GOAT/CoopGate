"""Descriptive interpretation of all fully audited V36 regression interventions."""
import json

import numpy as np

import diagnose_recovery_regressions as study


def derive():
    io=study.io;root=study.OUT;plan=io.read(study.PLAN)
    proof=io.read(root/'verification.json');complete=io.read(root/'complete.json')
    assert proof['complete_sha256']==io.digest(root/'complete.json')
    assert proof['observed_exit']['exit_code']==0 and proof['physical_replays_verified']==16
    assert proof['offline_jobs_recomputed']==8 and proof['all_forecasts_match_physical_replays']
    for name,expected in complete['files_sha256'].items():assert io.digest(root/name)==expected
    report=io.read(root/'report.json');cases=[];jobs=[]
    for case in report['cases']:
        difference=np.asarray(case['observed_difference'])
        assert case['assessable'] and case['identity_matches']
        cases.append(dict(case=case['case'],admissions=case['admissions'],
            cost_difference=difference[:,0].tolist(),grid_finish_difference=difference[:,1].tolist(),
            command_variation_difference=((difference[:,0]-difference[:,1])/.05).tolist(),
            positive_local_contributions=case['positive_local_contributions'],
            negative_local_contributions=case['negative_local_contributions'],identity_error=case['max_absolute_error']))
    for index,job in enumerate(plan['jobs']):
        data=io.read(root/'jobs'/f'{index:02d}'/'offline.json');actual=data['actual_pair']
        assert actual['both_complete']
        delta=np.asarray(actual['difference']);reference=np.asarray(data['reference_parameter_realized_path_pair']['difference'])
        jobs.append(dict(**job,job_index=index,actual_difference=delta.tolist(),
            actual_local_harm=bool(np.any(delta>1e-9)),reference_model_realized_path_difference=reference.tolist(),
            true_parameter_bank_pass=data['true_parameter_bank_pass'],
            true_parameter_bank_maximum=data['true_parameter_bank_maximum'],
            original_online_bank_maximum=data['original_online_bank_maximum']))
    affected=[j for j in jobs if j['actual_local_harm']]
    assert plan['jobs'][5]==dict(case_index=2,step=0) and cases[2]['case'][1]=='slow_yield' and cases[2]['case'][4]=='nominal'
    nominal=io.read(root/'jobs/05/offline.json');base=nominal['actual_branches']['baseline_suffix'];candidate=nominal['actual_branches']['candidate_then_baseline']
    b,c=np.asarray(base['values']),np.asarray(candidate['values'])
    crossing_delta=np.asarray(candidate['crossing'])-base['crossing']
    variation_delta=((c[:,0]-c[:,1])-(b[:,0]-b[:,1]))/.05
    return dict(source_complete_sha256=io.digest(root/'complete.json'),source_verification_sha256=io.digest(root/'verification.json'),
        source_scope='Every known bank harm in the completed 1920-run V36 bank grid',
        selected_cases=5,distinct_source_initials=4,adoption_points=8,actual_sumo_replays=16,
        all_five_existing_v31_identities_match=True,maximum_identity_error=max(c['identity_error'] for c in cases),
        harmful_local_points=len(affected),oracle_bank_rejects_harmful_local_points=sum(not j['true_parameter_bank_pass'] for j in affected),
        oracle_bank_rejects_total_points=sum(not j['true_parameter_bank_pass'] for j in jobs),
        oracle_veto_is_not_deployable_or_closed_loop_repair=True,cases=cases,jobs=jobs,
        nominal_grid_boundary_example=dict(case=cases[2]['case'],job_index=5,vehicle=2,
            baseline_substep_interpolated_crossing=base['crossing'][2],candidate_substep_interpolated_crossing=candidate['crossing'][2],
            interpolated_crossing_difference=float(crossing_delta[2]),grid_finish_difference=float(c[2,1]-b[2,1]),
            command_variation_difference=float(variation_delta[2]),original_cost_difference=float(c[2,0]-b[2,0]),
            alternative_interpolated_cost_difference=float(crossing_delta[2]+.05*variation_delta[2]),
            original_metric_and_harm_definition_changed=False),
        posthoc_descriptive_only=True,new_independent_initials=0,new_risk_certificate=False,
        interpretation='Oracle parameters and memory reject some archived harmful points but also useful points. Even the exact nominal model passes its finite bank at the grid-boundary counterexample. These are fixed-prefix interventions, not a deployed repair or population inference.')


if __name__=='__main__':
    study.io.committed(study.ROOT/'tools/report_recovery_regression_mechanisms.py')
    result=derive();study.io.exclusive_json(study.OUT/'interpretation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','jobs')}),flush=True)
