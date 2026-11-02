# derived unchanged full-outcome audit source sha256=4045216dcf0a4e759b9196b6ae855d2f8357c11da69969ddbd27c82246f36060
"""Recompute full completed outcomes and retain EVERY target false admission.

Descriptive diagnosis AFTER the completed frozen attempt4 study. No new SUMO,training,
threshold selection,error allocation or replacement deployment certificate.
"""
import json
import math
from pathlib import Path
import numpy as np
import mappo_coupled_formal_experiment as e
from mappo_completed_attempt4 import complete_evidence

OUT=e.BASE/'formal_mappo_attempt4_diagnosis_20261102'
SEEDS=('4703','4704','4705')


def failed(row):return bool(not row['completed'] or row['aborted'] or row['collision'] or row['contract_violations'])


def equal_motion(a,b):
    return len(a)==len(b) and all(x['t']==y['t'] and np.array_equal(x['command'],y['command']) and np.array_equal(x['x'],y['x']) for x,y in zip(a,b))


def run():
    if OUT.exists():raise FileExistsError('Retain original diagnosis')
    cert,comp=complete_evidence();plan=e.bound(e.FORMAL,False)
    summaries={};issues=[];records_seen=cases_seen=zero_equal=prefix_equal=thinning_equal=0
    def add(target,method,row,harm,adopted,gain):
        r=target.setdefault(method,dict(initials=0,adopted=0,harmful_adopted=0,execution_failures=0,gain_sum=0.))
        r['initials']+=1;r['adopted']+=int(adopted);r['harmful_adopted']+=int(harm and adopted)
        r['execution_failures']+=int(failed(row));r['gain_sum']+=float(gain)
    for stage in ('calibration','confirmation'):
        totals={};scenes={};plants={};pending={};completed=set()
        for shard in range(4):
            for case,record in e.reviewed_records(e.FORMAL,stage,shard,plan):
                records_seen+=1;key=case['cluster_id'];group=pending.setdefault(key,{})
                if record['method'] in group:raise ValueError('Duplicate diagnostic record')
                group[record['method']]=record
                if len(group)!=len(case['methods']):continue
                if key in completed:raise ValueError('Duplicate completed initial')
                completed.add(key);cases_seen+=1;base=group['baseline'];b=base['row']
                for method,r in group.items():
                    row=r['row'];adopted=row['learned_action_steps']>0
                    dc=np.asarray(row['per_vehicle_cost'])-b['per_vehicle_cost']
                    dt=np.asarray(row['per_vehicle_finish'])-b['per_vehicle_finish']
                    harm=failed(row) or failed(b) or bool((dc>1e-9).any() or (dt>1e-9).any())
                    gain=100*(b['team_cost']-row['team_cost'])/b['team_cost']
                    add(totals,method,row,harm,adopted,gain)
                    add(scenes.setdefault(case['scenario'],{}),method,row,harm,adopted,gain)
                    add(plants.setdefault(case['plant'],{}),method,row,harm,adopted,gain)
                    if method in SEEDS:
                        if not adopted and not failed(row) and not failed(b):
                            if not equal_motion(r['trace'],base['trace']):raise ValueError('Successful zero adoption differs physically from baseline')
                            zero_equal+=1
                        first=next((i for i,t in enumerate(r['trace']) if t['actual_learning_used']),len(r['trace']))
                        if not equal_motion(r['trace'][:first],base['trace'][:first]):raise ValueError('Pre-first-adoption prefix differs from baseline')
                        prefix_equal+=1
                        if harm and adopted:
                            admissions=[]
                            for i,t in enumerate(r['trace']):
                                if t['actual_learning_used']:
                                    admissions.append(dict(step=i,time_before=.5*i,state_before=row['initial'] if i==0 else r['trace'][i-1]['x'],
                                        previous=[0.,0.,0.] if i==0 else r['trace'][i-1]['command'],command=t['command'],baseline_command=t['baseline_command'],
                                        gain_score=t['selected_gain_confidence_score'],selected_forecast_differences=t['selected_bank_differences']))
                            issues.append(dict(stage=stage,cluster_id=key,seed=case['seed'],scenario=case['scenario'],plant=case['plant'],disturbance=case['disturbance'],
                                method=method,cost_regression=dc.tolist(),arrival_regression=dt.tolist(),mean_paired_percentage_gain=gain,
                                harmful_vehicle_indices=np.flatnonzero((dc>1e-9)|(dt>1e-9)).tolist(),actual_adoptions=row['learned_action_steps'],admissions=admissions,
                                source_shard=shard,source_block=case['index']//32,actual_plant_offline_only=row['plant']))
                    if method.startswith('thinned_'):
                        seed=method.split('_')[-1];chosen='nominal_'+seed if case['controller_coins'][seed] else 'baseline'
                        if not equal_motion(r['trace'],group[chosen]['trace']):raise ValueError('Actual randomized branch differs from its separately executed matched method')
                        thinning_equal+=1
                del pending[key]
            print(json.dumps(dict(stage=stage,shard=shard,records=records_seen,paired_initials=cases_seen)),flush=True)
        if pending or len(completed)!=(3072 if stage=='calibration' else 1024):raise ValueError('Incomplete full diagnostic scope')
        summaries[stage]=dict(methods=totals,by_scene=scenes,by_plant=plants)
    if records_seen!=37888 or cases_seen!=4096 or not 0<=zero_equal<=12288 or prefix_equal!=12288 or thinning_equal!=3072:
        raise ValueError('Original scope or exact fallback/branch equivalence differs from expected complete counts')
    for seed in SEEDS:
        r=summaries['calibration']['methods'][seed];old=cert['primary_risk']['members'][seed+':mixture']
        if r['adopted']!=old['adoptions'] or r['harmful_adopted']!=old['harmful_adoptions']:raise ValueError('Original calibration harm/adoption differs')
    for method,r in summaries['confirmation']['methods'].items():
        old=comp['confirmation_methods'][method]
        if any(r[k]!=old[k] for k in ('initials','adopted','harmful_adopted','execution_failures')) or not math.isclose(r['gain_sum']/r['initials'],old['mean_paired_percentage_gain'],rel_tol=0,abs_tol=1e-12):
            raise ValueError('Original complete confirmation statistics differ')
    for stage in summaries.values():
        for group in [stage['methods'],*stage['by_scene'].values(),*stage['by_plant'].values()]:
            for r in group.values():r['mean_paired_percentage_gain']=r.pop('gain_sum')/r['initials']
    unique={(x['stage'],x['cluster_id']) for x in issues}
    OUT.mkdir(exist_ok=False)
    e.write(OUT/'all_false_admissions.json',dict(all_target_harmful_adoptions=issues,method_initial_incidents=len(issues),independent_adverse_initials=len(unique),offline_diagnosis_only=True))
    e.write(OUT/'report.json',dict(complete_statistics_independently_match=True,original_records=records_seen,independent_initials=cases_seen,
        successful_zero_adoption_physical_trajectories_exactly_equal=zero_equal,all_primary_pre_first_adoption_prefixes_exact=prefix_equal,
        actual_thinned_branches_equal_separate_nominal_or_baseline=thinning_equal,by_stage=summaries,
        source_certificate_sha256=e.sha(e.FORMAL/'analysis/certificate.json'),source_comparisons_sha256=e.sha(e.FORMAL/'analysis/comparisons.json'),
        joint_deployment_certified=cert['joint_deployment_certified'],new_sumo_runs=0,new_independent_initials=0,new_inference=False,new_error_allocated=False,
        data_reuse='Post-completion diagnosis of full attempt4;any future candidate requires separate untouched validation and unused allocation'))
    print(json.dumps(dict(records=records_seen,initials=cases_seen,exact_zero_adoption=zero_equal,exact_prefixes=prefix_equal,exact_thinned_branches=thinning_equal,
        harmful_method_initial_incidents=len(issues),independent_adverse_initials=len(unique))),flush=True)


if __name__=='__main__':run()
