"""Complete-stage matching setup and final inference;never partial target effects."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from coopgate.episode_thinning_control import matching_probability
import mappo_coupled_formal_design as design
import mappo_coupled_formal_experiment as experiment
from mappo_formal_common import canonical_sha256
from mappo_coupled_joint_data import METHODS,episode_outcome,certificate_from_complete_records
from mappo_risk_attempt4 import ATTEMPT_ALPHA


def complete_rows(stage):
    root=experiment.FORMAL;plan=experiment.bound(root,False)
    receipt=experiment.read(root/stage/'native_stage_receipt.json')
    if receipt['stage']!=stage or receipt['plan_sha256']!=experiment.sha(root/'plan.json') or len(receipt['stages'])!=8:
        raise ValueError('Full four-collector/four-reviewer stage receipt required')
    expected={(action,s) for action in ('collect','review') for s in range(4)}
    if {(e['action'],e['shard']) for e in receipt['stages']}!=expected or any(e['actual_native_exit_code']!=0 for e in receipt['stages']):
        raise ValueError('Stage has missing,duplicate or nonzero native exits')
    records=[]
    for shard in range(4):
        out=experiment.stage_path(root,stage,shard);proof=experiment.read(out/'verification.json')
        event=next(e for e in receipt['stages'] if e['action']=='review' and e['shard']==shard)
        if event['verification_sha256']!=experiment.sha(out/'verification.json') or not proof['all_recorded_outcomes_replay']:
            raise ValueError('Native-reviewed stage data changed')
        for case,record in experiment.reviewed_records(root,stage,shard,plan):
            records.append({k:v for k,v in record.items() if k!='trace'})
    cases=experiment.stage_cases(root,stage,plan)
    keys={(r['cluster_id'],r['method']) for r in records};wanted={(c['cluster_id'],m) for c in cases for m in c['methods']}
    if keys!=wanted or len(records)!=len(wanted):raise ValueError('Missing/duplicate full-stage observations')
    return plan,cases,records


def matching():
    plan,cases,records=complete_rows('calibration');n=3072;lookup={(r['cluster_id'],r['method']):r['row'] for r in records}
    probabilities={};counts={};errors={}
    for seed in ('4703','4704','4705'):
        target=sum(lookup[c['cluster_id'],seed]['learned_action_steps']>0 for c in cases)
        nominal=sum(lookup[c['cluster_id'],'nominal_'+seed]['learned_action_steps']>0 for c in cases)
        counts[seed]=dict(target_adopted=target,nominal_adopted=nominal,independent_calibration_initials=n)
        try:probabilities[seed]=matching_probability(target,nominal,n)
        except ValueError as exc:probabilities[seed]=None;errors[seed]=str(exc)
    root=experiment.FORMAL
    result=dict(probabilities=probabilities,counts=counts,invalid_matching=errors,all_matching_valid=not errors,
        calibration_terminal_verified=True,calibration_native_receipt_sha256=experiment.sha(root/'calibration/native_stage_receipt.json'),
        target_policy_or_threshold_changed=False,calibration_effects_used_for_target_selection=False,
        inactive_invalid_control_is_not_reported_as_matched=True,confirmation_main_runs_even_if_invalid=True)
    experiment.write(root/'matching_controls.json',result)
    coins=design.stage_cases('confirmation',probabilities)
    experiment.write(root/'confirmation_control_binding.json',dict(matching_sha256=experiment.sha(root/'matching_controls.json'),
        calibration_native_receipt_sha256=experiment.sha(root/'calibration/native_stage_receipt.json'),
        cases_with_independent_coins_sha256=canonical_sha256(coins)))
    print(json.dumps(dict(matching_setup_complete=True,all_matching_valid=not errors,probabilities=probabilities)),flush=True)


def adoption_difference_interval(target,control,alpha):
    """Simultaneous bounds on discordant Bernoulli probabilities,then subtract."""
    a=np.asarray(target,bool);b=np.asarray(control,bool)
    if a.shape!=b.shape or a.ndim!=1 or not len(a):raise ValueError('Complete paired adoption indicators required')
    n=len(a);positive=int((a&~b).sum());negative=int((~a&b).sum())
    def interval(k):
        return (float(beta.ppf(alpha/4,k,n-k+1)) if k else 0.,
                float(beta.isf(alpha/4,k+1,n-k)) if k<n else 1.)
    lp,up=interval(positive);ln,un=interval(negative)
    return dict(initials=n,target_only=positive,control_only=negative,observed_difference=(positive-negative)/n,
        lower=lp-un,upper=up-ln,pointwise_error=alpha,method='Four one-sided CP discordance bounds;independent confirmation,conditional on calibration probability',
        exact_equality_or_equivalence_proved=False)


def finalize():
    plan,cal_cases,cal=complete_rows('calibration');_,confirm_cases,confirm=complete_rows('confirmation')
    # Only the declared primary four methods enter the unchanged certificate.
    # Secondary controllers stay in the complete dataset and comparison table.
    primary_cal=[r for r in cal if r['method'] in METHODS];primary_confirm=[r for r in confirm if r['method'] in METHODS]
    result=certificate_from_complete_records(dict(stages=dict(calibration=cal_cases,confirmation=confirm_cases)),primary_cal,primary_confirm)
    lookup={(r['cluster_id'],r['method']):r for r in confirm};comparison={};matching_info=experiment.read(experiment.FORMAL/'matching_controls.json')
    for method in confirm_cases[0]['methods']:
        adopted=[];harmful=[];failed=[];gains=[];times=[]
        for c in confirm_cases:
            record=lookup[c['cluster_id'],method];row=record['row'];baseline=lookup[c['cluster_id'],'baseline']['row']
            q=episode_outcome(row);b=episode_outcome(baseline)
            h=bool(q['failed'] or b['failed'] or np.any(q['costs']>b['costs']+1e-9) or np.any(q['arrivals']>b['arrivals']+1e-9))
            adopted.append(q['adopted']);harmful.append(h and q['adopted']);failed.append(q['failed'])
            gains.append(100*(b['team_cost']-q['team_cost'])/b['team_cost']);times.extend(call['wall_ms'] for call in record['calls'])
        comparison[method]=dict(initials=1024,adopted=sum(adopted),harmful_adopted=sum(harmful),execution_failures=sum(failed),
            conditional_harm_point=sum(harmful)/sum(adopted) if any(adopted) else None,
            mean_paired_percentage_gain=float(np.mean(gains)),callbacks=len(times),callback_sum_ms=float(np.sum(times)),
            callback_p95_ms=float(np.quantile(times,.95)),callback_max_ms=max(times),callback_over_500ms=sum(t>500 for t in times))
    differences={}
    for seed in ('4703','4704','4705'):
        a=[lookup[c['cluster_id'],seed]['row']['learned_action_steps']>0 for c in confirm_cases]
        b=[lookup[c['cluster_id'],'thinned_'+seed]['row']['learned_action_steps']>0 for c in confirm_cases]
        differences[seed]=adoption_difference_interval(a,b,ATTEMPT_ALPHA/3)
    result.update(algorithm='genuine jointly trained MAPPO',candidate_sha256=plan['candidate_sha256'],common_gain_floor=.0025,
        primary_actual_sumo_executions=16384,total_actual_sumo_executions=37888,independent_initials=4096,
        actual_same_adoption_controls_executed=True,matching_probabilities_valid=matching_info['all_matching_valid'],
        old_certificate_transferred=False,all_formal_thresholds_preserved=True,
        requested_combined_evidence_complete=bool(result['joint_deployment_certified'] and matching_info['all_matching_valid']))
    out=experiment.FORMAL/'analysis';out.mkdir(exist_ok=False)
    experiment.write(out/'certificate.json',result)
    experiment.write(out/'comparisons.json',dict(confirmation_methods=comparison,adoption_differences=differences,
        calibration_matching=matching_info,matched_rate_identity_is_in_expectation_not_exact_realization=True,
        difference_intervals_simultaneous_within_three_comparisons_error=ATTEMPT_ALPHA,
        comparator_intervals_are_not_new_deployment_certificates=True,no_hard_real_time_claim=True))
    print(json.dumps(dict(formal_analysis_complete=True,joint_deployment_certified=result['joint_deployment_certified'],
        requested_combined_evidence_complete=result['requested_combined_evidence_complete'],
        primary_runs=16384,total_runs=37888,independent_initials=4096)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('matching','finalize'));a=p.parse_args()
    (matching if a.action=='matching' else finalize)()
