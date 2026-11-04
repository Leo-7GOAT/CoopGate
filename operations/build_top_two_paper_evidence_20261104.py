"""Complete-data-only paper tables and exact physical fallback evidence.

No new inference, selection, alpha, learned model or simulation. Run only after
the recovery owner's actual zero exit and full native/replay/accounting audit.
"""
from itertools import groupby
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
from mappo_completed_top_two_attempt6 import complete_evidence
import recover_mappo_top_two_attempt6 as r
from mappo_completed_powered_attempt5 import complete_evidence as old_evidence
from mappo_top_two_joint_data import episode_outcome

OUT = r.original.BASE/'paper_top_two_attempt6_evidence_20261104'


def fallback_evidence(plan):
    result=dict(effective_original_schedule_records=0,successful_zero_adoption_exact_fallbacks=0,
        nonbaseline_episodes=0,paired_observed_prefixes_checked=0,exact_prefix_steps=0,
        target_prefixes_checked=0,baseline_or_method_failure_pairs=0,
        failed_zero_adoption_episodes_not_counted_as_successful_fallback=0)
    manifest=[]
    roles={stage:{seed:dict(initials=0,adopted=0,actual_command_change_steps=[0,0,0],
        multi_role_command_change_steps=0,proposal_indices=[set(),set(),set()])
        for seed in ('4703','4704','4705')} for stage in ('calibration','confirmation')}
    for stage in ('calibration','confirmation'):
        for shard in range(4):
            records=r.original.reviewed_records(r.OUT,stage,shard,plan)
            for cid,group in groupby(records,key=lambda value:value[0]['cluster_id']):
                values=list(group);cases=[c for c,_ in values]
                if not cases or any(c!=cases[0] for c in cases):raise ValueError('Full case identity changed')
                case=cases[0];rows={record['method']:record for _,record in values}
                if set(rows)!=set(case['methods']) or len(rows)!=len(values):
                    raise ValueError('Incomplete/duplicate original method family')
                result['effective_original_schedule_records']+=len(rows)
                baseline=rows['baseline'];base=episode_outcome(baseline['row'])
                for method,record in rows.items():
                    if method=='baseline':continue
                    q=episode_outcome(record['row']);trace=record['trace'];bt=baseline['trace']
                    if method in roles[stage]:
                        role=roles[stage][method];role['initials']+=1;role['adopted']+=int(q['adopted'])
                        state=record['row']['initial']
                        for step in trace:
                            changed=[state[i][0]<60 and abs(step['command'][i]-step['baseline_command'][i])>1e-9 for i in range(3)]
                            if any(changed)!=bool(step['actual_learning_used']):
                                raise ValueError('Actual role participation differs from recorded adoption')
                            for i in range(3):
                                role['actual_command_change_steps'][i]+=int(changed[i])
                                role['proposal_indices'][i].add(int(step['agent_actions'][i]))
                            role['multi_role_command_change_steps']+=int(sum(changed)>=2)
                            state=step['x']
                    result['nonbaseline_episodes']+=1
                    failed_pair=bool(base['failed'] or q['failed'])
                    result['baseline_or_method_failure_pairs']+=int(failed_pair)
                    first=next((i for i,t in enumerate(trace) if t['actual_learning_used']),len(trace))
                    # Every actually observed pre-adoption command/state prefix
                    # must agree with the common paired baseline. Failures stay
                    # in the data; unobserved steps are not invented.
                    length=min(first,len(bt))
                    for a,b in zip(trace[:length],bt[:length]):
                        if any(a[k]!=b[k] for k in ('t','command','x')):
                            raise ValueError('Before-adoption physical prefix differs from baseline')
                    result['paired_observed_prefixes_checked']+=1
                    result['exact_prefix_steps']+=length
                    result['target_prefixes_checked']+=int(method in r.original.runtime.TARGET_METHODS)
                    if not q['adopted']:
                        if failed_pair:
                            result['failed_zero_adoption_episodes_not_counted_as_successful_fallback']+=1
                        else:
                            if len(trace)!=len(bt) or any(any(a[k]!=b[k] for k in ('t','command','x')) for a,b in zip(trace,bt)):
                                raise ValueError('Successful zero-adoption episode does not exactly follow baseline')
                            result['successful_zero_adoption_exact_fallbacks']+=1
            p=r.OUT/stage/f'shard{shard}/collection_complete.json'
            manifest.append(dict(stage=stage,shard=shard,collection_complete_sha256=r.original.sha(p)))
    if result['effective_original_schedule_records']!=59392 or result['target_prefixes_checked']!=21504:
        raise ValueError('Missing whole original effective schedule')
    result.update(all_observed_prefixes_and_successful_zero_adoption_traces_exact=True,
                  zero_observed_failures_are_not_zero_population_risk=True,
                  unseen_states_or_future_noise_not_certified=True)
    for groups in roles.values():
        for role in groups.values():role['proposal_indices']=[sorted(values) for values in role['proposal_indices']]
    return result,manifest,roles


def run():
    cert,comp=complete_evidence()  # First effect access requires actual full completion.
    old,_=old_evidence()
    plan=r.original.bound(r.OUT,False)
    fallback,manifest,roles=fallback_evidence(plan)
    rows=[]
    for seed in ('4703','4704','4705'):
        methods=comp['confirmation_methods'];mixture=cert['primary_risk']['members'][seed+':mixture']
        strong=cert['primary_risk']['members'][seed+':strong_lag']
        adoption=cert['adoption']['members'][seed];gain=cert['paired_gain']['members'][seed]
        if (roles['calibration'][seed]['initials']!=6144 or roles['confirmation'][seed]['initials']!=1024
                or roles['calibration'][seed]['adopted']!=mixture['adoptions']
                or roles['confirmation'][seed]['adopted']!=methods[seed]['adopted']):
            raise ValueError('Whole-trace role accounting differs from frozen primary results')
        rows.append(dict(actor_seed=int(seed),mixture=mixture,strong_lag=strong,
            adoption=adoption,paired_gain=gain,
            target_confirmation=methods[seed],same_pool_nominal=methods['nominal_'+seed],
            same_pool_thinned_nominal=methods['thinned_'+seed],
            shared_mps_adapter=methods['mps_'+seed],shared_mamps_adapter=methods['mamps_'+seed],
            adoption_discordance_interval=comp['adoption_differences'][seed],
            same_adoption_means_expectation_not_exact_population_or_sample_equality=True))
    prior=r.original.BASE/'development/local_top_two_fresh_analysis_20261104'
    pilot=r.original.read(prior/'report.json');owner=r.original.read(prior/'owner_native_exit.json')
    if owner['actual_native_exit_code']!=0 or owner['report_sha256']!=r.original.sha(prior/'report.json'):
        raise ValueError('Complete actual fresh pilot analysis required')
    result=dict(completed=True,algorithm='Three decentralized MAPPO actors; centralized training-only critic; joint on-policy GAE/PPO',
        candidate='TopTwoHistoryGate',models=[4703,4704,4705],common_gain_floor=.0025,
        full_primary_certificate=cert,all_confirmation_comparisons=comp,primary_model_table=rows,
        development_paired_table=pilot['groups']['all'],development_is_not_confirmation=True,
        old_reused_data_promotion_screen_still_failed=True,
        earlier_attempt5=dict(joint_deployment_certified=old['joint_deployment_certified'],
            attempt=old['attempt'],spent_error=old['allocated_error'],
            source_certificate_sha256=r.original.sha(r.original.BASE/'formal_mappo_attempt5_powered_20261103/analysis/certificate.json')),
        exact_physical_fallback=fallback,deployed_role_participation=roles,complete_raw_collection_manifest=manifest,
        retained_actual_records=642,new_actual_sumo_executions=58750,full_effective_original_schedule=59392,
        independent_initials=7168,original_root_exit_code=None,unknown_lost_inflight_executions=True,
        total_physical_executions_including_lost_attempts_not_known=True,
        scientific_source_commit=plan['source_commit'],original_registration_commit='2dc23bc91f63d877e98719b9543e5b3b29dfe5e6',
        recovery_registration_commit='1e187a2a9e9e35886547f7261463e474e4bd0185',
        source_sha256=r.original.sha(__file__),certificate_sha256=r.original.sha(r.OUT/'analysis/certificate.json'),
        comparisons_sha256=r.original.sha(r.OUT/'analysis/comparisons.json'),
        callback_times_are_actual_mixed_original_and_recovery_batches=True,
        cross_batch_timing_speedup_or_hard_realtime_claim=False,communication_delay_metric=False,
        raw_unshielded_MAPPO_or_central_critic_causal_advantage_proved=False,
        fresh_whole_change_does_not_isolate_search_integration_and_rank_two_effects=True,
        no_new_statistical_inference=True,new_error_allocated=False,
        postprocessing_sources_sha256={name:r.original.sha(ROOT/name) for name in (
            'operations/build_top_two_paper_evidence_20261104.py',
            'tools/mappo_completed_top_two_attempt6.py',
            'tools/recover_mappo_top_two_attempt6.py')})
    OUT.mkdir(exist_ok=False);r.durable(OUT/'evidence.json',result)
    print(json.dumps(dict(completed=True,evidence_sha256=r.original.sha(OUT/'evidence.json'),
        joint_deployment_certified=cert['joint_deployment_certified'],exact_fallback=fallback)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check and not (r.OUT/'orchestrator_native_receipt.json').exists():
        print(json.dumps(dict(ready=False,formal_effects_read=False,paper_artifact_created=False)))
    else:run()
