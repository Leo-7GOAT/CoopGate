"""Post-exit full replay of curriculum training and the fresh three-way cohort.

Actual cost and completion determine utility. Original MCTS proposal acceptance
is recorded separately from adoption relative to the recovery reference.
"""
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.dmps_curriculum_initials import PHASES,phase_at,target_state
import train_dmps_curriculum as run

io=run.previous.base.io
OUT=run.ROOT/'artifacts/literature_reproduction/dmps_curriculum_training'
CONTRACT=OUT.parent/'dmps_curriculum_audit_contract.json'


def audit_execution_tree(output,kind,cases,*,curriculum_training=False,shadow_reference=False):
    calls=[];episodes=[];durations=[];horizon_rows=0;phase_counts={p:0 for p in PHASES}
    root=output/kind
    folders=sorted(root.glob('episode_*'),key=lambda p:int(p.name.split('_')[-1]))
    for index,folder in enumerate(folders):
        assert int(folder.name.split('_')[-1])==index
        if (folder/'reset_failure.json').exists():
            assert not curriculum_training
            assert io.read(folder/'reset_failure.json')['case']==cases[index]
            episodes.append(None);continue
        record=io.read(folder/'episode.json');assert record['case']==cases[index]
        initial=np.asarray(record['initial']);assert np.all(initial[:,0]<60)
        if curriculum_training:
            evidence=io.read(folder/'initialization.json');phase=phase_at(len(calls))
            assert evidence['phase']==phase and evidence['phase_name']==PHASES[phase]
            assert evidence['starting_actual_decisions']==len(calls)
            assert not evidence['placement_after_time_origin']
            np.testing.assert_array_equal(initial,evidence['actual_initial'])
            expected=target_state(cases[index]['seed'],phase)
            if phase==3:
                assert evidence['target_state'] is None and record['metrics']['scenario']=='nominal'
            else:
                np.testing.assert_array_equal(evidence['target_state'],expected)
                np.testing.assert_allclose(initial,expected,atol=1e-8,rtol=0)
                assert record['metrics']['scenario']=='curriculum_'+PHASES[phase]
            np.testing.assert_allclose(evidence['simulation_origin'],.1 if phase==3 else .2,atol=1e-10,rtol=0)
            assert record['failure'] is None and not record['metrics']['aborted']
            phase_counts[PHASES[phase]]+=len(record['decisions'])
            if index<len(folders)-1:assert record['terminated'] or record['truncated']
        else:
            assert not (folder/'initialization.json').exists()
            assert record['metrics']['scenario']=='nominal'
        review=run.previous.base.evidence.audit_episode(record)
        assert review==io.read(folder/'review.json')
        total_reward=0.;shadow_errors=[];shadow_differences=0
        reference=run.pairing.baseline.MemoizedRecoveryGate(None,cases[index]['plant'],'baseline') if shadow_reference else None
        for step,decision in enumerate(record['decisions']):
            assert io.read(folder/f'decision_{step:03d}.json')==decision
            call=io.read(root/f'call_{len(calls):04d}.json');assert call['index']==len(calls)
            state=np.asarray(decision['observation_state']);previous=np.asarray(decision['previous_command'])
            command=np.asarray(decision['trace']['command']);nxt=np.asarray(decision['trace']['x'])
            obs=np.append(np.column_stack((state[:,0]/150,state[:,1]/20,previous/5)).ravel(),step/50).astype(np.float32)
            next_obs=np.append(np.column_stack((nxt[:,0]/150,nxt[:,1]/20,command/5)).ravel(),(step+1)/50).astype(np.float32)
            np.testing.assert_array_equal(call['observation'],obs)
            np.testing.assert_array_equal(call['next_observation'],next_obs)
            assert call['executed']==decision['planner']['selected_action']
            assert call['proposed']==decision['planner']['proposal']
            assert call['intervention']==decision['planner']['intervention']
            reward=-float(np.mean((state[:,0]<60)*(.5+.05*np.abs(command-previous))))
            assert call['reward']==reward;total_reward+=reward
            done=step==len(record['decisions'])-1 and (record['terminated'] or record['truncated'])
            assert call['done']==done
            if curriculum_training and next_obs[-1]==1.:horizon_rows+=1;assert call['done']
            if reference is not None and not shadow_errors:
                try:
                    counterfactual,_=reference.select(state,previous)
                    shadow_differences+=int(np.any(np.abs(command[state[:,0]<60]-counterfactual[state[:,0]<60])>1e-9))
                except (RuntimeError,ValueError) as error:
                    # A reference outside its domain cannot establish zero adoption.
                    shadow_errors.append(dict(step=step,type=type(error).__name__,message=str(error)))
            durations.append(decision['wall_ms']);calls.append(call)
        if record['terminated'] or record['truncated']:
            np.testing.assert_allclose(-total_reward,np.mean(review['per_vehicle_cost']),atol=1e-8,rtol=0)
        metadata=dict(case=record['case'],review=review,full_task_end=record['terminated'] or record['truncated'],
            episode_sha256=io.digest(folder/'episode.json'))
        episodes.append(dict(metadata=metadata,capped_finish=record['capped_finish'],initial=record['initial'],
            original_proposal_accepted_steps=sum(d['planner']['intervention']==0 for d in record['decisions']),
            shadow_different_steps=shadow_differences,shadow_errors=shadow_errors,
            adoption_vs_reference=True if shadow_differences else (None if shadow_errors else False)))
    assert len(list(root.glob('call_*.json')))==len(calls)
    return dict(calls=calls,episodes=episodes,horizon_rows=horizon_rows,phase_decisions=phase_counts,
        original_planner_call_ms=dict(count=len(durations),mean=float(np.mean(durations)) if durations else None,
            p95=float(np.quantile(durations,.95)) if durations else None,max=max(durations) if durations else None))


def verify(output=OUT):
    done=io.read(output/'complete.json');plan=io.read(output/'protocol.json')
    io.committed(CONTRACT)
    contract=io.read(CONTRACT)
    assert contract['protocol_sha256']==io.digest(output/'protocol.json')
    assert contract['native_start_sha256']==io.digest(output/'native_start.json')
    for name,expected in contract['audit_sources_sha256'].items():
        io.committed(run.ROOT/name);assert io.digest(run.ROOT/name)==expected
    assert contract['frozen_before_any_test_episode'] and not contract['test_outcomes_read']
    assert io.runtime_identity()==plan['runtime']
    receipt=io.read(output/'observed_execution_exit.json')
    assert receipt['exit_code']==0 and receipt['complete_sha256']==io.digest(output/'complete.json')
    assert receipt['native_pid']==io.read(output/'native_start.json')['native_pid']
    assert plan['train_steps']==done['training_actual_decisions']==run.STEPS
    assert plan['training_cases']==run.training_cases() and plan['evaluation_cases']==run.evaluation_cases()
    assert plan['curriculum_boundaries']==[0,8192,16384,24576] and plan['phase_changes_at_reset_only']
    assert plan['evaluation_resets']=='Original nominal, never curriculum'
    assert plan['parents']==run.prerequisites() and not plan['privileged_forecasts']
    for name,expected in plan['source_sha256'].items():assert io.digest(run.ROOT/name)==expected
    for name,expected in done['files_sha256'].items():assert io.digest(output/name)==expected
    training=audit_execution_tree(output,'training',plan['training_cases'],curriculum_training=True)
    assert len(training['calls'])==run.STEPS
    metadata=[e['metadata'] for e in training['episodes']]
    assert metadata==io.read(output/'training_episodes.json')['episodes']
    replay={name:[] for name in ('obs_buf','act_buf','next_obs_buf','rew_buf','done_buf')}
    def append(obs,act,nxt,reward,flag):
        for key,value in zip(replay,(obs,act,nxt,reward,flag)):replay[key].append(value)
    for call in training['calls']:
        if call['intervention']!=0:append(call['observation'],call['proposed'],np.zeros(10),-32.,True)
        append(call['observation'],call['executed'],call['next_observation'],call['reward'],call['done'])
    stored=np.load(output/'replay.npz')
    for name,values in replay.items():np.testing.assert_array_equal(stored[name],np.asarray(values))
    assert done['replay_rows']==len(replay['obs_buf'])
    initial=torch.load(output/'initial_state.pt',map_location='cpu',weights_only=True)
    original=torch.load(run.OLD/'initial_state.pt',map_location='cpu',weights_only=True)
    final=torch.load(output/'trained_state.pt',map_location='cpu',weights_only=True)
    assert initial.keys()==original.keys()==final.keys()
    assert all(torch.equal(initial[k],original[k]) for k in initial)
    assert all(torch.isfinite(v).all() for v in final.values())
    changes={group:float(np.sqrt(sum(float(torch.sum((final[k].double()-initial[k].double())**2))
        for k in initial if k.startswith(group+'.')))) for group in
        ('actor','critic1','critic2','actor_targ','critic1_targ','critic2_targ')}
    optimizers=torch.load(output/'optimizers.pt',map_location='cpu',weights_only=True)
    updates={name:sorted({int(state['step']) for state in opt['state'].values()}) for name,opt in optimizers.items()}
    assert updates==dict(actor=[15352],critic1=[30704],critic2=[30704]) and done['gradient_updates']==30704
    evaluations={}
    for role in ('new','old'):
        data=audit_execution_tree(output,'evaluation_'+role,plan['evaluation_cases'],shadow_reference=True)
        saved=io.read(output/f'evaluation_{role}.json');results=saved['results']
        assert len(data['episodes'])==10 and [r['case'] for r in results]==plan['evaluation_cases']
        assert saved['episodes']==[e['metadata'] for e in data['episodes'] if e is not None]
        for index,(result,episode) in enumerate(zip(results,data['episodes'])):
            assert result['episode']==(episode['metadata'] if episode is not None else None)
            if episode is not None:assert result['actual_decisions']==episode['metadata']['review']['actual_decisions']
            if result['failure'] is not None:
                assert io.read(output/f'evaluation_{role}'/f'case_failure_{index:03d}.json')==result['failure']
        assert done[role+'_policy_conditions']==10
        evaluations[role]=data
    baseline_results=io.read(output/'baseline_results.json')['results']
    assert len(baseline_results)==done['baseline_conditions']==10
    paired={role:[] for role in evaluations};baseline_substeps=baseline_commands=0
    for index,(case,recorded) in enumerate(zip(plan['evaluation_cases'],baseline_results)):
        assert recorded['case']==case
        folder=output/'baseline'/f'{index:03d}';row=recorded['row']
        if row is None:
            assert recorded['failure'] and (folder/'failure.json').exists()
        else:
            record=io.read(folder/'episode.json');assert record['row']==row
            body=dict(case,cluster_id=f'curriculum_quality_{index:03d}',scenario='nominal')
            body.update(initial_sha256=run.pairing.baseline.design.shared.array_sha256(np.asarray(row['initial']),decimals=8),
                dynamics_sha256=run.pairing.baseline.design.shared.canonical_sha256(run.pairing.matched_plant_parameters(case['seed'],case['plant'])),
                disturbance_sha256=run.pairing.baseline.design.shared.array_sha256(run.pairing.disturbance_tape(case['seed'],case['disturbance'],.1,250)))
            review=run.pairing.baseline.audit_record(body,record,None)
            assert review==recorded['review']==io.read(folder/'review.json')
            baseline_substeps+=review['physical_substeps'];baseline_commands+=review['reference_replayed_commands']
        for role,data in evaluations.items():
            episode=data['episodes'][index]
            if row is not None and episode is not None:
                np.testing.assert_allclose(row['initial'],episode['initial'],atol=1e-8,rtol=0)
            learning=episode['metadata']['review'] if episode is not None else None
            finish=episode['capped_finish'] if episode is not None else None
            outcome=run.pairing.paired_outcome(row,learning,finish) if row is not None else dict(
                baseline_failed=True,learned_failed=learning is None or not learning['completed'],relative_harm=None,
                relative_performance_assessable=False,paired_cost_gain_percent=None)
            paired[role].append(dict(case=case,baseline_row=row,learning_review=learning,
                capped_finish=finish,adoption_vs_reference=episode['adoption_vs_reference'] if episode else None,
                shadow_errors=episode['shadow_errors'] if episode else [],**outcome))
    aggregate={}
    for role,rows in paired.items():
        complete=all(r['relative_performance_assessable'] for r in rows)
        before=sum(r['baseline_row']['team_cost'] for r in rows) if complete else None
        after=sum(float(np.mean(r['learning_review']['per_vehicle_cost'])) for r in rows) if complete else None
        aggregate[role]=dict(conditions=10,completed=sum(not r['learned_failed'] for r in rows),
            baseline_completed=sum(not r['baseline_failed'] for r in rows),
            harms_with_complete_baseline=sum(r['relative_harm'] is True for r in rows),
            adoption_observed=sum(r['adoption_vs_reference'] is True for r in rows),
            adoption_unassessable=sum(r['adoption_vs_reference'] is None for r in rows),
            failures_without_observed_reference_adoption=sum(r['learned_failed'] and r['adoption_vs_reference'] is not True for r in rows),
            original_proposal_accepted_steps=sum(e['original_proposal_accepted_steps'] for e in evaluations[role]['episodes'] if e is not None),
            actual_decisions=len(evaluations[role]['calls']),
            original_planner_call_ms=evaluations[role]['original_planner_call_ms'],
            all_scope_cost_gain_percent=100*(before-after)/before if complete else None)
    return dict(complete_sha256=io.digest(output/'complete.json'),observed_exit=receipt,
        audit_contract_sha256=io.digest(CONTRACT),
        all_physics_and_replay_arrays_reconstructed=True,matched_untrained_initialization=True,
        ordinary_test_resets_verified=True,all_source_and_file_hashes_verified=True,
        training_actual_decisions=run.STEPS,training_physical_substeps=5*run.STEPS,
        synthetic_penalty_rows=len(replay['obs_buf'])-run.STEPS,replay_rows=len(replay['obs_buf']),
        optimizer_steps=updates,parameter_l2_changes=changes,phase_actual_decisions=training['phase_decisions'],
        nonempty_training_episodes=sum(e['metadata']['review']['actual_decisions']>0 for e in training['episodes']),
        absorbing_horizon_rows_verified=training['horizon_rows'],
        baseline_physical_substeps=baseline_substeps,baseline_commands_replayed=baseline_commands,
        aggregate=aggregate,paired=paired,evaluation_conditions=30,independent_test_initials=2,
        new_risk_certificate=False,comparative_family_superiority_claimed=False,
        proposal_acceptance_is_reference_adoption=False,model_selection_performed=False)


if __name__=='__main__':
    torch.set_num_threads(1)
    result=verify();io.exclusive_json(OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('paired','parameter_l2_changes')}))
