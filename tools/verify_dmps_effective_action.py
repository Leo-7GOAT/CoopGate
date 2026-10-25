"""Full post-exit audit, frozen with the effective-input training recipe."""
import json

import numpy as np
import torch

from coopgate.dmps_effective_action import EffectiveCriticInputs,effective_action
from coopgate.dmps_nominal_model import command_for
from coopgate.merge_v4 import MergeConfig
import train_dmps_effective_action as run
import verify_dmps_curriculum as physical

io=run.nominal.base.io


def reconstruct_replay(calls):
    result={name:[] for name in ('obs_buf','act_buf','next_obs_buf','rew_buf','done_buf')}
    def append(obs,act,nxt,reward,done):
        for name,value in zip(result,(obs,act,nxt,reward,done)):result[name].append(value)
    for call in calls:
        if call['intervention']!=0:append(call['observation'],call['proposed'],np.zeros(10),-32.,True)
        append(call['observation'],call['executed'],call['next_observation'],call['reward'],call['done'])
    return {name:np.asarray(values) for name,values in result.items()}


def verify_hook_counts(proof):
    assert proof['optimizer_parameter_identity_unchanged'] and proof['all_four_critics_hooked']
    counts=proof['critic_calls']
    assert set(counts)==set(EffectiveCriticInputs.names)
    assert counts['critic2']==counts['critic1_targ']==counts['critic2_targ']==30704
    assert counts['critic1']>=30704+15352


def verify(output=run.OUT):
    plan=io.read(output/'protocol.json');done=io.read(output/'complete.json')
    receipt=io.read(output/'observed_execution_exit.json')
    assert receipt['exit_code']==0 and receipt['complete_sha256']==io.digest(output/'complete.json')
    assert receipt['native_pid']==io.read(output/'native_start.json')['native_pid']
    assert plan['runtime']==io.runtime_identity() and plan['parents']==run.prerequisites()
    for name,expected in plan['source_sha256'].items():
        io.committed(run.ROOT/name);assert io.digest(run.ROOT/name)==expected
    for name,expected in done['files_sha256'].items():assert io.digest(output/name)==expected
    assert plan['sole_recipe_change']=='Observable effective-command inputs for all four critics'
    assert not plan['curriculum_used'] and not plan['reward_shaping_used'] and not plan['demonstrations_used']
    assert plan['matched_initial_untrained_weights'] and plan['task_adapted_variant_not_original_method']
    assert plan['train_steps']==done['training_actual_decisions']==run.STEPS
    assert plan['training_cases']==run.nominal.training_cases() and plan['evaluation_cases']==run.evaluation_cases()
    training=physical.audit_execution_tree(output,'training',plan['training_cases'])
    assert len(training['calls'])==run.STEPS
    assert [e['metadata'] for e in training['episodes']]==io.read(output/'training_episodes.json')['episodes']
    expected=reconstruct_replay(training['calls']);stored=np.load(output/'replay.npz')
    for name,values in expected.items():np.testing.assert_array_equal(stored[name],values)
    assert done['replay_rows']==len(expected['obs_buf'])
    initial=torch.load(output/'initial_state.pt',map_location='cpu',weights_only=True)
    previous=torch.load(run.OLD/'initial_state.pt',map_location='cpu',weights_only=True)
    final=torch.load(output/'trained_state.pt',map_location='cpu',weights_only=True)
    assert initial.keys()==previous.keys()==final.keys()
    assert all(torch.equal(initial[k],previous[k]) for k in initial)
    assert all(torch.isfinite(v).all() for v in final.values())
    changes={group:float(np.sqrt(sum(float(torch.sum((final[k].double()-initial[k].double())**2))
        for k in initial if k.startswith(group+'.')))) for group in
        ('actor','critic1','critic2','actor_targ','critic1_targ','critic2_targ')}
    assert all(v>0 for v in changes.values())
    optimizers=torch.load(output/'optimizers.pt',map_location='cpu',weights_only=True)
    updates={name:sorted({int(state['step']) for state in opt['state'].values()}) for name,opt in optimizers.items()}
    assert updates==dict(actor=[15352],critic1=[30704],critic2=[30704]) and done['gradient_updates']==30704
    hooks=io.read(output/'effective_inputs_training.json');verify_hook_counts(hooks)
    evaluation_hooks=io.read(output/'effective_inputs_evaluation.json')
    assert evaluation_hooks['all_four_critics_hooked']
    assert all(evaluation_hooks['critic_calls'][k]>=v for k,v in hooks['critic_calls'].items())
    evaluations={}
    for role in ('new','old'):
        data=physical.audit_execution_tree(output,'evaluation_'+role,plan['evaluation_cases'],shadow_reference=True)
        saved=io.read(output/f'evaluation_{role}.json');results=saved['results']
        assert len(data['episodes'])==done[role+'_policy_conditions']==10
        assert [r['case'] for r in results]==plan['evaluation_cases']
        assert saved['episodes']==[e['metadata'] for e in data['episodes'] if e is not None]
        for index,(result,episode) in enumerate(zip(results,data['episodes'])):
            assert result['episode']==(episode['metadata'] if episode else None)
            if episode:assert result['actual_decisions']==episode['metadata']['review']['actual_decisions']
            if result['failure']:assert io.read(output/f'evaluation_{role}'/f'case_failure_{index:03d}.json')==result['failure']
        evaluations[role]=data
    # Independent numeric check of the observable decoding, not an added safety
    # certificate. The conservative float32 goal band is deliberately retained.
    command_errors=[];config=MergeConfig()
    for call in training['calls']+evaluations['new']['calls']:
        obs=torch.tensor(call['observation'],dtype=torch.float32)
        a=np.asarray(call['executed']);matrix=np.asarray(call['observation'])[:9].reshape(3,3)
        state=matrix[:,:2]*np.array([150.,20.]);previous=matrix[:,2]*5
        tensor_command=(-1.25+3.75*effective_action(obs,torch.tensor(a,dtype=torch.float32))).numpy()
        # Compare at the SAME reconstructed observable state; near-goal exact
        # float32 equality intentionally keeps the regular (not cruise) map.
        state[np.asarray(obs[(0,3,6),])==np.float32(.4),0]=np.nextafter(60.,-np.inf)
        expected_command=command_for(a,state,previous,config)
        command_errors.append(float(np.max(np.abs(tensor_command-expected_command))))
    assert max(command_errors)<=2e-6
    baselines=io.read(output/'baseline_results.json')['results'];paired={role:[] for role in evaluations}
    baseline_steps=baseline_commands=0
    assert len(baselines)==done['baseline_conditions']==10
    pairing=run.curriculum.pairing
    for index,(case,recorded) in enumerate(zip(plan['evaluation_cases'],baselines)):
        assert recorded['case']==case;folder=output/'baseline'/f'{index:03d}';row=recorded['row']
        if row is None:assert recorded['failure'] and (folder/'failure.json').exists()
        else:
            record=io.read(folder/'episode.json');assert record['row']==row
            body=dict(case,cluster_id=f'effective_quality_{index:03d}',scenario='nominal')
            body.update(initial_sha256=pairing.baseline.design.shared.array_sha256(np.asarray(row['initial']),decimals=8),
                dynamics_sha256=pairing.baseline.design.shared.canonical_sha256(pairing.matched_plant_parameters(case['seed'],case['plant'])),
                disturbance_sha256=pairing.baseline.design.shared.array_sha256(pairing.disturbance_tape(case['seed'],case['disturbance'],.1,250)))
            review=pairing.baseline.audit_record(body,record,None)
            assert review==recorded['review']==io.read(folder/'review.json')
            baseline_steps+=review['physical_substeps'];baseline_commands+=review['reference_replayed_commands']
        for role,data in evaluations.items():
            ep=data['episodes'][index]
            if row is not None and ep:np.testing.assert_allclose(row['initial'],ep['initial'],atol=1e-8,rtol=0)
            learning=ep['metadata']['review'] if ep else None;finish=ep['capped_finish'] if ep else None
            outcome=pairing.paired_outcome(row,learning,finish) if row is not None else dict(
                baseline_failed=True,learned_failed=learning is None or not learning['completed'],relative_harm=None,
                relative_performance_assessable=False,paired_cost_gain_percent=None)
            paired[role].append(dict(case=case,baseline_row=row,learning_review=learning,capped_finish=finish,
                adoption_vs_reference=ep['adoption_vs_reference'] if ep else None,**outcome))
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
            all_scope_cost_gain_percent=100*(before-after)/before if complete else None)
    return dict(complete_sha256=io.digest(output/'complete.json'),observed_exit=receipt,
        all_physics_and_replay_arrays_reconstructed=True,all_source_and_file_hashes_verified=True,
        matched_untrained_initialization=True,training_actual_decisions=run.STEPS,
        training_physical_substeps=run.STEPS*5,replay_rows=done['replay_rows'],
        synthetic_penalty_rows=done['replay_rows']-run.STEPS,optimizer_steps=updates,parameter_l2_changes=changes,
        effective_inputs_training=hooks,maximum_observable_command_map_error=max(command_errors),
        baseline_physical_substeps=baseline_steps,baseline_commands_replayed=baseline_commands,
        aggregate=aggregate,paired=paired,evaluation_conditions=30,independent_test_initials=2,
        task_adapted_variant_not_original_method=True,new_risk_certificate=False,
        comparative_family_superiority_claimed=False)


if __name__=='__main__':
    torch.set_num_threads(1);result=verify();io.exclusive_json(run.OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('paired','parameter_l2_changes')}))
