"""Frozen full audit of feasible actor outputs, physical replay and MCTS choices."""
import json

import numpy as np
import torch

from coopgate.dmps_effective_action import EffectiveCriticInputs
from coopgate.dmps_evaluation_replay import replay_episode
from coopgate.dmps_feasible_actor import FeasibleActorOutputs
import train_dmps_feasible_actor as run
import verify_dmps_curriculum as physical
from verify_dmps_effective_action import reconstruct_replay,verify_hook_counts

io=run.previous.nominal.base.io


def verify_network_hooks(proof):
    verify_hook_counts(proof)
    assert proof['both_actor_outputs_hooked']
    assert proof['actor_calls']==dict(actor=46072,actor_targ=30704)


def check_proposal_feasible(decision,call):
    prev=np.asarray(decision['previous_command']);raw=-1.25+3.75*np.asarray(call['proposed'])
    lo=np.maximum(-5.,prev-2.5);hi=np.minimum(2.5,prev+2.5)
    violation=float(max(0.,np.max(lo-raw),np.max(raw-hi)))
    assert violation<=2e-6,'Noise-free feasible actor request left its command interval'
    return violation


def verify(output=run.OUT):
    plan=io.read(output/'protocol.json');done=io.read(output/'complete.json');receipt=io.read(output/'observed_execution_exit.json')
    assert receipt['exit_code']==0 and receipt['complete_sha256']==io.digest(output/'complete.json')
    assert receipt['native_pid']==io.read(output/'native_start.json')['native_pid']
    assert plan['runtime']==io.runtime_identity() and plan['parents']==run.prerequisites()
    for name,expected in plan['source_sha256'].items():
        io.committed(run.ROOT/name);assert io.digest(run.ROOT/name)==expected
    for name,expected in done['files_sha256'].items():assert io.digest(output/name)==expected
    assert plan['sole_additional_recipe_change']==run.RECIPE and plan['retained_four_effective_command_critic_inputs']
    assert plan['matched_initial_untrained_weights'] and not plan['matched_initial_policy_functions']
    assert not plan['curriculum_used'] and not plan['reward_shaping_used'] and not plan['demonstrations_used']
    assert plan['task_adapted_variant_not_original_method'] and not plan['marl_training_claimed']
    assert plan['train_steps']==done['training_actual_decisions']==run.STEPS
    assert plan['training_cases']==run.previous.nominal.training_cases() and plan['evaluation_cases']==run.evaluation_cases()
    training=physical.audit_execution_tree(output,'training',plan['training_cases'])
    assert len(training['calls'])==run.STEPS
    assert [e['metadata'] for e in training['episodes']]==io.read(output/'training_episodes.json')['episodes']
    expected=reconstruct_replay(training['calls']);stored=np.load(output/'replay.npz')
    for name,values in expected.items():np.testing.assert_array_equal(stored[name],values)
    assert done['replay_rows']==len(expected['obs_buf'])
    initial=torch.load(output/'initial_state.pt',map_location='cpu',weights_only=True)
    previous=torch.load(run.PARENT/'initial_state.pt',map_location='cpu',weights_only=True)
    final=torch.load(output/'trained_state.pt',map_location='cpu',weights_only=True)
    assert initial.keys()==previous.keys()==final.keys() and all(torch.equal(initial[k],previous[k]) for k in initial)
    assert all(torch.isfinite(v).all() for v in final.values())
    changes={group:float(np.sqrt(sum(float(torch.sum((final[k].double()-initial[k].double())**2))
        for k in initial if k.startswith(group+'.')))) for group in
        ('actor','critic1','critic2','actor_targ','critic1_targ','critic2_targ')}
    assert all(value>0 for value in changes.values())
    optimizers=torch.load(output/'optimizers.pt',map_location='cpu',weights_only=True)
    updates={name:sorted({int(s['step']) for s in opt['state'].values()}) for name,opt in optimizers.items()}
    assert updates==dict(actor=[15352],critic1=[30704],critic2=[30704]) and done['gradient_updates']==30704
    hook_proof=io.read(output/'network_hooks_training.json');verify_network_hooks(hook_proof)
    evaluation_hooks=io.read(output/'network_hooks_evaluation.json')
    assert evaluation_hooks['both_actor_outputs_hooked'] and evaluation_hooks['actor_calls']['actor_targ']==30704
    control=io.read(output/'control_network_schema.json')
    assert control['all_four_critics_hooked'] and not control['feasible_actor_outputs_hooked']
    assert control['checkpoint_sha256']==io.digest(run.PARENT/'trained_state.pt')
    source=run.ROOT.parent/'reference_sources/dmps-f37df2d'
    module,metadata=run.previous.nominal.base.author.load(source);assert metadata==plan['author_sources']
    evaluations={};decision_replays={};proposal_errors=[]
    for role in ('new','old'):
        data=physical.audit_execution_tree(output,'evaluation_'+role,plan['evaluation_cases'],shadow_reference=True)
        saved=io.read(output/f'evaluation_{role}.json');results=saved['results']
        assert len(data['episodes'])==done[role+'_policy_conditions']==10
        assert [r['case'] for r in results]==plan['evaluation_cases']
        assert saved['episodes']==[e['metadata'] for e in data['episodes'] if e is not None]
        model=module.Model(10,3,torch.device('cpu'),ret_low=-32.,ret_high=0.)
        model.load_state_dict(final if role=='new' else torch.load(run.PARENT/'trained_state.pt',map_location='cpu',weights_only=True))
        model.eval();EffectiveCriticInputs(model)
        if role=='new':FeasibleActorOutputs(model)
        role_replays=[];cursor=0
        for index,(result,ep) in enumerate(zip(results,data['episodes'])):
            assert result['episode']==(ep['metadata'] if ep else None)
            if result['failure']:assert io.read(output/f'evaluation_{role}'/f'case_failure_{index:03d}.json')==result['failure']
            if ep is None:
                role_replays.append(dict(native_reset_failed=True,no_recorded_decisions=True));continue
            assert result['actual_decisions']==ep['metadata']['review']['actual_decisions']
            record=io.read(output/f'evaluation_{role}'/f'episode_{index:03d}'/'episode.json')
            length=len(record['decisions']);calls=data['calls'][cursor:cursor+length];cursor+=length
            role_replays.append(replay_episode(record,calls,model,module.MCTSPlanner,805801+index,result['failure']))
            if role=='new':proposal_errors.extend(check_proposal_feasible(d,c) for d,c in zip(record['decisions'],calls))
        assert cursor==len(data['calls'])
        evaluations[role]=data;decision_replays[role]=role_replays
    attempted=sum(r.get('attempted_actor_decisions',0) for r in decision_replays['new'])
    assert evaluation_hooks['attempted_new_policy_decisions']==attempted
    assert evaluation_hooks['actor_calls']['actor']==46072+attempted
    baselines=io.read(output/'baseline_results.json')['results'];assert len(baselines)==done['baseline_conditions']==10
    paired={role:[] for role in evaluations};baseline_steps=baseline_commands=0
    pairing=run.previous.curriculum.pairing
    for index,(case,saved) in enumerate(zip(plan['evaluation_cases'],baselines)):
        assert saved['case']==case;folder=output/'baseline'/f'{index:03d}';row=saved['row']
        if row is None:assert saved['failure'] and (folder/'failure.json').exists()
        else:
            record=io.read(folder/'episode.json');assert record['row']==row
            body=dict(case,cluster_id=f'effective_quality_{index:03d}',scenario='nominal')
            body.update(initial_sha256=pairing.baseline.design.shared.array_sha256(np.asarray(row['initial']),decimals=8),
                dynamics_sha256=pairing.baseline.design.shared.canonical_sha256(pairing.matched_plant_parameters(case['seed'],case['plant'])),
                disturbance_sha256=pairing.baseline.design.shared.array_sha256(pairing.disturbance_tape(case['seed'],case['disturbance'],.1,250)))
            review=pairing.baseline.audit_record(body,record,None)
            assert review==saved['review']==io.read(folder/'review.json')
            baseline_steps+=review['physical_substeps'];baseline_commands+=review['reference_replayed_commands']
        for role,data in evaluations.items():
            ep=data['episodes'][index]
            if row is not None and ep:np.testing.assert_allclose(row['initial'],ep['initial'],atol=1e-8,rtol=0)
            learning=ep['metadata']['review'] if ep else None;finish=ep['capped_finish'] if ep else None
            outcome=pairing.paired_outcome(row,learning,finish) if row is not None else dict(baseline_failed=True,
                learned_failed=learning is None or not learning['completed'],relative_harm=None,
                relative_performance_assessable=False,paired_cost_gain_percent=None)
            paired[role].append(dict(case=case,baseline_row=row,learning_review=learning,capped_finish=finish,
                adoption_vs_reference=ep['adoption_vs_reference'] if ep else None,**outcome))
    aggregate={}
    for role,rows in paired.items():
        assessable=all(r['relative_performance_assessable'] for r in rows)
        before=sum(r['baseline_row']['team_cost'] for r in rows) if assessable else None
        after=sum(float(np.mean(r['learning_review']['per_vehicle_cost'])) for r in rows) if assessable else None
        aggregate[role]=dict(conditions=10,completed=sum(not r['learned_failed'] for r in rows),
            baseline_completed=sum(not r['baseline_failed'] for r in rows),
            harms_with_complete_baseline=sum(r['relative_harm'] is True for r in rows),
            adoption_observed=sum(r['adoption_vs_reference'] is True for r in rows),
            adoption_unassessable=sum(r['adoption_vs_reference'] is None for r in rows),
            all_scope_cost_gain_percent=100*(before-after)/before if assessable else None)
    return dict(complete_sha256=io.digest(output/'complete.json'),observed_exit=receipt,
        all_physics_and_replay_arrays_reconstructed=True,all_source_and_file_hashes_verified=True,
        matched_untrained_initialization=True,training_actual_decisions=run.STEPS,training_physical_substeps=run.STEPS*5,
        replay_rows=done['replay_rows'],synthetic_penalty_rows=done['replay_rows']-run.STEPS,
        optimizer_steps=updates,parameter_l2_changes=changes,network_hooks_training=hook_proof,
        policy_and_mcts_replays=decision_replays,all_recorded_policy_decisions_replayed=True,
        maximum_new_actor_request_constraint_violation=max(proposal_errors,default=None),
        baseline_physical_substeps=baseline_steps,baseline_commands_replayed=baseline_commands,
        aggregate=aggregate,paired=paired,evaluation_conditions=30,independent_test_initials=2,
        task_adapted_variant_not_original_method=True,new_risk_certificate=False,
        comparative_family_superiority_claimed=False,marl_training_claimed=False)


if __name__=='__main__':
    torch.set_num_threads(1);result=verify();io.exclusive_json(run.OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('paired','parameter_l2_changes','policy_and_mcts_replays')}),flush=True)
