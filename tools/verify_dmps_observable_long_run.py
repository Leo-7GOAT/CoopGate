"""Post-exit audit of all longer-run data and failure-preserving evaluation.

Requires an observation of the actual tool exit, saved by the waiting caller;
the presence of a complete.json file alone is insufficient for this audit.
"""
import json
from pathlib import Path

import numpy as np
import torch

import train_dmps_observable_long_run as run

OUT=run.ROOT/'artifacts/literature_reproduction/dmps_observable_long_run'


def verify(output=OUT):
    io=run.base.io
    done=io.read(output/'complete.json');plan=io.read(output/'protocol.json')
    exit_proof=io.read(output/'observed_execution_exit.json')
    assert exit_proof['exit_code']==0
    assert exit_proof['native_pid']==io.read(output/'native_start.json')['native_pid']
    assert exit_proof['complete_sha256']==io.digest(output/'complete.json')
    assert done['training_actual_decisions']==plan['train_steps']==run.STEPS
    assert done['evaluation_conditions']==10 and not done['privileged_forecasts']
    assert plan['training_cases']==run.training_cases() and plan['evaluation_cases']==run.evaluation_cases()
    assert plan['parent_training_audit_sha256']==run.prerequisites()
    for name,expected in plan['source_sha256'].items():assert io.digest(run.ROOT/name)==expected
    for name,expected in done['files_sha256'].items():assert io.digest(output/name)==expected
    all_calls={};all_reviews={};times={};horizon_rows=0
    for kind in ('training','evaluation'):
        calls=[];reviews=[];durations=[]
        folders=sorted((output/kind).glob('episode_*'),key=lambda p:int(p.name.split('_')[-1]))
        schedule=plan['training_cases'] if kind=='training' else plan['evaluation_cases']
        for index,folder in enumerate(folders):
            assert int(folder.name.split('_')[-1])==index
            if (folder/'reset_failure.json').exists():
                assert kind=='evaluation'
                assert io.read(folder/'reset_failure.json')['case']==schedule[index]
                reviews.append(None);continue
            record=io.read(folder/'episode.json');assert record['case']==schedule[index]
            review=run.base.evidence.audit_episode(record)
            assert review==io.read(folder/'review.json');total_reward=0.
            for j,decision in enumerate(record['decisions']):
                assert io.read(folder/f'decision_{j:03d}.json')==decision
                call=io.read(output/kind/f'call_{len(calls):04d}.json')
                assert call['index']==len(calls)
                state=np.asarray(decision['observation_state']);previous=np.asarray(decision['previous_command'])
                command=np.asarray(decision['trace']['command']);nxt=np.asarray(decision['trace']['x'])
                observation=np.append(np.column_stack((state[:,0]/150,state[:,1]/20,previous/5)).ravel(),j/50).astype(np.float32)
                next_obs=np.append(np.column_stack((nxt[:,0]/150,nxt[:,1]/20,command/5)).ravel(),(j+1)/50).astype(np.float32)
                np.testing.assert_array_equal(call['observation'],observation)
                np.testing.assert_array_equal(call['next_observation'],next_obs)
                assert call['executed']==decision['planner']['selected_action']
                assert call['proposed']==decision['planner']['proposal']
                assert call['intervention']==decision['planner']['intervention']
                reward=-float(np.mean((state[:,0]<60)*(.5+.05*np.abs(command-previous))))
                assert call['reward']==reward;total_reward+=reward
                expected_done=j==len(record['decisions'])-1 and (record['terminated'] or record['truncated'])
                assert call['done']==expected_done
                if kind=='training' and next_obs[-1]==1.:
                    horizon_rows+=1;assert call['done']
                durations.append(decision['wall_ms']);calls.append(call)
            if record['terminated'] or record['truncated']:
                np.testing.assert_allclose(-total_reward,np.mean(review['per_vehicle_cost']),atol=1e-8,rtol=0)
            reviews.append(dict(case=record['case'],review=review,
                full_task_end=record['terminated'] or record['truncated'],
                episode_sha256=io.digest(folder/'episode.json')))
        assert len(list((output/kind).glob('call_*.json')))==len(calls)
        all_calls[kind]=calls;all_reviews[kind]=reviews;times[kind]=dict(count=len(durations),
            mean_ms=float(np.mean(durations)),p95_ms=float(np.quantile(durations,.95)),max_ms=max(durations))
    assert all_reviews['training']==io.read(output/'training_episodes.json')['episodes']
    assert len(all_calls['training'])==run.STEPS
    eval_file=io.read(output/'evaluation_episodes.json');results=eval_file['results']
    assert [r['case'] for r in results]==run.evaluation_cases() and len(all_reviews['evaluation'])==10
    for index,(result,review) in enumerate(zip(results,all_reviews['evaluation'])):
        assert result['episode']==review
        if review is not None:assert result['actual_decisions']==review['review']['actual_decisions']
        if result['failure'] is not None:
            assert io.read(output/f'case_failure_{index:03d}.json')==result['failure']
    assert sum(r is not None for r in all_reviews['evaluation'])==len(eval_file['episodes'])
    assert done['evaluation_actual_decisions']==len(all_calls['evaluation'])
    assert done['evaluation_failures']==sum(r['failure'] is not None for r in results)
    replay={name:[] for name in ('obs_buf','act_buf','next_obs_buf','rew_buf','done_buf')}
    def append(obs,act,nxt,reward,done_flag):
        for name,value in zip(replay,(obs,act,nxt,reward,done_flag)):replay[name].append(value)
    for call in all_calls['training']:
        if call['intervention']!=0:append(call['observation'],call['proposed'],np.zeros(10),-32.,True)
        append(call['observation'],call['executed'],call['next_observation'],call['reward'],call['done'])
    stored=np.load(output/'replay.npz')
    for name,values in replay.items():np.testing.assert_array_equal(stored[name],np.asarray(values))
    assert len(replay['obs_buf'])==done['replay_rows']
    initial=torch.load(output/'initial_state.pt',weights_only=True,map_location='cpu')
    final=torch.load(output/'trained_state.pt',weights_only=True,map_location='cpu')
    changes={group:float(np.sqrt(sum(float(torch.sum((final[k].double()-initial[k].double())**2))
        for k in initial if k.startswith(group+'.')))) for group in
        ('actor','critic1','critic2','actor_targ','critic1_targ','critic2_targ')}
    assert all(v>0 for v in changes.values()) and all(torch.isfinite(v).all() for v in final.values())
    optimizers=torch.load(output/'optimizers.pt',weights_only=True,map_location='cpu')
    updates={name:sorted({int(state['step']) for state in optimizer['state'].values()})
        for name,optimizer in optimizers.items()}
    assert updates==dict(actor=[15352],critic1=[30704],critic2=[30704])
    assert done['gradient_updates']==30704
    evaluation=[dict(case=r['case'],failure=r['failure'],actual_decisions=r['actual_decisions'],
        review=r['episode']['review'] if r['episode'] else None) for r in results]
    return dict(complete_sha256=io.digest(output/'complete.json'),observed_exit=exit_proof,
        source_and_file_hashes_verified=True,all_physics_and_replay_arrays_reconstructed=True,
        training_actual_decisions=run.STEPS,training_physical_substeps=5*run.STEPS,
        synthetic_rejected_proposals=len(replay['obs_buf'])-run.STEPS,replay_rows=len(replay['obs_buf']),
        absorbing_horizon_rows_verified=horizon_rows,optimizer_steps=updates,parameter_l2_changes=changes,
        evaluation=evaluation,evaluation_completed=sum(r['review'] is not None and r['review']['completed'] for r in evaluation),
        evaluation_failures=sum(r['failure'] is not None for r in evaluation),
        model_selection_performed=False,privileged_forecasts=False,new_risk_certificate=False,
        comparative_efficacy_claimed=False,selector_timings=times)


if __name__=='__main__':
    result=verify()
    run.base.io.exclusive_json(OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('evaluation','parameter_l2_changes')}))
