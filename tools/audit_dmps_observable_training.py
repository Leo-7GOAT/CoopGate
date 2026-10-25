"""Audit the fully completed training phase and preserve the failed evaluation."""
import json
from pathlib import Path

import numpy as np
import torch

import train_dmps_observable_development as run

OUT=run.ROOT/'artifacts/literature_reproduction/dmps_observable_training_development'


def verify(output=OUT):
    protocol=run.io.read(output/'protocol.json');failure=run.io.read(output/'failure.json')
    assert not (output/'complete.json').exists()
    for name,expected in protocol['source_sha256'].items():assert run.io.digest(run.ROOT/name)==expected
    assert protocol['training_cases']==run.training_cases() and protocol['evaluation_cases']==run.evaluation_cases()
    assert not protocol['privileged_forecasts'] and protocol['initialization']=='Original random Model; no prior checkpoint'
    assert failure==dict(type='RuntimeError',message='Original choice is outside nominal geometric safety',
        training_actual_decisions=256,evaluation_actual_decisions=414)
    group_reviews={};group_calls={}
    for kind in ('training','evaluation'):
        calls=[run.io.read(p) for p in sorted((output/kind).glob('call_*.json'))]
        assert [c['index'] for c in calls]==list(range(len(calls)))
        cursor=0;reviews=[]
        for index,folder in enumerate(sorted((output/kind).glob('episode_*'))):
            record=run.io.read(folder/'episode.json')
            schedule=protocol['training_cases'] if kind=='training' else protocol['evaluation_cases']
            assert record['case']==schedule[index]
            review=run.evidence.audit_episode(record)
            assert review==run.io.read(folder/'review.json')
            cumulative=0.
            for decision in record['decisions']:
                assert run.io.read(folder/f"decision_{decision['step']:03d}.json")==decision
                call=calls[cursor];cursor+=1
                state=np.asarray(decision['observation_state']);previous=np.asarray(decision['previous_command'])
                command=np.asarray(decision['trace']['command'])
                observation=np.append(np.column_stack((state[:,0]/150,state[:,1]/20,previous/5)).ravel(),
                    decision['step']/50).astype(np.float32)
                next_state=np.asarray(decision['trace']['x'])
                next_obs=np.append(np.column_stack((next_state[:,0]/150,next_state[:,1]/20,command/5)).ravel(),
                    (decision['step']+1)/50).astype(np.float32)
                np.testing.assert_array_equal(call['observation'],observation)
                np.testing.assert_array_equal(call['next_observation'],next_obs)
                assert call['executed']==decision['planner']['selected_action']
                assert call['proposed']==decision['planner']['proposal']
                assert call['intervention']==decision['planner']['intervention']
                expected_reward=-float(np.mean((state[:,0]<60)*(.5+.05*np.abs(command-previous))))
                assert call['reward']==expected_reward;cumulative+=expected_reward
            if record['terminated'] or record['truncated']:
                assert calls[cursor-1]['done']
                np.testing.assert_allclose(-cumulative,np.mean(review['per_vehicle_cost']),atol=1e-9,rtol=0)
            reviews.append(dict(case=record['case'],review=review,
                full_task_end=record['terminated'] or record['truncated'],
                episode_sha256=run.io.digest(folder/'episode.json')))
        assert cursor==len(calls)
        group_reviews[kind]=reviews;group_calls[kind]=calls
    assert len(group_calls['training'])==256 and len(group_calls['evaluation'])==414
    assert run.io.read(output/'training_episodes.json')['episodes']==group_reviews['training']
    # Rebuild all five replay arrays. Rejected proposals are synthetic absorbing
    # rows, kept separate from the subsequent actual selected-action transition.
    replay={name:[] for name in ('obs_buf','act_buf','next_obs_buf','rew_buf','done_buf')}
    def append(obs,act,nxt,reward,done):
        for name,value in zip(replay,(obs,act,nxt,reward,done)):replay[name].append(value)
    for call in group_calls['training']:
        if call['intervention']!=0:
            append(call['observation'],call['proposed'],np.zeros(10),-32.,True)
        append(call['observation'],call['executed'],call['next_observation'],call['reward'],call['done'])
    stored=np.load(output/'replay.npz')
    for name,values in replay.items():np.testing.assert_array_equal(stored[name],np.asarray(values))
    assert len(replay['obs_buf'])==261
    boundary_rows=[i for i,(o,d) in enumerate(zip(stored['next_obs_buf'],stored['done_buf'])) if o[-1]==1.]
    assert len(boundary_rows)==5 and np.all(stored['done_buf'][boundary_rows]==1.)
    initial=torch.load(output/'initial_state.pt',weights_only=True,map_location='cpu')
    final=torch.load(output/'trained_state.pt',weights_only=True,map_location='cpu')
    changes={group:float(np.sqrt(sum(float(torch.sum((final[k].double()-initial[k].double())**2))
        for k in initial if k.startswith(group+'.'))))
        for group in ('actor','critic1','critic2','actor_targ','critic1_targ','critic2_targ')}
    assert all(v>0 for v in changes.values()) and all(torch.isfinite(v).all() for v in final.values())
    optimizers=torch.load(output/'optimizers.pt',weights_only=True,map_location='cpu')
    updates={name:sorted({int(state['step']) for state in optimizer['state'].values()})
        for name,optimizer in optimizers.items()}
    assert updates==dict(actor=[54],critic1=[108],critic2=[108])
    assert len(group_reviews['evaluation'])==9 and group_reviews['evaluation'][-1]['review']['administrative_failure']
    return dict(training_phase_complete=True,overall_evaluation_complete=False,
        training_actual_decisions=256,training_physical_substeps=1280,gradient_updates=108,
        actor_updates=54,synthetic_rejected_proposals=5,replay_rows=261,
        absorbing_horizon_rows_verified=5,parameter_l2_changes=changes,optimizer_steps=updates,
        evaluation_executed_decisions=414,evaluation_schedule_conditions=10,
        evaluation_full_conditions=8,evaluation_failed_condition=8,evaluation_unexecuted_condition=9,
        group_reviews=group_reviews,privileged_forecasts=False,new_risk_certificate=False,
        comparative_efficacy_claimed=False,files_sha256={p.relative_to(output).as_posix():run.io.digest(p)
            for p in output.rglob('*') if p.is_file()})


if __name__=='__main__':
    result=verify()
    target=OUT.parent/'dmps_observable_training_audit.json'
    run.io.exclusive_json(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('group_reviews','files_sha256')}))
