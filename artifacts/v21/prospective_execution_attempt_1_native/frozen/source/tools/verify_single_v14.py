"""Recompute the prerequisite, audit complete trajectories, and replay SUMO."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig,transition_safe
from coopgate.robust_v9 import robust_certificate
from coopgate.disturbance_v9 import DisturbedMergeEnv,disturbance_tape
from coopgate.ego_v13 import ego_observations,EgoPolicyGate
from coopgate.policy_iteration_v14 import load_policy,IteratedPolicy
from coopgate.learned_rollout_v13 import actor_action
from coopgate.marl_v12 import action_mask,LEVELS,project_joint
from coopgate.single_v12 import constant_proposal
from experiment_ego_v13 import run_episode
from study_single_v14 import factories
from report_single_v14 import summarize
from verify_single_v12 import equivalent
from develop_v12 import initial as regenerate_initial

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/v14/single'


def verify(replay=False,require=False):
    torch.set_num_threads(1)
    p=json.loads((D/'protocol.json').read_text());training=json.loads((D/'training_protocol.json').read_text())
    assert hashlib.sha256((D/'training_protocol.json').read_bytes()).hexdigest()==p['training_protocol_sha256']
    assert p['initials']==training['test_initials'] and p['criteria']==training['criteria']
    assert p['bootstrap']==training['bootstrap'] and p['training_seeds']==training['training_seeds']
    assert len(p['initials'])==len({r['seed'] for r in p['initials']})==512
    assert all(45000000<=r['seed']<46000000 for r in p['initials'])
    assert p['primary_methods']==training['primary_methods'] and p['secondary_methods']==training['secondary_methods']
    assert p['disturbances']==training['disturbances']
    assert p['methods']==['rollout']+training['primary_methods']+training['secondary_methods']
    assert p['expected_runs']==len(p['initials'])*len(p['methods'])*len(p['disturbances'])
    for name,digest in p['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    if (D/'sha256.json').exists():
        for name,digest in json.loads((D/'sha256.json').read_text()).items():
            assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
    models={};parameter_changes={}
    for seed in p['training_seeds']:
        root=D/f'seed_{seed}'
        initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=True)
        model,meta=load_policy(root/'policy.pt')
        assert meta['seed']==seed+training['phase_b']['seed_offset']
        assert meta['averaging_window']==training['final_policy']['trailing_iterations']
        for phase in ('phase_a','phase_b'):
            conf=training[phase];record=json.loads((root/phase/'metadata.json').read_text())
            with (root/phase/'training.csv').open() as file:history=list(csv.DictReader(file))
            assert len(history)==conf['iterations']==record['iteration']
            assert record['visited_states']==conf['iterations']*conf['states']
            assert record['latest_only']==conf['latest_only']
            assert record['epochs']==conf['epochs'] and record['width']==training['width']
            assert record['seed']==seed+(training['phase_b']['seed_offset'] if phase=='phase_b' else 0)
            assert record['teacher_sha256']==training['teacher_sha256']
            if phase=='phase_b':assert record['warmstart_sha256']==hashlib.sha256((root/'phase_a/policy.pt').read_bytes()).hexdigest()
            assert record['evaluated_model_transitions']>record['visited_states']
            assert all(int(r['visited_states'])==(i+1)*conf['states'] for i,r in enumerate(history))
        last=training['phase_b']['iterations'];window=training['final_policy']['trailing_iterations']
        parents=[torch.load(root/'phase_b'/f'iteration_{i}.pt',map_location='cpu',weights_only=True)['model'] for i in range(last-window+1,last+1)]
        assert list(meta['averaging_parents'].values())==[hashlib.sha256((root/'phase_b'/f'iteration_{i}.pt').read_bytes()).hexdigest() for i in range(last-window+1,last+1)]
        for key,value in model.state_dict().items():
            average=torch.stack([state[key].double() for state in parents]).mean(0).to(value.dtype)
            assert torch.equal(value,average),key
        norms={part:sum(float((value-initial['model'][key]).square().sum()) for key,value in model.state_dict().items() if key.startswith(part))
               for part in ('encoder','advantage')}
        assert all(value>1e-6 for value in norms.values());parameter_changes[seed]=norms
        models[f'learned_{seed}']=model;models[f'last_{seed}']=load_policy(root/'last.pt')[0]
    fs=factories(p)
    probe=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for name in p['primary_methods']:
        instance=fs[name](probe)
        assert type(instance) is EgoPolicyGate and type(instance.policy) is IteratedPolicy
    rows=json.loads((D/'study/episodes.json').read_text());summary=summarize(p,rows)
    manifest=json.loads((D/'study/manifest.json').read_text())
    assert manifest['protocol_sha256']==hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest()
    assert manifest['sumo_runs']==len(rows) and manifest['single_worker']
    equivalent(summary,json.loads((D/'summary.json').read_text()))
    initial_records=json.loads((D/'study/initials.json').read_text())
    initials={r['seed']:np.array(r['state']) for r in initial_records}
    assert len(initials)==len(initial_records)==len(p['initials'])
    for item in p['initials']:
        np.testing.assert_allclose(initials[item['seed']],regenerate_initial(item['seed'],item['regime']),atol=1e-8,rtol=0)
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    c=MergeConfig();packed=parameters(c);audited=compared=0
    for index in p['audit_initial_indices']:
        item=p['initials'][index]
        for mode in p['disturbances']:
            tape=disturbance_tape(item['seed'],mode,.5,250)
            for method in p['methods']:
                trace=json.loads((D/f'study/trace_{index}_{mode}_{method}.json').read_text())
                state=initials[item['seed']].copy();previous=np.zeros(3);variation=np.zeros(3)
                finish=np.full(3,25.);active=np.ones(3,bool)
                for step,record in enumerate(trace):
                    assert record['t']==(step+1)*.5
                    np.testing.assert_array_equal(record['disturbance'],tape[step*5:(step+1)*5])
                    command=np.array(record['command'])
                    if method in models:
                        model=models[method]
                        selected=int(model.proposal(ego_observations(state,previous,step)[1:2],action_mask(previous)[1:2])[0])
                        fast=actor_action(state,previous,step,model.compiled_weights(),True,2)
                        proposed=constant_proposal(previous,packed);proposed[1]=fast
                        if state[1,0]>=c.goal:proposed[1]=min(c.accel,previous[1]+2.5)
                        compiled,_,ok=project_joint(state,previous,proposed,packed,.5)
                        assert ok
                        np.testing.assert_allclose(compiled,command,atol=1e-9,rtol=0)
                        assert selected==record['agent_actions'][1];compared+=1
                    assert robust_certificate(state,previous,command,packed,.5)
                    assert np.max(np.abs(command-previous))<=2.5+1e-9
                    variation+=active*np.abs(command-previous)
                    for error in record['disturbance']:
                        error=np.array(error);assert np.max(np.abs(error))<=.5+1e-9
                        old=state[:,0].copy();state[:,1]=np.clip(state[:,1]+(command+error)*.1,0,20);state[:,0]+=state[:,1]*.1
                        assert transition_safe(old[None],state[:,0][None],c)[0]
                    observed=np.array(record['x']);np.testing.assert_allclose(state,observed,atol=1e-8,rtol=0);state=observed
                    newly=active&(state[:,0]>=60);finish[newly]=record['t'];active[newly]=False;previous=command
                row=lookup[item['seed'],mode,method]
                np.testing.assert_allclose(variation,row['per_vehicle_variation'],atol=1e-9,rtol=0)
                np.testing.assert_allclose(finish,row['per_vehicle_finish'],atol=1e-9,rtol=0);audited+=1
    replays=0
    if replay:
        with DisturbedMergeEnv() as env:
            for index in p['replay_initial_indices']:
                item=p['initials'][index]
                for mode in p['disturbances']:
                    for method,factory in fs.items():
                        result,trace,_=run_episode(env,item['seed'],item['regime'],mode,factory,True)
                        expected=lookup[item['seed'],mode,method]
                        for key in ('ego_cost','ego_time','ego_variation','other_time','completed','collision','contract_violations','aborted'):
                            equivalent(result[key],expected[key])
                        saved=json.loads((D/f'study/trace_{index}_{mode}_{method}.json').read_text())
                        np.testing.assert_allclose([r['command'] for r in trace],[r['command'] for r in saved],atol=1e-8,rtol=0);replays+=1
    result=dict(validated_sumo_runs=len(rows),audited_traces=audited,compiled_actor_comparisons=compared,
                sumo_replays=replays,training_parameter_changes=parameter_changes,acceptance=summary['acceptance'],
                individual_ego_verified=summary['individual_ego_verified'],
                protocol_sha256=hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest(),
                episodes_sha256=hashlib.sha256((D/'study/episodes.json').read_bytes()).hexdigest())
    print(json.dumps(result),flush=True)
    if require and not summary['individual_ego_verified']:raise SystemExit('Frozen single-ego prerequisite is unmet')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');p.add_argument('--require-acceptance',action='store_true')
    p.add_argument('--out');a=p.parse_args();result=verify(a.replay,a.require_acceptance)
    if a.out:Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
