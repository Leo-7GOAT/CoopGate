"""Audit coupled training updates, recalculate branch targets, replay SUMO."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from coopgate.cooperative_iteration_v16 import load_coupled,unilateral_returns
from coopgate.ego_v13 import ego_observations
from coopgate.marl_v12 import action_mask
from coopgate.disturbance_v9 import DisturbedMergeEnv,disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig,transition_safe
from coopgate.robust_v9 import robust_certificate
from validate_cooperative_v16 import factories
from develop_v12 import initial as regenerate_initial
from experiment_ego_v13 import run_episode
from verify_single_v12 import equivalent

ROOT=Path(__file__).resolve().parents[1]


def audit_training(path):
    path=Path(path);folder=path.parent;model,meta=load_coupled(path)
    assert meta['updates']==meta['target_updates']==meta['cycles']*3
    if (folder/'sha256.json').exists():
        for name,digest in json.loads((folder/'sha256.json').read_text()).items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
    for name,digest in meta['source_sha256'].items():
        assert hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()==digest,name
    before,initial=load_coupled(folder/'initial.pt');target_samples=0
    if meta.get('warmstart'):
        parent=ROOT/meta['warmstart'];assert hashlib.sha256(parent.read_bytes()).hexdigest()==meta['warmstart_sha256']
        inherited=torch.load(parent,weights_only=True,map_location='cpu')['model']
        for key,value in before.state_dict().items():assert torch.equal(value,inherited[key])
    else:
        frozen_single=torch.load(ROOT/f'artifacts/v14/single/seed_{meta["ego_seed"]}/policy.pt',weights_only=True,map_location='cpu')
        for key,value in before.actors[1].state_dict().items():assert torch.equal(value,frozen_single['model'][key])
    with (folder/'training.csv').open() as f:history=list(csv.DictReader(f))
    assert len(history)==meta['target_updates']
    for u in range(1,meta['target_updates']+1):
        current,record=load_coupled(folder/f'update_{u}.pt');role=meta['actor_update_order'][(u-1)%3]
        assert record['updates']==u and record['role']==role
        for key,value in before.state_dict().items():
            if not key.startswith(f'actors.{role}.'):assert torch.equal(value,current.state_dict()[key]),key
        dataset=np.load(folder/f'data_{u}.npz');n=len(dataset['obs'])
        assert n==meta['states_per_update'];assert record['visited_states']==u*n
        obs=ego_observations(dataset['states'],dataset['previous'],dataset['steps'])[:,role]
        np.testing.assert_array_equal(obs,dataset['obs'])
        sample=np.linspace(0,n-1,min(8,n),dtype=int)
        costs,incumbent,_=unilateral_returns(dataset['states'][sample],dataset['previous'][sample],dataset['steps'][sample],before,role)
        np.testing.assert_allclose(costs,dataset['costs'][sample],atol=1e-9,rtol=0)
        np.testing.assert_array_equal(incumbent,dataset['incumbent'][sample]);target_samples+=len(sample)
        selected=current.actors[role].proposal(dataset['obs'],action_mask(dataset['previous'])[:,role])
        regret=dataset['costs'][np.arange(n),selected]-dataset['costs'].min(-1)
        equivalent(float(regret.mean()),record['training_action_regret'])
        assert all(np.isfinite(float(v)) for v in history[u-1].values())
        assert record['evaluated_model_transitions']>record['visited_states']
        before=current
    for key,value in before.state_dict().items():assert torch.equal(value,model.state_dict()[key])
    actual=[sum(float((value-initial['model'][key]).square().sum()) for key,value in model.state_dict().items()
        if key.startswith(f'actors.{i}.')) for i in range(3)]
    np.testing.assert_allclose(actual,meta['actor_squared_parameter_change'],atol=1e-9,rtol=0)
    return dict(updates=meta['updates'],counterfactual_states_recomputed=target_samples,actor_squared_changes=actual)


def verify(directory,replay=False):
    torch.set_num_threads(1);directory=Path(directory)
    protocol=json.loads((directory/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Changed runtime source: '+name
    paths=[];training={}
    for name,r in protocol['models'].items():
        path=ROOT/r['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        training[name]=audit_training(path);paths.append(path)
    fs,_=factories(paths);assert set(fs)==set(protocol['methods'])
    rows=json.loads((directory/'episodes.json').read_text());lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    seeds=range(protocol['seed_base'],protocol['seed_base']+protocol['n_initials'])
    assert len(rows)==len(lookup)==protocol['expected_runs']
    assert set(lookup)=={(s,m,p) for s in seeds for m in protocol['modes'] for p in protocol['methods']}
    for i,seed in enumerate(seeds):
        initial=regenerate_initial(seed,'nominal' if i<protocol['n_initials']//2 else 'dense')
        for mode in protocol['modes']:
            for method in protocol['methods']:
                row=lookup[seed,mode,method]
                np.testing.assert_allclose(row['initial'],initial,atol=1e-8,rtol=0)
                np.testing.assert_allclose(row['per_vehicle_cost'],np.array(row['per_vehicle_finish'])+.05*np.array(row['per_vehicle_variation']),atol=1e-10)
    c=MergeConfig();packed=parameters(c);audited=actions=0
    for i in protocol['audit_initial_indices']:
        seed=protocol['seed_base']+i
        for mode in protocol['modes']:
            tape=disturbance_tape(seed,mode,.5,250)
            for name,build in fs.items():
                row=lookup[seed,mode,name];x=np.array(row['initial']);prev=np.zeros(3);active=np.ones(3,bool)
                tv=np.zeros(3);finish=np.full(3,25.);controller=build(x)
                trace=json.loads((directory/f'trace_{i}_{mode}_{name}.json').read_text())
                for step,item in enumerate(trace):
                    assert item['t']==(step+1)*.5
                    np.testing.assert_array_equal(item['disturbance'],tape[step*5:(step+1)*5])
                    u,info=controller.select(x,prev);np.testing.assert_allclose(u,item['command'],atol=1e-9,rtol=0);actions+=1
                    if 'agent_actions' in info:assert info['agent_actions']==item['agent_actions']
                    assert robust_certificate(x,prev,u,packed,.5) and np.max(np.abs(u-prev))<=2.5+1e-9
                    tv+=active*np.abs(u-prev)
                    for error in item['disturbance']:
                        old=x[:,0].copy();x[:,1]=np.clip(x[:,1]+(u+np.array(error))*.1,0,20);x[:,0]+=x[:,1]*.1
                        assert transition_safe(old[None],x[:,0][None],c)[0]
                    np.testing.assert_allclose(x,item['x'],atol=1e-8,rtol=0);x=np.array(item['x'])
                    arrived=active&(x[:,0]>=60);finish[arrived]=item['t'];active[arrived]=False;prev=u
                np.testing.assert_allclose(tv,row['per_vehicle_variation'],atol=1e-9,rtol=0)
                np.testing.assert_allclose(finish,row['per_vehicle_finish'],atol=1e-9,rtol=0);audited+=1
    replays=0
    if replay:
        with DisturbedMergeEnv() as env:
            for i in (0,protocol['n_initials']//2):
                seed=protocol['seed_base']+i;regime='nominal' if i==0 else 'dense'
                for mode in protocol['modes']:
                    for name,build in fs.items():
                        actual,trace,_=run_episode(env,seed,regime,mode,build,True)
                        expected=lookup[seed,mode,name]
                        for key in ('per_vehicle_finish','per_vehicle_variation','completed','collision','contract_violations','aborted'):
                            equivalent(actual[key],expected[key])
                        saved=json.loads((directory/f'trace_{i}_{mode}_{name}.json').read_text())
                        np.testing.assert_allclose([r['command'] for r in trace],[r['command'] for r in saved],atol=1e-9,rtol=0);replays+=1
    result=dict(purpose='Development reconstruction; not a formal effectiveness verdict',runs=len(rows),training=training,
        saved_traces=audited,controller_action_reconstructions=actions,additional_sumo_replays=replays,
        protocol_sha256=hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest(),
        episodes_sha256=hashlib.sha256((directory/'episodes.json').read_bytes()).hexdigest())
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--replay',action='store_true');p.add_argument('--out')
    a=p.parse_args();result=verify(a.directory,a.replay)
    if a.out:Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
