"""Replay all saved multi-agent traces and a subset of actual SUMO episodes."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from coopgate.cooperative_v15 import load_joint,JointPolicyGate
from coopgate.ppo_numerics_v15 import require_complete_training
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.ego_v13 import EgoRolloutGate,EgoPolicyGate
from coopgate.policy_iteration_v14 import load_policy
from coopgate.disturbance_v9 import DisturbedMergeEnv,disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig,transition_safe
from coopgate.robust_v9 import robust_certificate
from develop_v12 import initial as regenerate_initial
from experiment_ego_v13 import run_episode
from verify_single_v12 import equivalent

ROOT=Path(__file__).resolve().parents[1]


def verify(directory,replay=False):
    directory=Path(directory);torch.set_num_threads(1)
    protocol=json.loads((directory/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Execution source changed: '+name
    rows=json.loads((directory/'episodes.json').read_text())
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    seeds=range(protocol['seed_base'],protocol['seed_base']+protocol['n_initials'])
    assert len(rows)==len(lookup)==protocol['expected_runs']
    assert set(lookup)=={(s,m,p) for s in seeds for m in protocol['modes'] for p in protocol['methods']}
    factories={'ego_planner':lambda x:EgoRolloutGate(),'team_planner':lambda x:JointRolloutGate()}
    changes={}
    for name,record in protocol['models'].items():
        path=directory/'models'/name/'policy.pt'
        if not path.exists():path=ROOT/record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        model,meta=load_joint(path);require_complete_training(meta)
        training=path.parent;initial=torch.load(training/'initial.pt',map_location='cpu',weights_only=True)
        for source,digest in meta['source_sha256'].items():
            assert hashlib.sha256((training/'source'/source).read_bytes()).hexdigest()==digest
        actual=[sum(float((value-initial['model'][key]).square().sum()) for key,value in model.state_dict().items()
            if key.startswith(f'actors.{i}.')) for i in range(3)]
        np.testing.assert_allclose(actual,meta['actor_parameter_delta_l2'],atol=1e-10,rtol=0)
        with (training/'training.csv').open() as f:history=list(csv.DictReader(f))
        assert int(history[-1]['joint_environment_steps'])==meta['target_steps']
        assert all(np.isfinite(float(v)) for row in history for v in row.values())
        assert all(b>a for a,b in zip([int(r['joint_environment_steps']) for r in history[:-1]],
                                    [int(r['joint_environment_steps']) for r in history[1:]]))
        episodes=json.loads((training/'episodes.json').read_text())
        assert len(episodes)==meta['episodes']
        for row in episodes:np.testing.assert_allclose(row['cost'],np.array(row['time'])+.05*np.array(row['variation']),atol=1e-10)
        changes[name]=actual;factories[name]=lambda x,m=model:JointPolicyGate(m)
    for name in protocol['methods']:
        if name.startswith('single_'):
            model,_=load_policy(ROOT/f'artifacts/v14/single/seed_{name[7:]}/policy.pt')
            factories[name]=lambda x,m=model:EgoPolicyGate(m)
    assert set(factories)==set(protocol['methods'])
    for i,seed in enumerate(seeds):
        initial=regenerate_initial(seed,'nominal' if i<protocol['n_initials']//2 else 'dense')
        for mode in protocol['modes']:
            for method in protocol['methods']:
                np.testing.assert_allclose(lookup[seed,mode,method]['initial'],initial,atol=1e-8,rtol=0)
    c=MergeConfig();packed=parameters(c);audited=comparisons=0
    for i in protocol['audit_initial_indices']:
        seed=protocol['seed_base']+i
        for mode in protocol['modes']:
            tape=disturbance_tape(seed,mode,.5,250)
            for name,factory in factories.items():
                row=lookup[seed,mode,name]
                state=np.array(row['initial']);previous=np.zeros(3);variation=np.zeros(3)
                active=np.ones(3,bool);finish=np.full(3,25.);controller=factory(state)
                trace=json.loads((directory/f'trace_{i}_{mode}_{name}.json').read_text())
                for step,item in enumerate(trace):
                    assert item['t']==(step+1)*.5
                    np.testing.assert_array_equal(item['disturbance'],tape[step*5:(step+1)*5])
                    command,info=controller.select(state,previous)
                    np.testing.assert_allclose(command,item['command'],atol=1e-9,rtol=0);comparisons+=1
                    if 'agent_actions' in info:assert info['agent_actions']==item['agent_actions']
                    assert robust_certificate(state,previous,command,packed,.5)
                    assert np.max(np.abs(command-previous))<=2.5+1e-9
                    variation+=active*np.abs(command-previous)
                    for error in item['disturbance']:
                        old=state[:,0].copy();state[:,1]=np.clip(state[:,1]+(command+np.array(error))*.1,0,20)
                        state[:,0]+=state[:,1]*.1
                        assert transition_safe(old[None],state[:,0][None],c)[0]
                    np.testing.assert_allclose(state,item['x'],atol=1e-8,rtol=0);state=np.array(item['x'])
                    arrived=active&(state[:,0]>=60);finish[arrived]=item['t'];active[arrived]=False;previous=command
                np.testing.assert_allclose(variation,row['per_vehicle_variation'],atol=1e-9,rtol=0)
                np.testing.assert_allclose(finish,row['per_vehicle_finish'],atol=1e-9,rtol=0);audited+=1
    replays=0
    if replay:
        with DisturbedMergeEnv() as env:
            for i in (0,protocol['n_initials']//2):
                seed=protocol['seed_base']+i;regime='nominal' if i==0 else 'dense'
                for mode in protocol['modes']:
                    for name,factory in factories.items():
                        result,trace,_=run_episode(env,seed,regime,mode,factory,True)
                        expected=lookup[seed,mode,name]
                        for key in ('per_vehicle_finish','per_vehicle_variation','completed','collision','contract_violations','aborted'):
                            equivalent(result[key],expected[key])
                        saved=json.loads((directory/f'trace_{i}_{mode}_{name}.json').read_text())
                        np.testing.assert_allclose([r['command'] for r in trace],[r['command'] for r in saved],atol=1e-9,rtol=0)
                        replays+=1
    result=dict(purpose='Development replay audit, not a MARL effectiveness verdict',runs=len(rows),
        saved_traces=audited,controller_action_reconstructions=comparisons,additional_sumo_replays=replays,
        training_actor_changes=changes,protocol_sha256=hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest(),
        episodes_sha256=hashlib.sha256((directory/'episodes.json').read_bytes()).hexdigest())
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--replay',action='store_true');p.add_argument('--out')
    a=p.parse_args();result=verify(a.directory,a.replay)
    if a.out:Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
