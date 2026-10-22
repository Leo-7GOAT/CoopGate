"""Verify V13 training, independent accounting, certificates and SUMO replay."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig,transition_safe
from coopgate.robust_v9 import robust_certificate
from coopgate.confidence_v12 import conformal_upper
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.ego_v13 import load_ego,ego_observations
from coopgate.learned_rollout_v13 import LearnedRolloutGate,actor_action
from coopgate.marl_v12 import action_mask,LEVELS
from experiment_ego_v13 import run_episode
from study_ego_v13 import factories
from report_ego_v13 import summarize
from verify_single_v12 import equivalent
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v13/ego'


def verify(replay=False,require=False):
    p=json.loads((D/'protocol.json').read_text());train=json.loads((D/'training_protocol.json').read_text())
    for mapping in (p['source_sha256'],train['source_sha256']):
        for name,digest in mapping.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    if (D/'sha256.json').exists():
        for name,digest in json.loads((D/'sha256.json').read_text()).items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
    changes={};policies={};compiled={};actor_comparisons=0
    for seed in p['training_seeds']:
        model=torch.load(D/f'seed_{seed}/policy.pt',map_location='cpu',weights_only=True)
        initial=torch.load(D/f'seed_{seed}/initial.pt',map_location='cpu',weights_only=True)
        assert model['algorithm']=='double_dqn_ego_v13' and model['steps']==2097152 and model['seed']==seed
        assert model['learned_vehicles']==[1] and model['observation_version']=='markov_joint_memory_v1'
        norms={part:sum(float((v-initial['model'][k]).square().sum()) for k,v in model['model'].items() if k.startswith(part))
            for part in ('encoder','value','advantage')}
        assert all(v>1e-6 for v in norms.values());changes[seed]=norms
        policies[seed]=load_ego(D/f'seed_{seed}/policy.pt')[0]
        compiled[seed]=LearnedRolloutGate(D/f'seed_{seed}/policy.pt')
        with (D/f'seed_{seed}/training.csv').open() as f:history=list(csv.DictReader(f))
        assert len(history)==64 and int(history[-1]['steps'])==2097152
    cal=json.loads((D/'calibration/calibration.json').read_text());res=json.loads((D/'calibration/residuals.json').read_text())
    for seed in p['training_seeds']:
        for group,entry in cal[str(seed)]['groups'].items():
            rr=[r for r in res if r['policy']==seed and r['group']==group]
            assert len(rr)==entry['n']
            for r in rr:assert np.isclose(max(r['residual']),r['score'])
            assert abs(conformal_upper([r['score'] for r in rr],.1)-entry['q'])<1e-10
            assert abs(conformal_upper([r['residual'][0] for r in rr],.1)-entry['q_cost_only'])<1e-10
    test_seeds={r['seed'] for r in p['initials']};cal_seeds={r['seed'] for r in res}
    assert len(test_seeds)==128 and len(cal_seeds)==256 and not test_seeds&cal_seeds
    assert min(test_seeds)>=32000000 and min(cal_seeds)>=31000000
    rows=json.loads((D/'study/episodes.json').read_text());summary=summarize(p,rows,cal)
    equivalent(summary,json.loads((D/'summary.json').read_text()))
    c=MergeConfig();packed=parameters(c);initials={r['seed']:np.array(r['state']) for r in json.loads((D/'study/initials.json').read_text())};audited=0
    for index,item in enumerate(p['initials'][:2]):
        for mode in p['disturbances']:
            for method in p['methods']:
                trace=json.loads((D/f'study/trace_{index}_{mode}_{method}.json').read_text());state=initials[item['seed']].copy()
                previous=np.zeros(3);variation=np.zeros(3);finish=np.full(3,25.);active=np.ones(3,bool)
                for step,t in enumerate(trace):
                    if method.startswith('raw_'):
                        seed=int(method.split('_')[1]);fast=compiled[seed]
                        original=policies[seed].proposal(ego_observations(state,previous,step)[1:2],action_mask(previous)[1:2])[0]
                        actual=actor_action(state,previous,step,fast.weights,fast.markov,fast.kind)
                        assert actual==LEVELS[original];actor_comparisons+=1
                    command=np.array(t['command']);assert robust_certificate(state,previous,command,packed,.5)
                    assert np.max(np.abs(command-previous))<=2.5+1e-9
                    variation+=active*np.abs(command-previous)
                    for error in t['disturbance']:
                        error=np.array(error);assert np.max(np.abs(error))<=.5+1e-9
                        old=state[:,0].copy();state[:,1]=np.clip(state[:,1]+(command+error)*.1,0,20);state[:,0]+=state[:,1]*.1
                        assert transition_safe(old[None],state[:,0][None],c)[0]
                    assert np.max(np.abs(state-np.array(t['x'])))<1e-8
                    newly=active&(state[:,0]>=60);finish[newly]=t['t'];active[newly]=False;previous=command
                row=next(r for r in rows if r['seed']==item['seed'] and r['disturbance']==mode and r['controller']==method)
                np.testing.assert_allclose(variation,row['per_vehicle_variation'],atol=1e-9,rtol=0)
                np.testing.assert_allclose(finish,row['per_vehicle_finish'],atol=1e-9,rtol=0);audited+=1
    replays=0
    if replay:
        fs=factories();item=p['initials'][0]
        with DisturbedMergeEnv() as env:
            for mode in p['disturbances']:
                for method,factory in fs.items():
                    actual,trace,_=run_episode(env,item['seed'],item['regime'],mode,factory,True)
                    expected=next(r for r in rows if r['seed']==item['seed'] and r['disturbance']==mode and r['controller']==method)
                    for key in ('ego_cost','ego_time','ego_variation','other_time','completed','collision','contract_violations','aborted'):
                        equivalent(actual[key],expected[key])
                    saved=json.loads((D/f'study/trace_0_{mode}_{method}.json').read_text())
                    np.testing.assert_allclose([r['command'] for r in trace],[r['command'] for r in saved],atol=1e-8,rtol=0);replays+=1
    print(json.dumps({'validated_sumo_runs':len(rows),'audited_traces':audited,'sumo_replays':replays,'compiled_actor_comparisons':actor_comparisons,
        'training_parameter_changes':changes,'acceptance':summary['acceptance'],
        'individual_ego_verified':summary['individual_ego_verified']}),flush=True)
    if require and not summary['acceptance']['all_passed']:raise SystemExit('A frozen V13 performance criterion is unmet')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--replay',action='store_true');a.add_argument('--require-acceptance',action='store_true')
    v=a.parse_args();verify(v.replay,v.require_acceptance)
