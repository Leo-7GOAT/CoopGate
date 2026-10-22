"""Verify provenance, training changes, statistical accounting and SUMO replay."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from coopgate.marl_v12 import MARLGate,load_policy
from coopgate.single_v12 import SingleRolloutGate
from coopgate.confidence_v12 import EpisodeConfidenceGate,conformal_upper
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.robust_v9 import robust_certificate
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig,transition_safe
from report_single_v12 import summarize
from experiment_single_v12 import run_episode

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v12/single'


def equivalent(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equivalent(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equivalent(x,y)
    elif isinstance(a,float):assert np.isclose(a,b,rtol=1e-9,atol=1e-9),(a,b)
    else:assert a==b,(a,b)


def verify(replay=False,require=False):
    protocol=json.loads((D/'protocol.json').read_text());c=MergeConfig();packed=parameters(c)
    for path,digest in protocol['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    if (D/'sha256.json').exists():
        for path,digest in json.loads((D/'sha256.json').read_text()).items():assert hashlib.sha256((D/path).read_bytes()).hexdigest()==digest,path
    train_protocol=json.loads((D/'training_protocol.json').read_text())
    for path,digest in train_protocol['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    policies={};bcs={};changes={}
    for seed in protocol['training_seeds']:
        policy,meta=load_policy(D/f'seed_{seed}/policy.pt');bc,_=load_policy(D/f'seed_{seed}/bc.pt')
        assert meta['algorithm']=='single_ego_ppo_v12' and meta['steps']==1048576 and meta['seed']==seed
        assert meta['learned_vehicles']==[1] and meta['finite_horizon_terminal']
        norms={part:sum(float((v-bc.state_dict()[k]).square().sum()) for k,v in policy.state_dict().items() if k.startswith(part)) for part in ('actor','critic')}
        assert all(v>1e-6 for v in norms.values());changes[seed]=norms
        with (D/f'seed_{seed}/training.csv').open() as f:history=list(csv.DictReader(f))
        assert len(history)==512 and int(history[-1]['steps'])==1048576
        assert all(int(history[i+1]['steps'])-int(history[i]['steps'])==2048 for i in range(len(history)-1))
        policies[seed]=policy;bcs[seed]=bc
    calibration=json.loads((D/'calibration/calibration.json').read_text())
    residuals=json.loads((D/'calibration/residuals.json').read_text())
    for seed in policies:
        for group,entry in calibration[str(seed)]['groups'].items():
            rr=[r['residual'] for r in residuals if r['policy']==seed and r['group']==group]
            assert len(rr)==entry['n'] and abs(conformal_upper(rr,.1)-entry['q'])<1e-12
    rows=json.loads((D/'study/episodes.json').read_text());summary=summarize(protocol,rows)
    equivalent(summary,json.loads((D/'summary.json').read_text()))
    train_seeds={r['seed'] for r in json.loads((D/'teacher/metadata.json').read_text())['rows']}
    calibration_seeds={r['seed'] for r in json.loads((D/'calibration/protocol.json').read_text())['episodes']}
    test_seeds={r['seed'] for r in protocol['initials']}
    assert not(train_seeds&calibration_seeds or train_seeds&test_seeds or calibration_seeds&test_seeds)
    initial_by_seed={r['seed']:np.array(r['state']) for r in json.loads((D/'study/initials.json').read_text())}
    audited=0
    for index,item in enumerate(protocol['initials'][:2]):
        for mode in protocol['disturbances']:
            for method in protocol['methods']:
                trace=json.loads((D/f'study/trace_{index}_{mode}_{method}.json').read_text())
                x=initial_by_seed[item['seed']].copy();prev=np.zeros(3);tv=0.
                for t in trace:
                    command=np.array(t['command']);assert robust_certificate(x,prev,command,packed,.5)
                    assert np.max(np.abs(command-prev))<=2.5+1e-9
                    tv+=np.abs(command-prev).sum()
                    for error in t['disturbance']:
                        error=np.array(error);assert np.max(np.abs(error))<=.5+1e-9
                        p=x[:,0].copy();x[:,1]=np.clip(x[:,1]+(command+error)*.1,0,20);x[:,0]+=x[:,1]*.1
                        assert transition_safe(p[None],x[:,0][None],c)[0]
                    assert np.max(np.abs(x-np.array(t['x'])))<1e-8
                    prev=command
                row=next(r for r in rows if r['seed']==item['seed'] and r['disturbance']==mode and r['controller']==method)
                assert abs(tv-row['variation'])<1e-8;audited+=1
    replays=0
    if replay:
        factories={'rule':lambda x:MARLGate(None,mode='constant'),'rollout':lambda x:SingleRolloutGate()}
        for seed in policies:
            factories[f'rl_{seed}']=lambda x,s=seed:MARLGate(policies[s],mode='single')
            factories[f'bc_{seed}']=lambda x,s=seed:MARLGate(bcs[s],mode='single')
            factories[f'gate_{seed}']=lambda x,s=seed:EpisodeConfidenceGate(policies[s],x,calibration[str(s)])
        with DisturbedMergeEnv() as env:
            item=protocol['initials'][0]
            for mode in protocol['disturbances']:
                for method,factory in factories.items():
                    actual,trace,_=run_episode(env,item['seed'],item['regime'],mode,factory,True)
                    expected=next(r for r in rows if r['seed']==item['seed'] and r['disturbance']==mode and r['controller']==method)
                    for key in ('mean_capped_finish_s','variation','completed','collision','contract_violations','aborted','ego_finish_s'):
                        equivalent(actual[key],expected[key])
                    saved=json.loads((D/f'study/trace_0_{mode}_{method}.json').read_text())
                    assert len(trace)==len(saved)
                    assert np.allclose([t['command'] for t in trace],[t['command'] for t in saved],atol=1e-9,rtol=0)
                    replays+=1
    print(json.dumps({'validated_runs':len(rows),'audited_traces':audited,'sumo_replays':replays,
        'learned_parameter_changes':changes,'acceptance':summary['acceptance']}),flush=True)
    if require and not summary['acceptance']['all_passed']:raise SystemExit('Frozen performance criteria not all met')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');p.add_argument('--require-acceptance',action='store_true')
    a=p.parse_args();verify(a.replay,a.require_acceptance)
