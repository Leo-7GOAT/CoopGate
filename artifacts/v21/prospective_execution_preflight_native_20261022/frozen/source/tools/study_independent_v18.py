"""Freeze and execute a paired three-seed study in disjoint SUMO shards.

Shards split initial states, never methods or disturbance modes. The final
assembly checks exact completeness and pairing. It issues no risk certificate.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.advantage_gate_v18 import JointAdvantageGate
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.disturbance_v9 import DisturbedMergeEnv
from experiment_ego_v13 import run_episode
from prerequisite_v15 import require_single
from verify_development_v14 import trace_audit

ROOT=Path(__file__).resolve().parents[1]


def factories(records):
    fs={'baseline':lambda x:JointRolloutGate()}
    for name,record in records.items():
        path=ROOT/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Frozen model changed')
        model,meta=load_coupled(path)
        if meta['updates']!=meta['target_updates'] or not all(v>1e-8 for v in meta['actor_squared_parameter_change']):
            raise ValueError('Incomplete or inactive joint learner')
        fs['raw_'+name]=lambda x,m=model:JointPolicyGate(m)
        fs['gate_'+name]=lambda x,m=model:JointAdvantageGate(m)
    return fs


def freeze(spec_path,out):
    spec_path,out=Path(spec_path),Path(out);spec=json.loads(spec_path.read_text())
    prerequisite=require_single(ROOT);models={}
    for name in spec['models']:
        path=ROOT/name;_,meta=load_coupled(path)
        seed=str(meta['seed'])
        if seed in models:raise ValueError('Duplicate training seed')
        models[seed]=dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ego_seed=meta['ego_seed'],
            training_seed=meta['seed'],updates=meta['updates'],target_updates=meta['target_updates'])
    fs=factories(models)
    if spec['expected_actual_sumo_runs']!=spec['n_initials']*len(spec['modes'])*len(fs):
        raise ValueError('Spec budget does not match the frozen methods')
    out.mkdir(parents=True,exist_ok=False)
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/experiment_ego_v13.py',
        ROOT/'tools/prerequisite_v15.py',ROOT/'tools/verify_development_v14.py']
    p=dict(spec=spec,spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),models=models,methods=list(fs),
        modes=spec['modes'],n_initials=spec['n_initials'],seed_base=spec['seed_base'],shards=spec['shards'],
        expected_sumo_runs=spec['expected_actual_sumo_runs'],audit_initial_indices=spec['audit_initial_indices'],
        prerequisite=prerequisite,source_sha256={str(f.resolve().relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    for source in p['source_sha256']:
        target=out/'source'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,target)
    print(json.dumps(dict(directory=str(out),runs=p['expected_sumo_runs'],models=models)),flush=True)


def verify_sources(directory,p):
    for source,digest in p['source_sha256'].items():
        if hashlib.sha256((directory/'source'/source).read_bytes()).hexdigest()!=digest:
            raise ValueError('Saved source changed: '+source)
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:
            raise ValueError('Runtime source changed: '+source)


def worker(directory,shard):
    torch.set_num_threads(1);directory=Path(directory);p=json.loads((directory/'protocol.json').read_text())
    if not 0<=shard<p['shards']:raise ValueError('Invalid shard')
    verify_sources(directory,p);fs=factories(p['models'])
    out=directory/f'shard_{shard}';out.mkdir(exist_ok=False);rows=[]
    ids=list(range(shard,p['n_initials'],p['shards']));started=time.perf_counter()
    with DisturbedMergeEnv() as env:
        for index,i in enumerate(ids):
            seed=p['seed_base']+i;regime='nominal' if i<p['n_initials']//2 else 'dense'
            for mi,mode in enumerate(p['modes']):
                for method in np.random.default_rng(seed+mi+1820).permutation(p['methods']):
                    result,trace,initial=run_episode(env,seed,regime,mode,fs[method],i in p['audit_initial_indices'])
                    cost=np.array(result['per_vehicle_finish'])+.05*np.array(result['per_vehicle_variation'])
                    result.update(controller=str(method),initial=initial.tolist(),per_vehicle_cost=cost.tolist(),team_cost=float(cost.mean()),
                        initial_index=i,learned_action_steps=sum(t.get('nominal_joint_action_admitted',False) for t in env.trace),
                        control_steps=len(env.trace),forecast_transitions=sum(t.get('performance_forecast_transitions',0) for t in env.trace),
                        internal_planning_transitions=sum(t.get('performance_model_transitions',0) for t in env.trace))
                    if not result['completed'] or result['collision'] or result['aborted'] or result['contract_violations']:
                        trace=env.trace
                    result['trace_saved']=trace is not None
                    if trace is not None:
                        trace_audit(trace,result)
                        (out/f'trace_{i}_{mode}_{method}.json').write_text(json.dumps(trace,indent=2)+'\n')
                    rows.append(result)
            (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(dict(shard=shard,initials=index+1,actual_sumo_runs=len(rows),elapsed_s=time.perf_counter()-started)),flush=True)
    expected=len(ids)*len(p['modes'])*len(fs)
    if len(rows)!=expected:raise RuntimeError('Incomplete shard')
    (out/'complete.json').write_text(json.dumps(dict(shard=shard,initial_indices=ids,actual_sumo_runs=len(rows),
        protocol_sha256=hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest(),
        episodes_sha256=hashlib.sha256((out/'episodes.json').read_bytes()).hexdigest()),indent=2)+'\n')


def finalize(directory):
    directory=Path(directory);p=json.loads((directory/'protocol.json').read_text());verify_sources(directory,p)
    rows=[];digest=hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest()
    for shard in range(p['shards']):
        folder=directory/f'shard_{shard}';done=json.loads((folder/'complete.json').read_text())
        assert done['protocol_sha256']==digest and done['initial_indices']==list(range(shard,p['n_initials'],p['shards']))
        assert done['episodes_sha256']==hashlib.sha256((folder/'episodes.json').read_bytes()).hexdigest()
        part=json.loads((folder/'episodes.json').read_text());assert len(part)==done['actual_sumo_runs']
        assert {r['initial_index'] for r in part}==set(done['initial_indices'])
        rows.extend(part)
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    expected={(p['seed_base']+i,m,k) for i in range(p['n_initials']) for m in p['modes'] for k in p['methods']}
    assert len(rows)==len(lookup)==p['expected_sumo_runs'] and set(lookup)==expected
    for i in range(p['n_initials']):
        same=[lookup[p['seed_base']+i,m,k] for m in p['modes'] for k in p['methods']]
        assert all(r['scenario']==('nominal' if i<p['n_initials']//2 else 'dense') for r in same)
        for row in same:
            np.testing.assert_allclose(row['initial'],same[0]['initial'],atol=1e-8,rtol=0)
            assert row['group']==same[0]['group']
    rows.sort(key=lambda r:(r['initial_index'],r['disturbance'],r['controller']))
    (directory/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
    result=dict(actual_sumo_runs=len(rows),initials=p['n_initials'],training_seeds=len(p['models']),
        physical={k:{metric:sum(r[metric] for r in rows if r['controller']==k) for metric in
            ('completed','collision','aborted','contract_violations')} for k in p['methods']},
        protocol_sha256=digest,episodes_sha256=hashlib.sha256((directory/'episodes.json').read_bytes()).hexdigest(),
        disturbance_performance_risk_certified=False)
    (directory/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('spec');f.add_argument('out')
    w=sub.add_parser('worker');w.add_argument('directory');w.add_argument('--shard',type=int,required=True)
    f=sub.add_parser('finalize');f.add_argument('directory')
    a=p.parse_args()
    if a.command=='freeze':freeze(a.spec,a.out)
    elif a.command=='worker':worker(a.directory,a.shard)
    else:finalize(a.directory)
