"""Audit archived development data without turning it into a formal result."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig,transition_safe
from coopgate.robust_v9 import robust_certificate
from coopgate.ego_v13 import EgoRolloutGate
from coopgate.disturbance_v9 import DisturbedMergeEnv
from experiment_ego_v13 import run_episode
from validate_policies_v14 import factory
from verify_single_v12 import equivalent

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/v14/development'


def trace_audit(trace,row):
    state=np.array(row['initial']);previous=np.zeros(3);active=np.ones(3,bool)
    finish=np.full(3,25.);variation=np.zeros(3);c=MergeConfig();packed=parameters(c)
    for item in trace:
        command=np.array(item['command']);assert robust_certificate(state,previous,command,packed,.5)
        variation+=active*np.abs(command-previous)
        for error in item['disturbance']:
            error=np.array(error);assert np.max(np.abs(error))<=.5+1e-9
            old=state[:,0].copy();state[:,1]=np.clip(state[:,1]+(command+error)*.1,0,20);state[:,0]+=state[:,1]*.1
            assert transition_safe(old[None],state[:,0][None],c)[0]
        observed=np.array(item['x']);np.testing.assert_allclose(state,observed,atol=1e-8,rtol=0);state=observed
        newly=active&(state[:,0]>=60);finish[newly]=item['t'];active[newly]=False;previous=command
    np.testing.assert_allclose(finish,row['per_vehicle_finish'],atol=1e-9,rtol=0)
    np.testing.assert_allclose(variation,row['per_vehicle_variation'],atol=1e-9,rtol=0)


def verify(replay=False):
    for name,digest in json.loads((D/'sha256.json').read_text()).items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
    manifest=json.loads((D/'manifest.json').read_text())
    for item in manifest['training_runs']:
        if item['kind']!='training':continue
        root=D/item['name'];meta=json.loads((root/'metadata.json').read_text())
        for name,digest in meta['source_sha256'].items():
            assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==digest
    audited=replayed=total=0
    for folder in sorted(D.glob('v14_validation_*')):
        p=json.loads((folder/'protocol.json').read_text());rows=json.loads((folder/'episodes.json').read_text());total+=len(rows)
        index={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
        assert len(index)==len(rows)==p['initials']*len(p['modes'])*len(p['methods'])
        fs={'rollout':lambda x:EgoRolloutGate()}
        for name,path in p['methods'].items():
            if name=='rollout':continue
            archived=D/Path(path.replace('\\','/')).relative_to('runs')
            assert hashlib.sha256(archived.read_bytes()).hexdigest()==p['policy_sha256'][path]
            fs[name]=factory(archived)
        for initial_index in range(2):
            seed=p['seed_base']+initial_index
            for mode in p['modes']:
                for method in p['methods']:
                    trace=json.loads((folder/f'trace_{initial_index}_{mode}_{method}.json').read_text())
                    trace_audit(trace,index[seed,mode,method]);audited+=1
        if replay:
            with DisturbedMergeEnv() as env:
                for initial_index in (0,p['initials']//2):
                    seed=p['seed_base']+initial_index;regime='nominal' if initial_index==0 else 'dense'
                    for mode in p['modes']:
                        for method,build in fs.items():
                            actual,trace,initial=run_episode(env,seed,regime,mode,build,True)
                            expected=index[seed,mode,method]
                            for key in ('ego_cost','ego_time','ego_variation','other_time','completed','collision','contract_violations','aborted'):
                                equivalent(actual[key],expected[key])
                            trace_audit(trace,{**actual,'initial':initial.tolist()});replayed+=1
    assert total==manifest['actual_sumo_runs']
    result=dict(validation_runs=total,saved_traces_audited=audited,additional_sumo_replays=replayed,
                purpose='development audit only; not single-ego formal acceptance')
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');a=p.parse_args();verify(a.replay)
