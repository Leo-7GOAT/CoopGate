"""Post-protocol equivalence/scaling supplement, not tuned efficacy evidence."""
from pathlib import Path
import argparse,json,time,csv
import numpy as np
from coopgate.recovery import RecoveryConfig,admissible_acceleration
from coopgate.chain_recovery import build_chain,chain_modes,brute_modes,ChainSupervisor
from coopgate.learning_v3 import load
from coopgate.platoon import PlatoonEnv

ROOT=Path(__file__).resolve().parents[1]


def main(OUT):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'results.json').exists():raise RuntimeError('Refusing to overwrite supplemental results')
    cfg=RecoveryConfig();rng=np.random.default_rng(2026101504);scaling=[]
    for n in (3,8,16,32,64):
        durations=[];select=[];brute=[]
        for repeat in range(20):
            states=np.zeros((n+1,5));states[:,0]=np.r_[1000-np.arange(1,n+1)*30,1000];states[:,2]=rng.uniform(12,19,n+1);states[:,4]=1
            previous=rng.uniform(-2.5,2.5,n);proposal=admissible_acceleration(rng.uniform(-5,2.5,n),previous,cfg)
            start=time.perf_counter();_,_,feasible,cost=build_chain(states,previous,proposal,cfg)
            part=time.perf_counter();mode=chain_modes(feasible,cost);stop=time.perf_counter()
            durations.append((stop-start)*1000);select.append((stop-part)*1000)
            if n<=8:
                start=time.perf_counter();exact=brute_modes(feasible,cost);brute.append((time.perf_counter()-start)*1000)
                assert (exact is None and mode is None) or np.array_equal(mode,exact)
        scaling.append({'vehicles':n,'pair_entries':2+4*(n-1),'joint_modes':2**n,'mean_total_ms':float(np.mean(durations)),
            'p95_total_ms':float(np.percentile(durations,95)),'mean_dp_ms':float(np.mean(select)),
            'mean_enumeration_selection_only_ms':float(np.mean(brute)) if brute else None})
    protocol=json.loads((ROOT/'artifacts/v3/final/protocol.json').read_text());rows=[];equivalence=[];timing={}
    for seed in (83,97,109):
        model,_=load(ROOT/f'artifacts/v3/final/seed{seed}/policy.pt')
        expected=json.loads((ROOT/f'artifacts/v3/final/seed{seed}/trace_test_joint_rl.json').read_text())
        with PlatoonEnv(cfg,'joint_rl') as env:
            obs=env.reset(**protocol['test_scenes'][0]);env.supervisor=ChainSupervisor(cfg,'joint_rl')
            for i,reference in enumerate(expected):
                obs,_,t,u,_=env.step(model.act(obs,True)[0])
                np.testing.assert_allclose(env.trace[-1]['commands'],reference['commands'],atol=1e-7)
                np.testing.assert_allclose(env.states,reference['states'],atol=1e-6)
            assert t or u
            equivalence.append({'seed':seed,'steps':len(expected),'identical_commands_and_states':True})
    model,_=load(ROOT/'artifacts/v3/final/seed83/policy.pt')
    for method in ('joint_mpc','joint_rl','chain_rl'):
        times=[]
        with PlatoonEnv(cfg,'joint_rl' if method=='chain_rl' else method) as env:
            for scene in protocol['test_scenes'][:5]:
                obs=env.reset(**scene)
                if method=='chain_rl':env.supervisor=ChainSupervisor(cfg,'joint_rl')
                while True:
                    start=time.perf_counter();a=model.act(obs,True)[0];obs,_,t,u,metrics=env.step(a)
                    times.append((time.perf_counter()-start)*1000)
                    if t or u:break
                rows.append({**metrics,'method':method})
        timing[method]={'steps':len(times),'mean_ms':float(np.mean(times)),'p95_ms':float(np.percentile(times,95)),
                        'max_ms':max(times),'over_500ms':int(np.sum(np.array(times)>500))}
    result={'scaling':scaling,'equivalence':equivalence,'timing':timing,'scope':'Single local simulator worker after main study. Scaling is synthetic fixed-order models, not a 64-car SUMO experiment; brute column excludes trajectory generation.'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2),newline='\n')
    with (OUT/'timing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'runs/v3_supplement_reproduction')
    main(parser.parse_args().out)
