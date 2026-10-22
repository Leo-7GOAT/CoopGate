"""Audit provenance, paired accounting, decisions and real perturbed replays."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv, disturbance_tape
from coopgate.experiment_v9 import prepare
from coopgate.learning_v6 import load_models
from coopgate.merge_v4 import MergeConfig

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v9'


def main(replay):
    hashes=json.loads((D/'SHA256SUMS.json').read_text())
    for name,digest in hashes.items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
    protocol=json.loads((D/'protocol.json').read_text());manifest=json.loads((D/'study/manifest.json').read_text())
    for source in (protocol,manifest):
        for name,digest in source['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    seeds=[seed for group in protocol['test_seeds'].values() for seed in group]
    assert len(seeds)==len(set(seeds))==200
    for v in ('v6','v7','v7_pair','v8'):
        old=json.loads((ROOT/f'artifacts/{v}/protocol.json').read_text())
        old_seeds=[s for key in ('data_seeds','calibration_seeds','test_seeds') for group in old.get(key,{}).values() for s in group]
        assert not set(seeds).intersection(old_seeds)
    rows=list(csv.DictReader((D/'study/episodes.csv').open()));methods=protocol['methods'];modes=protocol['disturbances']
    lookup={(r['scenario'],int(r['seed']),r['disturbance'],r['method']):r for r in rows}
    expected={(s,seed,mode,m) for s,group in protocol['test_seeds'].items() for seed in group for mode in modes for m in methods}
    assert len(rows)==len(lookup)==manifest['episodes']==5600 and set(lookup)==expected
    initials=json.loads((D/'study/initial_states.json').read_text())
    assert len(initials)==200 and {r['seed'] for r in initials}==set(seeds)
    models=load_models(ROOT/'artifacts/v6',protocol['model_seeds']);config=MergeConfig(**protocol['config'])
    for item in initials:
        s,seed,x=item['scenario'],item['seed'],np.array(item['state'])
        for method in methods:
            gate,info=prepare(method,models,x,np.zeros(3),config,protocol['radius'],seed)
            for mode_index,mode in enumerate(modes):
                r=lookup[s,seed,mode,method]
                order=np.random.default_rng(seed+903+mode_index).permutation(methods).tolist()
                assert int(r['execution_order'])==order.index(method)
                assert int(r['selected'])==info['selected'] and int(r['proposal'])==info['proposal']
                assert abs(float(r['nominal_predicted_cost_s'])-info['nominal_predicted_cost_s'])<1e-10
                assert float(r['max_command_slew'])<=config.jerk+1e-10
                assert (r['aborted']=='True')==bool(r['abort_reason'])
                if r['aborted']=='True':assert r['completed']=='False' and float(r['mean_capped_finish_s'])==config.horizon
                if mode=='clean':assert abs(float(r['mean_capped_finish_s'])-info['nominal_predicted_cost_s'])<1e-6
        assert lookup[s,seed,'clean','v8_cached']['selected']==lookup[s,seed,'clean','nominal_feedback']['selected']
    result=json.loads((D/'summary.json').read_text());target=[r for r in rows if r['method']=='robust_learned_pair']
    failed=lambda r:r['aborted']=='True' or r['collision']=='True' or int(r['contract_violations'])>0
    for method in methods:
        group=[r for r in rows if r['method']==method];summary=result['overall'][method]
        assert summary['failed_episodes']==sum(failed(r) for r in group)
        assert summary['completed']==sum(r['completed']=='True' for r in group)
        assert abs(summary['mean_cost_s']-np.mean([float(r['mean_capped_finish_s']) for r in group]))<1e-10
        assert abs(summary['warm_first_response_p95_ms']-np.percentile([float(r['first_response_ms']) for r in group],95))<1e-9
    rng=np.random.default_rng(2026101691)
    for regime in ('all','nominal','dense'):
        keys=[(s,seed) for s,group in protocol['test_seeds'].items() if regime in ('all',s) for seed in group]
        for method in methods:
            if method=='robust_learned_pair':continue
            d=np.array([np.mean([float(lookup[s,seed,mode,method]['mean_capped_finish_s'])-
                                float(lookup[s,seed,mode,'robust_learned_pair']['mean_capped_finish_s']) for mode in modes[1:]]) for s,seed in keys])
            ci=np.percentile(rng.choice(d,(10000,len(d)),replace=True).mean(1),[2.5,97.5])
            np.testing.assert_allclose(ci,result['contrasts_perturbed'][regime][method]['paired_cluster_ci95_s'],rtol=0,atol=1e-10)
    assert result['all_criteria_passed'] and all(result['criteria'].values())
    assert not any(failed(r) for r in target) and sum(r['completed']=='True' for r in target)/len(target)>=.98
    assert result['contrasts_perturbed']['all']['robust_constant']['gain_percent']>=3.
    assert result['contrasts_perturbed']['all']['robust_constant']['paired_cluster_ci95_s'][0]>0
    assert result['failure_reduction_vs_v8']>=.8 and result['overall']['robust_learned_pair']['warm_first_response_p95_ms']<=20.
    if replay:
        with DisturbedMergeEnv(config,'constant') as env:
            for regime,group in protocol['test_seeds'].items():
                seed=group[0]
                for mode in modes:
                    for method in methods:
                        env.reset_disturbed(seed,regime,mode,protocol['radius'])
                        env.gate,_=prepare(method,models,env.x,env.prev,config,protocol['radius'],seed)
                        while True:
                            _,_,done,truncated,metrics=env.step([4]*3)
                            if done or truncated:break
                        trace=json.loads((D/f'study/trace_{regime}_{mode}_{method}.json').read_text())
                        assert len(trace)==len(env.trace)
                        for actual,saved in zip(env.trace,trace):
                            np.testing.assert_allclose(actual['x'],saved['x'],rtol=0,atol=1e-6)
                            np.testing.assert_allclose(actual['command'],saved['command'],rtol=0,atol=1e-8)
                            np.testing.assert_array_equal(actual['disturbance'],saved['disturbance'])
                        row=lookup[regime,seed,mode,method]
                        for key in ('completed','collision','aborted'):assert metrics[key]==(row[key]=='True')
                        for key in ('contract_violations','outside_steps','uncertified_steps'):assert metrics[key]==int(row[key])
                        assert abs(metrics['mean_capped_finish_s']-float(row['mean_capped_finish_s']))<1e-6
                    print(f'Matched seven actual SUMO replays: {regime}/{mode}',flush=True)
    print(f'Verified {len(hashes)} artifacts, 200 independent initial states, 5600 executions and all six criteria.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');main(parser.parse_args().replay)
