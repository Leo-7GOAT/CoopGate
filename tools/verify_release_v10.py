"""Recompute provenance, complete paired accounting, statistics and SUMO traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.experiment_v10 import prepare
from coopgate.learning_v6 import load_models
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate
from coopgate.fast_v8 import parameters
from report_v10 import compute

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v10'


def nested_close(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:nested_close(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):nested_close(x,y)
    elif isinstance(a,float):np.testing.assert_allclose(a,b,rtol=0,atol=1e-9)
    else:assert a==b,(a,b)


def main(replay):
    hashes=json.loads((D/'SHA256SUMS.json').read_text())
    actual_files={str(p.relative_to(D)).replace('\\','/') for p in D.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    assert set(hashes)==actual_files
    for name,digest in hashes.items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
    protocol=json.loads((D/'protocol.json').read_text()); manifest=json.loads((D/'study/manifest.json').read_text())
    for record in (protocol,manifest):
        for name,digest in record['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    seeds=[s for group in protocol['test_seeds'].values() for s in group]
    assert len(seeds)==len(set(seeds))==200
    for version in ('v6','v7','v7_pair','v8','v9'):
        old=json.loads((ROOT/f'artifacts/{version}/protocol.json').read_text())
        old_seeds=[s for key in ('data_seeds','calibration_seeds','test_seeds') for group in old.get(key,{}).values() for s in group]
        assert not set(seeds).intersection(old_seeds)
    rows=list(csv.DictReader((D/'study/episodes.csv').open()));timings=json.loads((D/'study/timing.json').read_text())
    result=compute(protocol,rows,timings);nested_close(result,json.loads((D/'summary.json').read_text()))
    assert result['all_criteria_passed'] and all(result['criteria'].values())
    assert len(rows)==manifest['episodes']==6400
    lookup={(r['scenario'],int(r['seed']),r['disturbance'],r['method']):r for r in rows}
    initials=json.loads((D/'study/initial_states.json').read_text())
    assert len(initials)==200 and {r['seed'] for r in initials}==set(seeds)
    config=MergeConfig(**protocol['config']); packed=parameters(config)
    models=load_models(ROOT/'artifacts/v6',protocol['model_seeds']);methods=protocol['methods'];modes=protocol['disturbances']
    for item in initials:
        scenario,seed,x=item['scenario'],item['seed'],np.array(item['state'])
        for method in methods:
            _,info=prepare(method,models,x,np.zeros(3),config,protocol['radius'],seed)
            for mode_index,mode in enumerate(modes):
                row=lookup[scenario,seed,mode,method]
                order=np.random.default_rng(seed+1003+mode_index).permutation(methods).tolist()
                assert int(row['execution_order'])==order.index(method)
                assert int(row['selected'])==info['selected'] and int(row['proposal'])==info['proposal']
                assert abs(float(row['initial_base_cost_s'])-info['initial_base_cost_s'])<1e-10
                assert float(row['max_command_slew'])<=config.jerk+1e-10
                assert (row['aborted']=='True')==bool(row['abort_reason'])
                if row['aborted']=='True':assert row['completed']=='False' and float(row['mean_capped_finish_s'])==config.horizon
                if mode=='clean':
                    if method.startswith('v9_'):assert abs(float(row['mean_capped_finish_s'])-info['initial_base_cost_s'])<1e-6
                    else:assert float(row['mean_capped_finish_s'])<=info['initial_base_cost_s']+1e-6
    if (D/'pruning_audit.json').exists():
        audit=json.loads((D/'pruning_audit.json').read_text())
        assert audit['command_mismatches']==audit['value_mismatches']==0
        assert audit['unpruned_tail_steps']>audit['pruned_tail_steps']
    if replay:
        with DisturbedMergeEnv(config,'constant') as env:
            for scenario,group in protocol['test_seeds'].items():
                seed=group[0]
                for mode in modes:
                    for method in methods:
                        env.reset_disturbed(seed,scenario,mode,protocol['radius'])
                        initial=env.x.copy();env.gate,_=prepare(method,models,env.x,env.prev,config,protocol['radius'],seed)
                        while True:
                            _,_,done,truncated,metrics=env.step([4]*3)
                            if done or truncated:break
                        saved=json.loads((D/f'study/trace_{scenario}_{mode}_{method}.json').read_text())
                        assert len(saved)==len(env.trace)
                        x,prev=initial,np.zeros(3)
                        for actual,expected in zip(env.trace,saved):
                            np.testing.assert_allclose(actual['x'],expected['x'],rtol=0,atol=1e-6)
                            np.testing.assert_allclose(actual['command'],expected['command'],rtol=0,atol=1e-8)
                            np.testing.assert_array_equal(actual['disturbance'],expected['disturbance'])
                            assert robust_certificate(x,prev,np.array(actual['command']),packed,protocol['radius'])
                            for key in ('base_ticks','selected_ticks','active_profile','accepted_updates','trial_checks','pruned_trials','tail_feedback_steps'):
                                if key in actual:assert actual[key]==expected[key]
                            if 'selected_ticks' in actual:assert actual['selected_ticks']<=actual['base_ticks']
                            x,prev=np.array(actual['x']),np.array(actual['command'])
                        row=lookup[scenario,seed,mode,method]
                        for key in ('completed','aborted','collision'):assert metrics[key]==(row[key]=='True')
                        for key in ('contract_violations','outside_steps','uncertified_steps'):assert metrics[key]==int(row[key])
                        assert abs(metrics['mean_capped_finish_s']-float(row['mean_capped_finish_s']))<1e-6
                    print(f'Matched eight SUMO replays: {scenario}/{mode}',flush=True)
    print(f'Verified {len(hashes)} artifacts, 6400 executions, 200 clusters and all nine frozen criteria.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');main(parser.parse_args().replay)
