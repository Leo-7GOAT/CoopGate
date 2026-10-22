"""Verify V11 provenance, full accounting, metrics and native SUMO replays."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.experiment_v11 import prepare
from coopgate.learning_v6 import load_models
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate
from coopgate.fast_v8 import parameters
from report_v11 import compute
from verify_release_v10 import nested_close

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v11'


def main(replay,require_acceptance=False):
    hashes=json.loads((D/'SHA256SUMS.json').read_text())
    actual={str(p.relative_to(D)).replace('\\','/') for p in D.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    assert set(hashes)==actual
    for name,digest in hashes.items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
    protocol=json.loads((D/'protocol.json').read_text());manifest=json.loads((D/'study/manifest.json').read_text())
    for record in (protocol,manifest):
        for name,digest in record['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    seeds=[s for group in protocol['test_seeds'].values() for s in group]
    assert len(seeds)==len(set(seeds))==200
    for version in ('v6','v7','v7_pair','v8','v9','v10'):
        old=json.loads((ROOT/f'artifacts/{version}/protocol.json').read_text())
        old_seeds=[s for key in ('data_seeds','calibration_seeds','test_seeds') for group in old.get(key,{}).values() for s in group]
        assert not set(seeds).intersection(old_seeds)
    rows=list(csv.DictReader((D/'study/episodes.csv').open()));timings=json.loads((D/'study/timing.json').read_text())
    result=compute(protocol,rows,timings);nested_close(result,json.loads((D/'summary.json').read_text()))
    # Reproducibility is distinct from meeting a performance target. Preserve
    # negative acceptance outcomes, rather than changing thresholds to pass CI.
    assert result['all_criteria_passed']==all(result['criteria'].values())
    assert len(rows)==manifest['episodes']==7200 and manifest['single_worker']
    lookup={(r['scenario'],int(r['seed']),r['disturbance'],r['method']):r for r in rows}
    time_lookup={(r['scenario'],int(r['seed']),r['disturbance'],r['method']):r for r in timings}
    initials=json.loads((D/'study/initial_states.json').read_text())
    assert len(initials)==200 and {r['seed'] for r in initials}==set(seeds)
    config=MergeConfig(**protocol['config']);packed=parameters(config)
    models=load_models(ROOT/'artifacts/v6',protocol['model_seeds'])
    methods,modes=protocol['methods'],protocol['disturbances']
    for item in initials:
        scenario,seed,x=item['scenario'],item['seed'],np.array(item['state'])
        for method in methods:
            _,info=prepare(method,models,x,np.zeros(3),config,protocol['radius'],seed)
            for mode_index,mode in enumerate(modes):
                row=lookup[scenario,seed,mode,method];timing=time_lookup[scenario,seed,mode,method]
                order=np.random.default_rng(seed+1103+mode_index).permutation(methods).tolist()
                assert int(row['execution_order'])==order.index(method)
                assert int(row['selected'])==info['selected'] and int(row['proposal'])==info['proposal']
                assert abs(float(row['initial_v9_base_cost_s'])-info['initial_v9_base_cost_s'])<1e-10
                assert float(row['max_command_slew'])<=config.jerk+1e-10
                assert abs(float(row['peak_command_change_rate'])-float(row['max_command_slew']))<1e-10
                assert (row['aborted']=='True')==bool(row['abort_reason'])
                n=int(row['trace_steps']);assert n==len(timing['controller_ms'])
                if n:
                    assert abs(float(row['total_command_variation'])-float(row['mean_jerk'])*3*n*config.decision_dt)<1e-8
                assert abs(float(row['first_response_ms'])-float(row['selection_ms'])-(timing['cycle_ms'][0] if timing['cycle_ms'] else 0))<1e-8
                if row['aborted']=='True':
                    assert row['completed']=='False' and float(row['mean_capped_finish_s'])==config.horizon
                if mode=='clean':
                    assert float(row['mean_capped_finish_s'])<=info['initial_v9_base_cost_s']+1e-6
    if (D/'algorithm_audit.json').exists():
        audit=json.loads((D/'algorithm_audit.json').read_text())
        assert audit['command_mismatches']==audit['value_mismatches']==audit['projection_mismatches']==0
        assert audit['unpruned_tail_steps']>audit['pruned_tail_steps']
    if replay:
        count=0
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
                        x,prev,total_variation=initial,np.zeros(3),0.
                        for actual,expected in zip(env.trace,saved):
                            command=np.array(actual['command'])
                            np.testing.assert_allclose(actual['x'],expected['x'],rtol=0,atol=1e-6)
                            np.testing.assert_allclose(command,expected['command'],rtol=0,atol=1e-8)
                            np.testing.assert_array_equal(actual['disturbance'],expected['disturbance'])
                            assert robust_certificate(x,prev,command,packed,protocol['radius'])
                            if method.startswith('smooth_'):
                                np.testing.assert_array_equal(command/.625,np.rint(command/.625))
                            for key in ('base_ticks','selected_ticks','base_variation_units','selected_variation_units',
                                'active_profile','active_family','family_sweeps','accepted_updates','trial_checks','pruned_trials','tail_feedback_steps'):
                                if key in actual:assert actual[key]==expected[key]
                            if 'selected_variation_units' in actual:
                                assert actual['selected_ticks']<=actual['base_ticks']
                                assert (actual['selected_ticks'],actual['selected_variation_units'])<=(actual['base_ticks'],actual['base_variation_units']) or method=='smooth_time_only'
                            total_variation+=float(np.abs(command-prev).sum())
                            x,prev=np.array(actual['x']),command
                        row=lookup[scenario,seed,mode,method]
                        for key in ('completed','aborted','collision'):assert metrics[key]==(row[key]=='True')
                        for key in ('contract_violations','outside_steps','uncertified_steps'):assert metrics[key]==int(row[key])
                        for key in ('mean_capped_finish_s','mean_jerk','max_command_slew'):
                            assert abs(metrics[key]-float(row[key]))<1e-6
                        assert abs(total_variation-float(row['total_command_variation']))<1e-8
                        count+=1
                    print(f'Matched nine SUMO replays: {scenario}/{mode}',flush=True)
        assert count==72
    passed=sum(result['criteria'].values());failed=[k for k,v in result['criteria'].items() if not v]
    print(f'Verified {len(hashes)} artifacts, 7200 executions and 200 clusters. Frozen performance criteria: {passed}/10 passed.')
    if failed:print('Unmet criteria retained: '+', '.join(failed))
    if require_acceptance and failed:
        raise SystemExit('Frozen performance acceptance is not fully met')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true')
    parser.add_argument('--require-acceptance',action='store_true',help='Also require every frozen performance criterion to pass')
    args=parser.parse_args();main(args.replay,args.require_acceptance)
