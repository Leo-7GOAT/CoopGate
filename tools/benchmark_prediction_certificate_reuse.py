"""Fixed development-only SUMO equivalence and callback timing comparison."""
import argparse
import json
from pathlib import Path
from types import FunctionType

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import EpisodeStore, observations
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.memoized_recovery_v37 import MemoizedRecoveryGate
from coopgate.reuse_prediction_certificate import CertificateReuseGate
from coopgate.recovery_gate_audit_v35 import review_joint_trace
from audit_buffered_scope_v25 import check_calls
import execute_ood_validation_v39 as io
import execute_fair_failure_v44 as frozen_reference
import freeze_fair_comparison_v39 as design
import run_joint_risk_v42 as runtime

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('src/coopgate/reuse_prediction_certificate.py','tests/test_reuse_prediction_certificate.py',
         'tools/benchmark_prediction_certificate_reuse.py','docs/prediction_certificate_reuse_protocol.md')
IMPLEMENTATIONS=dict(memoized=MemoizedRecoveryGate,certificate_reuse=CertificateReuseGate)


def non_timing(record):
    return dict(row={k:v for k,v in record['row'].items() if k not in ('setup_ms','controller_ms')},
        trace=[{k:v for k,v in item.items() if k not in ('gate_ms','prediction_certificate_reuse')}
               for item in record['trace']],errors=[c['error'] for c in record['calls']])


def run_episode(case,rule,implementation,model,env):
    def factory(m,p,unused): return IMPLEMENTATIONS[implementation](m,p,rule)
    function=FunctionType(runtime.execute_episode.__code__,
        {**runtime.execute_episode.__globals__,'MemoizedRecoveryGate':factory},
        argdefs=runtime.execute_episode.__defaults__)
    result=function(case,case['policy_id'],model,env)
    result['row']['rule']=rule
    return result


def review(record,model):
    check_calls(record['row'],record['trace'],record['calls'])
    return review_joint_trace(record['row'],record['trace'],model,replay=True)


def run(output):
    for name in SOURCES: io.committed(ROOT/name)
    _,reference=frozen_reference.bound_inputs()
    source_hashes={**reference['source_sha256'],**{name:io.digest(ROOT/name) for name in SOURCES}}
    models_meta=design.load_models(ROOT)
    torch.set_num_threads(1)
    models={key:load_coupled(ROOT/value['path'])[0] for key,value in models_meta.items()}
    cases=[]
    for model in ('1802','1812','1822'):
        for profile in ('nominal','gain_drag','lag_mild','lag_strong'):
            for rule in ('nominal_performance','bank_performance'):
                index=len(cases)
                order=list(IMPLEMENTATIONS)
                if index%2: order.reverse()
                cases.append(dict(index=index,model=model,profile=profile,rule=rule,order=order))
    output.mkdir(parents=True,exist_ok=False)
    io.exclusive_json(output/'plan.json',dict(cases=cases,seed=139890001,scenario='nominal',disturbance='clean',
        actual_sumo_runs=48,formal_independent_initials=0,models=models_meta,
        source_sha256=source_hashes,reference_binding_sha256=io.digest(frozen_reference.BINDING),runtime=io.runtime_identity(),
        timing_scope='Single measurement per pair on shared desktop; no hard realtime claim'))
    sha=io.digest(output/'plan.json')
    warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for item in cases:
        for cls in IMPLEMENTATIONS.values():
            cls(models[item['model']],item['profile'],item['rule']).select(warm,np.zeros(3))
    entries=[]
    for item in cases:
        case=dict(cluster_id=f"certificate-reuse-dev-{item['index']:03d}",seed=139890001,
                  scenario='nominal',plant=item['profile'],disturbance='clean',policy_id=item['model'])
        directory=output/f"{item['index']:03d}";directory.mkdir()
        metadata=dict(plan_sha256=sha,block_index=item['index'])
        store=EpisodeStore(directory/'observations.sqlite',metadata)
        results={};reviews=[]
        try:
            with MatchedShiftedMergeEnv(plant=item['profile']) as env:
                for implementation in item['order']:
                    result=run_episode(case,item['rule'],implementation,models[item['model']],env)
                    store.append([item['index'],implementation],{k:v for k,v in result.items() if k!='trace'},result['trace'])
                    reviews.append(dict(implementation=implementation,**review(result,models[item['model']])))
                    results[implementation]=non_timing(result)
            if results['memoized']!=results['certificate_reuse']:
                raise ValueError('Any original non-timing mismatch rejects the pair')
            store.seal([[item['index'],name] for name in item['order']])
        finally:
            store.close()
        entry=dict(**metadata,reviews=reviews,non_timing_exact=True,actual_sumo_runs=2,
                   database_sha256=io.digest(directory/'observations.sqlite'),
                   seal_sha256=io.digest(directory/'observations.sqlite.seal.json'))
        io.exclusive_json(directory/'complete.json',entry)
        entries.append(dict(index=item['index'],files_sha256={n:io.digest(directory/n) for n in
                            ('observations.sqlite','observations.sqlite.seal.json','complete.json')}))
        print(json.dumps(dict(completed_pairs=len(entries),expected_pairs=24)),flush=True)
    timings={rule:{name:[] for name in IMPLEMENTATIONS} for rule in ('nominal_performance','bank_performance')}
    proofs=dict(evaluations=0,reuses=0,logical_substeps=0,computed_substeps=0)
    for item,entry in zip(cases,entries):
        directory=output/f"{item['index']:03d}"
        assert all(io.digest(directory/n)==h for n,h in entry['files_sha256'].items())
        metadata=dict(plan_sha256=sha,block_index=item['index'])
        paired={}
        for key,body,trace in observations(directory/'observations.sqlite',metadata,
                                         [[item['index'],name] for name in item['order']]):
            name=key[-1]
            paired[name]=non_timing(dict(**body,trace=trace))
            timings[item['rule']][name].extend(c['wall_ms'] for c in body['calls'])
            if name=='certificate_reuse':
                for step in trace:
                    counts=step['prediction_certificate_reuse']
                    assert counts['logical_substeps']==step['recovery_certificate_substeps']
                    assert counts['computed_substeps']<=counts['logical_substeps']
                    for field in proofs: proofs[field]+=counts[field]
        assert paired['memoized']==paired['certificate_reuse']
    summary={}
    for rule,methods in timings.items():
        summary[rule]={name:dict(calls=len(values),total_ms=sum(values),mean_ms=float(np.mean(values)),
                p95_ms=float(np.quantile(values,.95)),max_ms=max(values),over_500ms=sum(v>500 for v in values))
                      for name,values in methods.items()}
        old,new=summary[rule]['memoized'],summary[rule]['certificate_reuse']
        assert old['calls']==new['calls']
        summary[rule]['total_time_reduction_percent']=100*(1-new['total_ms']/old['total_ms'])
    for name,expected in source_hashes.items():
        if io.digest(ROOT/name)!=expected:
            raise ValueError('Source changed during development execution')
    if io.digest(output/'plan.json')!=sha: raise ValueError('Development plan changed')
    io.exclusive_json(output/'report.json',dict(plan_sha256=sha,actual_sumo_runs=48,paired_runs=24,
        all_original_non_timing_exact=True,new_risk_certificate=False,timing=summary,
        prediction_certificate_work=proofs,blocks=entries,formal_independent_initials=0))
    io.exclusive_json(output/'complete.json',dict(plan_sha256=sha,report_sha256=io.digest(output/'report.json'),
                                               actual_sumo_runs=48,paired_runs=24))
    print(json.dumps(dict(timing=summary,prediction_certificate_work=proofs)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)
