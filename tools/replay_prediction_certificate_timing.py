"""Order-balanced callback replay of the complete fixed development trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import observations
from audit_buffered_scope_v25 import TimedGate
import benchmark_prediction_certificate_reuse as source

io,ROOT=source.io,source.ROOT
SCRIPT='tools/replay_prediction_certificate_timing.py'
DOC='docs/prediction_certificate_balanced_timing.md'
NAMES=('memoized','certificate_reuse')


def order_for(case,step,repeat):
    return NAMES if (case+step+repeat)%2==0 else NAMES[::-1]


def normalized(result):
    command,info=result
    values={k:v for k,v in info.items() if k not in ('gate_ms','prediction_certificate_reuse')}
    return np.asarray(command).tobytes(),values


def run(parent,output):
    for name in (SCRIPT,DOC): io.committed(ROOT/name)
    plan,report,done=[io.read(parent/n) for n in ('plan.json','report.json','complete.json')]
    plan_sha=io.digest(parent/'plan.json')
    if (done['report_sha256']!=io.digest(parent/'report.json') or done['plan_sha256']!=plan_sha
            or report['plan_sha256']!=plan_sha or report['actual_sumo_runs']!=48
            or len(plan['cases'])!=24 or len(report['blocks'])!=24
            or not report['all_original_non_timing_exact']):
        raise ValueError('Complete fixed 24-pair development evidence required')
    hashes={**plan['source_sha256'],**{n:io.digest(ROOT/n) for n in (SCRIPT,DOC)}}
    for n,h in hashes.items():
        if io.digest(ROOT/n)!=h: raise ValueError('Source changed: '+n)
    for model in plan['models'].values():
        if io.digest(ROOT/model['path'])!=model['sha256']: raise ValueError('Model changed')
    output.mkdir(parents=True,exist_ok=False)
    io.exclusive_json(output/'plan.json',dict(parent_plan_sha256=plan_sha,parent_report_sha256=done['report_sha256'],
        source_sha256=hashes,models=plan['models'],rounds=2,cases=24,new_sumo_runs=0,
        new_independent_initials=0,order='(case+step+round)%2',created_after_first_timing_results=True))
    torch.set_num_threads(1)
    models={k:load_coupled(ROOT/v['path'])[0] for k,v in plan['models'].items()}
    warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for item in plan['cases']:
        for cls in source.IMPLEMENTATIONS.values():
            cls(models[item['model']],item['profile'],item['rule']).select(warm,np.zeros(3))
    logs=[];expected=0
    for item,block in zip(plan['cases'],report['blocks']):
        index=item['index'];directory=parent/f'{index:03d}'
        assert block['index']==index
        for n,h in block['files_sha256'].items():
            if io.digest(directory/n)!=h: raise ValueError('Input block changed')
        saved={key[-1]:dict(**body,trace=trace) for key,body,trace in observations(directory/'observations.sqlite',
            dict(plan_sha256=plan_sha,block_index=index),[[index,n] for n in item['order']])}
        if source.non_timing(saved[NAMES[0]])!=source.non_timing(saved[NAMES[1]]):
            raise ValueError('Historical paired outputs differ')
        original=saved['memoized'];expected+=4*len(original['calls'])
        case_logs=[]
        for repeat in (0,1):
            gates={n:TimedGate(cls(models[item['model']],item['profile'],item['rule']))
                   for n,cls in source.IMPLEMENTATIONS.items()}
            state=np.array(original['row']['initial']);previous=np.zeros(3)
            for step,call in enumerate(original['calls']):
                order=order_for(index,step,repeat);outputs={}
                for name in order:
                    result=None;error=None
                    try: result=gates[name].select(state,previous)
                    except Exception as exc: error=dict(type=type(exc).__name__,message=str(exc))
                    if error!=call['error']:
                        raise ValueError('Recorded terminal failure changed')
                    if error is None:
                        command,info=result;reference=original['trace'][step]
                        if np.asarray(command).tobytes()!=np.array(reference['command']).tobytes():
                            raise ValueError('Recorded command changed')
                        for k,v in normalized(result)[1].items():
                            if reference.get(k)!=v: raise ValueError('Recorded evidence changed: '+k)
                        outputs[name]=normalized(result)
                        evidence_sha=hashlib.sha256(json.dumps(outputs[name][1],sort_keys=True,allow_nan=False).encode()).hexdigest()
                    else: evidence_sha=None
                    case_logs.append(dict(case=index,rule=item['rule'],repeat=repeat,step=step,
                        order=list(order),implementation=name,wall_ms=gates[name].calls[-1]['wall_ms'],
                        error=error,evidence_sha256=evidence_sha))
                if outputs and outputs[NAMES[0]]!=outputs[NAMES[1]]:
                    raise ValueError('Implementations differ on the same input')
                if call['error'] is None:
                    state=np.array(original['trace'][step]['x'])
                    previous=np.array(original['trace'][step]['command'])
        io.exclusive_json(output/f'{index:03d}.json',dict(case=index,calls=case_logs))
        logs.extend(case_logs)
        print(json.dumps(dict(replayed_cases=index+1,expected_cases=24)),flush=True)
    if len(logs)!=expected: raise ValueError('Incomplete expected callbacks')
    timing={}
    for repeat in (None,0,1):
        tag='combined' if repeat is None else f'round_{repeat}'
        timing[tag]={}
        for rule in ('nominal_performance','bank_performance'):
            summary={}
            for name in NAMES:
                values=[r['wall_ms'] for r in logs if r['rule']==rule and r['implementation']==name
                        and (repeat is None or r['repeat']==repeat)]
                summary[name]=dict(calls=len(values),total_ms=sum(values),mean_ms=float(np.mean(values)),
                                  p95_ms=float(np.quantile(values,.95)),max_ms=max(values),over_500ms=sum(v>500 for v in values))
            summary['total_time_reduction_percent']=100*(1-summary[NAMES[1]]['total_ms']/summary[NAMES[0]]['total_ms'])
            timing[tag][rule]=summary
    for n,h in hashes.items():
        if io.digest(ROOT/n)!=h: raise ValueError('Source changed during replay')
    io.exclusive_json(output/'report.json',dict(callbacks=len(logs),all_exact=True,new_sumo_runs=0,
        new_independent_initials=0,timing=timing,
        files_sha256={p.name:io.digest(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(callbacks=len(logs),timing=timing)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.input,args.output)
