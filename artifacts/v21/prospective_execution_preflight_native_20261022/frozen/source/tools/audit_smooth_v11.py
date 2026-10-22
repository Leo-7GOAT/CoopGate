"""Post-study differential audit on predetermined trace states, not new scenes."""
import itertools
import json
from pathlib import Path
import numpy as np
from coopgate.fast_v8 import parameters
from coopgate.learning_v6 import load_models,predict
from coopgate.merge_v4 import MergeConfig
from coopgate.planning_v5 import portfolio
from coopgate.robust_v9 import feedback,robust_certificate
from coopgate.smooth_v11 import SmoothGate,binary_feedback_ordered,ternary_feedback

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v11'


def full_projection(x,prev,k,c,radius,p):
    proposal=np.minimum(np.maximum(np.where((k<p['coast_steps'])&(np.arange(3)!=p['first'])&(x[:,0]<c[11]),0.,c[5]),
        np.maximum(-c[4],prev-c[6]*c[1])),np.minimum(c[5],prev+c[6]*c[1]))
    backup=np.maximum(-c[4],prev-c[6]*c[1]);weights=np.array(p['weights'])
    best=np.inf;chosen=None
    for digits in itertools.product(range(3),repeat=3):
        u=np.array([backup[i] if d==0 else prev[i] if d==1 else proposal[i] for i,d in enumerate(digits)])
        if robust_certificate(x,prev,u,c,radius):
            score=sum(weights[i]*(u[i]-proposal[i])**2-1e-8*u[i] for i in range(3))
            if score<best:best,chosen=score,u
    if chosen is None:raise RuntimeError('Reference has no feasible action')
    return chosen


def main():
    protocol=json.loads((D/'protocol.json').read_text());c=MergeConfig(**protocol['config']);packed=parameters(c)
    initials=json.loads((D/'study/initial_states.json').read_text())
    models=load_models(ROOT/'artifacts/v6',protocol['model_seeds']);radius=protocol['radius']
    records=[]
    for regime,seeds in protocol['test_seeds'].items():
        initial=np.array(next(r['state'] for r in initials if r['seed']==seeds[0]))
        proposal=int(predict(models,initial[None])[0][0])
        for mode in protocol['disturbances']:
            trace=json.loads((D/f'study/trace_{regime}_{mode}_smooth_learned.json').read_text())
            for k in range(0,len(trace),2):
                x=initial if k==0 else np.array(trace[k-1]['x'])
                prev=np.zeros(3) if k==0 else np.array(trace[k-1]['command'])
                records.append((x,prev,k,proposal,regime,mode))
    command_bad=value_bad=projection_bad=0;fast_steps=slow_steps=0
    timings_fast=[];timings_slow=[];details=[]
    for x,prev,k,proposal,regime,mode in records:
        fast=SmoothGate(x,prev,c,radius,'pair',proposal,pruning=True)
        slow=SmoothGate(x,prev,c,radius,'pair',proposal,pruning=False)
        fast.step=slow.step=k
        u,a=fast.select(x,prev,None);v,b=slow.select(x,prev,None)
        command_bad+=not np.array_equal(u,v)
        value_bad+=any(a[key]!=b[key] for key in ('base_ticks','base_variation_units','selected_ticks','selected_variation_units','active_profile','active_family'))
        fast_steps+=a['tail_feedback_steps'];slow_steps+=b['tail_feedback_steps']
        # Both old and learned profiles, both families, all 27 reference vertices.
        for pid in dict.fromkeys((0,proposal)):
            p=portfolio()[pid];weights=np.array(p['weights'])
            projection_bad+=not np.array_equal(binary_feedback_ordered(x,prev,packed,radius,weights,p['first'],p['coast_steps'],k),
                feedback(x,prev,packed,radius,weights,p['first'],p['coast_steps'],k))
            projection_bad+=not np.array_equal(ternary_feedback(x,prev,packed,radius,weights,p['first'],p['coast_steps'],k),
                full_projection(x,prev,k,packed,radius,p))
        ft=[];st=[]
        for repeat in range(5):
            for gate,out in ((fast,ft),(slow,st)) if repeat%2==0 else ((slow,st),(fast,ft)):
                gate.step=k;_,info=gate.select(x,prev,None);out.append(info['gate_ms'])
        timings_fast.append(float(np.median(ft)));timings_slow.append(float(np.median(st)))
        details.append({'regime':regime,'disturbance':mode,'step':k,'pruned_tail_steps':a['tail_feedback_steps'],
            'unpruned_tail_steps':b['tail_feedback_steps'],'pruned_median_ms':float(np.median(ft)),
            'unpruned_median_ms':float(np.median(st))})
    result={'diagnostic_states':len(records),'sampling':'Every second decision from the first seed of each regime and all four error modes; eight smooth_learned traces. Correlated diagnostic states, not independent test samples.',
        'timing_repeats_per_state':5,'command_mismatches':int(command_bad),'value_mismatches':int(value_bad),
        'projection_mismatches':int(projection_bad),'pruned_tail_steps':fast_steps,'unpruned_tail_steps':slow_steps,
        'tail_step_reduction_percent':100*(1-fast_steps/slow_steps),
        'pruned_state_median_p95_ms':float(np.percentile(timings_fast,95)),
        'unpruned_state_median_p95_ms':float(np.percentile(timings_slow,95)),
        'states':details}
    (D/'algorithm_audit.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='states'},indent=2))


if __name__=='__main__':main()
