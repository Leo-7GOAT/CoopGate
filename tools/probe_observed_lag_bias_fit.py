"""Reused adverse-prefix diagnosis; no new SUMO or deployment inference.

Every attempt4 harmful model-initial is retained, including unsupported plants
and reset-time cases. Predictions see only measured prior states/commands.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from coopgate.observed_lag_bias_fit import fit_observed_history
from coopgate.episode_store_v21 import observations
from coopgate.actuator_recovery_v34 import class_bounds
from coopgate.memoized_recovery_v37 import make_cache,CAPACITY
from coopgate.scalar_nominal_recovery import forecast_kernel
from coopgate.fast_v8 import parameters
from coopgate.merge_v4 import MergeConfig
import mappo_coupled_formal_experiment as e
from mappo_completed_attempt4 import complete_evidence


def run():
    torch.set_num_threads(1);complete_evidence();started=time.perf_counter()
    source=e.BASE/'formal_mappo_attempt4_diagnosis_20261102'
    receipt=e.read(source/'owner_native_exit.json')
    if receipt['actual_native_exit_code']!=0 or receipt['report_sha256']!=e.sha(source/'report.json'):
        raise ValueError('Terminal original complete diagnosis required')
    issues=e.read(source/'all_false_admissions.json')['all_target_harmful_adoptions']
    out=e.BASE/'observed_lag_bias_prefix_probe_20261103';out.mkdir(exist_ok=False)
    packed=parameters(MergeConfig());results=[]
    block_cache={}
    for issue in issues:
        key=(issue['stage'],issue['source_shard'],issue['cluster_id'])
        if key not in block_cache:
            root=e.FORMAL/issue['stage']/f"shard{issue['source_shard']}"
            complete=e.read(root/'collection_complete.json')
            block=next(b for b in complete['blocks'] if any(k[0]==issue['cluster_id'] for k in b['keys']))
            db=root/block['block']/'observations.sqlite'
            if e.sha(db)!=block['database_sha256']:raise ValueError('Original block changed')
            block_cache[key]={}
            for case,r,t in observations(db,block['metadata'],block['keys']):
                if case[0]==issue['cluster_id']:
                    r['trace']=t;block_cache[key][case[1]]=r
        record=block_cache[key][issue['method']];row=record['row'];trace=record['trace']
        result={k:issue[k] for k in ('stage','cluster_id','seed','method','plant','disturbance')};result['admissions']=[]
        for admission in issue['admissions']:
            step=admission['step'];item=trace[step]
            if issue['plant']=='gain_drag':
                result['admissions'].append(dict(step=step,supported=False,reason='Non-unit gain/nonzero drag class'))
                continue
            states=[np.array(row['initial'])]+[np.array(t['x']) for t in trace[:step]]
            commands=np.array([t['command'] for t in trace[:step]],dtype=float).reshape((-1,3))
            fit=fit_observed_history(states,commands,issue['plant'])
            if fit is None:
                result['admissions'].append(dict(step=step,supported=False,reason='No usable observed history'))
                continue
            rho,bias,memory=fit['retention'],fit['bias'],fit['memory']
            state=states[-1];previous=np.zeros(3) if step==0 else commands[-1]
            mlo=np.array(item['actuator_memory_lower']);mhi=np.array(item['actuator_memory_upper'])
            if np.any(memory<mlo-1e-10) or np.any(memory>mhi+1e-10):raise ValueError('Fitted memory outside declared common interval')
            args=(state,previous,memory,mlo,mhi,step,np.tile(bias,(250,1)),np.ones(3),np.zeros(3),rho,class_bounds(issue['plant']),packed)
            cache,stats=make_cache();before=forecast_kernel(*args,np.zeros(3),False,cache,stats,CAPACITY)
            cache,stats=make_cache();after=forecast_kernel(*args,np.array(item['command']),True,cache,stats,CAPACITY)
            delta=after[0]-before[0];phase=after[1]-before[1];valid=bool(before[2] and after[2])
            score=float(-delta[:,0].mean()/before[0][:,0].mean()) if valid else None
            rejected=bool(not valid or np.any(delta>1e-10) or np.any(phase>.05+1e-10) or score<.0025)
            result['admissions'].append(dict(step=step,supported=True,retention=rho.tolist(),bias=bias.tolist(),fit_squared_error=fit['fit_squared_error'].tolist(),
                complete_pair=valid,differences=delta.tolist(),phase_differences=phase.tolist(),score=score,would_reject_fixed_recorded_command=rejected))
        results.append(result)
        print(json.dumps(dict(seed=issue['seed'],method=issue['method'],first=result['admissions'][0])),flush=True)
    report=dict(method_initial_incidents=len(results),first_recorded_admissions_rejected=sum(r['admissions'][0].get('would_reject_fixed_recorded_command',False) for r in results),
        all_cases=results,extra_sumo_executions=0,new_independent_initials=0,new_error_allocated=False,
        deployment_certificate=False,closed_loop_repair_proved=False,
        source_sha256={p:e.sha(e.ROOT/p) for p in ('tools/probe_observed_lag_bias_fit.py','src/coopgate/observed_lag_bias_fit.py')},
        diagnosis_sha256=e.sha(source/'all_false_admissions.json'),elapsed_s=time.perf_counter()-started)
    e.write(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('all_cases','source_sha256')}),flush=True)


if __name__=='__main__':run()
