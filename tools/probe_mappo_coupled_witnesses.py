"""Exhaustive post-completion prefix diagnosis; no SUMO or new inference.

Parent owns all four child handles and records actual terminal codes. Each
child retains every path, including incomplete predictions, without truncating
the predetermined adverse-prefix scope after a favorable witness is found.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib
import numpy as np
from coopgate.actuator_recovery_v34 import class_bounds
from coopgate.coupled_corner_witness import coupled_library, history_memory
from coopgate.episode_store_v21 import observations
from coopgate.fast_v8 import parameters
from coopgate.memoized_recovery_v37 import make_cache, CAPACITY
from coopgate.merge_v4 import MergeConfig
from coopgate.scalar_nominal_recovery import forecast_kernel
import mappo_formal_experiment as e
from plot_mappo_formal_results import complete_evidence

OUT=e.BASE/'development/coupled_corner_probe_20261102'
DIAG=e.BASE/'formal_mappo_attempt3_diagnosis_20261102/all_false_admissions.json'
PROFILES=('nominal','gain_drag','lag_mild','lag_strong')
SOURCE=[Path(__file__).resolve(),e.ROOT/'src/coopgate/coupled_corner_witness.py',
        e.ROOT/'docs/mappo_coupled_probe_protocol_20261102.md']


def write(path,value):
    e.write(path,value)


def freeze():
    complete_evidence()
    e.bound(e.FORMAL,False)
    OUT.mkdir(parents=True,exist_ok=False)
    issues=e.read(DIAG)['all_target_harmful_adoptions']
    write(OUT/'plan.json',dict(kind='post_completion_development_prefix_probe',
        sources={str(p.relative_to(e.ROOT)):e.sha(p) for p in SOURCE},
        diagnosis_sha256=e.sha(DIAG),certificate_sha256=e.sha(e.FORMAL/'analysis/certificate.json'),
        incidents=len(issues),prefixes=sum(len(x['admissions']) for x in issues),
        counts={p:sum(1 for _ in coupled_library(p)) for p in PROFILES},
        no_sumo=True,new_independent_initials=0,alpha_allocated=False,certification=False))


def guard():
    plan=e.read(OUT/'plan.json')
    if any(e.sha(e.ROOT/p)!=h for p,h in plan['sources'].items()) or e.sha(DIAG)!=plan['diagnosis_sha256']:
        raise ValueError('Frozen probe source/diagnosis changed')
    complete_evidence()
    e.bound(e.FORMAL,False)
    return plan


def load_block(issue):
    root=e.stage_path(e.FORMAL,issue['stage'],issue['source_shard'])
    # The historical diagnosis's source_block is the frozen global index//32.
    block=e.read(root/'collection_complete.json')['blocks'][issue['source_block']]
    db=root/block['block']/'observations.sqlite'
    if e.sha(db)!=block['database_sha256']:raise ValueError('Changed source database')
    return {key:dict(record,trace=trace) for key,record,trace in observations(db,block['metadata'],block['keys'])}


def worker(profile):
    guard();started=time.perf_counter();directory=OUT/profile;directory.mkdir(exist_ok=False)
    issues=[x for x in e.read(DIAG)['all_target_harmful_adoptions'] if x['plant']==profile]
    library=list(coupled_library(profile));packed=parameters(MergeConfig());bounds=class_bounds(profile)
    blocks={};results=[]
    for incident in issues:
        blockkey=(incident['stage'],incident['source_shard'],incident['source_block'])
        if blockkey not in blocks:blocks[blockkey]=load_block(incident)
        record=blocks[blockkey][(incident['cluster_id'],incident['method'])];trace=record['trace']
        prefixes=[]
        for admission in incident['admissions']:
            step=admission['step'];entry=trace[step]
            state=np.asarray(admission['state_before'],dtype=float);previous=np.asarray(admission['previous'],dtype=float)
            command=np.asarray(admission['command'],dtype=float)
            mlo=np.asarray(entry['actuator_memory_lower']);mhi=np.asarray(entry['actuator_memory_upper'])
            cache,stats=make_cache();paths=[]
            for index,(gain,drag,rho,bias) in enumerate(library):
                memory=history_memory([t['command'] for t in trace[:step]],rho)
                if np.any(memory<mlo-1e-12) or np.any(memory>mhi+1e-12):
                    raise ValueError('Command-history reconstruction escaped observer interval')
                tape=np.ascontiguousarray(np.tile(bias,(250,1)))
                args=(state,previous,memory,mlo,mhi,step,tape,gain,drag,rho,bounds,packed)
                before=forecast_kernel(*args,np.zeros(3),False,cache,stats,CAPACITY)
                after=forecast_kernel(*args,command,True,cache,stats,CAPACITY)
                complete=bool(before[2] and after[2]);delta=after[0]-before[0]
                phase=after[1]-before[1];denominator=float(before[0][:,0].mean())
                score=float(-delta[:,0].mean()/denominator) if complete and denominator>0 else None
                per_vehicle=bool(complete and np.any(delta>1e-10))
                phase_harm=bool(complete and np.any(phase>.05+1e-10))
                margin_reject=bool(complete and (score is None or score<.0025))
                paths.append(dict(path=index,baseline_complete=bool(before[2]),candidate_complete=bool(after[2]),
                    baseline_error=int(before[3]),candidate_error=int(after[3]),
                    planning_complete=bool(before[21] and after[21]),
                    differences=delta.tolist() if complete else None,
                    phase_differences=[float(v) if np.isfinite(v) else None for v in phase] if complete else None,
                    gain_score=score,per_vehicle_witness=per_vehicle,phase_witness=phase_harm,
                    margin_rejection=margin_reject,unassessable=not complete,
                    would_reject=bool(not complete or per_vehicle or phase_harm or margin_reject)))
            prefixes.append(dict(step=step,paths=paths,cache_stats=stats.tolist(),
                witness_paths=[p['path'] for p in paths if p['per_vehicle_witness'] or p['phase_witness']],
                rejection_paths=[p['path'] for p in paths if p['would_reject']]))
        results.append(dict(stage=incident['stage'],cluster_id=incident['cluster_id'],method=incident['method'],
            prefixes=prefixes,any_prefix_witness=any(x['witness_paths'] for x in prefixes),
            first_prefix_witness=bool(prefixes[0]['witness_paths']),
            first_prefix_rejection=bool(prefixes[0]['rejection_paths'])))
        print(json.dumps(dict(profile=profile,incidents=len(results),expected=len(issues),elapsed_s=time.perf_counter()-started)),flush=True)
    raw=json.dumps(dict(profile=profile,library=[dict(gain=g.tolist(),drag=d.tolist(),rho=r.tolist(),bias=b.tolist()) for g,d,r,b in library],
        incidents=results),allow_nan=False,separators=(',',':')).encode()
    with (directory/'all_paths.json.zlib').open('xb') as f:f.write(zlib.compress(raw))
    write(directory/'report.json',dict(profile=profile,native_pid=os.getpid(),incidents=len(results),
        prefixes=sum(len(x['prefixes']) for x in results),library_size=len(library),
        first_prefix_witness=sum(x['first_prefix_witness'] for x in results),
        first_prefix_rejection=sum(x['first_prefix_rejection'] for x in results),
        any_prefix_witness=sum(x['any_prefix_witness'] for x in results),
        all_paths_sha256=e.sha(directory/'all_paths.json.zlib'),elapsed_s=time.perf_counter()-started,
        closed_loop_repair_demonstrated=False,new_independent_initials=0))


def run():
    freeze();children=[]
    for profile in PROFILES:
        stream=(OUT/(profile+'.log')).open('xb')
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--profile',profile],cwd=e.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        write(OUT/(profile+'_started.json'),dict(native_pid=process.pid,owner_pid=os.getpid()))
        children.append((profile,process,stream))
    receipts=[]
    for profile,process,stream in children:
        code=process.wait();stream.close();receipt=dict(profile=profile,native_pid=process.pid,actual_native_exit_code=code)
        write(OUT/(profile+'_native_exit.json'),receipt);receipts.append(receipt)
    if any(r['actual_native_exit_code']!=0 for r in receipts):raise RuntimeError('Probe failed;all logs preserved')
    plan=guard();reports={p:e.read(OUT/p/'report.json') for p in PROFILES}
    if sum(r['incidents'] for r in reports.values())!=plan['incidents'] or sum(r['prefixes'] for r in reports.values())!=plan['prefixes']:
        raise ValueError('Adverse-prefix scope incomplete')
    write(OUT/'report.json',dict(completed=True,children=receipts,profiles=reports,
        original_study_joint_certified=False,new_sumo_runs=0,new_independent_initials=0,
        prospective_deployment_certified=False,closed_loop_repair_demonstrated=False))
    print(json.dumps(dict(completed=True,profiles=reports)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=PROFILES);args=parser.parse_args()
    worker(args.profile) if args.profile else run()
