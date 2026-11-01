"""Paired whole-gate replay/callback measurement, no extra SUMO trials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from coopgate.certificate_cached_recovery import CertificateCachedRecoveryGate
from coopgate.mappo_deployment import load_deployment_actor
from coopgate.memoized_recovery_v37 import MemoizedRecoveryGate

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('actual_learning_used','actual_learning_used_ever','baseline_command','baseline_evidence',
        'agent_actions','actor_proposal','composition','repair_weight','selected_bank_differences',
        'selected_phase_differences','finite_bank_performance_margin','certified','in_set',
        'actuator_memory_lower','actuator_memory_upper')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(source,out,scope):
    source,out=Path(source),Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    proof=json.loads((source/'verification.json').read_text())
    receipt=json.loads((source/'native_receipt.json').read_text())
    original=json.loads((source/'plan.json').read_text())
    complete=json.loads((source/'collection_complete.json').read_text())
    if (not proof['all_match'] or receipt['review_actual_native_exit_code']!=0
            or sha(source/'episodes.jsonl')!=complete['episodes_sha256']):
        raise ValueError('Completed reviewed source required')
    model,identity=load_deployment_actor(original['checkpoint'])
    if identity!=original['model']:raise ValueError('Source MAPPO changed')
    rows=[json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['row']['rule'] in ('nominal_performance','bank_performance')]
    if scope=='smoke':
        rows=[r for r in rows if r['row']['plant_name']=='nominal'
              and r['row']['scenario'] in ('nominal','dense') and r['row']['disturbance']=='clean']
    if len(rows)!=(4 if scope=='smoke' else 160):raise ValueError('Fixed complete replay scope required')
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'docs/certificate_cached_recovery_20261101_protocol.md']
    frozen={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    plan=dict(scope=scope,source=str(source.resolve()),source_episodes_sha256=sha(source/'episodes.jsonl'),
        model=identity,keys=[r['key'] for r in rows],implementations=['memoized','certificate_cached'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=frozen,actual_extra_sumo_runs=0,extra_independent_initials=0,
        timing='Independent whole callbacks in rotating/reversing per-episode order; concurrent experiments may be active, no isolated/hard-real-time claim')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for cls in (MemoizedRecoveryGate,CertificateCachedRecoveryGate):
        for profile in sorted({r['row']['plant_name'] for r in rows}):
            cls(model,profile,'nominal_performance').select(warm,np.zeros(3))
    times={};work={};calls=0;started=time.perf_counter()
    with (out/'callbacks.jsonl').open('x',encoding='utf-8') as stream:
        for index,record in enumerate(rows):
            row=record['row'];order=[('memoized',MemoizedRecoveryGate),('certificate_cached',CertificateCachedRecoveryGate)]
            if index%2:order.reverse()
            for name,cls in order:
                gate=cls(model,row['plant_name'],row['rule']);state=np.asarray(row['initial']);prev=np.zeros(3)
                key=row['rule']+'/'+name
                times.setdefault(key,[]);work.setdefault(key,dict(planning=0,rejections=0,bound_substeps=0,certificate_requests=0,certificate_hits=0,certificate_misses=0))
                for step,item in enumerate(record['trace']):
                    start=time.perf_counter();error=None;actual=None;info=None
                    try:actual,info=gate.select(state,prev)
                    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc))
                    wall_ms=1000*(time.perf_counter()-start)
                    saved=dict(key=record['key'],implementation=name,step=step,wall_ms=wall_ms,error=error,
                               command=None if actual is None else actual.tolist(),info=info)
                    stream.write(json.dumps(saved,allow_nan=False)+'\n');stream.flush()
                    if error:raise RuntimeError('Unexpected gate failure; evidence preserved: '+str(error))
                    np.testing.assert_array_equal(actual,item['command'])
                    for field in FIELDS:
                        if info[field]!=item[field]:raise ValueError('Admission semantics changed: '+field)
                    times[key].append(wall_ms);calls+=1
                    work[key]['planning']+=info['proposal_cache']['computed_successful_transitions']
                    work[key]['rejections']+=info.get('optimistic_rejected_forecasts',0)
                    work[key]['bound_substeps']+=info.get('optimistic_bound_physical_substeps',0)
                    for field in ('requests','hits','misses'):
                        work[key]['certificate_'+field]+=info.get('nominal_certificate_cache',{}).get(field,0)
                    state=np.asarray(item['x']);prev=actual
            for path,digest in frozen.items():
                if sha(ROOT/path)!=digest:raise ValueError('Frozen prototype dependency changed: '+path)
            print(json.dumps(dict(episodes_replayed=index+1,expected=len(rows),callbacks=calls)),flush=True)
    summary={}
    for key,values in times.items():
        arr=np.asarray(values);summary[key]=dict(callbacks=len(arr),sum_ms=float(arr.sum()),
            p95_ms=float(np.quantile(arr,.95)),max_ms=float(arr.max()),over_500ms=int((arr>500).sum()),**work[key])
    result=dict(all_commands_and_admission_fields_match=True,source_episodes=len(rows),
        whole_controller_replays=2*len(rows),callback_count=calls,scope=scope,summary=summary,
        extra_sumo_executions=0,extra_independent_initials=0,elapsed_s=time.perf_counter()-started,
        callbacks_sha256=sha(out/'callbacks.jsonl'),risk_certificate=False,
        meaning='Exact paired development replay; no new efficacy population or hard-real-time guarantee')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--scope',choices=('smoke','full'),default='full');args=parser.parse_args()
    run(args.source,args.out,args.scope)
