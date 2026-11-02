"""Whole-parameter-gate paired replay; no extra SUMO or independent trials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from coopgate.mappo_deployment import load_deployment_actor
from coopgate.parameter_scenario_gate import ParameterScenarioGate
from coopgate.scalar_nominal_recovery import ScalarNominalParameterGate

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/mappo_ctde_20261101'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def run(out,scope):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    models={s:load_deployment_actor(BASE/f'development/joint_{s}/policy.pt')[0] for s in (4703,4704,4705)}
    records=[];source_hashes={}
    for shard in range(4):
        source=BASE/f'development/parameter_scenario_grid_20261102_shard{shard}'
        complete=json.loads((source/'collection_complete.json').read_text());proof=json.loads((source/'verification.json').read_text())
        if not proof['all_match'] or sha(source/'episodes.jsonl')!=complete['episodes_sha256']:raise ValueError('Complete reviewed source required')
        source_hashes[source.name]=sha(source/'episodes.jsonl')
        records.extend(json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines())
    records=[r for r in records if r['row']['rule']=='parameter_scenario_confidence']
    records.sort(key=lambda r:tuple(r['key']))
    if scope=='smoke':
        records=[r for r in records if r['key'][0]==97001001 and r['key'][2]=='lag_strong' and r['key'][3]=='bias']
    if len(records)!=(3 if scope=='smoke' else 240):raise ValueError('Fixed complete replay grid required')
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/build_scalar_nominal_recovery.py',ROOT/'tests/test_scalar_nominal_recovery.py',ROOT/'docs/scalar_nominal_recovery_20261102_protocol.md']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    write(out/'plan.json',dict(scope=scope,source_hashes=source_hashes,source_sha256=hashes,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        episodes=len(records),whole_controller_replays=2*len(records),extra_sumo_runs=0,extra_independent_initials=0,
        timing='Full independent callbacks including exceptions; per-episode alternating order; shared desktop, no isolation/WCET claim',
        risk_certificate=False))
    warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for cls in (ParameterScenarioGate,ScalarNominalParameterGate):
        for profile in sorted({r['row']['plant_name'] for r in records}):cls(models[4703],profile).select(warm,np.zeros(3))
    timings={};calls=0;started=time.perf_counter()
    with (out/'callbacks.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for index,record in enumerate(records):
            order=[('original',ParameterScenarioGate),('scalar',ScalarNominalParameterGate)]
            if index%2:order.reverse()
            row=record['row'];model_seed=int(row['method_id'].split('_')[-1])
            for name,cls in order:
                gate=cls(models[model_seed],row['plant_name']);state=np.array(row['initial']);prev=np.zeros(3)
                values=timings.setdefault(str(model_seed)+'/'+name,[])
                for step,item in enumerate(record['trace']):
                    start=time.perf_counter();error=None;command=info=None
                    try:command,info=gate.select(state,prev)
                    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc))
                    ms=1000*(time.perf_counter()-start)
                    stream.write(json.dumps(dict(key=record['key'],implementation=name,step=step,wall_ms=ms,error=error,
                        command=None if command is None else command.tolist(),info=info),allow_nan=False)+'\n');stream.flush()
                    if error:raise RuntimeError('Unexpected exception persisted: '+str(error))
                    if command.tobytes()!=np.asarray(item['command']).tobytes():raise ValueError('Command bytes changed')
                    for field,value in info.items():
                        if field!='gate_ms' and item.get(field)!=value:raise ValueError('Original non-timing field changed: '+field)
                    values.append(ms);calls+=1;state=np.array(item['x']);prev=command
                if row['aborted']:raise ValueError('This complete source has no registered aborted replay branch')
            print(json.dumps(dict(episodes_replayed=index+1,expected=len(records),callbacks=calls)),flush=True)
    for name,digest in hashes.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    summary={}
    for key,values in timings.items():
        a=np.asarray(values);summary[key]=dict(callbacks=len(a),sum_ms=float(a.sum()),p95_ms=float(np.quantile(a,.95)),max_ms=float(a.max()),over_500ms=int((a>500).sum()))
    report=dict(all_commands_and_original_fields_match=True,episodes=len(records),whole_controller_replays=2*len(records),
        calls=calls,summary=summary,extra_sumo_runs=0,extra_independent_initials=0,risk_certificate=False,
        elapsed_s=time.perf_counter()-started,callbacks_sha256=sha(out/'callbacks.jsonl'))
    write(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--scope',choices=('smoke','full'),default='full');args=p.parse_args()
    run(args.out,args.scope)
