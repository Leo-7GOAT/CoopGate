"""Actual closed-loop margin fixtures; every floor and fixed MAPPO seed retained."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from coopgate.gain_margin_gate import ScalarGainMarginGate
from coopgate.gain_margin_audit import review_gain_trace
from coopgate.mappo_deployment import make_recovery_gate
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.recovery_gate_audit_v35 import relative_outcome,cell_summary
from audit_buffered_scope_v25 import TimedGate,check_calls
from experiment_ego_v13 import run_episode
from evaluate_scalar_nominal_sumo import load_models,sha,write

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'
FLOORS=(0,25,50,100)
METHODS=('baseline',)+tuple(f'gain_{s}_{f}' for s in (4703,4704,4705) for f in FLOORS)


def cases():
    return [(96902001,'slow_yield','lag_strong',mode) for mode in ('clean','uniform','pulse')]+[(97110131,'dense','lag_strong','clean')]


def gate(method,models,profile):
    if method=='baseline':return make_recovery_gate(None,profile,'baseline')
    _,seed,floor=method.split('_');return ScalarGainMarginGate(models[int(seed)],profile,int(floor)/10000.)


def original_records():
    lookup={};hashes={}
    for name in ('parameter_scenario_regression_20261102','scalar_nominal_sumo_20261102'):
        p=BASE/'development'/name;proof=json.loads((p/'verification.json').read_text());complete=json.loads((p/'collection_complete.json').read_text())
        if not proof['all_match'] or sha(p/'episodes.jsonl')!=complete['episodes_sha256']:raise ValueError('Complete reviewed unchanged zero-floor reference required')
        hashes[name]=sha(p/'episodes.jsonl')
        for line in (p/'episodes.jsonl').read_text().splitlines():
            r=json.loads(line);method=r['row']['method_id']
            if method=='baseline':lookup[tuple(r['key'][:4])+('baseline',)]=r
            if method.startswith('parameter_') or method.startswith('scalar_'):
                lookup[tuple(r['key'][:4])+(method.split('_')[-1],)]=r
    return lookup,hashes


def zero_parity(record,lookup):
    method=record['row']['method_id']
    if method!='baseline' and not method.endswith('_0'):return
    seed='baseline' if method=='baseline' else method.split('_')[1]
    ref=lookup[tuple(record['key'][:4])+(seed,)]
    for key,value in ref['row'].items():
        if key not in ('setup_ms','controller_ms','method_id') and record['row'].get(key)!=value:
            raise ValueError('Zero-floor original row differs: '+key)
    if len(ref['trace'])!=len(record['trace']):raise ValueError('Zero-floor trace length differs')
    for old,new in zip(ref['trace'],record['trace']):
        for key,value in old.items():
            if key!='gate_ms' and new.get(key)!=value:raise ValueError('Zero-floor original trace differs: '+key)


def collect(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    models,identities=load_models();original,original_hashes=original_records()
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/build_gain_margin_gate.py',
        ROOT/'tools/run_gain_margin_stage.py',ROOT/'tools/evaluate_scalar_nominal_sumo.py',
        ROOT/'tools/audit_buffered_scope_v25.py',ROOT/'tools/experiment_ego_v13.py',
        ROOT/'tests/test_gain_margin_gate.py',ROOT/'docs/gain_margin_20261102_protocol.md']
    plan=dict(cases=cases(),methods=METHODS,expected_executions=52,model=identities,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},original_data_sha256=original_hashes,
        reused_independent_initials=2,extra_independent_initials=0,formal_calibration=False,risk_certificate=False,
        timing='Full independent callbacks including exceptions,rotating/reversing all13 methods;shared desktop,no hard-real-time guarantee',
        scope='Reused mechanism fixtures only,NOT generalization. No offline removal of decisions/episodes.')
    write(out/'plan.json',plan);warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for method in METHODS:gate(method,models,'lag_strong').select(warm,np.zeros(3))
    count=0;started=time.perf_counter()
    with MatchedShiftedMergeEnv(plant='lag_strong') as env,(out/'episodes.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for index,(seed,scene,profile,mode) in enumerate(cases()):
            offset=index%len(METHODS);methods=METHODS[offset:]+METHODS[:offset]
            if index%2:methods=methods[::-1]
            for method in methods:
                holder=[]
                def factory(_):
                    g=TimedGate(gate(method,models,profile));holder.append(g);return g
                row,trace,initial=run_episode(env,seed,scene,mode,factory,True)
                costs=np.array(row['per_vehicle_finish'])+.05*np.array(row['per_vehicle_variation'])
                row.update(initial=initial.tolist(),plant_name=profile,method_id=method,
                    rule='baseline' if method=='baseline' else 'parameter_scenario_confidence',
                    scenario_max_failed_paths=None if method=='baseline' else 0,
                    certificate_contract='bounded_actuator_memory_stopping',per_vehicle_cost=costs.tolist(),
                    team_cost=float(costs.mean()),control_steps=len(trace),
                    learned_action_steps=sum(bool(t['actual_learning_used']) for t in trace),
                    gain_floor=None if method=='baseline' else int(method.split('_')[-1])/10000.)
                record=dict(key=[seed,scene,profile,mode,method],row=row,trace=trace,calls=holder[0].calls)
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();count+=1
                check_calls(row,trace,record['calls']);review_gain_trace(row,trace);zero_parity(record,original)
            print(json.dumps(dict(completed=count,expected=52,elapsed_s=time.perf_counter()-started)),flush=True)
    if count!=52:raise ValueError('Incomplete52-run fixture grid')
    write(out/'collection_complete.json',dict(executions=52,episodes_sha256=sha(out/'episodes.jsonl'),elapsed_s=time.perf_counter()-started))


def review(out):
    out=Path(out);torch.set_num_threads(1);plan=json.loads((out/'plan.json').read_text());complete=json.loads((out/'collection_complete.json').read_text())
    if sha(out/'episodes.jsonl')!=complete['episodes_sha256']:raise ValueError('Raw evidence changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    models,identities=load_models();original,hashes=original_records()
    if identities!=plan['model'] or hashes!=plan['original_data_sha256']:raise ValueError('Model/reference changed')
    records=[json.loads(s) for s in (out/'episodes.jsonl').read_text().splitlines()];lookup={tuple(r['key']):r for r in records}
    expected={tuple(c)+(m,) for c in cases() for m in METHODS}
    if len(records)!=52 or len(lookup)!=52 or set(lookup)!=expected:raise ValueError('Missing or duplicate fixture')
    commands=substeps=0
    for record in records:
        row,trace=record['row'],record['trace'];check_calls(row,trace,record['calls']);zero_parity(record,original)
        proof=review_gain_trace(row,trace);commands+=proof['commands'];substeps+=proof['substeps']
        g=gate(row['method_id'],models,row['plant_name']);state=np.array(row['initial']);prev=np.zeros(3)
        for item in trace:
            command,info=g.select(state,prev)
            if command.tobytes()!=np.asarray(item['command']).tobytes():raise ValueError('New controller replay command differs')
            for key,value in info.items():
                if key!='gate_ms' and item.get(key)!=value:raise ValueError('Full margin evidence differs: '+key)
            state=np.array(item['x']);prev=command
        if row['aborted']:raise ValueError('Accurate aborted prefix retained;mechanism fixture failed')
    methods={}
    for method in METHODS:
        pairs=[]
        for case in cases():
            r=lookup[tuple(case)+(method,)];base=lookup[tuple(case)+('baseline',)]['row']
            pairs.append(dict(**relative_outcome(r['row'],base),row=r['row'],baseline=base,calls=r['calls']))
        methods[method]=cell_summary(pairs)
    write(out/'report.json',dict(methods=methods,executions=52,reused_independent_initials=2,
        extra_independent_initials=0,risk_certificate=False,generalization_proved=False))
    proof=dict(all_match=True,actual_sumo_episodes=52,commands_matched=commands,independent_physical_substeps=substeps,
        full_controller_replays=52,zero_floor_preserves_original_actual_runs=True,extra_independent_initials=0,
        risk_certificate=False,report_sha256=sha(out/'report.json'))
    write(out/'verification.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('collect','review'));p.add_argument('--out',required=True);a=p.parse_args()
    (collect if a.action=='collect' else review)(a.out)
