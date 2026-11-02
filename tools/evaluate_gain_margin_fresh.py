"""Fresh independent initial DEVELOPMENT,not formal calibration or certification."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from coopgate.pruned_gain_margin_gate import PrunedGainMarginGate
from coopgate.pruned_gain_margin_audit import review_pruned_gain_trace
from coopgate.mappo_deployment import make_recovery_gate
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.recovery_gate_audit_v35 import relative_outcome,cell_summary
from audit_buffered_scope_v25 import TimedGate,check_calls
from experiment_ego_v13 import run_episode
from evaluate_scalar_nominal_sumo import load_models,sha,write,SCENES,PROFILES,MODES

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'
METHODS=('baseline',)+tuple(f'gain_{s}_{f}' for s in (4703,4704,4705) for f in (0,25,50,100))


def schedule(shard):
    if shard not in range(4):raise ValueError('Fixed four-way partition required')
    all_cases=[(97250001+1000*s+100*p+m,scene,profile,mode) for s,scene in enumerate(SCENES)
        for p,profile in enumerate(PROFILES) for m,mode in enumerate(MODES)]
    if len(all_cases)!=80 or len({c[0] for c in all_cases})!=80:raise ValueError('Every factorial cell must have a distinct initial seed')
    return all_cases[shard::4]


def factory(method,models,profile):
    if method=='baseline':return make_recovery_gate(None,profile,'baseline')
    _,seed,floor=method.split('_');return PrunedGainMarginGate(models[int(seed)],profile,int(floor)/10000.)


def prerequisite():
    p=BASE/'development/pruned_gain_margin_replay_20261102'
    r=json.loads((p/'report.json').read_text());native=json.loads((p/'native_receipt.json').read_text())
    if not r['all_commands_and_complete_admission_evidence_match'] or r['source_episodes']!=48 or native['actual_native_exit_code']!=0:
        raise ValueError('Complete observed-terminal-zero exact pruning replay required')
    return dict(report_sha256=sha(p/'report.json'),native_sha256=sha(p/'native_receipt.json'))


def summarize(records,case_list):
    lookup={tuple(r['key']):r for r in records};methods={}
    for method in METHODS:
        pairs=[];percent=[]
        for case in case_list:
            r=lookup[tuple(case)+(method,)];base=lookup[tuple(case)+('baseline',)]['row']
            pairs.append(dict(**relative_outcome(r['row'],base),row=r['row'],baseline=base,calls=r['calls']))
            percent.append(100*(base['team_cost']-r['row']['team_cost'])/base['team_cost'])
        methods[method]=cell_summary(pairs)
        methods[method]['mean_paired_percentage_cost_gain']=float(np.mean(percent))
    return methods


def collect(out,shard):
    dependency=prerequisite();out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    models,identities=load_models();cases=schedule(shard)
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/run_gain_margin_fresh.py',
        ROOT/'tools/evaluate_scalar_nominal_sumo.py',ROOT/'tools/audit_buffered_scope_v25.py',ROOT/'tools/experiment_ego_v13.py',
        ROOT/'docs/gain_margin_fresh_20261102_protocol.md']
    plan=dict(shard=shard,shards=4,cases=cases,methods=METHODS,expected_executions=260,model=identities,prerequisite=dependency,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},independent_initials=20,
        formal_calibration=False,risk_certificate=False,error_allocation=None,
        timing='Each entire callback including exceptions;concurrent four-worker shared desktop,no isolation/WCET or communication latency claim')
    write(out/'plan.json',plan);warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for profile in PROFILES:
        for method in METHODS:factory(method,models,profile).select(warm,np.zeros(3))
    count=0;started=time.perf_counter()
    with ExitStack() as stack,(out/'episodes.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        envs={p:stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in PROFILES}
        for index,(seed,scene,profile,mode) in enumerate(cases):
            offset=index%len(METHODS);ordered=METHODS[offset:]+METHODS[:offset]
            if index%2:ordered=ordered[::-1]
            for method in ordered:
                holder=[]
                def make(_):
                    g=TimedGate(factory(method,models,profile));holder.append(g);return g
                row,trace,initial=run_episode(envs[profile],seed,scene,mode,make,True)
                costs=np.array(row['per_vehicle_finish'])+.05*np.array(row['per_vehicle_variation'])
                row.update(initial=initial.tolist(),plant_name=profile,method_id=method,
                    rule='baseline' if method=='baseline' else 'parameter_scenario_confidence',
                    scenario_max_failed_paths=None if method=='baseline' else 0,
                    certificate_contract='bounded_actuator_memory_stopping',per_vehicle_cost=costs.tolist(),team_cost=float(costs.mean()),
                    control_steps=len(trace),learned_action_steps=sum(bool(t['actual_learning_used']) for t in trace),
                    gain_floor=None if method=='baseline' else int(method.split('_')[-1])/10000.)
                record=dict(key=[seed,scene,profile,mode,method],row=row,trace=trace,calls=holder[0].calls)
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();count+=1
                check_calls(row,trace,record['calls']);review_pruned_gain_trace(row,trace)
            print(json.dumps(dict(completed=count,expected=260,elapsed_s=time.perf_counter()-started)),flush=True)
    if count!=260:raise ValueError('Incomplete260-execution shard')
    write(out/'collection_complete.json',dict(executions=count,episodes_sha256=sha(out/'episodes.jsonl'),elapsed_s=time.perf_counter()-started))


def review(out,shard):
    out=Path(out);torch.set_num_threads(1);plan=json.loads((out/'plan.json').read_text());complete=json.loads((out/'collection_complete.json').read_text())
    if plan['shard']!=shard or sha(out/'episodes.jsonl')!=complete['episodes_sha256']:raise ValueError('Shard/raw evidence differs')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    models,identities=load_models()
    if identities!=plan['model']:raise ValueError('MAPPO identities changed')
    records=[json.loads(s) for s in (out/'episodes.jsonl').read_text().splitlines()];lookup={tuple(r['key']):r for r in records}
    cases=schedule(shard);expected={tuple(c)+(m,) for c in cases for m in METHODS}
    if len(records)!=260 or len(lookup)!=260 or set(lookup)!=expected:raise ValueError('Missing/duplicate/unexpected shard outcome')
    commands=substeps=0
    for r in records:
        row,trace=r['row'],r['trace'];key=r['key']
        if (row['seed'],row['scenario'],row['plant_name'],row['disturbance'],row['method_id'])!=tuple(key):
            raise ValueError('Outcome row/controller identity differs from scheduled case')
        expected_floor=None if key[-1]=='baseline' else int(key[-1].split('_')[-1])/10000.
        if row['gain_floor']!=expected_floor:raise ValueError('Floor identity differs')
        check_calls(row,trace,r['calls']);p=review_pruned_gain_trace(row,trace);commands+=p['commands'];substeps+=p['substeps']
        g=factory(row['method_id'],models,row['plant_name']);state=np.array(row['initial']);prev=np.zeros(3)
        for item in trace:
            command,info=g.select(state,prev)
            if command.tobytes()!=np.asarray(item['command']).tobytes():raise ValueError('Full controller command replay differs')
            for name,value in info.items():
                if name!='gate_ms' and item.get(name)!=value:raise ValueError('Full controller evidence replay differs: '+name)
            state=np.array(item['x']);prev=command
        if row['aborted']:raise ValueError('Accurate aborted prefix retained;development verification failed')
    write(out/'report.json',dict(methods=summarize(records,cases),executions=260,independent_initials=20,
        formal_calibration=False,risk_certificate=False))
    proof=dict(all_match=True,actual_sumo_episodes=260,commands_matched=commands,independent_physical_substeps=substeps,
        full_controller_replays=260,independent_initials=20,risk_certificate=False,report_sha256=sha(out/'report.json'))
    write(out/'verification.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('collect','review'));p.add_argument('--out',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    (collect if a.action=='collect' else review)(a.out,a.shard)
