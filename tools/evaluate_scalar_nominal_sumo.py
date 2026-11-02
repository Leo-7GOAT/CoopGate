"""Fresh actual SUMO pairs for the exact scalar implementation, not certification."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from coopgate.mappo_deployment import load_deployment_actor,make_recovery_gate
from coopgate.parameter_scenario_gate import ParameterScenarioGate
from coopgate.scalar_nominal_recovery import ScalarNominalParameterGate
from coopgate.parameter_gate_audit import review_parameter_trace
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.recovery_gate_audit_v35 import relative_outcome,cell_summary
from audit_buffered_scope_v25 import TimedGate,check_calls
from experiment_ego_v13 import run_episode

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'
SCENES=('nominal','dense','slow_yield','asymmetric_speed','tight_headway')
PROFILES=('nominal','gain_drag','lag_mild','lag_strong');MODES=('clean','uniform','bias','pulse')
METHODS=('baseline',)+tuple(f'{kind}_{seed}' for seed in (4703,4704,4705) for kind in ('original','scalar'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def schedule():
    return [(97110001+100*s+10*p,scene,profile,MODES[(s+p)%4])
            for s,scene in enumerate(SCENES) for p,profile in enumerate(PROFILES)]


def load_models():
    pairs={s:load_deployment_actor(BASE/f'development/joint_{s}/policy.pt') for s in (4703,4704,4705)}
    return {s:p[0] for s,p in pairs.items()},{str(s):p[1] for s,p in pairs.items()}


def factory(method,models,profile):
    if method=='baseline':return make_recovery_gate(None,profile,'baseline')
    kind,seed=method.split('_');cls=ParameterScenarioGate if kind=='original' else ScalarNominalParameterGate
    return cls(models[int(seed)],profile)


def require_replay():
    p=BASE/'development/scalar_nominal_full_20261102'
    r=json.loads((p/'report.json').read_text());native=json.loads((p/'native_receipt.json').read_text())
    if not r['all_commands_and_original_fields_match'] or r['episodes']!=240 or native['actual_native_exit_code']!=0:
        raise ValueError('Complete240-trajectory replay and observed terminal zero required')
    return dict(report_sha256=sha(p/'report.json'),native_receipt_sha256=sha(p/'native_receipt.json'))


def collect(out):
    dependency=require_replay();out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    models,identities=load_models();cases=schedule()
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/audit_buffered_scope_v25.py',ROOT/'tools/experiment_ego_v13.py',ROOT/'docs/scalar_nominal_sumo_20261102_protocol.md']
    plan=dict(cases=cases,methods=METHODS,expected_executions=140,independent_initials=20,
        model=identities,prerequisite=dependency,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        risk_certificate=False,purpose='Fresh physical/controller equivalence and complete-callback cost; not formal superiority or calibration',
        timing='Full callback including exceptions; methods rotate and reverse per case; shared desktop, not isolated/WCET')
    write(out/'plan.json',plan);warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for profile in PROFILES:
        for method in METHODS:factory(method,models,profile).select(warm,np.zeros(3))
    started=time.perf_counter();count=0
    with ExitStack() as stack,(out/'episodes.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        envs={p:stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in PROFILES}
        for index,(seed,scene,profile,mode) in enumerate(cases):
            offset=index%len(METHODS);methods=METHODS[offset:]+METHODS[:offset]
            if index%2:methods=methods[::-1]
            for method in methods:
                holder=[]
                def make(_):
                    gate=TimedGate(factory(method,models,profile));holder.append(gate);return gate
                row,trace,initial=run_episode(envs[profile],seed,scene,mode,make,True)
                costs=np.array(row['per_vehicle_finish'])+.05*np.array(row['per_vehicle_variation'])
                row.update(initial=initial.tolist(),plant_name=profile,method_id=method,
                    rule='baseline' if method=='baseline' else 'parameter_scenario_confidence',
                    scenario_max_failed_paths=None if method=='baseline' else 0,
                    certificate_contract='bounded_actuator_memory_stopping',per_vehicle_cost=costs.tolist(),
                    team_cost=float(costs.mean()),control_steps=len(trace),
                    learned_action_steps=sum(bool(t['actual_learning_used']) for t in trace))
                record=dict(key=[seed,scene,profile,mode,method],row=row,trace=trace,calls=holder[0].calls)
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();count+=1
                check_calls(row,trace,record['calls']);review_parameter_trace(row,trace,None)
            print(json.dumps(dict(completed=count,expected=140,elapsed_s=time.perf_counter()-started)),flush=True)
    if count!=140:raise ValueError('Incomplete140 execution grid')
    write(out/'collection_complete.json',dict(executions=count,episodes_sha256=sha(out/'episodes.jsonl'),elapsed_s=time.perf_counter()-started))


def behavior(record):
    return dict(row={k:v for k,v in record['row'].items() if k not in ('setup_ms','controller_ms','method_id')},
        trace=[{k:v for k,v in t.items() if k!='gate_ms'} for t in record['trace']],
        terminal_errors=[c['error'] for c in record['calls']])


def review(out):
    out=Path(out);torch.set_num_threads(1);plan=json.loads((out/'plan.json').read_text())
    complete=json.loads((out/'collection_complete.json').read_text())
    if sha(out/'episodes.jsonl')!=complete['episodes_sha256']:raise ValueError('Raw outcomes changed')
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    models,identities=load_models()
    if identities!=plan['model']:raise ValueError('Model identities changed')
    records=[json.loads(s) for s in (out/'episodes.jsonl').read_text().splitlines()];lookup={tuple(r['key']):r for r in records}
    expected={tuple(c)+(m,) for c in schedule() for m in METHODS}
    if len(records)!=140 or len(lookup)!=140 or set(lookup)!=expected:raise ValueError('Missing/duplicate case')
    commands=substeps=0
    for record in records:
        row,trace=record['row'],record['trace'];check_calls(row,trace,record['calls'])
        proof=review_parameter_trace(row,trace,None);commands+=proof['commands'];substeps+=proof['substeps']
        # Reconstruct the complete original controller, not just new-code output.
        canonical=row['method_id'].replace('scalar_','original_')
        gate=factory(canonical,models,row['plant_name']);state=np.array(row['initial']);prev=np.zeros(3)
        for item in trace:
            command,info=gate.select(state,prev)
            if command.tobytes()!=np.asarray(item['command']).tobytes():raise ValueError('Original command replay differs')
            for key,value in info.items():
                if key!='gate_ms' and item.get(key)!=value:raise ValueError('Original full evidence replay differs: '+key)
            state=np.array(item['x']);prev=command
        if row['aborted']:raise ValueError('Accurate aborted output retained; integration failed, not a fixed-N certificate')
    for case in schedule():
        for seed in (4703,4704,4705):
            if behavior(lookup[tuple(case)+(f'original_{seed}',)])!=behavior(lookup[tuple(case)+(f'scalar_{seed}',)]):
                raise ValueError('Actual SUMO behavior/evidence differs')
    summaries={}
    for method in METHODS:
        pairs=[]
        for case in schedule():
            r=lookup[tuple(case)+(method,)];base=lookup[tuple(case)+('baseline',)]['row']
            pairs.append(dict(**relative_outcome(r['row'],base),row=r['row'],baseline=base,calls=r['calls']))
        summaries[method]=cell_summary(pairs)
    write(out/'report.json',dict(methods=summaries,executions=140,independent_initials=20,risk_certificate=False,
        actual_sumo_pair_equivalence=True,raw_sha256=sha(out/'episodes.jsonl')))
    proof=dict(all_match=True,actual_sumo_episodes=140,commands_matched=commands,independent_physical_substeps=substeps,
        complete_original_gate_replays=140,original_vs_scalar_pairs=60,independent_initials=20,risk_certificate=False,
        report_sha256=sha(out/'report.json'))
    write(out/'verification.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('collect','review'));p.add_argument('--out',required=True);a=p.parse_args()
    (collect if a.action=='collect' else review)(a.out)
