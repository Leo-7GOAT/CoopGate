"""Actual SUMO integration of audited MAPPO and unchanged recovery gates.

Separate collect/review invocations preserve native-exit accountability. This
is a development integration study, never a risk or real-time certificate.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from coopgate.mappo_deployment import load_deployment_actor, make_recovery_gate
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.recovery_gate_audit_v35 import relative_outcome, cell_summary
from coopgate.scenario_gate_audit import review_scenario_trace
from coopgate.scenario_confidence_gate import ScenarioConfidenceGate
from coopgate.parameter_scenario_gate import ParameterScenarioGate
from coopgate.parameter_gate_audit import review_parameter_trace
RULES=('baseline',)+tuple(f'{kind}_{seed}' for seed in (4703,4704,4705) for kind in ('original','parameter'))
from audit_buffered_scope_v25 import TimedGate, check_calls
from experiment_ego_v13 import run_episode

ROOT=Path(__file__).resolve().parents[1]
PROFILES=('nominal','gain_drag','lag_mild','lag_strong')
SCENES=('nominal','dense','slow_yield','asymmetric_speed','tight_headway')
MODES=('clean','uniform','bias','pulse')



def load_models():
    loaded={seed:load_deployment_actor(ROOT/f'artifacts/mappo_ctde_20261101/development/joint_{seed}/policy.pt') for seed in (4703,4704,4705)}
    return {s:v[0] for s,v in loaded.items()},{str(s):v[1] for s,v in loaded.items()}

def factory_gate(method,models,profile):
    if method=='baseline':return make_recovery_gate(None,profile,'baseline')
    kind,seed=method.split('_')
    cls=ScenarioConfidenceGate if kind=='original' else ParameterScenarioGate
    return cls(models[int(seed)],profile,0)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def schedule(n,seed_base,scope,shard=0,shards=1):
    if shards not in (1,4) or not 0<=shard<shards or scope in ('smoke','regression') and shards!=1:
        raise ValueError('Explicit one/four-way partition, no sharded smoke')
    if not 0<n<1000 or scope not in ('smoke','integration','regression'):
        raise ValueError('Explicit bounded integration or numerical smoke required')
    if scope=='smoke' and n!=1:
        raise ValueError('Smoke has exactly one initial, plant, mode and scene')
    if scope=='regression':return [(96902001,'slow_yield','lag_strong',mode) for mode in ('clean','uniform','pulse')]
    scenes=SCENES if scope=='integration' else SCENES[:1]
    profiles=PROFILES if scope=='integration' else PROFILES[:1]
    modes=MODES if scope=='integration' else MODES[:1]
    complete=[(seed_base+1000*s+i,scene,profile,mode)
            for s,scene in enumerate(scenes) for i in range(n)
            for profile in profiles for mode in modes]
    return complete[shard::shards]


def collect(out,checkpoint,n=1,seed_base=97000001,scope='integration',runtime_context='Not isolated; concurrent workload unspecified',shard=0,shards=1):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    models,identity=load_models()
    cases=schedule(n,seed_base,scope,shard,shards)
    sources=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),
        ROOT/'tools/audit_buffered_scope_v25.py',ROOT/'tools/experiment_ego_v13.py',ROOT/'docs/scenario_score_sumo_20261102_protocol.md',
        ROOT/'docs/scenario_score_development_20261102_protocol.md',ROOT/'docs/parameter_scenario_sumo_20261102_protocol.md',
        ROOT/'docs/parameter_scenario_development_20261102_protocol.md']
    plan=dict(scope=scope,n=n,seed_base=seed_base,shard=shard,shards=shards,checkpoint=None,
        model=identity,cases=cases,methods=list(RULES),expected_executions=len(cases)*len(RULES),
        independent_initials=len({(k[0],k[1]) for k in cases}),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        independent_risk_certificate=False,old_certificate_transferred=False,
        purpose='Development integration and selector-specific actual callback measurement; not superiority confirmation',
        physical_contract='Declared actuator-class bounds and command-history stopping certificate',
        online_information='Observed kinematics and executed commands only; no true plant or future tape',
        timing='Each full callback including terminal exceptions; no shared evidence cost and no hard-real-time claim',
        runtime_context=runtime_context,runtime_isolation_claim=False,
        inference='Repeated profiles/modes are not additional independent initials; conditional harm is descriptive')
    write(out/'plan.json',plan)
    warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for profile in sorted({k[2] for k in cases}):
        for rule in RULES:
            factory_gate(rule,models,profile).select(warm,np.zeros(3))
    count=0;started=time.perf_counter()
    with ExitStack() as stack, (out/'episodes.jsonl').open('x',encoding='utf-8') as stream:
        envs={p:stack.enter_context(MatchedShiftedMergeEnv(plant=p)) for p in sorted({k[2] for k in cases})}
        for block,(seed,scene,profile,mode) in enumerate(cases):
            offset=block%len(RULES);ordered=RULES[offset:]+RULES[:offset]
            if block%2:
                ordered=ordered[::-1]
            for rule in ordered:
                holder=[]
                def factory(_):
                    gate=TimedGate(factory_gate(rule,models,profile))
                    holder.append(gate);return gate
                row,trace,initial=run_episode(envs[profile],seed,scene,mode,factory,True)
                costs=np.asarray(row['per_vehicle_finish'])+.05*np.asarray(row['per_vehicle_variation'])
                row.update(initial=initial.tolist(),plant_name=profile,rule='baseline' if rule=='baseline' else ('scenario_confidence' if rule.startswith('original_') else 'parameter_scenario_confidence'),method_id=rule,
                    scenario_max_failed_paths=None if rule=='baseline' else 0,
                    certificate_contract='bounded_actuator_memory_stopping',per_vehicle_cost=costs.tolist(),
                    team_cost=float(costs.mean()),control_steps=len(trace),
                    learned_action_steps=sum(bool(t['actual_learning_used']) for t in trace))
                record=dict(key=[seed,scene,profile,mode,rule],row=row,trace=trace,calls=holder[0].calls)
                # Preserve outcomes BEFORE any reviewer may reject them.
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                count+=1
                check_calls(row,trace,record['calls'])
                (review_parameter_trace if row['rule']=='parameter_scenario_confidence' else review_scenario_trace)(row,trace,None)
            print(json.dumps(dict(completed=count,expected=plan['expected_executions'],elapsed_s=time.perf_counter()-started)),flush=True)
    if count!=plan['expected_executions']:
        raise ValueError('Incomplete integration grid')
    write(out/'collection_complete.json',dict(executions=count,episodes_sha256=sha(out/'episodes.jsonl'),
        elapsed_s=time.perf_counter()-started,native_exit_observed_by='Caller must observe this process exit before starting review'))


def review(out):
    out=Path(out);torch.set_num_threads(1)
    plan=json.loads((out/'plan.json').read_text())
    complete=json.loads((out/'collection_complete.json').read_text())
    if complete['episodes_sha256']!=sha(out/'episodes.jsonl'):
        raise ValueError('Collected outcomes changed')
    for name,digest in plan['sources_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Frozen integration source changed: '+name)
    models,identity=load_models()
    if identity!=plan['model']:
        raise ValueError('Model or PPO audit changed')
    rows=[json.loads(s) for s in (out/'episodes.jsonl').read_text().splitlines()]
    lookup={tuple(r['key']):r for r in rows}
    expected={tuple(case)+(rule,) for case in plan['cases'] for rule in RULES}
    if set(lookup)!=expected or len(rows)!=len(expected):
        raise ValueError('Missing, duplicate or unexpected outcome')
    commands=substeps=terminal=0
    for record in rows:
        row,trace=record['row'],record['trace']
        check_calls(row,trace,record['calls'])
        physical=(review_parameter_trace if row['rule']=='parameter_scenario_confidence' else review_scenario_trace)(row,trace,None)
        commands+=physical['commands'];substeps+=physical['substeps']
        gate=factory_gate(row['method_id'],models,row['plant_name'])
        state=np.asarray(row['initial']);previous=np.zeros(3)
        for item in trace:
            actual,info=gate.select(state,previous)
            np.testing.assert_array_equal(actual,item['command'])
            for key,value in info.items():
                if key!='gate_ms' and item.get(key)!=value:
                    raise ValueError('Full memoized recovery replay differs: '+key)
            state=np.asarray(item['x']);previous=actual
        if row['aborted']:
            terminal+=1
            try:
                gate.select(state,previous)
            except (RuntimeError,ValueError) as exc:
                if str(exc)!=row['abort_reason']:
                    raise ValueError('Terminal failure identity differs') from exc
            else:
                raise ValueError('Recorded terminal failure did not recur')
    summaries={}
    for rule in RULES:
        pairs=[]
        for case in plan['cases']:
            record=lookup[tuple(case)+(rule,)];base=lookup[tuple(case)+('baseline',)]['row']
            pairs.append(dict(**relative_outcome(record['row'],base),row=record['row'],baseline=base,calls=record['calls']))
        summaries[rule]=cell_summary(pairs)
    write(out/'report.json',dict(methods=summaries,independent_initials=plan['independent_initials'],
        executions=len(rows),scope=plan['scope'],risk_certified=False,
        meaning='Descriptive integration; smoke is not efficacy evidence, repeated conditions are not independent trials'))
    proof=dict(all_match=True,actual_sumo_episodes=len(rows),commands_matched=commands,
        independent_physical_substeps=substeps,terminal_errors_replayed=terminal,
        additional_independent_initials=0,models=identity,
        report_sha256=sha(out/'report.json'),risk_certified=False)
    write(out/'verification.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('collect','review'))
    parser.add_argument('--out',required=True);parser.add_argument('--checkpoint')
    parser.add_argument('--n',type=int,default=1);parser.add_argument('--seed-base',type=int,default=97000001)
    parser.add_argument('--scope',choices=('smoke','integration','regression'),default='regression')
    parser.add_argument('--runtime-context',default='Not isolated; concurrent workload unspecified')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    if args.action=='collect':
        pass # Fixed three-model family; no best-seed checkpoint argument
        collect(args.out,args.checkpoint,args.n,args.seed_base,args.scope,args.runtime_context,args.shard,args.shards)
    else:
        review(args.out)
