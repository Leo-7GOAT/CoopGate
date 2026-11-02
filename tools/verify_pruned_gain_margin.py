"""All48 source controller trajectories,complete paired independent callbacks."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from coopgate.gain_margin_gate import ScalarGainMarginGate
from coopgate.pruned_gain_margin_gate import PrunedGainMarginGate
from coopgate.gain_margin_audit import validate_gain_evidence
from coopgate.pruned_gain_margin_audit import validate_prefix_evidence
from evaluate_scalar_nominal_sumo import load_models,sha,write

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/mappo_ctde_20261101'
FIELDS=('actual_learning_used','actual_learning_used_ever','baseline_command','baseline_evidence',
    'baseline_recovery_proposed','agent_actions','actor_proposal','composition','repair_weight',
    'selected_bank_differences','selected_phase_differences','finite_bank_performance_margin','certified','in_set',
    'actuator_bounds','actuator_memory_lower','actuator_memory_upper','nominal_evidence','candidate_count','original_candidate_count',
    'prediction_parameters','reconstructed_actuator','parameter_scenarios','reconstructed_parameter_memories',
    'forecast_bank_size','scenario_pass_fraction','scenario_max_failed_paths','performance_confidence_kind',
    'gain_confidence_floor','selected_gain_confidence_score','selected_baseline_forecast_team_costs')


def run():
    source=BASE/'development/gain_margin_fixtures_20261102';out=BASE/'development/pruned_gain_margin_replay_20261102'
    out.mkdir(exist_ok=False);torch.set_num_threads(1)
    original=json.loads((source/'plan.json').read_text());proof=json.loads((source/'verification.json').read_text())
    receipt=json.loads((source/'native_receipt.json').read_text());complete=json.loads((source/'collection_complete.json').read_text())
    if not proof['all_match'] or receipt['review_actual_native_exit_code']!=0 or sha(source/'episodes.jsonl')!=complete['episodes_sha256']:
        raise ValueError('Complete observed-terminal-zero reviewed source required')
    models,identities=load_models()
    if identities!=original['model']:raise ValueError('MAPPO identity changed')
    rows=[json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines()]
    rows=sorted([r for r in rows if r['row']['rule']!='baseline'],key=lambda r:tuple(r['key']))
    if len(rows)!=48:raise ValueError('All48 learning-controller trajectories required')
    files=list((ROOT/'src/coopgate').glob('*.py'))+[Path(__file__),ROOT/'tools/build_pruned_gain_margin.py',ROOT/'tools/evaluate_scalar_nominal_sumo.py',ROOT/'tests/test_pruned_gain_margin.py',ROOT/'docs/pruned_gain_margin_20261102_protocol.md']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    write(out/'plan.json',dict(source_sha256=hashes,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_episodes_sha256=sha(source/'episodes.jsonl'),source_episodes=48,whole_controller_replays=96,
        models=identities,extra_sumo_runs=0,extra_independent_initials=0,risk_certificate=False,
        preserved_fields=FIELDS,timing='Full callbacks,per-episode alternating order,shared desktop,no isolated/WCET claim'))
    warm=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])
    for cls in (ScalarGainMarginGate,PrunedGainMarginGate):
        for floor in (0.,.0025,.005,.01):cls(models[4703],'lag_strong',floor).select(warm,np.zeros(3))
    timings={};work={};count=0;started=time.perf_counter()
    with (out/'callbacks.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for index,record in enumerate(rows):
            row=record['row'];seed=int(row['method_id'].split('_')[1]);floor=row['gain_floor']
            order=[('naive',ScalarGainMarginGate),('pruned',PrunedGainMarginGate)]
            if index%2:order.reverse()
            for name,cls in order:
                g=cls(models[seed],row['plant_name'],floor);state=np.array(row['initial']);prev=np.zeros(3)
                key=f'{seed}/{floor}/{name}';times=timings.setdefault(key,[])
                counter=work.setdefault(key,dict(forecasts=0,computed_planning_transitions=0,prefix_rejections=0))
                for step,item in enumerate(record['trace']):
                    start=time.perf_counter();error=None;command=info=None
                    try:command,info=g.select(state,prev)
                    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc))
                    ms=1000*(time.perf_counter()-start)
                    stream.write(json.dumps(dict(key=record['key'],implementation=name,step=step,wall_ms=ms,error=error,
                        command=None if command is None else command.tolist(),info=info),allow_nan=False)+'\n');stream.flush()
                    if error:raise RuntimeError('Unexpected exception persisted: '+str(error))
                    if command.tobytes()!=np.asarray(item['command']).tobytes():raise ValueError('Command bytes changed')
                    for field in FIELDS:
                        if info[field]!=item[field]:raise ValueError('Complete admission semantics changed: '+field)
                    accepted=[e for e in info['gain_margin_evidence'] if e['passed']]
                    if accepted!=[e for e in item['gain_margin_evidence'] if e['passed']]:raise ValueError('Full accepted gain proof changed')
                    (validate_gain_evidence if name=='naive' else validate_prefix_evidence)(info)
                    if floor==0.:
                        for field,value in item.items():
                            if field not in ('t','x','command','disturbance','innovation','actuator_states','realized_error_in_contract','gate_ms') and info.get(field)!=value:
                                raise ValueError('Zero-floor original field changed: '+field)
                    times.append(ms);count+=1;counter['forecasts']+=info['forecast_calls']
                    counter['computed_planning_transitions']+=info['proposal_cache']['computed_successful_transitions']
                    counter['prefix_rejections']+=len(info.get('gain_margin_prefix_rejections',[]))
                    state=np.array(item['x']);prev=command
            if row['aborted']:raise ValueError('Unexpected aborted fixture source')
            print(json.dumps(dict(episodes_replayed=index+1,expected=48,callbacks=count)),flush=True)
    for name,digest in hashes.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen replay source changed: '+name)
    summary={}
    for key,times in timings.items():
        a=np.array(times);summary[key]=dict(calls=len(a),sum_ms=float(a.sum()),p95_ms=float(np.quantile(a,.95)),max_ms=float(a.max()),over_500ms=int((a>500).sum()),**work[key])
    result=dict(all_commands_and_complete_admission_evidence_match=True,source_episodes=48,whole_controller_replays=96,callbacks=count,
        extra_sumo_runs=0,extra_independent_initials=0,risk_certificate=False,summary=summary,
        callbacks_sha256=sha(out/'callbacks.jsonl'),elapsed_s=time.perf_counter()-started)
    write(out/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':run()
