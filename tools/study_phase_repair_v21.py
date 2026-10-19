"""Fresh SUMO development of two fixed arrival tolerances and an exact reference."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from coopgate.arrival_phase_v20 import arrival_phase_suffix
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import JointRolloutGate, joint_rollout
from coopgate.merge_v4 import MergeConfig
from coopgate.phase_repair_v21 import PhaseRepairGate
from coopgate.repair_portfolio_v20 import RepairPortfolioGate
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest, save
from recover_checkpoint_v19 import atomic_bytes
from resume_frozen_risk_v19 import build_record, schedule
from review_repair_closed_loop_v20 import check_forecast_record, check_selected, outcome_counts, reconstruct_candidates
from start_final_v19 import require_committed, verify_manifest
from study_gate_v19 import read_protocol, regime_for
from study_repair_v20 import REFERENCE_FIELDS, trace_metadata, verify_trace
from study_temporal_v20 import independent_ranges, summarize

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'artifacts/v21/phase_development'
MODELS=(1802,1812,1822)
REFERENCE_KEYS=tuple(k for k in REFERENCE_FIELDS if k not in (
    'forecast_transitions','internal_transitions','decision_sha256'))


def methods():
    result={'baseline':{'kind':'baseline'}}
    for seed in MODELS:
        for name, variant, tolerance in (('repair','original',None),('reference','phase',None),
                                         ('phase05','phase',.05),('phase10','phase',.1)):
            result[f'{name}_{seed}']=dict(kind='repair',variant=variant,seed=seed,bank='sampled',
                                        bank_size=33,phase_tolerance_seconds=tolerance)
    return result


def factories(protocol):
    result={'baseline':lambda x:JointRolloutGate()}
    models={}
    for seed,record in protocol['models'].items():
        if digest(ROOT/record['path'])!=record['sha256']:
            raise ValueError('Frozen training model changed')
        model,meta=load_coupled(ROOT/record['path'])
        if meta['updates']!=meta['target_updates']:
            raise ValueError('Incomplete training model')
        models[seed]=model
    for name,method in protocol['spec']['methods'].items():
        if method['kind']=='baseline':
            continue
        model=models[str(method['seed'])]
        if method['variant']=='original':
            result[name]=lambda x,m=model:RepairPortfolioGate(m,'sampled',True)
        elif method['variant']=='phase':
            result[name]=lambda x,m=model,t=method['phase_tolerance_seconds']:PhaseRepairGate(m,t)
        else:
            raise ValueError('Unknown frozen controller variant')
    return result


def phase_record_check(item, method):
    if method['kind']=='baseline' or method['variant']=='original':
        return
    tolerance=method['phase_tolerance_seconds']
    if item['phase_tolerance_seconds']!=tolerance or item['phase_filter_enabled']!=(tolerance is not None):
        raise ValueError('Executed a different phase condition')
    phases=np.asarray(item['selected_phase_differences'])
    if item['portfolio_joint_action_admitted'] and tolerance is not None:
        if phases.shape!=(33,3) or not np.isfinite(phases).all() or np.any(phases>tolerance+1e-10):
            raise ValueError('An admitted phase action lacks a complete valid forecast')
    elif phases.size:
        raise ValueError('Fallback or disabled reference has selected phase evidence')


def freeze():
    if STUDY.exists():
        raise FileExistsError('Preserve the existing phase study')
    _,old=read_protocol(ROOT/'artifacts/v20/temporal_repair_extension')
    preflight=ROOT/'artifacts/v21/phase_preflight'
    verify_manifest(preflight)
    checked=json.loads((preflight/'audit.json').read_text())
    if not checked['all_original_and_reference_checks_pass'] or checked['completed_runs']!=13:
        raise ValueError('Actual SUMO preflight has not passed')
    preflight_sources=json.loads((preflight/'plan.json').read_text())['source_sha256']
    for source,expected in preflight_sources.items():
        if source.startswith('src/') and digest(ROOT/source)!=expected:
            raise ValueError('Controller source changed after preflight: '+source)
    spec=dict(purpose='Fresh closed-loop phase-gate development with full-candidate search and reference identity',
        data_role='development',seed_base=63300001,n_initials=128,shards=3,sampling='iid_mixture',
        modes=['clean','uniform','bias','pulse'],methods=methods(),all_traces_saved=True,audit_initial_indices=[],
        expected_actual_sumo_runs=128*4*13,mode_selection_stream=21107,bootstrap_stream=21113,bootstrap_samples=5000,
        phase_tolerances_seconds=[.05,.1],reference_phase_filter_enabled=False,
        report_scope='All three fixed models and both tolerances; pointwise paired development intervals, no new risk certificate.',
        timing_scope='Concurrent desktop workload; every complete call measured including phase forecasts; no deadline cutoff or hard-real-time claim.',
        checks='Full original-reference trajectory identity; 50ms episode adoption subset of 100ms subset of original under error-free forecasts; all candidates and each distinct admitted bank independently reconstructed.',
        selection_basis='Completed V20 exhaustive local screen and complete margin grid; no new study outcomes used.')
    prior=independent_ranges(spec['seed_base'],spec['n_initials'])
    sources=dict(old['source_sha256'])
    for path in (ROOT/'src/coopgate').glob('*.py'):
        sources[path.relative_to(ROOT).as_posix()]=digest(path)
    names=('src/coopgate/arrival_phase_v20.py','src/coopgate/phase_repair_v21.py',
        'src/coopgate/fast_repair_v20.py','src/coopgate/cached_portfolio_v19.py',
        'src/coopgate/bounded_portfolio_v19.py','src/coopgate/deduplicated_planner_v19.py',
        'tools/study_phase_repair_v21.py','tools/advance_phase_repair_v21.py',
        'tools/review_repair_closed_loop_v20.py','tools/study_repair_v20.py',
        'tests/test_phase_repair_v21.py','tests/test_phase_study_v21.py','tests/test_fast_repair_v20.py')
    for name in names:
        sources[name]=digest(ROOT/name)
    protocol=dict(spec=spec,models=old['models'],source_sha256=sources,disjoint_prior_protocols=prior,
        independent_review_frozen_with_controller=True,final_confirmation_outcomes_read=False,
        actual_sumo_preflight_manifest_sha256=digest(preflight/'sha256.json'),
        development_screen_sha256=digest(ROOT/'artifacts/v20/arrival_phase_screen/sha256.json'),
        development_margin_grid_sha256=digest(ROOT/'artifacts/v20/arrival_phase_margin_analysis/sha256.json'))
    if len(factories(protocol))!=13:
        raise ValueError('Every model and reference must remain')
    STUDY.mkdir(parents=True)
    save(STUDY/'protocol.json',protocol)
    for name in sources:
        target=STUDY/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    print(json.dumps(dict(expected_actual_sumo_runs=spec['expected_actual_sumo_runs'],
                          independent_initials=spec['n_initials'],protocol_sha256=digest(STUDY/'protocol.json'))),flush=True)


def worker(shard):
    torch.set_num_threads(1)
    _,protocol=read_protocol(STUDY)
    spec=protocol['spec']
    require_committed([STUDY/'protocol.json',*[ROOT/p for p in protocol['source_sha256']]])
    out=STUDY/f'shard_{shard}'
    out.mkdir(exist_ok=False)
    fs,rows,started=factories(protocol),[],time.perf_counter()
    cases=schedule(spec,shard)
    warm=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for factory in fs.values():
        factory(warm).select(warm,np.zeros(3))
    with DisturbedMergeEnv() as env:
        for index,mode,name in cases:
            row,trace,initial=run_episode(env,spec['seed_base']+index,regime_for(spec,index),mode,fs[name],True)
            method=spec['methods'][name]
            row,adopted,checks,failures=build_record(row,trace,initial,index,name,method,spec)
            row.update(trace_metadata(trace))
            save(out/f'trace_{index}_{mode}_{name}.json',trace)
            rows.append(row)
            atomic_bytes(out/'episodes.json',(json.dumps(rows,indent=2)+'\n').encode('utf-8'))
            verify_trace(trace,row,method,adopted,checks,failures)
            for item in trace:
                phase_record_check(item,method)
            if len(rows)%len(fs)==0:
                print(json.dumps(dict(shard=shard,pid=os.getpid(),initial_index=index,mode=mode,
                    actual_sumo_runs=len(rows),expected_shard_runs=len(cases),elapsed_s=time.perf_counter()-started)),flush=True)
    save(out/'complete.json',dict(actual_sumo_runs=len(rows),episodes_sha256=digest(out/'episodes.json'),
        protocol_sha256=digest(STUDY/'protocol.json'),pid=os.getpid()))


def paired_checks(rows,spec):
    lookup={(r['initial_index'],r['disturbance'],r['controller']):r for r in rows}
    expected={(i,m,n) for i in range(spec['n_initials']) for m in spec['modes'] for n in spec['methods']}
    if len(lookup)!=len(rows) or set(lookup)!=expected:
        raise ValueError('Missing or duplicated study case')
    output={}
    draws=np.random.default_rng(spec['bootstrap_stream']).integers(spec['n_initials'],size=(spec['bootstrap_samples'],spec['n_initials']))
    for seed in MODELS:
        values={k:[] for k in ('phase05','phase10')}
        subset_checks,reference_pairs=0,0
        for index in range(spec['n_initials']):
            chosen=int(np.random.default_rng(np.random.SeedSequence([spec['seed_base']+index,spec['mode_selection_stream']])).integers(len(spec['modes'])))
            for mi,mode in enumerate(spec['modes']):
                group={prefix:lookup[index,mode,f'{prefix}_{seed}'] for prefix in ('repair','reference','phase05','phase10')}
                for key in REFERENCE_KEYS:
                    np.testing.assert_equal(group['repair'][key],group['reference'][key],err_msg='Reference identity: '+key)
                reference_pairs+=1
                if not any(r['forecast_error_count'] or r['generation_error_count'] for r in group.values()):
                    adoption=[group[p]['learned_action_steps']>0 for p in ('phase05','phase10','repair')]
                    if adoption!=sorted(adoption):
                        raise ValueError('A stricter eligible action set created an episode adoption')
                    subset_checks+=1
                if mi==chosen:
                    for prefix in values:
                        values[prefix].append(group['repair']['team_cost']-group[prefix]['team_cost'])
        output[str(seed)]=dict(reference_pairs=reference_pairs,error_free_adoption_subset_pairs=subset_checks,
            actual_incremental_gain={k:dict(mean_seconds=float(np.mean(v)),pointwise_bootstrap95=np.quantile(
                np.asarray(v)[draws].mean(1),[.025,.975]).tolist()) for k,v in values.items()})
    return output


def independent_review(rows,protocol):
    torch.set_num_threads(1)
    for record in protocol['models'].values():
        if digest(ROOT/record['path'])!=record['sha256']:
            raise ValueError('Frozen model changed before independent review')
    models={seed:load_coupled(ROOT/record['path'])[0] for seed,record in protocol['models'].items()}
    packed=parameters(MergeConfig())
    _,tapes=temporal_stress_bank('sampled')
    cache,counts={},dict(physical_traces=0,candidate_families=0,candidate_commands=0,
        distinct_admitted_forecasts=0,admitted_decisions=0,phase_admitted_decisions=0,reference_trajectory_pairs=0)
    for number,row in enumerate(rows):
        method=protocol['spec']['methods'][row['controller']]
        path=STUDY/f"shard_{row['initial_index']%protocol['spec']['shards']}/trace_{row['initial_index']}_{row['disturbance']}_{row['controller']}.json"
        trace=json.loads(path.read_text())
        rebuilt,adopted,checks,failures=build_record(dict(row),trace,np.asarray(row['initial']),row['initial_index'],row['controller'],method,protocol['spec'])
        for key,value in rebuilt.items():
            np.testing.assert_equal(value,row[key])
        verify_trace(trace,row,method,adopted,checks,failures)
        counts['physical_traces']+=1
        actor=JointPolicyGate(models[str(method['seed'])]) if method['kind']!='baseline' else None
        state,previous=np.asarray(row['initial']),np.zeros(3)
        if row['controller'].startswith('reference_'):
            original_path=path.with_name(path.name.replace('_reference_','_repair_'))
            original=json.loads(original_path.read_text())
            if len(original)!=len(trace):
                raise ValueError('Reference trajectory length differs')
            for actual,expected in zip(trace,original):
                for key in ('command','x','disturbance','composition','repair_weight','selected_bank_differences'):
                    np.testing.assert_equal(actual[key],expected[key])
            counts['reference_trajectory_pairs']+=1
        for step,item in enumerate(trace):
            phase_record_check(item,method)
            command=np.asarray(item['command'])
            if actor is not None:
                baseline=joint_rollout(state,previous,step,packed)[0]
                actor.step=step
                _,actor_info=actor.select(state,previous)
                np.testing.assert_array_equal(item['actor_proposal'],actor_info['proposal'])
                np.testing.assert_array_equal(item['baseline_command'],baseline)
                candidates,original_count=reconstruct_candidates(state,previous,np.asarray(actor_info['proposal']),baseline,packed,True)
                check_selected(item,baseline,candidates,original_count)
                counts['candidate_families']+=1
                counts['candidate_commands']+=len(candidates)
            if item.get('portfolio_joint_action_admitted',False):
                key=(step,state.tobytes(),previous.tobytes(),command.tobytes())
                if key not in cache:
                    differences,phases,complete=[],[],True
                    for tape in tapes:
                        base,bt,bc=arrival_phase_suffix(state,previous,step,tape,packed,command,False)
                        candidate,ct,cc=arrival_phase_suffix(state,previous,step,tape,packed,command,True)
                        differences.append(candidate-base)
                        phases.append(ct-bt)
                        complete=complete and bc and cc
                    cache[key]=(np.asarray(differences),np.asarray(phases),complete)
                    counts['distinct_admitted_forecasts']+=1
                differences,phases,complete=cache[key]
                check_forecast_record(differences,item['selected_bank_differences'])
                counts['admitted_decisions']+=1
                if method.get('phase_tolerance_seconds') is not None:
                    if not complete:
                        raise ValueError('Admitted phase forecast did not complete')
                    np.testing.assert_allclose(phases,item['selected_phase_differences'],atol=1e-10,rtol=0)
                    if np.any(phases>method['phase_tolerance_seconds']+1e-10):
                        raise ValueError('Independently reconstructed phase condition fails')
                    counts['phase_admitted_decisions']+=1
            state,previous=np.asarray(item['x']),command
        if number%40==0:
            print(json.dumps(dict(verified_runs=number+1,total_runs=len(rows))),flush=True)
    return counts


def validate_worker_exits(exits,launch,shards):
    pids=[row['pid'] for row in exits]
    if (len(exits)!=shards or len(set(pids))!=shards or set(pids)!=set(launch['worker_pids']) or
            len(launch['worker_pids'])!=shards or any(row['exit_code']!=0 for row in exits)):
        raise ValueError('Every original worker must exit successfully, with no missing or substituted handle')


def finalize(logs):
    if (STUDY/'summary.json').exists():
        raise FileExistsError('Preserve completed study analysis')
    _,protocol=read_protocol(STUDY)
    exits=json.loads((Path(logs)/'worker_exits.json').read_text())['workers']
    launch=json.loads((Path(logs)/'launch.json').read_text())
    if launch['protocol_sha256']!=digest(STUDY/'protocol.json'):
        raise ValueError('Execution launch used a different protocol')
    validate_worker_exits(exits,launch,protocol['spec']['shards'])
    spec,rows=protocol['spec'],[]
    for shard in range(spec['shards']):
        out=STUDY/f'shard_{shard}'
        receipt=json.loads((out/'complete.json').read_text())
        if receipt['protocol_sha256']!=digest(STUDY/'protocol.json') or receipt['episodes_sha256']!=digest(out/'episodes.json'):
            raise ValueError('Completed shard changed')
        part=json.loads((out/'episodes.json').read_text())
        if [(r['initial_index'],r['disturbance'],r['controller']) for r in part]!=schedule(spec,shard) or len(part)!=receipt['actual_sumo_runs']:
            raise ValueError('Completed schedule differs')
        rows.extend(part)
    if len(rows)!=spec['expected_actual_sumo_runs']:
        raise ValueError('Incomplete study')
    paired=paired_checks(rows,spec)
    reviewed=independent_review(rows,protocol)
    results=summarize(rows,spec)
    recounted={}
    for name in spec['methods']:
        recounted[name]=outcome_counts(rows,spec,name)
        for key,value in recounted[name].items():
            np.testing.assert_allclose(value,results[name][key],atol=1e-12,rtol=1e-12)
    rows.sort(key=lambda r:(r['initial_index'],r['disturbance'],r['controller']))
    save(STUDY/'episodes.json',rows)
    summary=dict(data_role='fresh paired development',independent_initials=spec['n_initials'],
        actual_sumo_runs=len(rows),results=results,paired_results=paired,independent_review_counts=reviewed,
        independently_recounted_outcomes=recounted,risk_certified=False,hard_real_time_certified=False,
        protocol_sha256=digest(STUDY/'protocol.json'),episodes_sha256=digest(STUDY/'episodes.json'))
    save(STUDY/'summary.json',summary)
    save(STUDY/'sha256.json',{p.relative_to(STUDY).as_posix():digest(p) for p in STUDY.rglob('*') if p.is_file() and p.name!='sha256.json'})
    print(json.dumps(dict(completed=True,actual_sumo_runs=len(rows),summary_sha256=digest(STUDY/'summary.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('freeze')
    sub.add_parser('worker').add_argument('shard',type=int)
    sub.add_parser('finalize').add_argument('logs')
    args=parser.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='worker':worker(args.shard)
    else:finalize(args.logs)
