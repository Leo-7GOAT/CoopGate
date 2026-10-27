"""Compare serial/parallel original audits on 64 already-completed V36 traces."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import torch
from coopgate.episode_store_v21 import EpisodeStore, inventory, observations
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import matched_plant_parameters
import parallel_joint_risk_replay_20261027 as replay

ORIGIN=ROOT/'artifacts/v38/interrupted_block_grid_recovery/v36'
FIXTURES=ROOT/'runs/parallel_joint_replay_20261027'
OUTPUT=replay.OUT/'benchmark'
BLOCKS=tuple(range(106,122))


def make_jobs():
    if FIXTURES.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve prior benchmark prefix; no automatic retry')
    proof=replay.original.io.read(ORIGIN/'completion_verification.json')
    if (proof.get('recovery_exit_code') != 0 or not proof.get('exact_full_schedule')
            or not proof.get('independent_reviews_replayed')
            or replay.original.io.digest(ORIGIN/'analysis/report.json') != proof['report_sha256']):
        raise ValueError('Completed original grid evidence required')
    FIXTURES.mkdir(parents=True)
    OUTPUT.mkdir(parents=True)
    jobs,provenance=[],[]
    for index in BLOCKS:
        source=ORIGIN/'blocks'/f'{index:05d}'/'observations.sqlite'
        if replay.original.io.digest(source) != proof['block_files_sha256'][f'{index:05d}']['observations.sqlite']:
            raise ValueError('Completed old source block changed')
        state=inventory(source)
        selected=[]
        for key,body,trace in observations(source,state['metadata'],state['cases']):
            row=body['row']
            if row['rule'] not in ('baseline','bank_performance'):
                continue
            method='baseline' if row['rule']=='baseline' else str(row['model'])
            selected.append((method,deepcopy(body),deepcopy(trace)))
        if len(selected)!=4 or {a[0] for a in selected}!=set(replay.original.collect.METHODS):
            raise ValueError('The fixed old block must contain the whole four-method family')
        row=selected[0][1]['row']; shared=replay.original.design.shared
        case=dict(stage='operational_replay',index=index,cluster_id=f'old-v36:{index:05d}',
            seed=row['seed'],scenario=row['scenario'],plant=row['plant_name'],
            disturbance=row['disturbance'],methods=[a[0] for a in selected],
            initial_sha256=shared.array_sha256(shared.planned_initial(row['seed'],row['scenario']),decimals=8),
            dynamics_sha256=shared.canonical_sha256(matched_plant_parameters(row['seed'],row['plant_name'],.1)),
            disturbance_sha256=shared.array_sha256(disturbance_tape(row['seed'],row['disturbance'],.1,250)))
        meta=dict(operational_equivalence_only=True,old_block=index)
        directory=FIXTURES/f'{index:05d}';directory.mkdir()
        store=EpisodeStore(directory/'observations.sqlite',meta)
        try:
            for method,body,trace in selected:
                body.update(cluster_id=case['cluster_id'],method=method)
                store.append([case['cluster_id'],method],body,trace)
            store.seal(replay.original.collect.schedule(case))
        finally:
            store.close()
        replay.original.io.exclusive_json(directory/'complete.json',dict(operational_fixture=True))
        jobs.append(dict(case=case,directory=str(directory),metadata=meta))
        provenance.append(dict(old_block=index,old_database_sha256=replay.original.io.digest(source),
            old_case_keys=[list(key) for key in state['cases']],
            derived_database_sha256=replay.original.io.digest(directory/'observations.sqlite')))
    return jobs,provenance


def main():
    torch.set_num_threads(1)
    models=replay.original.plan_inputs()['models']
    jobs,provenance=make_jobs()
    replay.initialize(models)
    start=time.perf_counter()
    serial=[replay.audit_block(job) for job in jobs]
    serial_seconds=time.perf_counter()-start
    start=time.perf_counter()
    parallel=replay.replay_blocks(jobs,models)
    parallel_seconds=time.perf_counter()-start
    if serial!=parallel:
        raise ValueError('Serial and parallel original audit results differ')
    value=dict(old_blocks=list(BLOCKS),audited_old_episodes=64,new_sumo_runs=0,
        new_independent_initials=0,all_original_audit_results_exact=True,
        workers=replay.WORKERS,serial_shared_host_seconds=serial_seconds,
        parallel_shared_host_seconds=parallel_seconds,
        observed_shared_host_speedup=serial_seconds/parallel_seconds,
        all_failures_retained=True,standalone_latency_or_new_scientific_claim=False,
        source_sha256={name:replay.original.io.digest(ROOT/name) for name in replay.SOURCES},
        old_grid_completion_sha256=replay.original.io.digest(ORIGIN/'completion_verification.json'),
        provenance=provenance)
    replay.original.io.exclusive_json(OUTPUT/'report.json',value)
    print(json.dumps({k:value[k] for k in ('audited_old_episodes','new_sumo_runs',
        'all_original_audit_results_exact','serial_shared_host_seconds',
        'parallel_shared_host_seconds','observed_shared_host_speedup')}),flush=True)


if __name__=='__main__':main()
