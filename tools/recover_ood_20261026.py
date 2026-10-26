"""Suffix-only continuation of the unchanged frozen V39 OOD experiment.

No controller, scenario, inference or runtime budget changes. Old complete
blocks are copied verbatim; partial blobs are retained exactly. Replay the
last committed boundary before any missing run; retain all ordinary failures.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import torch

from coopgate.episode_store_v21 import EpisodeStore, encode, inventory, unpack
import execute_ood_validation_v39 as io
import recover_prospective_v24 as storage
import snapshot_interruption_20261026 as archive

ROOT = io.ROOT
PRIOR = io.OUT
OUT = ROOT/'artifacts/recovery_20261026/ood_recovery'
FROZEN = OUT/'recovery.json'
SCRIPT = 'tools/recover_ood_20261026.py'
SOURCES = (SCRIPT, 'tests/test_ood_recovery_20261026.py',
           'tools/recover_prospective_v24.py', 'tools/snapshot_interruption_20261026.py')


def non_timing(value):
    if isinstance(value, dict):
        return {k:non_timing(v) for k,v in value.items()
                if not k.endswith('_ms') and k != 'episode_wall_seconds'}
    if isinstance(value, list):
        return [non_timing(v) for v in value]
    return value


def assert_stopped():
    processes = io.read_processes() if hasattr(io, 'read_processes') else storage.original.native_snapshot()
    for p in processes:
        command = (p.get('CommandLine') or '').replace('\\','/')
        if p['ProcessId'] not in (os.getpid(), os.getppid()) and any(
                name in command for name in ('tools/execute_ood_validation_v39.py', SCRIPT)):
            raise ValueError('Original/recovery OOD process remains live')


def freeze():
    assert_stopped()
    if OUT.exists():
        raise FileExistsError('Preserve every interrupted recovery instead of overwriting')
    plan, binding = io.binding_inputs()
    if any((PRIOR/'execution'/n).exists() for n in ('exit.json','collection_complete.json','blocks_manifest.json')):
        raise ValueError('Terminal evidence exists; inspect it before an interruption recovery')
    for name in SOURCES:
        io.committed(ROOT/name)
    entries=[]
    for directory in sorted((PRIOR/'blocks').iterdir()):
        index=int(directory.name)
        if index != len(entries):
            raise ValueError('Original OOD blocks are not a contiguous frozen prefix')
        group=plan['groups'][index]; expected=io.schedule(group)
        metadata=io.metadata(group,io.digest(io.PLAN),io.digest(io.BINDING))
        db=directory/'observations.sqlite'; state=inventory(db)
        count=len(state['cases'])
        if state['metadata'] != metadata or state['cases'] != expected[:count] or not 0 < count <= 13:
            raise ValueError('Original OOD block identity/prefix differs')
        markers=[(directory/n).exists() for n in ('complete.json','observations.sqlite.seal.json')]
        if any(markers) and not all(markers):
            raise ValueError('Partially sealed block requires an explicit decision')
        complete=all(markers)
        if complete:
            done=io.read(directory/'complete.json'); seal=io.read(directory/'observations.sqlite.seal.json')
            if (count != 13 or done['database_sha256'] != io.digest(db)
                    or done['seal_sha256'] != io.digest(directory/'observations.sqlite.seal.json')
                    or seal['database_sha256'] != io.digest(db)
                    or seal['cases'] != expected or len(done['reviews']) != 13
                    or any(done[k] != v for k,v in metadata.items())):
                raise ValueError('Original sealed OOD block changed')
        else:
            storage.prefix_records(db,metadata,expected,io.digest(db))
        entries.append(dict(index=index,records=count,complete=complete,database_sha256=io.digest(db),
                            inventory=state))
    if not entries or any(not e['complete'] for e in entries[:-1]):
        raise ValueError('Only the final block may be an interrupted prefix')
    files=archive.files_at(PRIOR)
    OUT.mkdir(parents=True)
    parts=[]
    for index,paths in enumerate(archive.parts_for([PRIOR/name for name in files])):
        target=OUT/f'original_{index:03d}.zip'
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in paths:z.write(path,path.relative_to(PRIOR).as_posix())
        parts.append(dict(name=target.name,sha256=io.digest(target)))
    if archive.files_at(PRIOR) != files:
        raise ValueError('Interrupted OOD prefix changed during preservation')
    sources={**binding['source_sha256'],**{n:io.digest(ROOT/n) for n in SOURCES}}
    kept=sum(e['records'] for e in entries)
    value=dict(utc=io.utc(),plan_sha256=io.digest(io.PLAN),binding_sha256=io.digest(io.BINDING),
        source_sha256=sources,original_files_sha256=files,archive_parts=parts,entries=entries,
        retained_roles=kept,missing_roles=plan['budget']['actual_sumo_runs']-kept,
        boundary_replays=sum(not e['complete'] for e in entries),
        original_exit_status='unknown',partial_efficacy_outcomes_read=False,
        statistical_sample_size_unchanged=True,scientific_functions_unchanged=True,
        original_scientific_source='tools/execute_ood_validation_v39.py',
        recovery_rule='Retain all exact blobs; replay boundary first; append original missing schedule only.')
    io.exclusive_json(FROZEN,value)
    print(json.dumps({k:value[k] for k in ('retained_roles','missing_roles','boundary_replays')}),flush=True)


def inputs():
    plan,_=io.binding_inputs(); io.committed(FROZEN); frozen=io.read(FROZEN)
    if frozen['plan_sha256'] != io.digest(io.PLAN) or frozen['binding_sha256'] != io.digest(io.BINDING):
        raise ValueError('Recovery changed original design/binding')
    for name,expected in frozen['source_sha256'].items():
        if io.digest(ROOT/name) != expected:raise ValueError('Frozen recovery source changed: '+name)
        io.committed(ROOT/name)
    if archive.files_at(PRIOR) != frozen['original_files_sha256']:
        raise ValueError('Original interrupted OOD files changed')
    for part in frozen['archive_parts']:
        io.committed(OUT/part['name'])
        if io.digest(OUT/part['name']) != part['sha256']:raise ValueError('Preserved archive changed')
    return plan,frozen


def decoded_prefix(entry,group,ps,bs):
    db=PRIOR/'blocks'/f"{entry['index']:05d}"/'observations.sqlite'
    _,raw=storage.prefix_records(db,io.metadata(group,ps,bs),io.schedule(group),entry['database_sha256'])
    return db,raw


def replay_boundary(entry,plan,group,models,env,ps,bs):
    _,raw=decoded_prefix(entry,group,ps,bs)
    _,key,rb,tb,rh,th=raw[-1]
    case=json.loads(key); old,old_trace=unpack(rb,rh),unpack(tb,th)
    method=dict(rule=case[-2],model=case[-1])
    record,trace=io.run_case(plan,group,method,models,env)
    directory=OUT/'boundary'/f"{entry['index']:05d}"
    directory.mkdir(parents=True)
    store=EpisodeStore(directory/'observations.sqlite',dict(boundary_replay=True,**io.metadata(group,ps,bs)))
    try:
        store.append(case,record,trace)
        review=io.auditor.audit_episode(plan,group,method,record,trace,
                                      None if method['model']=='none' else models[method['model']])
        if non_timing([old,old_trace]) != non_timing([record,trace]):
            raise ValueError('Interrupted OOD boundary failed exact non-timing replay')
        store.seal([case])
    finally:store.close()
    io.exclusive_json(directory/'verified.json',dict(exact_non_timing_match=True,
        original_database_sha256=entry['database_sha256'],review=review))


def complete_partial(entry,plan,group,models,env,ps,bs,run_fn=None,audit_fn=None):
    run_fn=run_fn or io.run_case; audit_fn=audit_fn or io.auditor.audit_episode
    db,raw=decoded_prefix(entry,group,ps,bs)
    directory=OUT/'blocks'/f"{entry['index']:05d}"; directory.mkdir()
    expected=io.schedule(group); metadata=io.metadata(group,ps,bs)
    store,_=storage.copy_prefix(db,directory/'observations.sqlite',metadata,expected,entry['database_sha256'])
    reviews=[]
    try:
        for i,method in enumerate(group['method_order']):
            if i < len(raw):
                _,_,rb,tb,rh,th=raw[i]; record,trace=unpack(rb,rh),unpack(tb,th)
            else:
                record,trace=run_fn(plan,group,method,models,env)
                store.append(expected[i],record,trace)
            model=None if method['model']=='none' else models[method['model']]
            reviews.append(dict(case=expected[i],**audit_fn(plan,group,method,record,trace,model)))
        store.seal(expected)
    finally:store.close()
    after=inventory(directory/'observations.sqlite')
    for k in ('cases','row_sha256','trace_sha256'):
        if after[k][:entry['records']] != entry['inventory'][k]:raise ValueError('Retained prefix changed')
    io.exclusive_json(directory/'complete.json',dict(**metadata,actual_sumo_runs=len(reviews),reviews=reviews,
        database_sha256=io.digest(directory/'observations.sqlite'),
        seal_sha256=io.digest(directory/'observations.sqlite.seal.json')))


def worker():
    execution=OUT/'execution'; startup=execution/'worker.native.json'
    io.exclusive_bytes(startup,encode(dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=io.utc())))
    ack=execution/'worker.acknowledged.json'; deadline=time.monotonic()+180
    while not ack.exists():
        if time.monotonic()>deadline:raise TimeoutError('No acknowledged native worker; no SUMO started')
        time.sleep(.2)
    proof=io.read(ack)
    if (proof['native_pid']!=os.getpid() or proof['parent_pid']!=os.getppid()
            or proof['native_start_sha256']!=io.digest(startup)):
        raise ValueError('Native acknowledgment differs')
    plan,frozen=inputs(); ps,bs=io.digest(io.PLAN),io.digest(io.BINDING)
    torch.set_num_threads(1)
    models={n:io.load_coupled(ROOT/m['path'])[0] for n,m in plan['models'].items()}
    io.warm_gates(plan,models)
    (OUT/'blocks').mkdir()
    entries={e['index']:e for e in frozen['entries']}; manifests=[]; total=0
    with io.OODMismatchMergeEnv() as env:
        for entry in frozen['entries']:
            if not entry['complete']:
                replay_boundary(entry,plan,plan['groups'][entry['index']],models,env,ps,bs)
        for group in plan['groups']:
            index=group['block_index']; entry=entries.get(index)
            destination=OUT/'blocks'/f'{index:05d}'
            if entry and entry['complete']:
                source=PRIOR/'blocks'/f'{index:05d}'
                shutil.copytree(source,destination)
                if archive.files_at(source)!=archive.files_at(destination):raise ValueError('Complete copy changed')
            elif entry:
                complete_partial(entry,plan,group,models,env,ps,bs)
            else:
                io.collect_block(plan,group,destination,models,env,ps,bs)
            manifests.append(io.block_manifest_entry(destination,index)); total+=13
            io.atomic_bytes(execution/'progress.json',encode(dict(utc=io.utc(),native_pid=os.getpid(),
                completed_blocks=index+1,expected_blocks=len(plan['groups']),completed_runs=total,
                expected_runs=plan['budget']['actual_sumo_runs'],retained_roles=frozen['retained_roles'])))
    if total!=plan['budget']['actual_sumo_runs']:raise ValueError('Full original OOD scope incomplete')
    io.exclusive_json(execution/'blocks_manifest.json',dict(plan_sha256=ps,execution_binding_sha256=bs,
        blocks=len(manifests),actual_sumo_runs=total,entries=manifests,utc=io.utc()))
    io.exclusive_json(execution/'collection_complete.json',dict(plan_sha256=ps,execution_binding_sha256=bs,
        native_pid=os.getpid(),actual_sumo_runs=total,blocks=len(manifests),
        retained_roles=frozen['retained_roles'],new_sumo_runs=frozen['missing_roles'],
        boundary_replays=frozen['boundary_replays'],original_exit_status='unknown',
        blocks_manifest_sha256=io.digest(execution/'blocks_manifest.json'),utc=io.utc()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','watch','worker','summarize'))
    command=parser.parse_args().command
    if command=='freeze':return freeze()
    # Routing only. Original plan, binding, controller, audit and inference stay.
    io.OUT,io.SCRIPT=OUT,SCRIPT
    if command=='worker':return worker()
    inputs()
    return getattr(io,command)()


if __name__=='__main__':main()
