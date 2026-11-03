"""Recover external interruption without replacing any durable observation.

Original files remain untouched. Complete sealed blocks are copied byte-exact;
an unsealed transaction prefix is checksum-verified and preserved in order.
Only missing planned rows execute SUMO. Lost in-flight attempts have unknown
exit/count and are explicitly not relabelled as successful observations.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import torch
from coopgate.episode_store_v21 import EpisodeStore,inventory,readonly,unpack,observations,seal_path
import run_coupled_holdout_ablation as a
from mappo_native_launch import command

ORIGINAL=a.OUT
OUT=a.e.BASE/'coupled_holdout_ablation_recovery_20261103'


def existing(db,metadata,keys):
    inv=inventory(db)
    if inv['metadata']!=metadata or inv['cases']!=keys[:len(inv['cases'])]:
        raise ValueError('Interrupted records must be an exact scheduled prefix')
    if seal_path(db).exists():return list(observations(db,metadata,keys))
    with readonly(db) as connection:
        return [(tuple(json.loads(k)),unpack(row,rsha),unpack(trace,tsha)) for k,row,trace,rsha,tsha in
            connection.execute('SELECT case_key,row_blob,trace_blob,row_sha256,trace_sha256 FROM episodes ORDER BY position')]


def prepare():
    a.bound(True)
    source=Path(__file__).resolve();raw=subprocess.check_output(['git','show','HEAD:'+source.relative_to(a.e.ROOT).as_posix()],cwd=a.e.ROOT)
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=a.e.sha(source):raise ValueError('Commit recovery implementation first')
    OUT.mkdir(exist_ok=False);files={};durable=0
    for s in range(4):
        started=a.e.read(ORIGINAL/f'execution/shard{s}_started.json')
        check=subprocess.run(['powershell','-NoProfile','-Command',f'if (Get-Process -Id {int(started["native_pid"])} -ErrorAction SilentlyContinue) {{exit 1}}'],capture_output=True)
        if check.returncode:raise ValueError('An explicitly owned original worker still exists;do not duplicate it')
        for path in (ORIGINAL/f'shard{s}').rglob('*'):
            if path.is_file():files[path.relative_to(a.e.ROOT).as_posix()]=a.e.sha(path)
        for db in (ORIGINAL/f'shard{s}').glob('block*/observations.sqlite'):durable+=len(inventory(db)['cases'])
    for name in ('plan.json','implementation_binding.json'):
        with (OUT/name).open('xb') as f:f.write((ORIGINAL/name).read_bytes())
    a.e.write(OUT/'recovery_registration.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.e.ROOT,text=True).strip(),
        recovery_source_sha256=a.e.sha(source),original_files=files,prior_durable_observations=durable,
        original_root_session=99073,original_native_exit_code=None,original_handle_missing=True,
        explicitly_owned_workers_absent=True,external_interruption=True,lost_inflight_attempt_count_unknown=True,
        original_files_modified=False,planned_observations=3072,new_error_allocated=False,new_independent_initials=0))
    print(json.dumps(dict(prior_durable=durable,missing_planned=3072-durable,original_exit_unknown=True)))


def guard():
    plan=a.bound(True);reg=a.e.read(OUT/'recovery_registration.json')
    if reg['recovery_source_sha256']!=a.e.sha(__file__):raise ValueError('Recovery code changed')
    for name,digest in reg['original_files'].items():
        if a.e.sha(a.e.ROOT/name)!=digest:raise ValueError('Interrupted evidence changed')
    if a.e.sha(OUT/'plan.json')!=a.e.sha(ORIGINAL/'plan.json'):raise ValueError('Original schedule changed')
    return plan,reg


def worker(shard):
    torch.set_num_threads(1);plan,reg=guard();models=a.e.load_models()
    root=OUT/f'shard{shard}';root.mkdir(exist_ok=False);cases=plan['cases'][shard::4];blocks=[];retained=fresh=0
    with ExitStack() as stack:
        envs={p:stack.enter_context(a.runtime.MatchedShiftedMergeEnv(plant=p)) for p in a.e.design.PLANTS}
        for index in range(0,len(cases),8):
            batch=cases[index:index+8];expected=[]
            for case in batch:
                offset=case['index']%3;methods=plan['methods'][offset:]+plan['methods'][:offset]
                expected.extend([[case['cluster_id'],m] for m in methods])
            metadata=dict(plan_sha256=a.e.sha(OUT/'plan.json'),shard=shard,index=index//8,cases=batch)
            directory=root/f'block{index//8:03d}';directory.mkdir(exist_ok=False)
            source=ORIGINAL/f'shard{shard}'/directory.name/'observations.sqlite'
            prior=existing(source,metadata,expected) if source.exists() else []
            if source.exists() and seal_path(source).exists():
                for old,new in ((source,directory/'observations.sqlite'),(seal_path(source),directory/'observations.sqlite.seal.json')):
                    with new.open('xb') as f:f.write(old.read_bytes())
                    if a.e.sha(old)!=a.e.sha(new):raise ValueError('Sealed copy differs')
                retained+=len(prior)
            else:
                store=EpisodeStore(directory/'observations.sqlite',metadata)
                try:
                    for key,record,trace in prior:store.append(key,record,trace);retained+=1
                    for cluster,method in expected[len(prior):]:
                        case=next(c for c in batch if c['cluster_id']==cluster)
                        record=a.runtime.execute_episode(case,method,models[method],envs[case['plant']],controller_factory=a.old_factory)
                        payload={k:v for k,v in record.items() if k!='trace'};payload['variant']='original_uncoupled_gate'
                        store.append([cluster,method],payload,record['trace']);fresh+=1
                    store.seal(expected)
                finally:store.close()
            for key,record,trace in observations(directory/'observations.sqlite',metadata,expected):
                record['trace']=trace;case=next(c for c in batch if c['cluster_id']==key[0])
                a.runtime.audit_record(case,record,models[record['method']],reference_factory=a.old_factory)
                a.episode_outcome(record['row'])
            proof=dict(block=directory.name,metadata=metadata,keys=expected,database_sha256=a.e.sha(directory/'observations.sqlite'),
                seal_sha256=a.e.sha(directory/'observations.sqlite.seal.json'),full_controller_and_physics_replay=True)
            a.e.write(directory/'reviewed.json',proof);blocks.append(proof)
            print(json.dumps(dict(shard=shard,reviewed=retained+fresh,retained=retained,new_sumo=fresh,expected=768)),flush=True)
    a.e.write(root/'report.json',dict(native_pid=os.getpid(),blocks=blocks,executions=768,retained=retained,new_sumo=fresh))


def run():
    plan,reg=guard();execution=OUT/'execution';execution.mkdir(exist_ok=False);children=[];receipts=[]
    for s in range(4):
        stream=(execution/f'shard{s}.log').open('xb');child=subprocess.Popen(command(Path(__file__),'worker','--shard',s),cwd=a.e.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        a.e.write(execution/f'shard{s}_started.json',dict(native_pid=child.pid));children.append((s,child,stream))
    for s,child,stream in children:
        code=child.wait();stream.close();event=dict(shard=s,native_pid=child.pid,actual_native_exit_code=code)
        a.e.write(execution/f'shard{s}_native_exit.json',event);receipts.append(event)
    if any(x['actual_native_exit_code'] for x in receipts):raise RuntimeError('Retain failed recovery;do not overwrite or resample')
    guard();summaries=[a.e.read(OUT/f'shard{s}/report.json') for s in range(4)]
    retained=sum(x['retained'] for x in summaries);fresh=sum(x['new_sumo'] for x in summaries)
    if retained!=reg['prior_durable_observations'] or retained+fresh!=3072:raise ValueError('Lost or duplicate observation')
    a.OUT=OUT;a.finalize(receipts)
    (OUT/'report.json').rename(OUT/'paired_record_analysis.json')
    report=a.e.read(OUT/'paired_record_analysis.json');report.pop('actual_additional_sumo_executions')
    report.update(complete_recorded_old_gate_outcomes=3072,prior_durable_observations_retained=retained,
        actual_resumed_sumo_executions=fresh,original_root_exit_unknown=True,
        total_attempt_count_including_lost_inflight_not_identifiable=True,
        original_interrupted_evidence_unchanged=True,recovery_registration_sha256=a.e.sha(OUT/'recovery_registration.json'))
    a.e.write(OUT/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','worker'));p.add_argument('--shard',type=int,choices=range(4));x=p.parse_args()
    (prepare if x.action=='prepare' else run if x.action=='run' else lambda:worker(x.shard))()
