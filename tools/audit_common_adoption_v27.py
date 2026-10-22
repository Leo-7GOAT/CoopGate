"""Reconstruct one adoption event for all 18,816 completed historical episodes.

No simulator or live calibration outcomes are used. Original admission events
and certificates remain unchanged. Compact source states and actions are saved
with every audit so the common baseline can be reconstructed independently.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from coopgate.common_adoption_v27 import BaselineReplayCache, audit_trace, changed_active_commands
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from execute_prospective_v23 import exclusive_json, runtime_identity, sources_committed, utc
from present_completed_v21 import load_complete
from profile_phase_runtime_v24 import committed, digest
from recover_checkpoint_v19 import atomic_bytes
import recover_interrupted_v21 as original

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v27/common_adoption_audit'
SCOPE=ROOT/'artifacts/v25/buffered_complete_scope/plan.json'
COMPARISON=ROOT/'artifacts/v27/physical_switch_comparison/plan.json'
OLD=ROOT/'artifacts/v21/interrupted_recovery_20261022/results'
NEW_SOURCES=('src/coopgate/common_adoption_v27.py','tests/test_common_adoption_v27.py',
             'tools/audit_common_adoption_v27.py','tests/test_common_adoption_audit_v27.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def object_hash(value):
    return hashlib.sha256(encode(value)).hexdigest()


def validate_cases(study,rows,spec):
    expected=sorted(case for shard in range(spec['shards']) for case in original.MODULES[study].schedule(spec,shard))
    actual=[original.identity(study,row) for row in rows]
    if sorted(actual)!=expected or len(actual)!=len(set(actual)):
        raise ValueError('All original methods, models and cases must be retained')


def freeze():
    summaries,protocols,_=load_complete()
    scope=read(SCOPE)
    records={}
    inputs={}
    for study,protocol in protocols.items():
        directory=OLD/study
        rows=read(directory/'episodes.json')
        validate_cases(study,rows,protocol['spec'])
        assert digest(directory/'episodes.json')==summaries[study]['episodes_sha256']
        provenance={tuple(item['case']):item for item in read(directory/'provenance.json')}
        for name in ('episodes.json','provenance.json','summary.json','sha256.json'):
            path=directory/name
            inputs[path.relative_to(ROOT).as_posix()]=digest(path)
        for row in rows:
            case=original.identity(study,row)
            source=provenance[case]
            if digest(ROOT/source['path'])!=source['sha256']:
                raise ValueError('Completed source trace changed')
            records.setdefault((study,case[0]),[]).append(dict(case=list(case),
                trace_path=source['path'],trace_sha256=source['sha256'],source_row_sha256=object_hash(row)))
    groups=[]
    for group in scope['groups']:
        key=(group['study'],group['initial_index'])
        groups.append(dict(block_index=len(groups),study=key[0],initial_index=key[1],
            local_index=group['local_index'],scenario=group['scenario'],
            cases=sorted(records.pop(key),key=lambda r:r['case'])))
    if records or len(groups)!=288 or sum(len(g['cases']) for g in groups)!=18816:
        raise ValueError('Full historical grid required')
    sources=dict(scope['source_sha256'])
    for name,expected in sources.items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Existing source changed: '+name)
    sources.update({name:digest(ROOT/name) for name in NEW_SOURCES})
    sources_committed([ROOT/name for name in sources])
    plan=dict(purpose=__doc__,source_sha256=sources,inputs_sha256=inputs,
        scope_plan_sha256=digest(SCOPE),comparison_plan_sha256=digest(COMPARISON),
        groups=groups,protocols=protocols,runtime=runtime_identity(),blocks=288,old_episodes=18816,
        new_sumo_runs=0,new_independent_initials=0,
        event='At least one not-yet-arrived vehicle has an executed command differing from the same-state full baseline by >1e-9; any such step adopts at episode level',
        cache='Exact measured state, previous actually executed command, step and fixed baseline configuration; cache reset for each independent initial; output arrays copied',
        retained='Original row and its original learned_action_steps, every observed state/action, reconstructed baseline commands, per-agent differences and source hashes',
        failure='No source filtering or silent recovery. A reconstruction error stops with committed prefix retained; source controller failures remain in all counts.',
        timing='Offline reconstruction only; no controller latency comparison',
        priority='Below-normal Windows process priority for this audit only, preserving resources for the live experiments',
        active_calibration_outcomes_read=False,new_risk_certificate=False,
        original_event_definitions_or_certificates_changed=False)
    OUT.mkdir(parents=True,exist_ok=False)
    exclusive_json(OUT/'plan.json',plan)
    print(json.dumps(dict(plan_sha256=digest(OUT/'plan.json'),old_episodes=18816,new_sumo_runs=0)),flush=True)


def inputs():
    committed(OUT/'plan.json')
    plan=read(OUT/'plan.json')
    if (digest(SCOPE)!=plan['scope_plan_sha256'] or digest(COMPARISON)!=plan['comparison_plan_sha256']
            or runtime_identity()!=plan['runtime']):
        raise ValueError('Frozen comparison/scope/runtime differs')
    for name,expected in {**plan['source_sha256'],**plan['inputs_sha256']}.items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Frozen source/input changed: '+name)
    sources_committed([ROOT/name for name in plan['source_sha256']])
    return plan


def metadata(group):
    return dict(plan_sha256=digest(OUT/'plan.json'),block_index=group['block_index'],
                study=group['study'],initial_index=group['initial_index'])


def compact_trace(trace,audits):
    if len(trace)!=len(audits):
        raise ValueError('Every original control step requires an audit')
    return [dict(**audit,command=item['command'],x=item['x']) for item,audit in zip(trace,audits,strict=True)]


def check_compact(row,trace,record):
    if object_hash(row)!=record['source_row_sha256'] or record['original_learning_steps']!=row['learned_action_steps']:
        raise ValueError('Original row or adoption field changed')
    if len(trace)!=row['control_steps']:
        raise ValueError('Incomplete observed control history')
    state=np.asarray(row['initial'])
    total=0
    for step,item in enumerate(trace):
        if item['step']!=step:
            raise ValueError('Noncontiguous control history')
        flags=changed_active_commands(state,item['command'],item['baseline_command'])
        if flags.tolist()!=item['changed_active_vehicles'] or bool(flags.any())!=item['adopted']:
            raise ValueError('Common adoption event differs')
        total+=int(flags.any())
        state=np.asarray(item['x'])
    if total!=record['common_learning_steps'] or (total>0)!=record['common_episode_adopted']:
        raise ValueError('Common episode adoption denominator differs')
    if row['controller']=='baseline' and total:
        raise ValueError('Original baseline differs from its own reconstruction')


def run():
    plan=inputs()
    execution=OUT/'execution'
    execution.mkdir(exist_ok=False)
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetPriorityClass.argtypes=(ctypes.c_void_p,ctypes.c_uint32)
        kernel.SetPriorityClass.restype=ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(),subprocess.BELOW_NORMAL_PRIORITY_CLASS):
            raise ctypes.WinError(ctypes.get_last_error())
    exclusive_json(execution/'start.json',dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=utc(),
        plan_sha256=digest(OUT/'plan.json'),priority='below_normal' if os.name=='nt' else 'default'))
    torch.set_num_threads(1)
    rows={s:{original.identity(s,r):r for r in read(OLD/s/'episodes.json')} for s in plan['protocols']}
    blocks=OUT/'blocks'
    blocks.mkdir(exist_ok=False)
    completed=0
    for group in plan['groups']:
        directory=blocks/f"{group['block_index']:05d}"
        directory.mkdir()
        store=EpisodeStore(directory/'observations.sqlite',metadata(group))
        cache=BaselineReplayCache()
        started=time.perf_counter()
        steps=0
        try:
            for entry in group['cases']:
                row=rows[group['study']][tuple(entry['case'])]
                if object_hash(row)!=entry['source_row_sha256'] or digest(ROOT/entry['trace_path'])!=entry['trace_sha256']:
                    raise ValueError('Source observation changed during reconstruction')
                trace=read(ROOT/entry['trace_path'])
                audits=audit_trace(row['initial'],trace,cache)
                data=compact_trace(trace,audits)
                common=sum(item['adopted'] for item in audits)
                record=dict(original_row=row,source_row_sha256=entry['source_row_sha256'],
                    source_trace_path=entry['trace_path'],source_trace_sha256=entry['trace_sha256'],
                    original_learning_steps=row['learned_action_steps'],common_learning_steps=common,
                    common_episode_adopted=bool(common),source_controller_failed=bool(not row['completed']
                        or row['collision'] or row['aborted'] or row['contract_violations']))
                # Preserve a completed reconstruction before consistency assertions.
                store.append(entry['case'],record,data)
                check_compact(row,data,record)
                steps+=len(data);completed+=1
            store.seal([entry['case'] for entry in group['cases']])
        finally:
            store.close()
        exclusive_json(directory/'complete.json',dict(**metadata(group),old_episodes=len(group['cases']),
            control_steps=steps,baseline_evaluations=cache.evaluations,cache_hits=cache.hits,
            offline_wall_seconds=time.perf_counter()-started,
            database_sha256=digest(directory/'observations.sqlite'),seal_sha256=digest(directory/'observations.sqlite.seal.json')))
        atomic_bytes(execution/'progress.json',encode(dict(native_pid=os.getpid(),utc=utc(),
            completed_blocks=group['block_index']+1,expected_blocks=288,completed_episodes=completed,expected_episodes=18816)))
        print(json.dumps(dict(completed_blocks=group['block_index']+1,completed_episodes=completed)),flush=True)
    assert completed==18816
    exclusive_json(execution/'collection_complete.json',dict(plan_sha256=digest(OUT/'plan.json'),
        native_pid=os.getpid(),old_episodes=18816,blocks=288,utc=utc()))


def complete_records(plan):
    receipt=read(OUT/'execution/observation.json')
    started=read(OUT/'execution/start.json')
    done=read(OUT/'execution/collection_complete.json')
    fingerprint=digest(OUT/'plan.json')
    if (receipt.get('exit_code')!=0 or receipt.get('plan_sha256')!=fingerprint
            or started['plan_sha256']!=fingerprint or done['plan_sha256']!=fingerprint
            or done['native_pid']!=started['native_pid'] or done['old_episodes']!=18816 or done['blocks']!=288):
        raise ValueError('Actual successful exit and full historical scope required')
    count=0
    for group in plan['groups']:
        directory=OUT/'blocks'/f"{group['block_index']:05d}"
        checked=read(directory/'complete.json')
        if (checked['database_sha256']!=digest(directory/'observations.sqlite')
                or checked['seal_sha256']!=digest(directory/'observations.sqlite.seal.json')
                or checked['old_episodes']!=len(group['cases'])):
            raise ValueError('Completed audit block changed')
        for case,record,trace in observations(directory/'observations.sqlite',metadata(group),[e['case'] for e in group['cases']]):
            check_compact(record['original_row'],trace,record)
            count+=1
            yield group,case,record,trace
    if count!=18816:
        raise ValueError('Incomplete common-event audit')


def summarize():
    plan=inputs()
    groups={}
    steps=0
    for group,case,record,trace in complete_records(plan):
        key=(group['study'],record['original_row']['controller'])
        item=groups.setdefault(key,dict(study=key[0],controller=key[1],old_episodes=0,
            original_adopted_episodes=0,common_adopted_episodes=0,changed_episode_events=0,
            changed_step_counts=0,source_failed_episodes=0))
        item['old_episodes']+=1
        item['original_adopted_episodes']+=record['original_learning_steps']>0
        item['common_adopted_episodes']+=record['common_episode_adopted']
        item['changed_episode_events']+=(record['original_learning_steps']>0)!=record['common_episode_adopted']
        item['changed_step_counts']+=record['original_learning_steps']!=record['common_learning_steps']
        item['source_failed_episodes']+=record['source_controller_failed']
        steps+=len(trace)
    output=OUT/'analysis'
    output.mkdir(exist_ok=False)
    report=dict(plan_sha256=digest(OUT/'plan.json'),exit_observation_sha256=digest(OUT/'execution/observation.json'),
        old_episodes=18816,control_steps=steps,controllers=list(groups.values()),
        new_sumo_runs=0,new_independent_initials=0,new_risk_certificate=False,
        original_event_definitions_or_certificates_changed=False,active_calibration_outcomes_read=False,
        scope='Counts above are mode/plant episodes, not new independent initials. Use original clusters and declared actual modes for subsequent comparisons.')
    exclusive_json(output/'report.json',report)
    exclusive_json(output/'sha256.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(dict(old_episodes=18816,control_steps=steps,controller_groups=len(groups))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','run','summarize'))
    args=parser.parse_args()
    {'freeze':freeze,'run':run,'summarize':summarize}[args.action]()
