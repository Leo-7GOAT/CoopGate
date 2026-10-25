"""Freeze and collect a complete 3x3x13 timestamp-preprocessing SUMO probe."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from types import FunctionType

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import inventory, observations
from coopgate.ood_mismatch_v39 import OODMismatchMergeEnv
from coopgate.timestamped_control_probe import VARIANTS, PublicSensorBridge, TimestampPort, TimestampedGate
import audit_aged_observation_envelope as parent
import audit_ood_validation_v39 as original_audit
import execute_ood_validation_v39 as io

ROOT = parent.ROOT
OUT = ROOT/'artifacts/diagnostics/timestamped_control_probe'
PLAN = OUT/'protocol.json'
EXPECTED_RUNS = 117
SOURCES = ('src/coopgate/timestamped_control_probe.py','tools/execute_timestamped_control_probe.py',
    'tools/verify_timestamped_control_probe.py','tools/watch_timestamped_control_probe.py',
    'tests/test_timestamped_control_probe.py','docs/timestamped_control_probe_protocol.md',
    'src/coopgate/aged_observation_envelope.py','src/coopgate/bounded_observation_v45.py',
    'tools/watch_dmps_effective_action.py')


def private(function, **changes):
    result = FunctionType(function.__code__,{**function.__globals__,**changes},
        function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def parent_inputs():
    proof = io.read(parent.OUT/'verification.json')
    assert proof['audit_exit_code']==0 and proof['all_791_frames_checked']
    assert proof['report_sha256']==io.digest(parent.OUT/'report.json')
    manifest = io.read(parent.PARENT/'manifest.json')
    assert manifest['test_exit_code']==0 and manifest['actual_sumo_runs']==39
    paths=[parent.OUT/'report.json',parent.OUT/'verification.json',parent.PARENT/'manifest.json']
    for name,sha in manifest['files_sha256'].items():
        path=parent.PARENT/name
        assert path.resolve().is_relative_to(parent.PARENT) and io.digest(path)==sha
        paths.append(path)
    for path in paths:io.committed(path)
    return {p.relative_to(ROOT).as_posix():io.digest(p) for p in paths}


def freeze():
    if OUT.exists():raise FileExistsError('Preserve first complete timestamp probe')
    parents=parent_inputs();_,binding=io.binding_inputs()
    hashes={**binding['source_sha256'],**{name:io.digest(ROOT/name) for name in SOURCES}}
    for name in hashes:io.committed(ROOT/name)
    jobs=[dict(index=3*v+i,variant=variant,source_index=i) for v,variant in enumerate(VARIANTS) for i in range(3)]
    OUT.mkdir(parents=True,exist_ok=False)
    io.exclusive_json(PLAN,dict(source_sha256=hashes,parent_sha256=parents,runtime=io.runtime_identity(),
        variants=list(VARIANTS),jobs=jobs,expected_sumo_runs=EXPECTED_RUNS,expected_blocks=9,
        reused_development_initials=3,new_independent_initials=0,posthoc_development=True,
        public_inputs='Same received state, source timestamps, known measurement error/model envelope, executed command prefix; all variants form the same envelope.',
        controller='Unchanged V35 gate at transformed point state, still lag_strong; no state-set safety certificate',
        comparison='All three fixed policies and baseline/three adapters; raw variant must reproduce every prior V45 non-timing field.',
        failures='Retain every abort, timeout, collision and geometry violation; no resampling or subset gain claims.',
        timing='Shared desktop resources; SUMO waits; no callback cutoff or isolated latency ranking.',
        formal_studies_unchanged=True,partial_formal_efficacy_read=False,new_risk_certificate=False,
        utc=io.utc()))
    print(json.dumps(dict(protocol_sha256=io.digest(PLAN),expected_sumo_runs=EXPECTED_RUNS,launched=False)),flush=True)


def inputs():
    io.committed(PLAN);p=io.read(PLAN)
    assert p['variants']==list(VARIANTS) and p['expected_sumo_runs']==EXPECTED_RUNS and p['expected_blocks']==9
    assert p['jobs']==[dict(index=3*v+i,variant=variant,source_index=i) for v,variant in enumerate(VARIANTS) for i in range(3)]
    assert p['runtime']==io.runtime_identity() and p['parent_sha256']==parent_inputs()
    for name,sha in p['source_sha256'].items():
        io.committed(ROOT/name);assert io.digest(ROOT/name)==sha
    return p


def source_case(index):
    folder=parent.PARENT/f'{index:05d}'
    return io.read(folder/'development_plan.json'),io.read(folder/'development_group.json')


def metadata(group, variant, plan_sha):
    return {**io.metadata(group,plan_sha,plan_sha),'preprocessing_variant':variant}


def native_functions(variant):
    def run(plan,group,method,models,env):
        def route(method,models):
            port=TimestampPort()
            env.perception=PublicSensorBridge(env.perception,port)
            model=None if method['model']=='none' else models[method['model']]
            return io.TimedGate(TimestampedGate(model,'lag_strong',method['rule'],variant,port))
        record,trace=private(io.run_case,route_gate=route)(plan,group,method,models,env)
        record['observation_adapter_inputs']=deepcopy(env.gate.gate.receipts)
        return record,trace
    def audit(plan,group,method,record,trace,model):
        port=TimestampPort();holder=[]
        def factory(model,profile,rule):
            gate=TimestampedGate(model,profile,rule,variant,port);holder.append(gate);return gate
        def received(*args,**kwargs):
            result=original_audit._expected_observation(*args,**kwargs)
            port.publish(result[1]);return result
        review=private(original_audit.audit_episode,RecoveryJointGate=factory,
                       _expected_observation=received)(plan,group,method,record,trace,model)
        assert len(holder)==1
        if holder[0].receipts!=record['observation_adapter_inputs']:
            raise ValueError('Independent public preprocessing replay differs')
        receipts=holder[0].receipts
        assert len(receipts)==len(trace)+int(record['row']['aborted'])
        maximum=memory_error=0.;covered=0
        for step,receipt in enumerate(receipts):
            envelope=receipt['envelope']
            if envelope is None:
                assert receipt['estimator_error'] is not None;continue
            state=np.asarray(trace[step]['true_state_before'] if step<len(trace)
                             else record['row']['terminal_attempt']['true_state'])
            memory=np.zeros(3) if step==0 else np.asarray(trace[step-1]['actuator_states'][-1])
            maximum=max(maximum,float(np.max(np.asarray(envelope['lower'])-state)),
                        float(np.max(state-np.asarray(envelope['upper']))))
            memory_error=max(memory_error,float(np.max(np.asarray(envelope['memory_lower'])-memory)),
                             float(np.max(memory-np.asarray(envelope['memory_upper']))))
            assert maximum<=2e-8 and memory_error<=2e-10
            covered+=1
        return dict(**review,public_preprocessing_replayed=True,received_frames=len(receipts),
            envelope_frames=covered,maximum_state_violation=maximum,maximum_memory_violation=memory_error,
            original_physical_certificate_restored=False)
    return run,audit


def non_timing(value):
    if isinstance(value,dict):
        return {k:non_timing(v) for k,v in value.items() if not k.endswith('_ms') and k not in
                ('age_adapter','observation_adapter_inputs','episode_wall_seconds','velocity_projection')}
    if isinstance(value,list):return [non_timing(v) for v in value]
    return value


def compare_raw(directory,group,plan_sha):
    old=parent.PARENT/f"{group['block_index']:05d}"
    old_meta=inventory(old/'observations.sqlite')['metadata']
    before=list(observations(old/'observations.sqlite',old_meta,io.schedule(group)))
    after=list(observations(directory/'observations.sqlite',metadata(group,'projected_raw',plan_sha),io.schedule(group)))
    assert len(before)==len(after)==13
    for (key,body,trace),(new_key,new_body,new_trace) in zip(before,after):
        assert key==new_key and non_timing(body)==non_timing(new_body) and non_timing(trace)==non_timing(new_trace)


def run():
    if (OUT/'native_start.json').exists() or (OUT/'blocks').exists():
        raise FileExistsError('No implicit resume or overwrite')
    p=inputs();plan_sha=io.digest(PLAN)
    io.exclusive_json(OUT/'native_start.json',dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=io.utc()))
    (OUT/'blocks').mkdir()
    original,_=source_case(0)
    torch.set_num_threads(1)
    models={name:load_coupled(ROOT/item['path'])[0] for name,item in original['models'].items()}
    entries=[]
    try:
        with OODMismatchMergeEnv() as env:
            for job in p['jobs']:
                plan,group=source_case(job['source_index']);variant=job['variant']
                native,audit=native_functions(variant)
                block=OUT/'blocks'/f"{job['index']:03d}"
                collect=private(io.collect_block,metadata=lambda g,a,b:metadata(g,variant,plan_sha))
                reviews=collect(plan,group,block,models,env,plan_sha,plan_sha,run_fn=native,audit_fn=audit)
                if variant=='projected_raw':compare_raw(block,group,plan_sha)
                entries.append(dict(**job,complete_sha256=io.digest(block/'complete.json')))
                progress=dict(completed_blocks=len(entries),expected_blocks=9,completed_sumo_runs=13*len(entries),
                              expected_sumo_runs=EXPECTED_RUNS,native_pid=os.getpid(),utc=io.utc())
                io.atomic_bytes(OUT/'progress.json',json.dumps(progress).encode())
                print(json.dumps(progress),flush=True)
    except Exception as error:
        io.exclusive_json(OUT/'failure.json',dict(type=type(error).__name__,message=str(error),
            complete_blocks=len(entries),prefix_preserved=True,utc=io.utc()))
        raise
    assert inputs()==p and io.digest(PLAN)==plan_sha and len(entries)==9
    io.exclusive_json(OUT/'complete.json',dict(actual_sumo_runs=EXPECTED_RUNS,blocks=entries,
        protocol_sha256=plan_sha,all_failures_retained=True,raw_non_timing_identity_verified=True,
        files_sha256={x.relative_to(OUT).as_posix():io.digest(x) for x in OUT.rglob('*') if x.is_file()}))
    print(json.dumps(dict(actual_sumo_runs=EXPECTED_RUNS,collection_complete=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run'))
    globals()[parser.parse_args().command]()
