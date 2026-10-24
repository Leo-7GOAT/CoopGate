"""Pinned original DMPS planner against real, privileged SUMO state clones."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from coopgate.dmps_sumo_snapshot import DMPSSumoSnapshotEnv
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.merge_v4 import transition_safe
import audit_dmps_source_contract as upstream
import execute_ood_validation_v39 as io
import execute_fair_failure_v44 as frozen_reference

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('src/coopgate/dmps_sumo_snapshot.py','tools/probe_dmps_sumo_snapshot.py',
         'tests/test_dmps_sumo_snapshot.py','docs/dmps_sumo_snapshot_protocol.md')


def plain(value):
    if isinstance(value,np.ndarray):
        return dict(dtype=str(value.dtype),shape=list(value.shape),values=plain(value.tolist()))
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    if isinstance(value,(list,tuple)):return [plain(x) for x in value]
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if hasattr(value,'__dict__'):return dict(type=type(value).__qualname__,fields=plain(vars(value)))
    return value


def capture(env):
    raw=env.raw
    native=[(raw.conn.vehicle.getDistance(v),raw.conn.vehicle.getSpeed(v),
             raw.conn.vehicle.getLanePosition(v),raw.conn.vehicle.getRoadID(v),raw.conn.vehicle.getRouteIndex(v))
            for v in raw.ids]
    return dict(host=plain({k:v for k,v in raw.__dict__.items() if k not in env._EXCLUDED}),
                rng=plain(env.rng.bit_generator.state),native=native,time=raw.conn.simulation.getTime())


def equivalent(a,b):
    # Host memory is restored exactly. Native serialization/reintegration is
    # checked within the existing route-coordinate numerical tolerance.
    assert a['host']==b['host'] and a['rng']==b['rng'] and a['time']==b['time']
    for old,new in zip(a['native'],b['native']):
        np.testing.assert_allclose(old[:3],new[:3],atol=1e-8,rtol=0)
        assert old[3:]==new[3:]


def trajectory_equivalent(a,b):
    # Advancing after native save/load can change floating roundoff; every
    # numeric field is checked, not only the final positions.
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:trajectory_equivalent(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):trajectory_equivalent(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=0)
    else:assert a==b


def restore_checks(env):
    root=env.get_state();original=capture(env)
    sample=env.action_space.sample()
    env.step(np.array([1.,-1.,0.]))
    child=env.get_state();first=capture(env)
    env.step(np.array([-1.,0.,1.]))
    leaf=capture(env)
    env.set_state(child);equivalent(first,capture(env))
    env.step(np.array([-1.,0.,1.]))
    trajectory_equivalent(leaf,capture(env))
    env.set_state(root);equivalent(original,capture(env))
    np.testing.assert_array_equal(env.action_space.sample(),sample)
    env.step(np.array([1.,-1.,0.]))
    trajectory_equivalent(first,capture(env))
    env.set_state(root);equivalent(original,capture(env))
    return root


def audit(env):
    raw=env.raw;c=raw.c
    x=raw.initial.copy();memory=np.zeros(3);previous=np.zeros(3)
    tape=disturbance_tape(env.seed,env.disturbance,raw.innovation_radius,len(raw.tape))
    plant=raw.plant;tau=np.array(plant['tau']);retention=np.zeros(3)
    retention[tau>0]=np.exp(-c.dt/tau[tau>0]);gain=np.array(plant['gain']);drag=np.array(plant['drag'])
    cursor=violations=0;reward=0.
    for step,entry in enumerate(raw.trace):
        u=np.array(entry['command']);assert np.all(u>=-5-1e-10) and np.all(u<=2.5+1e-10)
        assert np.all(np.abs(u-previous)<=2.5+1e-10)
        reward-=float(np.mean((x[:,0]<c.goal)*(.5+.05*np.abs(u-previous))))
        for sub in range(5):
            np.testing.assert_array_equal(entry['innovation'][sub],tape[cursor])
            memory=retention*memory+(1-retention)*u
            applied=gain*memory-drag*x[:,1]**2+tape[cursor]
            np.testing.assert_allclose(entry['actuator_states'][sub],memory,atol=1e-9,rtol=0)
            np.testing.assert_allclose(entry['disturbance'][sub],applied-u,atol=1e-9,rtol=0)
            old=x[:,0].copy();x[:,1]=np.clip(x[:,1]+c.dt*applied,0,c.vmax);x[:,0]+=c.dt*x[:,1]
            violations+=int(not transition_safe(old[None],x[:,0][None],c)[0]);cursor+=1
        np.testing.assert_allclose(entry['x'],x,atol=1e-8,rtol=0)
        assert entry['t']==(step+1)*.5
        x=np.array(entry['x']);previous=u
    assert raw.substep==cursor and raw.violations==violations
    return dict(actual_decisions=len(raw.trace),physical_substeps=cursor,geometry_violations=violations,
                actual_collision=bool(raw.collision),reconstructed_reward=reward)


def load(source):
    content=source.read_bytes()
    if hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()!=upstream.BLOB:
        raise ValueError('Pinned upstream mcts.py changed')
    spec=importlib.util.spec_from_file_location('original_dmps_sumo_probe',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run(source,output):
    for name in SOURCES:io.committed(ROOT/name)
    _,binding=frozen_reference.bound_inputs()
    hashes={**binding['source_sha256'],**{name:io.digest(ROOT/name) for name in SOURCES}}
    module=load(source);source_sha=io.digest(source)
    identity=io.runtime_identity()
    cases=[dict(index=2*p+s,seed=139892001+s,plant=plant) for p,plant in
           enumerate(('nominal','gain_drag','lag_mild','lag_strong')) for s in (0,1)]
    output.mkdir(parents=True,exist_ok=False)
    io.exclusive_json(output/'plan.json',dict(cases=cases,scenario='nominal',disturbance='uniform',
        source_sha256=hashes,upstream_commit=upstream.COMMIT,upstream_blob=upstream.BLOB,upstream_sha256=source_sha,
        runtime=identity,privileged_true_simulator_forecasts=True,critic='constant zero',training_steps=0,
        max_actual_decisions_per_prefix=12,new_risk_certificate=False,efficacy_claimed=False))
    torch.set_num_threads(1)
    def critic(obs,acts):return torch.zeros((len(obs),1),device=obs.device)
    entries=[]
    for case in cases:
        directory=output/f"{case['index']:03d}";directory.mkdir()
        np.random.seed(804800+case['index'])
        with DMPSSumoSnapshotEnv(case['seed'],plant=case['plant'],disturbance='uniform') as env:
            try:
                planner=module.MCTSPlanner(env,.99,4,4,2,torch.device('cpu'),critic,single_action=True)
                planner.reset()
                initial=restore_checks(env)
                (directory/'initial_native.xml').write_bytes(initial.native)
                io.exclusive_json(directory/'initial_host.json',plain(initial.host))
                decisions=[];reward=0.
                for step in range(12):
                    before=capture(env)
                    check=bool(planner.check_action(np.ones(3)))
                    equivalent(before,capture(env))
                    old_length=len(env.raw.trace);old_time=env.raw.t
                    start=time.perf_counter()
                    data,(actions,intervention)=planner.take_step(np.ones(3))
                    wall_ms=(time.perf_counter()-start)*1000
                    assert len(data)==len(actions)==1 and len(env.raw.trace)==old_length+1
                    assert env.raw.t==old_time+.5
                    np.testing.assert_array_equal(planner.obs,env.obs())
                    obs,rew,done,info=data[0];reward+=rew
                    record=dict(step=step,proposed_recoverable=check,intervention=int(intervention),
                                action=np.asarray(actions[0]).tolist(),wall_ms=wall_ms,reward=rew,done=bool(done),
                                trace=deepcopy(env.raw.trace[-1]),metrics=info)
                    io.exclusive_json(directory/f'decision_{step:03d}.json',record)
                    decisions.append(record)
                    if done:break
                review=audit(env)
                np.testing.assert_allclose(review['reconstructed_reward'],reward,atol=1e-9,rtol=0)
                io.exclusive_json(directory/'complete.json',dict(case=case,review=review,
                    interventions=sum(d['intervention'] for d in decisions),state_restoration_checks=len(decisions),
                    snapshots_saved=env.snapshots_saved,snapshots_loaded=env.snapshots_loaded,
                    simulator_decisions_including_forecasts=env.simulator_steps,actual_prefix_not_full_evaluation=True,
                    files_sha256={p.name:io.digest(p) for p in directory.iterdir() if p.is_file()}))
            except Exception as error:
                io.exclusive_json(directory/'failure.json',dict(type=type(error).__name__,message=str(error),state=capture(env)))
                raise
        entries.append(dict(case=case,complete_sha256=io.digest(directory/'complete.json')))
        print(json.dumps(dict(completed_prefixes=len(entries),expected_prefixes=8)),flush=True)
    for name,expected in hashes.items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Source changed: '+name)
    if io.digest(source)!=source_sha or io.runtime_identity()!=identity:raise ValueError('Upstream or runtime changed')
    io.exclusive_json(output/'complete.json',dict(cases=entries,plan_sha256=io.digest(output/'plan.json'),
        original_planner_source_unchanged=True,actual_sumo_prefixes=8,training_steps=0,
        privileged_true_simulator_forecasts=True,new_risk_certificate=False,efficacy_claimed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
