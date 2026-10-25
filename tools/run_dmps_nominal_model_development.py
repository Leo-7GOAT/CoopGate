"""Fixed observable-only DMPS deployment probe, not a trained efficacy study."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from coopgate.dmps_nominal_model import ObservableDMPS
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv, matched_plant_parameters
from coopgate.merge_v4 import MergeConfig, transition_safe
import execute_ood_validation_v39 as io
import load_original_dmps as author

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'artifacts/literature_reproduction/original_dmps_training_development'
SOURCES=('src/coopgate/dmps_nominal_model.py','tests/test_dmps_nominal_model.py',
         'tools/run_dmps_nominal_model_development.py','tests/test_dmps_nominal_development.py',
         'docs/dmps_nominal_model_protocol.md','tools/load_original_dmps.py')


def schedule():
    settings=[('nominal','clean')]+[(p,'uniform') for p in ('nominal','gain_drag','lag_mild','lag_strong')]
    return [dict(index=2*i+s,seed=139895001+s,plant=plant,disturbance=mode)
            for i,(plant,mode) in enumerate(settings) for s in (0,1)]


def validate_parent():
    done=io.read(PARENT/'complete.json');protocol=io.read(PARENT/'protocol.json')
    io.committed(PARENT/'complete.json')
    for name,expected in done['files_sha256'].items():
        if io.digest(PARENT/name)!=expected:raise ValueError('Parent training file changed: '+name)
    for name,expected in protocol['source_sha256'].items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Parent training source changed: '+name)
    if done['training_actual_decisions']!=32 or done['gradient_updates']!=4:
        raise ValueError('Expected fixed immature development checkpoint')
    return protocol,io.digest(PARENT/'complete.json')


class SelectedCommand:
    """Transport the selected action to SUMO; perform no hidden safety repair."""
    def __init__(self,command,info,wall_ms):
        self.command=np.asarray(command).copy();self.info=info;self.wall_ms=wall_ms

    def select(self,state,previous,ignored):
        return self.command.copy(),dict(in_set=False,certified=False,
            intervention=self.info['intervention'],gate_ms=self.wall_ms,
            proof_scope=self.info['prediction_contract'])


def audit_episode(record):
    """Reconstruct true physics AFTER execution, outside the policy interface."""
    c=MergeConfig();case=record['case'];metrics=record['metrics']
    plant=matched_plant_parameters(case['seed'],case['plant'])
    assert metrics['plant']==plant and metrics['method']=='original_dmps_nominal_model'
    tape=disturbance_tape(case['seed'],case['disturbance'],.1,250)
    state=np.asarray(record['initial']).copy();memory=np.zeros(3);previous=np.zeros(3)
    tau=np.asarray(plant['tau']);retention=np.zeros(3)
    retention[tau>0]=np.exp(-c.dt/tau[tau>0])
    gain=np.asarray(plant['gain']);drag=np.asarray(plant['drag'])
    cursor=violations=0;maximum_prediction_error=0.;errors=[]
    finish=np.full(3,c.horizon);active=np.ones(3,dtype=bool);variation=np.zeros(3)
    for index,entry in enumerate(record['decisions']):
        np.testing.assert_allclose(entry['observation_state'],state,atol=1e-8,rtol=0)
        np.testing.assert_array_equal(entry['previous_command'],previous)
        assert entry['step']==index
        trace=entry['trace'];command=np.asarray(trace['command'])
        assert command.shape==(3,) and np.isfinite(command).all()
        assert np.all(command>=-5-1e-10) and np.all(command<=2.5+1e-10)
        assert np.all(np.abs(command-previous)<=2.5+1e-10)
        expected=np.clip(-1.25+3.75*np.asarray(entry['planner']['selected_action']),-5,2.5)
        expected=np.clip(expected,previous-2.5,previous+2.5)
        expected=np.where(state[:,0]>=c.goal,np.minimum(2.5,previous+2.5),expected)
        np.testing.assert_array_equal(command,expected)
        assert not trace['certified'] and not trace['in_set']
        assert len(entry['actual_substeps'])==5
        nominal=state.copy();variation+=active*np.abs(command-previous)
        for sub in range(5):
            np.testing.assert_array_equal(trace['innovation'][sub],tape[cursor])
            memory=retention*memory+(1-retention)*command
            applied=gain*memory-drag*state[:,1]**2+tape[cursor]
            np.testing.assert_allclose(trace['actuator_states'][sub],memory,atol=1e-9,rtol=0)
            np.testing.assert_allclose(trace['disturbance'][sub],applied-command,atol=1e-9,rtol=0)
            before=state[:,0].copy()
            state[:,1]=np.clip(state[:,1]+.1*applied,0,20);state[:,0]+=.1*state[:,1]
            violations+=int(not transition_safe(before[None],state[:,0][None],c)[0]);cursor+=1
            np.testing.assert_allclose(entry['actual_substeps'][sub],state,atol=1e-8,rtol=0)
            nominal[:,1]=np.clip(nominal[:,1]+.1*command,0,20);nominal[:,0]+=.1*nominal[:,1]
            if case['plant']=='nominal' and case['disturbance']=='clean':
                np.testing.assert_allclose(entry['actual_substeps'][sub],nominal,atol=1e-8,rtol=0)
        np.testing.assert_allclose(entry['planner']['predicted_state'],nominal,atol=1e-8,rtol=0)
        np.testing.assert_allclose(trace['x'],state,atol=1e-8,rtol=0)
        err=float(np.max(np.abs(nominal-state)));errors.append(err)
        maximum_prediction_error=max(maximum_prediction_error,err)
        assert trace['t']==(index+1)*.5
        newly=active & (state[:,0]>=60);finish[newly]=trace['t'];active[newly]=False
        state=np.asarray(trace['x']).copy();previous=command
    assert violations==metrics['contract_violations']
    np.testing.assert_allclose(finish,record['capped_finish'],atol=1e-9,rtol=0)
    np.testing.assert_allclose(maximum_prediction_error,metrics['max_prediction_error'],atol=1e-8,rtol=0)
    completed=bool(not active.any() and not metrics['collision'] and not metrics['aborted'] and not violations)
    return dict(actual_decisions=len(record['decisions']),physical_substeps=cursor,
        completed=completed,geometry_violations=violations,collision=metrics['collision'],
        administrative_failure=record['failure'],maximum_prediction_error=maximum_prediction_error,
        prediction_errors=errors,per_vehicle_cost=(finish+.05*variation).tolist(),
        nominal_model_decisions=sum(d['planner']['nominal_model_decisions'] for d in record['decisions']))


def run_case(case,model,planner_class,directory):
    np.random.seed(804950+case['index']);policy=ObservableDMPS(model,planner_class)
    with MatchedShiftedMergeEnv(plant=case['plant']) as raw:
        try:
            raw.reset_disturbed(case['seed'],'nominal',case['disturbance'],.5)
        except (RuntimeError,ValueError) as error:
            failure=dict(type=type(error).__name__,message=str(error),stage='reset')
            io.exclusive_json(directory/'failure.json',dict(case=case,failure=failure))
            return dict(case=case,completed=False,reset_failure=failure)
        raw.method='original_dmps_nominal_model'
        reads=[];native_read=raw.read
        def observed_read():
            value=native_read();reads.append(value.tolist());return value
        raw.read=observed_read
        decisions=[];failure=None;terminal=truncated=False
        for step in range(50):
            observed=raw.x.copy();previous=raw.prev.copy();reads.clear();start=time.perf_counter()
            try:
                command,info=policy.choose(observed,previous,step)
            except (RuntimeError,ValueError) as error:
                failure=dict(type=type(error).__name__,message=str(error),stage='planning',step=step,
                    observation_state=observed.tolist(),previous_command=previous.tolist())
                raw.abort_reason=str(error);break
            elapsed=(time.perf_counter()-start)*1000
            # Policy sees only the copies above. It never owns this physical env.
            assert not reads and len(raw.trace)==step and raw.t==step*.5
            np.testing.assert_array_equal(raw.x,observed);np.testing.assert_array_equal(raw.prev,previous)
            raw.gate=SelectedCommand(command,info,elapsed)
            _,_,terminal,truncated,_=raw.step([2]*3)
            assert len(raw.trace)==step+1 and len(reads)==5
            entry=dict(step=step,observation_state=observed.tolist(),previous_command=previous.tolist(),
                planner=info,wall_ms=elapsed,actual_substeps=deepcopy(reads),trace=deepcopy(raw.trace[-1]))
            io.exclusive_json(directory/f'decision_{step:03d}.json',entry);decisions.append(entry)
            if terminal or truncated or raw.violations:break
        record=dict(case=case,initial=raw.initial.tolist(),metrics=raw.metrics(),decisions=decisions,
            capped_finish=np.nan_to_num(raw.finish,nan=25.).tolist(),failure=failure,
            terminated=bool(terminal),truncated=bool(truncated))
        # Persist failures and raw trajectories BEFORE reviewing any assertions.
        io.exclusive_json(directory/'episode.json',record)
        review=audit_episode(record)
        io.exclusive_json(directory/'review.json',review)
        return dict(case=case,**review)


def run(source,output):
    parent,parent_hash=validate_parent()
    for name in SOURCES:io.committed(ROOT/name)
    hashes={**parent['source_sha256'],**{name:io.digest(ROOT/name) for name in SOURCES}}
    module,metadata=author.load(source)
    torch.set_num_threads(1);torch.manual_seed(804950)
    model=module.Model(10,3,torch.device('cpu'),ret_low=-32.,ret_high=0.)
    checkpoint=PARENT/'trained_state.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
    identity=io.runtime_identity();output.mkdir(parents=True,exist_ok=False)
    io.exclusive_json(output/'plan.json',dict(cases=schedule(),source_sha256=hashes,
        parent_complete_sha256=parent_hash,checkpoint_sha256=io.digest(checkpoint),author_sources=metadata,
        upstream_commit=author.COMMIT,runtime=identity,privileged_forecasts=False,training_steps=0,
        inherited_checkpoint_training_steps=32,new_risk_certificate=False,comparative_efficacy_claimed=False))
    results=[]
    for case in schedule():
        folder=output/f"{case['index']:03d}";folder.mkdir()
        result=run_case(case,model,module.MCTSPlanner,folder)
        io.exclusive_json(folder/'complete.json',dict(result=result,
            files_sha256={p.name:io.digest(p) for p in folder.iterdir() if p.is_file()}))
        results.append(dict(case=case,complete_sha256=io.digest(folder/'complete.json'),result=result))
        print(json.dumps(dict(completed_conditions=len(results),expected_conditions=10)),flush=True)
    if author.verify(source)!=metadata or io.runtime_identity()!=identity:
        raise ValueError('Author source/runtime changed')
    if validate_parent()[1]!=parent_hash:raise ValueError('Checkpoint evidence changed')
    for name,expected in hashes.items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Bound source changed: '+name)
    io.exclusive_json(output/'complete.json',dict(cases=results,plan_sha256=io.digest(output/'plan.json'),
        privileged_forecasts=False,new_risk_certificate=False,comparative_efficacy_claimed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
