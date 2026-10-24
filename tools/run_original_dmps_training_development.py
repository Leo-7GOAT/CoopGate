"""Exercise unchanged author replay/actor/critics on validated privileged SUMO.

Tiny integration run, explicitly not a trained comparative baseline. Requires
all eight fixed native-state interface probes to finish before any training.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from coopgate.dmps_sumo_process import DMPSProcessEnv
import execute_ood_validation_v39 as io
import load_original_dmps as author
import probe_dmps_sumo_snapshot as probe

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'artifacts/literature_reproduction/dmps_sumo_process_development'
SOURCES=('tools/load_original_dmps.py','tools/run_original_dmps_training_development.py',
         'tests/test_load_original_dmps.py','docs/original_dmps_training_development_protocol.md')


def validated_parent():
    done=io.read(PARENT/'complete.json');plan=io.read(PARENT/'plan.json')
    if done['plan_sha256']!=io.digest(PARENT/'plan.json') or len(done['cases'])!=8 or done['actual_sumo_prefixes']!=8:
        raise ValueError('Complete eight-case interface evidence required before training')
    if [x['case'] for x in done['cases']]!=plan['cases']:
        raise ValueError('Parent fixed schedule changed')
    for entry in done['cases']:
        folder=PARENT/f"{entry['case']['index']:03d}"
        if io.digest(folder/'complete.json')!=entry['complete_sha256']:
            raise ValueError('Parent completion receipt changed')
        receipt=io.read(folder/'complete.json')
        for name,expected in receipt['files_sha256'].items():
            if io.digest(folder/name)!=expected:raise ValueError('Parent interface observation changed')
        if receipt['state_restoration_checks']!=receipt['review']['actual_decisions']:
            raise ValueError('Incomplete actual-state restoration checks')
    for name,expected in plan['source_sha256'].items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Parent source changed')
    return plan,io.digest(PARENT/'complete.json')


class RecordedEnv(DMPSProcessEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.episodes=[]

    def record_episode(self):
        if hasattr(self.raw,'trace') and self.raw.trace:
            reviewed=probe.audit(self)
            self.episodes.append(dict(seed=self.seed,initial=self.raw.initial.tolist(),plant=self.raw.plant,
                trace=deepcopy(self.raw.trace),metrics=self.raw.metrics(),review=reviewed))

    def reset(self):
        self.record_episode()
        return super().reset()


def state(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


def run(source,output):
    parent,parent_sha=validated_parent()
    for name in SOURCES:io.committed(ROOT/name)
    metadata=author.verify(source)
    hashes={**parent['source_sha256'],**{name:io.digest(ROOT/name) for name in SOURCES}}
    output.mkdir(parents=True,exist_ok=False)
    io.exclusive_json(output/'protocol.json',dict(parent_complete_sha256=parent_sha,upstream_commit=author.COMMIT,
        author_sources=metadata,source_sha256=hashes,training_seed=804900,training_initial=139893001,
        evaluation_initial=139893002,train_steps=32,max_eval_steps=12,expected_gradient_updates=4,
        warmup_steps=8,batch_size=8,train_interval=8,num_grad_steps=2,ret_low=-32.,ret_high=0.,
        unsafe_penalty=-32.,num_iters=4,branch_factor=4,path_len=2,
        privileged_true_simulator_forecasts=True,comparative_efficacy_claimed=False,new_risk_certificate=False,
        runtime=io.runtime_identity()))
    torch.set_num_threads(1);np.random.seed(804900);torch.manual_seed(804900)
    module,_=author.load(source)
    with RecordedEnv(139893001,disturbance='uniform') as train_env, RecordedEnv(139893002,disturbance='uniform') as eval_env:
        trainer=module.MCTSTrainer(train_env,eval_env,torch.device('cpu'),replay_buffer_size=128,
            num_iters=4,branch_factor=4,path_len=2,ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,
            train_interval=8,num_grad_steps=2,batch_size=8,warmup_steps=8,
            eval_freq=10**9,model_save_freq=10**9,model_save_dir=None)
        initial=state(trainer.model);torch.save(initial,output/'initial_state.pt')
        original_take=trainer.planner.take_step
        train_calls=[]
        def observed_take(action,*args,**kwargs):
            before=trainer.planner.obs.copy();start=time.perf_counter()
            result=original_take(action,*args,**kwargs)
            data,(actions,code)=result
            assert len(data)==len(actions)==1
            np.testing.assert_array_equal(data[0][0],train_env.obs())
            record=dict(index=len(train_calls),observation=before.tolist(),proposed=np.asarray(action).tolist(),
                executed=np.asarray(actions[0]).tolist(),intervention=int(code),
                next_observation=data[0][0].tolist(),reward=float(data[0][1]),done=bool(data[0][2]),
                actual_trace=deepcopy(train_env.raw.trace[-1]),wall_ms=(time.perf_counter()-start)*1000)
            io.exclusive_json(output/f"train_call_{len(train_calls):03d}.json",record)
            train_calls.append(record)
            return result
        trainer.planner.take_step=observed_take
        try:
            trainer.run(32)
            train_env.record_episode()
            assert len(train_calls)==32 and trainer.num_updates==4
            assert trainer.buf.size==32+sum(x['intervention']!=0 for x in train_calls)
            assert sum(x['review']['actual_decisions'] for x in train_env.episodes)==32
            final=state(trainer.model)
            changes={group:any(not torch.equal(initial[k],v) for k,v in final.items() if k.startswith(group+'.'))
                     for group in ('actor','critic1','critic2','actor_targ','critic1_targ','critic2_targ')}
            assert all(changes.values()) and all(torch.isfinite(v).all() for v in final.values())
            torch.save(final,output/'trained_state.pt')
            np.savez_compressed(output/'replay.npz',**{name:getattr(trainer.buf,name)[:trainer.buf.size]
                for name in ('obs_buf','act_buf','next_obs_buf','rew_buf','done_buf')})
            io.exclusive_json(output/'training_episodes.json',dict(episodes=train_env.episodes))
            # Explicitly synchronize the evaluation planner; the upstream
            # evaluate() env-only reset bug is not silently copied into this run.
            observation=trainer.test_planner.reset();evaluation=[]
            for index in range(12):
                with torch.no_grad():
                    action=trainer.model.actor(torch.as_tensor(observation,dtype=torch.float32)).numpy()
                data,(actions,code)=trainer.test_planner.take_step(action)
                assert len(data)==len(actions)==1
                observation,reward,done,_=data[0]
                entry=dict(index=index,action=np.asarray(actions[0]).tolist(),intervention=int(code),
                           reward=float(reward),done=bool(done),actual_trace=deepcopy(eval_env.raw.trace[-1]))
                io.exclusive_json(output/f'eval_call_{index:03d}.json',entry);evaluation.append(entry)
                if done:break
            eval_env.record_episode()
            io.exclusive_json(output/'evaluation_episodes.json',dict(episodes=eval_env.episodes))
        except Exception as error:
            io.exclusive_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
                completed_train_calls=len(train_calls),num_updates=trainer.num_updates))
            raise
    if author.verify(source)!=metadata or io.digest(PARENT/'complete.json')!=parent_sha:
        raise ValueError('Original source or parent evidence changed')
    if io.runtime_identity()!=io.read(output/'protocol.json')['runtime']:
        raise ValueError('Simulation runtime changed during training development')
    for name,expected in hashes.items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Source changed during training')
    io.exclusive_json(output/'complete.json',dict(training_actual_decisions=32,evaluation_actual_decisions=len(evaluation),
        gradient_updates=trainer.num_updates,replay_rows=trainer.buf.size,parameter_groups_changed=changes,
        privileged_true_simulator_forecasts=True,comparative_efficacy_claimed=False,new_risk_certificate=False,
        files_sha256={p.name:io.digest(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(training_actual_decisions=32,gradient_updates=trainer.num_updates,
                         evaluation_actual_decisions=len(evaluation),parameter_groups_changed=changes)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
