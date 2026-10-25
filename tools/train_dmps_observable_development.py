"""From-scratch author TD3 training with nominal-only search and real SUMO data.

Fixed integration stage, not a converged or budget-matched efficacy comparison.
Finite 25 s task ends are absorbing for learning, including the horizon itself.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from coopgate.dmps_execution_bridge import ModelCommitPlanner
from coopgate.dmps_nominal_model import command_for
from coopgate.matched_shift_v21 import MatchedShiftedMergeEnv
from coopgate.merge_v4 import MergeConfig
import execute_fair_failure_v44 as frozen
import execute_ood_validation_v39 as io
import load_original_dmps as author
import run_dmps_nominal_model_development as evidence

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('src/coopgate/dmps_execution_bridge.py','src/coopgate/dmps_nominal_model.py',
    'src/coopgate/dmps_nominal_model_v2.py','tools/train_dmps_observable_development.py',
    'tools/load_original_dmps.py','tools/run_dmps_nominal_model_development.py',
    'tests/test_dmps_execution_bridge.py','tests/test_dmps_observable_training.py',
    'docs/dmps_observable_training_protocol.md')


def training_cases():
    return [dict(index=i,seed=139896001+i,plant='nominal',disturbance='uniform') for i in range(257)]


def evaluation_cases():
    return [dict(index=2*i+s,seed=139897001+s,plant=plant,disturbance=mode)
        for i,(plant,mode) in enumerate([('nominal','clean')]+[(p,'uniform') for p in
            ('nominal','gain_drag','lag_mild','lag_strong')]) for s in (0,1)]


class ObservableSumoExecution:
    def __init__(self,cases,output,sampling_seed):
        self.c=MergeConfig();self.cases=cases;self.output=output;self.cursor=0;self.raw=None
        self.num_steps=50;self.action_rng=np.random.default_rng(sampling_seed)
        self.observation_space=SimpleNamespace(shape=(10,))
        self.action_space=SimpleNamespace(shape=(3,),sample=lambda:self.action_rng.uniform(-1,1,3))
        self.episodes=[];self.calls=[];self.recorded=False;self.failure=None

    def obs(self):
        raw=self.raw
        return np.append(np.column_stack((raw.x[:,0]/150,raw.x[:,1]/20,raw.prev/5)).ravel(),raw.t/25).astype(np.float32)

    def observables(self):
        return self.raw.x.copy(),self.raw.prev.copy(),round(self.raw.t/.5)

    def reset(self):
        self.finish_record()
        if self.raw is not None:self.raw.close();self.raw=None
        if self.cursor>=len(self.cases):raise RuntimeError('Fixed episode seed schedule exhausted')
        self.case=self.cases[self.cursor];self.cursor+=1
        self.directory=self.output/f"episode_{self.cursor-1:03d}";self.directory.mkdir()
        self.recorded=False;self.decisions=[];self.reads=[];self.failure=None
        self.terminal=self.truncated=False
        self.raw=MatchedShiftedMergeEnv(plant=self.case['plant'])
        try:
            self.raw.reset_disturbed(self.case['seed'],'nominal',self.case['disturbance'],.5)
        except Exception as error:
            io.exclusive_json(self.directory/'reset_failure.json',dict(case=self.case,
                type=type(error).__name__,message=str(error)))
            self.recorded=True
            raise
        self.raw.method='original_dmps_nominal_model'
        native_read=self.raw.read
        def observed_read():
            state=native_read();self.reads.append(state.tolist());return state
        self.raw.read=observed_read
        return self.obs()

    def execute(self,action,context):
        raw=self.raw;x,previous,step=self.observables()
        assert not self.recorded and not self.terminal and not self.truncated and step==len(self.decisions)
        command=command_for(action,x,previous,self.c)
        np.testing.assert_array_equal(context['selected_action'],action)
        self.reads.clear();raw.gate=evidence.SelectedCommand(command,context,context['wall_ms'])
        _,_,terminal,truncated,_=raw.step([2]*3)
        assert len(raw.trace)==step+1 and len(self.reads)==5
        self.terminal=bool(terminal or raw.violations);self.truncated=bool(truncated)
        reward=-float(np.mean((x[:,0]<60)*(.5+.05*np.abs(command-previous))))
        entry=dict(step=step,observation_state=x.tolist(),previous_command=previous.tolist(),planner=context,
            wall_ms=context['wall_ms'],actual_substeps=deepcopy(self.reads),trace=deepcopy(raw.trace[-1]))
        io.exclusive_json(self.directory/f'decision_{step:03d}.json',entry);self.decisions.append(entry)
        done=self.terminal or self.truncated
        call=dict(index=len(self.calls),observation=np.append(np.column_stack((x[:,0]/150,x[:,1]/20,previous/5)).ravel(),step/50).astype(np.float32).tolist(),
            proposed=context['proposal'],executed=np.asarray(action).tolist(),intervention=context['intervention'],
            next_observation=self.obs().tolist(),reward=reward,done=done)
        io.exclusive_json(self.output/f'call_{len(self.calls):04d}.json',call);self.calls.append(call)
        return self.obs(),reward,done,raw.metrics()

    def finish_record(self):
        if self.raw is None or self.recorded:return
        raw=self.raw
        record=dict(case=self.case,initial=raw.initial.tolist(),metrics=raw.metrics(),decisions=self.decisions,
            capped_finish=np.nan_to_num(raw.finish,nan=25.).tolist(),failure=self.failure,
            terminated=self.terminal,truncated=self.truncated)
        io.exclusive_json(self.directory/'episode.json',record)
        review=evidence.audit_episode(record)
        io.exclusive_json(self.directory/'review.json',review)
        self.episodes.append(dict(case=self.case,review=review,full_task_end=self.terminal or self.truncated,
            episode_sha256=io.digest(self.directory/'episode.json')))
        self.recorded=True

    def close(self):
        try:self.finish_record()
        finally:
            if self.raw is not None:self.raw.close();self.raw=None


def model_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def install_bridges(trainer,planner_class):
    trainer.planner=ModelCommitPlanner(trainer.env,planner_class,trainer.model.critic1)
    trainer.test_planner=ModelCommitPlanner(trainer.test_env,planner_class,trainer.model.critic1)
    # Original trainer excludes a done occurring exactly at max_ep_len from
    # its absorbing replay flag. This task includes the 25 s horizon in its MDP.
    trainer.max_ep_len=trainer.env.num_steps+1


def run(source,output):
    for name in SOURCES:io.committed(ROOT/name)
    _,binding=frozen.bound_inputs()
    hashes={**binding['source_sha256'],**{name:io.digest(ROOT/name) for name in SOURCES}}
    module,metadata=author.load(source);identity=io.runtime_identity()
    torch.set_num_threads(1);torch.manual_seed(805101);np.random.seed(805101)
    output.mkdir(parents=True,exist_ok=False);(output/'training').mkdir();(output/'evaluation').mkdir()
    io.exclusive_json(output/'protocol.json',dict(source_sha256=hashes,author_sources=metadata,
        runtime=identity,training_cases=training_cases(),evaluation_cases=evaluation_cases(),
        initialization='Original random Model; no prior checkpoint',training_seed=805101,
        train_steps=256,warmup_steps=32,batch_size=32,train_interval=8,num_grad_steps=4,
        expected_gradient_updates=108,ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,
        actor_lr=.0003,critic_lr=.0003,gamma=.99,tau=.005,policy_delay=2,
        num_iters=4,branch_factor=4,path_len=2,all_finite_horizon_ends_absorbing=True,
        training_action_sampler_seed=805102,privileged_forecasts=False,
        comparative_efficacy_claimed=False,new_risk_certificate=False))
    train=ObservableSumoExecution(training_cases(),output/'training',805102)
    test=ObservableSumoExecution(evaluation_cases(),output/'evaluation',805103)
    try:
        trainer=module.MCTSTrainer(train,test,torch.device('cpu'),replay_buffer_size=1024,
            num_iters=4,branch_factor=4,path_len=2,ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,
            train_interval=8,num_grad_steps=4,batch_size=32,warmup_steps=32,
            eval_freq=10**9,model_save_freq=10**9,model_save_dir=None)
        install_bridges(trainer,module.MCTSPlanner)
        initial=model_state(trainer.model);torch.save(initial,output/'initial_state.pt')
        trainer.run(256);train.finish_record()
        assert len(train.calls)==256 and trainer.num_updates==108
        assert trainer.buf.size==256+sum(c['intervention']!=0 for c in train.calls)
        final=model_state(trainer.model);torch.save(final,output/'trained_state.pt')
        torch.save(dict(actor=trainer.actor_opt.state_dict(),critic1=trainer.critic1_opt.state_dict(),
            critic2=trainer.critic2_opt.state_dict()),output/'optimizers.pt')
        np.savez_compressed(output/'replay.npz',**{name:getattr(trainer.buf,name)[:trainer.buf.size]
            for name in ('obs_buf','act_buf','next_obs_buf','rew_buf','done_buf')})
        io.exclusive_json(output/'training_episodes.json',dict(episodes=train.episodes))
        trainer.model.eval()
        for case in evaluation_cases():
            observation=trainer.test_planner.reset()
            for _ in range(50):
                with torch.no_grad():
                    proposal=trainer.model.actor(torch.as_tensor(observation,dtype=torch.float32)).numpy()
                data,_=trainer.test_planner.take_step(proposal)
                observation,_,done,_=data[0]
                if done:break
            test.finish_record()
            print(json.dumps(dict(evaluation_conditions=test.cursor,expected_conditions=10)),flush=True)
        io.exclusive_json(output/'evaluation_episodes.json',dict(episodes=test.episodes))
    except Exception as error:
        for env in (train,test):
            if env.raw is not None and not env.recorded:
                env.failure=dict(type=type(error).__name__,message=str(error));env.raw.abort_reason=str(error)
                env.finish_record()
        io.exclusive_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
            training_actual_decisions=len(train.calls),evaluation_actual_decisions=len(test.calls)))
        raise
    finally:
        train.close();test.close()
    for name,expected in hashes.items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Bound source changed: '+name)
    assert author.verify(source)==metadata and io.runtime_identity()==identity
    io.exclusive_json(output/'complete.json',dict(training_actual_decisions=len(train.calls),
        evaluation_actual_decisions=len(test.calls),gradient_updates=trainer.num_updates,replay_rows=trainer.buf.size,
        training_interventions=sum(c['intervention']!=0 for c in train.calls),
        evaluation_interventions=sum(c['intervention']!=0 for c in test.calls),
        full_evaluation_conditions=10,privileged_forecasts=False,comparative_efficacy_claimed=False,
        new_risk_certificate=False,files_sha256={p.relative_to(output).as_posix():io.digest(p)
            for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(dict(training_actual_decisions=len(train.calls),gradient_updates=trainer.num_updates,
        evaluation_actual_decisions=len(test.calls))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
