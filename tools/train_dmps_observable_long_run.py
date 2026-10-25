"""Fixed longer from-scratch DMPS run; evaluate every declared failure case.

Development of a trained existing-method baseline. This does not replace the
frozen comparative studies or prove convergence or matched-training superiority.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import audit_dmps_observable_training as parent
import train_dmps_observable_development as base

ROOT=base.ROOT
STEPS=32768
SOURCES=base.SOURCES+('tools/audit_dmps_observable_training.py',
    'tools/train_dmps_observable_long_run.py','tests/test_dmps_observable_long_run.py',
    'docs/dmps_observable_long_run_protocol.md')


class GaussianExploration:
    def __init__(self,seed):self.rng=np.random.default_rng(seed)
    def gen(self):return self.rng.normal(0.,.15,3)


def training_cases():
    return [dict(index=i,seed=139898001+i,plant='nominal',disturbance='clean') for i in range(STEPS+1)]


def evaluation_cases():
    return [dict(index=2*i+s,seed=149899001+s,plant=plant,disturbance=mode)
        for i,(plant,mode) in enumerate([('nominal','clean')]+[(p,'uniform') for p in
            ('nominal','gain_drag','lag_mild','lag_strong')]) for s in (0,1)]


class ProgressExecution(base.ObservableSumoExecution):
    def execute(self,action,context):
        result=super().execute(action,context)
        if len(self.calls)%128==0:
            progress=dict(stage='training',actual_decisions=len(self.calls),expected_decisions=STEPS,
                episode_resets=self.cursor,native_pid=os.getpid(),utc=base.io.utc())
            base.io.atomic_bytes(self.output.parent/'training_progress.json',json.dumps(progress).encode())
            print(json.dumps(progress),flush=True)
        return result


def evaluate_all(env,planner,actor,cases,output):
    results=[]
    for case in cases:
        start=len(env.calls);failure=None;previous_episodes=len(env.episodes)
        try:
            observation=planner.reset()
            assert env.case==case
            for _ in range(50):
                with torch.no_grad():
                    proposal=actor(torch.as_tensor(observation,dtype=torch.float32)).numpy()
                data,_=planner.take_step(proposal)
                observation,_,done,_=data[0]
                if done:break
        except (RuntimeError,ValueError) as error:
            failure=dict(type=type(error).__name__,message=str(error),case=case)
            env.failure=failure
            if env.raw is not None and hasattr(env.raw,'abort_reason'):
                env.raw.abort_reason=str(error)
            if hasattr(planner,'model_env'):
                failure['nominal_state']=planner.model_env.x.tolist()
                failure['nominal_command']=planner.model_env.previous.tolist()
                failure['nominal_failed']=bool(planner.model_env.failed)
            base.io.exclusive_json(output/f"case_failure_{case['index']:03d}.json",failure)
        finally:
            # Audit failures are not swallowed; they invalidate the evidence.
            env.finish_record()
        reviewed=env.episodes[-1] if len(env.episodes)>previous_episodes else None
        results.append(dict(case=case,failure=failure,actual_decisions=len(env.calls)-start,episode=reviewed))
        print(json.dumps(dict(evaluated_conditions=len(results),expected_conditions=len(cases))),flush=True)
    return results


def prerequisites():
    audit_path=parent.OUT.parent/'dmps_observable_training_audit.json'
    base.io.committed(audit_path)
    audit=base.io.read(audit_path)
    if not audit['training_phase_complete'] or audit['overall_evaluation_complete']:
        raise ValueError('Expected completed training and preserved evaluation failure')
    for name,expected in audit['files_sha256'].items():
        if base.io.digest(parent.OUT/name)!=expected:raise ValueError('Preserved training evidence changed')
    return base.io.digest(audit_path)


def run(source,output):
    prior_hash=prerequisites()
    for name in SOURCES:base.io.committed(ROOT/name)
    _,binding=base.frozen.bound_inputs()
    hashes={**binding['source_sha256'],**{name:base.io.digest(ROOT/name) for name in SOURCES}}
    module,metadata=base.author.load(source);identity=base.io.runtime_identity()
    torch.set_num_threads(1);torch.manual_seed(805200);np.random.seed(805200)
    output.mkdir(parents=True,exist_ok=False);(output/'training').mkdir();(output/'evaluation').mkdir()
    base.io.exclusive_json(output/'native_start.json',dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=base.io.utc()))
    base.io.exclusive_json(output/'protocol.json',dict(source_sha256=hashes,author_sources=metadata,
        parent_training_audit_sha256=prior_hash,runtime=identity,train_steps=STEPS,
        training_cases=training_cases(),evaluation_cases=evaluation_cases(),
        initialization='Original random Model; no prior checkpoint',training_seed=805200,
        warmup_steps=2048,batch_size=128,train_interval=16,num_grad_steps=16,
        expected_gradient_updates=30704,expected_actor_updates=15352,
        ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,gamma=.99,
        exploration_standard_deviation=.15,training_sampler_seed=805201,exploration_seed=805202,
        all_finite_horizon_ends_absorbing=True,privileged_forecasts=False,
        comparative_efficacy_claimed=False,new_risk_certificate=False))
    train=ProgressExecution(training_cases(),output/'training',805201)
    test=base.ObservableSumoExecution(evaluation_cases(),output/'evaluation',805203)
    try:
        trainer=module.MCTSTrainer(train,test,torch.device('cpu'),replay_buffer_size=2*STEPS,
            num_iters=4,branch_factor=4,path_len=2,ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,
            train_interval=16,num_grad_steps=16,batch_size=128,warmup_steps=2048,
            act_noise=GaussianExploration(805202),eval_freq=10**9,model_save_freq=10**9,model_save_dir=None)
        base.install_bridges(trainer,module.MCTSPlanner)
        torch.save(base.model_state(trainer.model),output/'initial_state.pt')
        trainer.run(STEPS);train.finish_record()
        assert len(train.calls)==STEPS and trainer.num_updates==30704
        assert trainer.buf.size==STEPS+sum(c['intervention']!=0 for c in train.calls)
        torch.save(base.model_state(trainer.model),output/'trained_state.pt')
        torch.save(dict(actor=trainer.actor_opt.state_dict(),critic1=trainer.critic1_opt.state_dict(),
            critic2=trainer.critic2_opt.state_dict()),output/'optimizers.pt')
        np.savez_compressed(output/'replay.npz',**{name:getattr(trainer.buf,name)[:trainer.buf.size]
            for name in ('obs_buf','act_buf','next_obs_buf','rew_buf','done_buf')})
        base.io.exclusive_json(output/'training_episodes.json',dict(episodes=train.episodes))
        base.io.exclusive_json(output/'training_complete.json',dict(actual_decisions=STEPS,
            gradient_updates=trainer.num_updates,checkpoint_sha256=base.io.digest(output/'trained_state.pt'),
            replay_sha256=base.io.digest(output/'replay.npz'),privileged_forecasts=False,utc=base.io.utc()))
        trainer.model.eval()
        results=evaluate_all(test,trainer.test_planner,trainer.model.actor,evaluation_cases(),output)
        base.io.exclusive_json(output/'evaluation_episodes.json',dict(episodes=test.episodes,results=results))
    except Exception as error:
        for env in (train,test):
            if env.raw is not None and not env.recorded:
                env.failure=dict(type=type(error).__name__,message=str(error));env.raw.abort_reason=str(error)
                env.finish_record()
        base.io.exclusive_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
            training_actual_decisions=len(train.calls),evaluation_actual_decisions=len(test.calls)))
        raise
    finally:
        train.close();test.close()
    for name,expected in hashes.items():
        if base.io.digest(ROOT/name)!=expected:raise ValueError('Bound source changed: '+name)
    assert base.author.verify(source)==metadata and base.io.runtime_identity()==identity
    assert prerequisites()==prior_hash
    base.io.exclusive_json(output/'complete.json',dict(training_actual_decisions=STEPS,
        evaluation_actual_decisions=len(test.calls),evaluation_conditions=10,
        gradient_updates=trainer.num_updates,replay_rows=trainer.buf.size,
        evaluation_failures=sum(r['failure'] is not None for r in results),
        privileged_forecasts=False,comparative_efficacy_claimed=False,new_risk_certificate=False,
        files_sha256={p.relative_to(output).as_posix():base.io.digest(p) for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(dict(training_actual_decisions=STEPS,evaluation_conditions=10)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
