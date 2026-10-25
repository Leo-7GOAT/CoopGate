"""Fixed-budget execution-semantic critic repair, with fresh matched controls.

This task-adapted DMPS variant changes only the four critic input maps relative
to the nominal training recipe. It is not an original-method reproduction win.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from coopgate.dmps_effective_action import EffectiveCriticInputs
import train_dmps_curriculum as curriculum
import train_dmps_observable_long_run as nominal

ROOT=nominal.ROOT
STEPS=nominal.STEPS
PARENT=ROOT/'artifacts/literature_reproduction/dmps_curriculum_training'
OLD=curriculum.OLD
OUT=ROOT/'artifacts/literature_reproduction/dmps_effective_action_training'
DIAG=PARENT.parent/'dmps_curriculum_value_diagnosis.json'
PROBE=PARENT.parent/'dmps_effective_action_native_probe'
SOURCES=curriculum.SOURCES+('src/coopgate/dmps_effective_action.py',
    'tools/train_dmps_effective_action.py','tools/verify_dmps_effective_action.py',
    'tests/test_dmps_effective_action.py','tests/test_verify_dmps_effective_action.py',
    'tools/probe_dmps_effective_action_training.py','docs/dmps_effective_action_protocol.md',
    'tools/train_dmps_curriculum.py','tools/pair_dmps_recovery_baseline.py',
    'tools/verify_dmps_curriculum.py','tools/diagnose_dmps_curriculum_value.py')


def evaluation_cases():
    return [dict(index=2*i+s,seed=179902001+s,plant=plant,disturbance=mode)
        for i,(plant,mode) in enumerate([('nominal','clean')]+[(p,'uniform') for p in
            ('nominal','gain_drag','lag_mild','lag_strong')]) for s in (0,1)]


def prerequisites():
    io=nominal.base.io
    for folder in (OLD,PARENT):
        for name in ('complete.json','verification.json','observed_execution_exit.json'):io.committed(folder/name)
        proof=io.read(folder/'verification.json')
        assert proof['complete_sha256']==io.digest(folder/'complete.json')
        assert proof['all_physics_and_replay_arrays_reconstructed'] and proof['observed_exit']['exit_code']==0
    io.committed(DIAG);diagnosis=io.read(DIAG)
    assert diagnosis['parent_complete_sha256']==io.digest(PARENT/'complete.json')
    assert diagnosis['parent_audit_sha256']==io.digest(PARENT/'verification.json')
    assert diagnosis['posthoc_descriptive_diagnosis'] and not diagnosis['causal_training_failure_proven']
    for name in ('complete.json','observed_execution_exit.json'):io.committed(PROBE/name)
    probe=io.read(PROBE/'complete.json');exit_proof=io.read(PROBE/'observed_execution_exit.json')
    assert exit_proof['exit_code']==0 and exit_proof['complete_sha256']==io.digest(PROBE/'complete.json')
    assert probe['actual_training_decisions']==128 and probe['all_actual_physics_and_replay_verified']
    for name,expected in probe['files_sha256'].items():assert io.digest(PROBE/name)==expected
    return dict(nominal_audit_sha256=io.digest(OLD/'verification.json'),
        curriculum_audit_sha256=io.digest(PARENT/'verification.json'),
        diagnosis_sha256=io.digest(DIAG),initial_weights_sha256=io.digest(OLD/'initial_state.pt'),
        raw_control_weights_sha256=io.digest(OLD/'trained_state.pt'),
        native_probe_complete_sha256=io.digest(PROBE/'complete.json'))


class ProgressExecution(nominal.base.ObservableSumoExecution):
    def execute(self,action,context):
        result=super().execute(action,context)
        if self.raw.collision or self.raw.violations:
            raise RuntimeError('Training physical failure; preserve prefix without replacement')
        if len(self.calls)%128==0:
            progress=dict(stage='training',actual_decisions=len(self.calls),expected_decisions=STEPS,
                episode_resets=self.cursor,native_pid=os.getpid(),utc=nominal.base.io.utc())
            nominal.base.io.atomic_bytes(self.output.parent/'training_progress.json',json.dumps(progress).encode())
            print(json.dumps(progress),flush=True)
        return result


def evaluate_role(env,planner,actor,output):
    results=[]
    for case in evaluation_cases():
        np.random.seed(805501+case['index'])
        results.extend(nominal.evaluate_all(env,planner,actor,[case],output))
    return results


def baseline_case(case,folder):
    pairing=curriculum.pairing;io=nominal.base.io;failure=record=review=None
    body=dict(case,cluster_id=f"effective_quality_{case['index']:03d}",scenario='nominal')
    with pairing.BaselineEnv(plant=case['plant']) as env:
        try:record=pairing.baseline.execute_episode(body,'baseline',None,env)
        except (RuntimeError,ValueError) as error:
            failure=dict(type=type(error).__name__,message=str(error))
            io.exclusive_json(folder/'failure.json',dict(case=case,failure=failure))
        if record is not None:
            io.exclusive_json(folder/'episode.json',record)
            body.update(initial_sha256=pairing.baseline.design.shared.array_sha256(np.asarray(record['row']['initial']),decimals=8),
                dynamics_sha256=pairing.baseline.design.shared.canonical_sha256(pairing.matched_plant_parameters(case['seed'],case['plant'])),
                disturbance_sha256=pairing.baseline.design.shared.array_sha256(pairing.disturbance_tape(case['seed'],case['disturbance'],.1,250)))
            review=pairing.baseline.audit_record(body,record,None)
            io.exclusive_json(folder/'review.json',review)
    return dict(case=case,failure=failure,row=record['row'] if record else None,review=review)


def run(source,output=OUT):
    base=nominal.base;io=base.io;parents=prerequisites()
    for name in SOURCES:io.committed(ROOT/name)
    _,binding=base.frozen.bound_inputs()
    hashes={**binding['source_sha256'],**{name:io.digest(ROOT/name) for name in SOURCES}}
    module,metadata=base.author.load(source);identity=io.runtime_identity()
    torch.set_num_threads(1);torch.manual_seed(805200);np.random.seed(805200)
    output.mkdir(parents=True,exist_ok=False)
    for name in ('training','evaluation_new','evaluation_old','baseline'):(output/name).mkdir()
    io.exclusive_json(output/'native_start.json',dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=io.utc()))
    io.exclusive_json(output/'protocol.json',dict(source_sha256=hashes,author_sources=metadata,runtime=identity,
        parents=parents,train_steps=STEPS,training_cases=nominal.training_cases(),evaluation_cases=evaluation_cases(),
        training_seed=805200,training_sampler_seed=805201,exploration_seed=805202,
        exploration_standard_deviation=.15,warmup_steps=2048,batch_size=128,train_interval=16,num_grad_steps=16,
        expected_gradient_updates=30704,expected_actor_updates=15352,all_finite_horizon_ends_absorbing=True,
        ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,gamma=.99,
        sole_recipe_change='Observable effective-command inputs for all four critics',
        task_adapted_variant_not_original_method=True,matched_initial_untrained_weights=True,
        curriculum_used=False,reward_shaping_used=False,demonstrations_used=False,
        checkpoint_selection='Final fixed-budget checkpoint only',
        policy_evaluation_search_seed='805501 + fixed condition index, reset for each policy',
        evaluation_roles=['effective_input_policy','previous_nominal_policy','recovery_baseline'],
        primary_development_target='All ten ordinary-merge tasks complete; failures prohibit all-scope gain',
        independent_test_initials=2,privileged_forecasts=False,new_risk_certificate=False,
        comparative_family_superiority_claimed=False))
    train=ProgressExecution(nominal.training_cases(),output/'training',805201)
    new_env=base.ObservableSumoExecution(evaluation_cases(),output/'evaluation_new',805503)
    old_env=base.ObservableSumoExecution(evaluation_cases(),output/'evaluation_old',805503)
    try:
        trainer=module.MCTSTrainer(train,new_env,torch.device('cpu'),replay_buffer_size=2*STEPS,
            num_iters=4,branch_factor=4,path_len=2,ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,
            train_interval=16,num_grad_steps=16,batch_size=128,warmup_steps=2048,
            act_noise=nominal.GaussianExploration(805202),eval_freq=10**9,model_save_freq=10**9,model_save_dir=None)
        base.install_bridges(trainer,module.MCTSPlanner)
        initial=base.model_state(trainer.model)
        reference=torch.load(OLD/'initial_state.pt',map_location='cpu',weights_only=True)
        assert initial.keys()==reference.keys() and all(torch.equal(initial[k],reference[k]) for k in initial)
        parameter_ids=[id(p) for p in trainer.model.parameters()]
        hooks=EffectiveCriticInputs(trainer.model)
        assert parameter_ids==[id(p) for p in trainer.model.parameters()]
        assert all(torch.equal(initial[k],v) for k,v in trainer.model.state_dict().items())
        torch.save(initial,output/'initial_state.pt')
        trainer.run(STEPS);train.finish_record()
        assert len(train.calls)==STEPS and trainer.num_updates==30704
        assert trainer.buf.size==STEPS+sum(c['intervention']!=0 for c in train.calls)
        assert hooks.calls['critic1_targ']==hooks.calls['critic2_targ']==30704
        assert hooks.calls['critic2']==30704 and hooks.calls['critic1']>=30704+15352
        io.exclusive_json(output/'effective_inputs_training.json',dict(critic_calls=dict(hooks.calls),
            optimizer_parameter_identity_unchanged=True,all_four_critics_hooked=True))
        torch.save(base.model_state(trainer.model),output/'trained_state.pt')
        torch.save(dict(actor=trainer.actor_opt.state_dict(),critic1=trainer.critic1_opt.state_dict(),
            critic2=trainer.critic2_opt.state_dict()),output/'optimizers.pt')
        np.savez_compressed(output/'replay.npz',**{name:getattr(trainer.buf,name)[:trainer.buf.size]
            for name in ('obs_buf','act_buf','next_obs_buf','rew_buf','done_buf')})
        io.exclusive_json(output/'training_episodes.json',dict(episodes=train.episodes))
        io.exclusive_json(output/'training_complete.json',dict(actual_decisions=STEPS,
            gradient_updates=trainer.num_updates,checkpoint_sha256=io.digest(output/'trained_state.pt'),utc=io.utc()))
        trainer.model.eval()
        new_results=evaluate_role(new_env,trainer.test_planner,trainer.model.actor,output/'evaluation_new')
        io.exclusive_json(output/'effective_inputs_evaluation.json',dict(critic_calls=dict(hooks.calls),
            all_four_critics_hooked=True))
        old_model=module.Model(10,3,torch.device('cpu'),ret_low=-32.,ret_high=0.)
        old_model.load_state_dict(torch.load(OLD/'trained_state.pt',map_location='cpu',weights_only=True));old_model.eval()
        old_planner=base.ModelCommitPlanner(old_env,module.MCTSPlanner,old_model.critic1)
        old_results=evaluate_role(old_env,old_planner,old_model.actor,output/'evaluation_old')
        io.exclusive_json(output/'evaluation_new.json',dict(episodes=new_env.episodes,results=new_results))
        io.exclusive_json(output/'evaluation_old.json',dict(episodes=old_env.episodes,results=old_results))
        baselines=[]
        for case in evaluation_cases():
            folder=output/'baseline'/f"{case['index']:03d}";folder.mkdir()
            baselines.append(baseline_case(case,folder))
        io.exclusive_json(output/'baseline_results.json',dict(results=baselines))
    except Exception as error:
        for env in (train,new_env,old_env):
            if env.raw is not None and not env.recorded:
                env.failure=dict(type=type(error).__name__,message=str(error));env.raw.abort_reason=str(error)
                env.finish_record()
        io.exclusive_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
            training_actual_decisions=len(train.calls),new_evaluation_decisions=len(new_env.calls),old_evaluation_decisions=len(old_env.calls)))
        raise
    finally:
        train.close();new_env.close();old_env.close()
    for name,expected in hashes.items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Bound source changed: '+name)
    assert base.author.verify(source)==metadata and io.runtime_identity()==identity and prerequisites()==parents
    io.exclusive_json(output/'complete.json',dict(training_actual_decisions=STEPS,gradient_updates=trainer.num_updates,
        replay_rows=trainer.buf.size,new_policy_conditions=10,old_policy_conditions=10,baseline_conditions=10,
        task_adapted_variant_not_original_method=True,privileged_forecasts=False,new_risk_certificate=False,
        comparative_family_superiority_claimed=False,
        files_sha256={p.relative_to(output).as_posix():io.digest(p) for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(dict(training_actual_decisions=STEPS,all_evaluation_conditions=30)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();run(args.source,args.output)
