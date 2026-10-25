"""Fixed-budget actor reparameterization, relative to the failed critic repair.

Only online/target actor output maps change. This task-adapted development
experiment is neither original-method superiority nor a MARL training claim.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from coopgate.dmps_effective_action import EffectiveCriticInputs
from coopgate.dmps_feasible_actor import FeasibleActorOutputs
import train_dmps_effective_action as previous

ROOT=previous.ROOT
STEPS=previous.STEPS
PARENT=previous.OUT
OUT=PARENT.parent/'dmps_feasible_actor_training'
PROBE=PARENT.parent/'dmps_feasible_actor_native_probe'
SOURCES=previous.SOURCES+('src/coopgate/dmps_feasible_actor.py','tools/train_dmps_feasible_actor.py',
    'tools/verify_dmps_feasible_actor.py','tests/test_dmps_feasible_actor.py',
    'tests/test_verify_dmps_feasible_actor.py','docs/dmps_feasible_actor_protocol.md',
    'src/coopgate/dmps_evaluation_replay.py','tests/test_dmps_evaluation_replay.py',
    'tools/probe_dmps_feasible_actor.py')
RECIPE='Online and target actor outputs parameterize the current feasible command interval'


def evaluation_cases():
    return [dict(index=2*i+s,seed=199904001+s,plant=plant,disturbance=mode)
        for i,(plant,mode) in enumerate([('nominal','clean')]+[(p,'uniform') for p in
            ('nominal','gain_drag','lag_mild','lag_strong')]) for s in (0,1)]


def parent_evidence():
    io=previous.nominal.base.io
    for name in ('complete.json','verification.json','observed_execution_exit.json','initial_state.pt','trained_state.pt'):
        io.committed(PARENT/name)
    proof=io.read(PARENT/'verification.json');done=io.read(PARENT/'complete.json')
    assert proof['complete_sha256']==io.digest(PARENT/'complete.json') and proof['observed_exit']['exit_code']==0
    assert proof['all_physics_and_replay_arrays_reconstructed'] and proof['all_source_and_file_hashes_verified']
    assert proof['training_actual_decisions']==STEPS and proof['matched_untrained_initialization']
    for name in ('initial_state.pt','trained_state.pt'):assert io.digest(PARENT/name)==done['files_sha256'][name]
    return dict(parent_audit_sha256=io.digest(PARENT/'verification.json'),
        parent_complete_sha256=io.digest(PARENT/'complete.json'),
        initial_weights_sha256=io.digest(PARENT/'initial_state.pt'),control_weights_sha256=io.digest(PARENT/'trained_state.pt'))


def prerequisites():
    io=previous.nominal.base.io;parents=parent_evidence()
    for name in ('complete.json','observed_execution_exit.json'):io.committed(PROBE/name)
    probe=io.read(PROBE/'complete.json');receipt=io.read(PROBE/'observed_execution_exit.json')
    assert receipt['exit_code']==0 and receipt['complete_sha256']==io.digest(PROBE/'complete.json')
    assert probe['actual_training_decisions']==128 and probe['all_actual_physics_and_replay_verified']
    assert probe['actor_calls']==dict(actor=118,actor_targ=44)
    for name,expected in probe['files_sha256'].items():assert io.digest(PROBE/name)==expected
    return dict(**parents,native_probe_complete_sha256=io.digest(PROBE/'complete.json'))


def evaluate_role(env,planner,actor,output):
    results=[]
    for case in evaluation_cases():
        np.random.seed(805801+case['index'])
        results.extend(previous.nominal.evaluate_all(env,planner,actor,[case],output))
    return results


def run(source,output=OUT):
    nominal=previous.nominal;base=nominal.base;io=base.io;parents=prerequisites()
    for name in SOURCES:io.committed(ROOT/name)
    _,binding=base.frozen.bound_inputs()
    hashes={**binding['source_sha256'],**{name:io.digest(ROOT/name) for name in SOURCES}}
    module,metadata=base.author.load(source);runtime=io.runtime_identity()
    torch.set_num_threads(1);torch.manual_seed(805200);np.random.seed(805200)
    output.mkdir(parents=True,exist_ok=False)
    for name in ('training','evaluation_new','evaluation_old','baseline'):(output/name).mkdir()
    io.exclusive_json(output/'native_start.json',dict(native_pid=os.getpid(),parent_pid=os.getppid(),utc=io.utc()))
    io.exclusive_json(output/'protocol.json',dict(source_sha256=hashes,author_sources=metadata,runtime=runtime,
        parents=parents,train_steps=STEPS,training_cases=nominal.training_cases(),evaluation_cases=evaluation_cases(),
        training_seed=805200,training_sampler_seed=805201,exploration_seed=805202,
        exploration_standard_deviation=.15,warmup_steps=2048,batch_size=128,train_interval=16,num_grad_steps=16,
        expected_gradient_updates=30704,expected_actor_updates=15352,
        expected_training_actor_calls=46072,expected_training_target_actor_calls=30704,
        all_finite_horizon_ends_absorbing=True,ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,gamma=.99,
        sole_additional_recipe_change=RECIPE,retained_four_effective_command_critic_inputs=True,
        task_adapted_variant_not_original_method=True,matched_initial_untrained_weights=True,
        matched_initial_policy_functions=False,curriculum_used=False,reward_shaping_used=False,demonstrations_used=False,
        checkpoint_selection='Final fixed-budget checkpoint only',
        policy_evaluation_search_seed='805801 + fixed condition index, reset for each policy',
        evaluation_roles=['feasible_actor_policy','previous_effective_critic_policy','recovery_baseline'],
        control_has_critic_hooks=True,control_has_feasible_actor_hooks=False,
        primary_development_target='All ten ordinary-merge tasks complete; failures prohibit all-scope gain',
        independent_test_initials=2,privileged_forecasts=False,new_risk_certificate=False,
        comparative_family_superiority_claimed=False,marl_training_claimed=False))
    train=previous.ProgressExecution(nominal.training_cases(),output/'training',805201)
    new_env=base.ObservableSumoExecution(evaluation_cases(),output/'evaluation_new',805803)
    old_env=base.ObservableSumoExecution(evaluation_cases(),output/'evaluation_old',805803)
    try:
        trainer=module.MCTSTrainer(train,new_env,torch.device('cpu'),replay_buffer_size=2*STEPS,
            num_iters=4,branch_factor=4,path_len=2,ret_low=-32.,ret_high=0.,unsafe_penalty=-32.,
            train_interval=16,num_grad_steps=16,batch_size=128,warmup_steps=2048,
            act_noise=nominal.GaussianExploration(805202),eval_freq=10**9,model_save_freq=10**9,model_save_dir=None)
        base.install_bridges(trainer,module.MCTSPlanner)
        initial=base.model_state(trainer.model)
        reference=torch.load(PARENT/'initial_state.pt',map_location='cpu',weights_only=True)
        assert initial.keys()==reference.keys() and all(torch.equal(initial[k],reference[k]) for k in initial)
        ids=[id(p) for p in trainer.model.parameters()]
        critics=EffectiveCriticInputs(trainer.model);actors=FeasibleActorOutputs(trainer.model)
        assert ids==[id(p) for p in trainer.model.parameters()]
        assert all(torch.equal(initial[k],v) for k,v in trainer.model.state_dict().items())
        torch.save(initial,output/'initial_state.pt')
        trainer.run(STEPS);train.finish_record()
        assert len(train.calls)==STEPS and trainer.num_updates==30704
        assert trainer.buf.size==STEPS+sum(c['intervention']!=0 for c in train.calls)
        assert critics.calls['critic1_targ']==critics.calls['critic2_targ']==critics.calls['critic2']==30704
        assert critics.calls['critic1']>=30704+15352 and actors.calls==dict(actor=46072,actor_targ=30704)
        io.exclusive_json(output/'network_hooks_training.json',dict(critic_calls=dict(critics.calls),actor_calls=dict(actors.calls),
            optimizer_parameter_identity_unchanged=True,all_four_critics_hooked=True,both_actor_outputs_hooked=True))
        torch.save(base.model_state(trainer.model),output/'trained_state.pt')
        torch.save(dict(actor=trainer.actor_opt.state_dict(),critic1=trainer.critic1_opt.state_dict(),critic2=trainer.critic2_opt.state_dict()),output/'optimizers.pt')
        np.savez_compressed(output/'replay.npz',**{name:getattr(trainer.buf,name)[:trainer.buf.size]
            for name in ('obs_buf','act_buf','next_obs_buf','rew_buf','done_buf')})
        io.exclusive_json(output/'training_episodes.json',dict(episodes=train.episodes))
        io.exclusive_json(output/'training_complete.json',dict(actual_decisions=STEPS,gradient_updates=trainer.num_updates,
            checkpoint_sha256=io.digest(output/'trained_state.pt'),utc=io.utc()))
        trainer.model.eval();new_results=evaluate_role(new_env,trainer.test_planner,trainer.model.actor,output/'evaluation_new')
        io.exclusive_json(output/'network_hooks_evaluation.json',dict(critic_calls=dict(critics.calls),actor_calls=dict(actors.calls),
            attempted_new_policy_decisions=actors.calls['actor']-46072,both_actor_outputs_hooked=True))
        old_model=module.Model(10,3,torch.device('cpu'),ret_low=-32.,ret_high=0.)
        old_model.load_state_dict(torch.load(PARENT/'trained_state.pt',map_location='cpu',weights_only=True));old_model.eval()
        old_critics=EffectiveCriticInputs(old_model) # Restore the parent's REQUIRED critic semantics.
        old_planner=base.ModelCommitPlanner(old_env,module.MCTSPlanner,old_model.critic1)
        old_results=evaluate_role(old_env,old_planner,old_model.actor,output/'evaluation_old')
        io.exclusive_json(output/'control_network_schema.json',dict(critic_calls=dict(old_critics.calls),
            all_four_critics_hooked=True,feasible_actor_outputs_hooked=False,checkpoint_sha256=io.digest(PARENT/'trained_state.pt')))
        io.exclusive_json(output/'evaluation_new.json',dict(episodes=new_env.episodes,results=new_results))
        io.exclusive_json(output/'evaluation_old.json',dict(episodes=old_env.episodes,results=old_results))
        baselines=[]
        for case in evaluation_cases():
            folder=output/'baseline'/f"{case['index']:03d}";folder.mkdir()
            baselines.append(previous.baseline_case(case,folder))
        io.exclusive_json(output/'baseline_results.json',dict(results=baselines))
    except Exception as error:
        for env in (train,new_env,old_env):
            if env.raw is not None and not env.recorded:
                env.failure=dict(type=type(error).__name__,message=str(error));env.raw.abort_reason=str(error);env.finish_record()
        io.exclusive_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),training_actual_decisions=len(train.calls),
            new_evaluation_decisions=len(new_env.calls),old_evaluation_decisions=len(old_env.calls)))
        raise
    finally:train.close();new_env.close();old_env.close()
    for name,expected in hashes.items():
        if io.digest(ROOT/name)!=expected:raise ValueError('Bound source changed: '+name)
    assert base.author.verify(source)==metadata and io.runtime_identity()==runtime and prerequisites()==parents
    io.exclusive_json(output/'complete.json',dict(training_actual_decisions=STEPS,gradient_updates=trainer.num_updates,
        replay_rows=trainer.buf.size,new_policy_conditions=10,old_policy_conditions=10,baseline_conditions=10,
        task_adapted_variant_not_original_method=True,new_risk_certificate=False,marl_training_claimed=False,
        files_sha256={p.relative_to(output).as_posix():io.digest(p) for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(dict(training_actual_decisions=STEPS,all_evaluation_conditions=30)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();run(args.source,args.output)
