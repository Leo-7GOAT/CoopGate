import numpy as np
import pytest
import torch
from dataclasses import asdict

from coopgate.mappo_ctde import MAPPOCTDE, PPOConfig, PPOVector, gae, losses, load_mappo


def test_gae_bootstrap_and_reset_are_distinct_and_no_cross_episode_leakage():
    rewards=np.array([[1.], [2.], [100.]], np.float32)
    values=np.zeros((3, 1, 3), np.float32)
    future=np.full_like(values, 10.)
    terminal=np.array([[False], [False], [True]])
    boundary=np.array([[False], [True], [True]])
    adv, ret=gae(rewards, values, future, terminal, boundary, gamma=1., lam=1.)
    np.testing.assert_allclose(adv[:, 0, 0], [23., 12., 100.])
    np.testing.assert_array_equal(adv, ret)
    # The final continuing rollout step bootstraps; it is not a reset.
    adv, _=gae(rewards[:1], values[:1], future[:1], terminal[:1], boundary[:1])
    np.testing.assert_allclose(adv, 11.)


@pytest.mark.parametrize('kinematic', [False, True])
def test_decentralized_actor_and_centralized_critic_information_paths(kinematic):
    torch.manual_seed(71)
    model=MAPPOCTDE(PPOConfig(width=32, kinematic_features=kinematic))
    obs=torch.randn(4, 3, 18)
    state=torch.randn(4, 13, requires_grad=True)
    mask=torch.ones(4, 3, 13, dtype=torch.bool)
    before=model.distribution(obs, mask).logits.detach().clone()
    first=model.value(obs, state)
    assert first.shape == (4, 3)
    first.sum().backward()
    assert state.grad[:, :9].abs().sum() > 0
    assert not any(p.grad is not None for p in model.actor.parameters())
    changed=model.value(obs, state.detach()+1.)
    assert not torch.allclose(first, changed)
    torch.testing.assert_close(before, model.distribution(obs, mask).logits)
    # Changing a neighbor's separate observation cannot change this actor's
    # logits. Neighbor information inside its own local vector remains allowed.
    altered=obs.clone(); altered[:, 2] += 20.
    torch.testing.assert_close(before[:, 0], model.distribution(altered, mask).logits[:, 0])


def test_proposal_likelihood_and_masked_ppo_gradients():
    torch.manual_seed(72)
    cfg=PPOConfig(width=32)
    model=MAPPOCTDE(cfg)
    obs=torch.randn(12, 3, 18)
    state=torch.randn(12, 13)
    mask=torch.ones(12, 3, 13, dtype=torch.bool); mask[..., 0]=False
    with torch.no_grad():
        dist=model.distribution(obs, mask); actions=dist.sample()
        old_logp=dist.log_prob(actions); old_value=model.value(obs, state)
    assert (actions != 0).all()
    batch=dict(obs=obs, state=state, mask=mask, actions=actions,
               old_logp=old_logp, old_value=old_value,
               advantage=torch.randn(12, 3), returns=old_value+1.,
               active=torch.ones(12, 3, dtype=torch.bool))
    total, detail=losses(model, batch, cfg)
    assert detail['approximate_kl'] == 0
    assert detail['clip_fraction'] == 0
    total.backward()
    assert sum(float(p.grad.square().sum()) for p in model.actor.parameters()) > 0
    assert sum(float(p.grad.square().sum()) for p in model.critic.parameters()) > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match='Empty'):
        model.distribution(obs, torch.zeros_like(mask))


def test_vector_uses_new_reset_state_and_single_actions_are_explicitly_fixed():
    env=PPOVector(4, 73, 'single')
    _, _, mask=env.inputs()
    sampled=mask.argmax(-1)
    actions=env.execute(sampled, mask)
    highest=np.where(mask, np.arange(13), -1).max(-1)
    np.testing.assert_array_equal(actions[:, (0, 2)], highest[:, (0, 2)])
    np.testing.assert_array_equal(actions[:, 1], sampled[:, 1])
    env.steps[:] = 49
    following, reward, boundary, active, episodes=env.step(actions)
    assert boundary.all() and len(episodes) == 4
    assert not active[:, (0, 2)].any() and active[:, 1].all()
    assert np.isfinite(reward).all()
    np.testing.assert_allclose(following[1][:, -1], 1.)
    # Returned final observation must not be fed to the next sampled action.
    np.testing.assert_allclose(env.inputs()[1][:, -1], 0.)


def test_training_checkpoint_is_actual_on_policy_ppo(tmp_path):
    from tools.train_mappo_ctde import train
    cfg=PPOConfig(width=16, n_envs=4, rollout=16, epochs=2, minibatch=32)
    out=tmp_path/'smoke'
    result=train(out, 128, 74, 'joint', config=cfg)
    model, meta=load_mappo(out/'policy.pt')
    assert meta['algorithm'] == 'mappo_ctde_20261101'
    assert meta['steps'] == meta['target_steps'] == 128
    assert all(n > 0 for n in result['actor_samples'])
    assert result['actor_delta_squared_l2'] > 0
    saved=torch.load(out/'rollout_before_update_128.pt', weights_only=False)
    model.load_state_dict(saved['before_model'])
    b=saved['batch']
    with torch.no_grad():
        lp=model.distribution(b['obs'], b['mask']).log_prob(b['actions'])
    torch.testing.assert_close(lp, b['old_logp'])
    assert meta['no_teacher_actions'] and meta['no_policy_iteration']


def test_expert_initializer_is_disclosed_and_ppo_never_needs_teacher_dataset(tmp_path):
    from tools.train_mappo_ctde import train
    cfg=PPOConfig(width=16, n_envs=4, rollout=16, epochs=2, minibatch=32)
    initial=MAPPOCTDE(cfg)
    path=tmp_path/'bc.pt'
    torch.save(dict(algorithm='planner_bc_initializer_20261101', ppo_config=asdict(cfg),
                    model=dict(actor=initial.actor.state_dict())), path)
    # No teacher dataset exists: this path must collect its own on-policy data.
    result=train(tmp_path/'ppo', 128, 75, 'single', config=cfg, expert_initial=path)
    assert not result['no_teacher_actions']
    assert result['no_teacher_actions_in_ppo_updates']
    assert result['actor_delta_squared_l2'] > 0
    assert result['actor_samples'][0] == result['actor_samples'][2] == 0
    assert result['actor_samples'][1] > 0
    saved=torch.load(tmp_path/'ppo'/'initial.pt', weights_only=True)
    for key, value in initial.actor.state_dict().items():
        torch.testing.assert_close(value, saved['model']['actor.'+key])


def test_policy_iteration_checkpoint_cannot_be_called_mappo(tmp_path):
    path=tmp_path/'pi.pt'
    torch.save(dict(algorithm='coupled_policy_iteration_v16'), path)
    with pytest.raises(ValueError, match='Not a new on-policy PPO'):
        load_mappo(path)


def test_reset_curriculum_carries_command_memory_and_global_clock(tmp_path):
    from coopgate.mappo_reset_curriculum import ResetCurriculumVector
    path=tmp_path/'states.npz'
    state=np.array([[[-100., 10.], [-95., 9.], [-140., 10.]]])
    previous=np.full((1,3),1.25)
    np.savez(path, state=state, previous=previous, step=np.array([10]))
    env=ResetCurriculumVector(path, n=64, seed=76)
    assert 0 < env.curriculum_resets < 64
    reset=env.steps==10
    np.testing.assert_array_equal(env.x[reset], np.broadcast_to(state,(reset.sum(),3,2)))
    np.testing.assert_array_equal(env.previous[reset],np.broadcast_to(previous,(reset.sum(),3)))
    np.testing.assert_allclose(env.inputs()[1][reset, -1],.2)
    assert np.all(env.time==0.) and np.all(env.variation==0.)
    _, _, mask=env.inputs()
    actions=np.where(mask,np.arange(13),-1).max(-1)
    following, reward, boundary, active, finished=env.step(actions)
    assert np.isfinite(reward).all()
    assert not active[:,(0,2)].any()
    assert not boundary.any()


def test_accelerated_arrival_matches_physics_and_preserves_local_information():
    from coopgate.mappo_ctde import accelerating_arrival
    # Reach vmax in 4s after 60m; another 40m takes 2s.
    torch.testing.assert_close(accelerating_arrival(torch.tensor([60.,100.]),torch.tensor([10.,10.])),torch.tensor([4.,6.]))
    model=MAPPOCTDE(PPOConfig(width=32,kinematic_features=True,accelerated_features=True))
    obs=torch.zeros(4,3,18);obs[...,0]=-.6;obs[...,1]=.5
    obs[...,6]=obs[...,12]=1.
    mask=torch.ones(4,3,13,dtype=torch.bool)
    original=model.distribution(obs,mask).logits.detach().clone()
    state=torch.randn(4,13)
    value=model.value(obs,state)
    (value.mean()+model.distribution(obs,mask).entropy().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    altered=obs.clone();altered[:,2,1]=.8
    torch.testing.assert_close(original[:,0],model.distribution(altered,mask).logits[:,0])


def test_phase_actor_routes_only_on_own_clock_and_both_heads_receive_gradients():
    cfg=PPOConfig(width=16,phase_actor=True)
    model=MAPPOCTDE(cfg)
    obs=torch.zeros(2,3,18);obs[1,:,5]=.2
    mask=torch.ones(2,3,13,dtype=torch.bool)
    with torch.no_grad():
        model.actor.initial[-1].bias[0]=1.
        model.actor.flow[-1].bias[1]=1.
    logits=model.distribution(obs,mask).logits
    assert (logits[0].argmax(-1)==0).all()
    assert (logits[1].argmax(-1)==1).all()
    model.distribution(obs,mask).entropy().sum().backward()
    for head in (model.actor.initial,model.actor.flow):
        assert sum(float(p.grad.square().sum()) for p in head.parameters())>0


def test_startup_expert_costs_match_original_strong_planner_value_and_argmin(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    from tools.initialize_mappo_phase import initial_action_costs
    from tools.develop_v12 import initial
    from coopgate.ego_v13 import EgoRolloutGate
    from coopgate.fast_v8 import parameters
    from coopgate.merge_v4 import MergeConfig
    from coopgate.marl_v12 import LEVELS
    for i in range(16):
        x=initial(951099001+i,'nominal' if i%2==0 else 'dense')
        costs,work=initial_action_costs(x,parameters(MergeConfig()))
        _,info=EgoRolloutGate().select(x,np.zeros(3))
        index=np.argmin(np.abs(LEVELS-info['ego_proposal']))
        assert abs(costs[index]-info['rollout_value'])<1e-8
        assert abs(costs[index]-costs.min())<1e-8
        assert work>=9


def test_separate_actor_roles_have_local_inputs_and_independent_gradients():
    cfg=PPOConfig(width=16,role_actors=True,actor_relu=True,actor_logit_scale=5.)
    model=MAPPOCTDE(cfg)
    obs=torch.randn(8,3,18);mask=torch.ones(8,3,13,dtype=torch.bool)
    baseline=model.distribution(obs,mask).logits.detach()
    altered=obs.clone();altered[:,0]+=10.
    torch.testing.assert_close(baseline[:,1],model.distribution(altered,mask).logits[:,1])
    model.distribution(obs,mask).entropy()[:,1].sum().backward()
    norms=[sum(float(p.grad.square().sum()) for p in actor.parameters()) for actor in model.actor.roles]
    assert norms[1]>0 and norms[0]==norms[2]==0
    np.testing.assert_array_equal(model.proposal(obs[:,1:2],mask[:,1:2])[:,0],
                                  model.proposal(obs,mask)[:,1])


def test_small_ego_rate_does_not_freeze_any_joint_actor(tmp_path):
    from tools.train_mappo_ctde import train
    cfg=PPOConfig(width=16,n_envs=4,rollout=16,epochs=2,minibatch=32,
                  role_actors=True,actor_lr=1e-4,ego_actor_lr=1e-5)
    out=tmp_path/'actual_two_rate_joint'
    final=train(out,128,78,'joint',config=cfg)
    initial=torch.load(out/'initial.pt',weights_only=True)['model']
    for i in range(3):
        delta=sum(float((v-initial[k]).square().sum()) for k,v in final['model'].items()
                  if k.startswith(f'actor.roles.{i}.'))
        assert delta>0
    assert [g['lr'] for g in final['optimizer']['param_groups']]==[1e-4,1e-5,1e-4,cfg.critic_lr]
