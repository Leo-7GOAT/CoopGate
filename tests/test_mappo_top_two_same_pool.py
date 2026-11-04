from pathlib import Path
import numpy as np
import pytest
import torch
from coopgate.mappo_deployment import load_deployment_actor
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate, compositions
from experiments.top_two_candidates.gate import TopTwoHistoryGate
import mappo_top_two_formal_runtime as runtime


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    return load_deployment_actor(root / 'artifacts/mappo_ctde_20261101/development/joint_4703/policy.pt')[0]


@pytest.mark.parametrize('profile', ('nominal', 'gain_drag', 'lag_mild', 'lag_strong'))
def test_target_and_nominal_controls_receive_identical_candidates_at_identical_states(model, profile):
    target = TopTwoHistoryGate(model, profile, runtime.COUPLED_BANK)
    nominal = runtime.ScalarNominalOnlyGate(model, profile)
    x = np.array([[-140., 12.], [-220., 10.], [-190., 12.]])
    previous = np.zeros(3)
    _, a = target.select(x, previous)
    _, b = nominal.select(x, previous)
    assert a['top_two_candidates'] == b['top_two_candidates']
    assert a['candidate_count'] == b['candidate_count']
    assert b['nominal_control_uses_same_ranked_candidate_provider']
    assert target.step == target.baseline.step == nominal.step == nominal.baseline.step == 1
    assert nominal._ranked_select.__code__ is RecoveryJointGate.select.__code__
    assert RecoveryJointGate.select.__globals__['compositions'] is compositions
    assert nominal._ranked_select.__globals__['compositions'].__self__ is nominal
    assert not b['deployed_critic_used'] and not b['logits_are_feasibility_probabilities']


def test_true_thinning_branch_receives_same_pool_control(model):
    controller = runtime.make_controller('thinned_4703', model,
        dict(plant='lag_strong', controller_coins={'4703': True}))
    x = np.array([[-140., 12.], [-220., 10.], [-190., 12.]])
    _, info = controller.select(x, np.zeros(3))
    assert info['nominal_control_uses_same_ranked_candidate_provider']
    assert info['candidate_proposal_source'] == 'local_masked_top_two'
