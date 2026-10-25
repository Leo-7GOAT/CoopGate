import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from coopgate.dmps_effective_action import EffectiveCriticInputs
from coopgate.dmps_evaluation_replay import replay_episode
import load_original_dmps as author


@pytest.fixture(scope='module')
def fixture():
    root=Path(__file__).resolve().parents[1]
    source=root.parent/'reference_sources/dmps-f37df2d'
    parent=root/'artifacts/literature_reproduction/dmps_effective_action_training'
    if not source.exists() or not parent.exists():pytest.skip('Sealed author/native fixtures unavailable')
    module,_=author.load(source);torch.set_num_threads(1)
    model=module.Model(10,3,torch.device('cpu'),ret_low=-32.,ret_high=0.)
    model.load_state_dict(torch.load(parent/'trained_state.pt',map_location='cpu',weights_only=True));model.eval()
    EffectiveCriticInputs(model)
    return module,model,parent


@pytest.mark.parametrize('index',[0,8])
def test_loaded_parent_policy_and_original_mcts_reproduce_complete_native_episode(fixture,index):
    module,model,parent=fixture
    record=json.loads((parent/'evaluation_new'/f'episode_{index:03d}'/'episode.json').read_text())
    calls=[json.loads((parent/'evaluation_new'/f'call_{index*50+j:04d}.json').read_text()) for j in range(50)]
    before=np.random.get_state()
    result=replay_episode(record,calls,model,module.MCTSPlanner,805501+index,None)
    after=np.random.get_state()
    assert result['replayed_physical_commands']==50 and result['new_sumo_executions']==0
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])


def test_wrong_recorded_execution_is_rejected_before_fake_transition_is_returned(fixture):
    module,model,parent=fixture
    record=json.loads((parent/'evaluation_new/episode_000/episode.json').read_text())
    calls=[json.loads((parent/'evaluation_new'/f'call_{j:04d}.json').read_text()) for j in range(50)]
    bad=copy.deepcopy(calls);bad[0]['executed'][0]+=.01
    with pytest.raises(AssertionError):replay_episode(record,bad,model,module.MCTSPlanner,805501,None)
