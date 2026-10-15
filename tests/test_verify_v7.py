import numpy as np
from coopgate.merge_v4 import MergeConfig
from coopgate.planning_v5 import plan_episode
import coopgate.verify_v7 as verified


def test_two_profile_verification_matches_full_teacher_and_never_worsens(monkeypatch):
    c = MergeConfig(); state = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])
    full = plan_episode(state, np.zeros(3), c)
    for proposal in (0, int(np.argmin(full.costs)), int(np.argmax(full.costs))):
        monkeypatch.setattr(verified, 'predict', lambda models, states: (np.array([proposal]), np.array([-99.])))
        selected, pair, info = verified.verify_proposal([], state, np.zeros(3), c)
        np.testing.assert_allclose(pair.costs, full.costs[[0, proposal]], atol=1e-8)
        assert full.costs[selected] <= full.costs[0]
        assert info['proposal'] == proposal
        if full.costs[proposal] >= full.costs[0]:
            assert selected == 0
