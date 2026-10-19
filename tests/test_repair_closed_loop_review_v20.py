import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import review_repair_closed_loop_v20 as review


def test_incomplete_or_other_car_harm_cannot_pass_forecast_audit():
    values = np.zeros((33, 3, 2))
    values[:, :, 0] -= .1
    review.check_forecast_record(values, values.tolist())
    with pytest.raises(ValueError, match='Incomplete'):
        review.check_forecast_record(values[:-1], values[:-1].tolist())
    bad = values.copy()
    bad[32, 2, 1] = .5
    with pytest.raises(ValueError, match='performance condition'):
        review.check_forecast_record(bad, bad.tolist())
    bad = values.copy()
    bad[0, :, 0] = 0.
    with pytest.raises(ValueError, match='performance condition'):
        review.check_forecast_record(bad, bad.tolist())


def test_exact_selected_candidate_identity_and_no_baseline_adoption():
    baseline, command = np.zeros(3), np.array([.2, .4, .5])
    candidates = [(2, 1., command), (2, .5, command * .5)]
    item = dict(portfolio_joint_action_admitted=True, candidate_count=2, original_candidate_count=1,
                command=(command * .5).tolist(), composition=2, repair_weight=.5,
                forecast_errors=[], generation_errors=[])
    review.check_selected(item, baseline, candidates, 1)
    wrong = copy.deepcopy(item)
    wrong['repair_weight'] = .25
    with pytest.raises(ValueError, match='identity'):
        review.check_selected(wrong, baseline, candidates, 1)
    wrong = copy.deepcopy(item)
    wrong['command'] = baseline.tolist()
    with pytest.raises(AssertionError):
        review.check_selected(wrong, baseline, candidates, 1)


def test_errors_and_removed_candidates_are_not_silently_accepted():
    item = dict(portfolio_joint_action_admitted=False, candidate_count=0, original_candidate_count=1,
                command=[0., 0., 0.], composition=0, repair_weight=None, forecast_errors=[], generation_errors=[])
    with pytest.raises(ValueError, match='coverage'):
        review.check_selected(item, np.zeros(3), [(1, 1., np.ones(3))], 1)
    item['candidate_count'] = 1
    item['forecast_errors'] = [{'error': 'failed forecast'}]
    with pytest.raises(ValueError, match='investigation'):
        review.check_selected(item, np.zeros(3), [(1, 1., np.ones(3))], 1)


def test_disjointness_includes_the_running_temporal_study(tmp_path, monkeypatch):
    temporal = tmp_path / 'artifacts/v20/temporal_development'
    temporal.mkdir(parents=True)
    (temporal / 'protocol.json').write_text(json.dumps(dict(spec=dict(seed_base=100, n_initials=16))))
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    monkeypatch.setattr(review, 'STUDY', tmp_path / 'artifacts/v20/repair_development')
    with pytest.raises(ValueError, match='Overlapping'):
        review.verify_disjoint_ranges(dict(spec=dict(seed_base=110, n_initials=16)))
    assert len(review.verify_disjoint_ranges(dict(spec=dict(seed_base=116, n_initials=16)))) == 1
