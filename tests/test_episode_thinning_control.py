import numpy as np
import pytest
from coopgate.episode_thinning_control import matching_probability,independent_coins,EpisodeThinnedNominal


def test_matching_ratio_is_identified_only_on_valid_common_calibration_counts():
    assert matching_probability(29,55,80)==29/55
    for args in ((2,1,10),(0,0,10),(1,3,2),(True,3,10)):
        with pytest.raises(ValueError):matching_probability(*args)


def test_controller_coins_have_a_separate_reproducible_stage_and_model_stream():
    a=independent_coins(.5,64,stream=271828,model_index=0,stage=1)
    np.testing.assert_array_equal(a,independent_coins(.5,64,stream=271828,model_index=0,stage=1))
    assert not np.array_equal(a,independent_coins(.5,64,stream=271828,model_index=1,stage=1))
    assert not np.array_equal(a,independent_coins(.5,64,stream=271828,model_index=0,stage=2))


def test_coin_branch_does_not_manufacture_learning_adoption_or_compute_both_rules():
    calls=[]
    class Inner:
        def select(self,*args,**kwargs):return np.zeros(3),dict(rule=calls[-1],actual_learning_used=False)
    def factory(model,profile,rule):calls.append(rule);return Inner()
    for coin,rule in ((True,'nominal_performance'),(False,'baseline')):
        gate=EpisodeThinnedNominal(object(),'nominal',coin,factory=factory)
        _,info=gate.select(None,None)
        assert calls[-1]==info['rule']==rule and not info['actual_learning_used']
        assert info['episode_coin_admit_nominal']==coin and not info['coin_is_actual_adoption']
    assert calls==['nominal_performance','baseline']
