from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_dmps_nominal_model_development import SelectedCommand,schedule


def test_fixed_schedule_covers_clean_and_all_mismatch_pairs_without_substitution():
    cases=schedule()
    assert len(cases)==10 and [c['index'] for c in cases]==list(range(10))
    assert [c['disturbance'] for c in cases[:2]]==['clean']*2
    assert {tuple(c[k] for k in ('plant','disturbance','seed')) for c in cases[2:]}=={
        (p,'uniform',s) for p in ('nominal','gain_drag','lag_mild','lag_strong') for s in (139895001,139895002)}


def test_action_transport_does_not_substitute_a_coopgate_command_or_certificate():
    command=np.array([2.5,-5.,.125])
    gate=SelectedCommand(command,dict(intervention=1,prediction_contract='nominal only'),3.)
    command[:]=0
    actual,info=gate.select(None,None,None)
    np.testing.assert_array_equal(actual,[2.5,-5.,.125])
    assert not info['certified'] and not info['in_set'] and info['intervention']==1
    actual[:]=100
    np.testing.assert_array_equal(gate.select(None,None,None)[0],[2.5,-5.,.125])
