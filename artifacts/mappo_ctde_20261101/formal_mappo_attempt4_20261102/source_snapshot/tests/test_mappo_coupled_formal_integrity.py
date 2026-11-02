from copy import deepcopy
from pathlib import Path
import sys
import json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mappo_coupled_formal_experiment as e
import mappo_coupled_formal_design as d
from mappo_coupled_formal_finalize import adoption_difference_interval


@pytest.mark.parametrize('kind',('missing','duplicate','foreign'))
def test_complete_schedule_is_checked_before_any_block_is_read(tmp_path,kind):
    plan=dict(validation_only=True,cases=dict(validation=d.validation_cases()))
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    p=tmp_path/'validation/shard0';p.mkdir(parents=True)
    rows=[] if kind=='missing' else [dict(block='block0000',keys=[],metadata={})]*(2 if kind=='duplicate' else 1)
    (p/'collection_complete.json').write_text(json.dumps(dict(blocks=rows)))
    with pytest.raises(ValueError,match='block|schedule'):
        list(e.reviewed_records(tmp_path,'validation',0,plan))


def test_matching_difference_handles_identical_zero_and_discordant_events():
    a=np.array([True]*25+[False]*75);same=adoption_difference_interval(a,a,.01)
    assert same['observed_difference']==0 and same['lower']<0<same['upper']
    b=np.zeros(100,bool);r=adoption_difference_interval(a,b,.01)
    assert r['target_only']==25 and r['control_only']==0 and r['observed_difference']==.25
    assert r['lower']<.25<r['upper'] and not r['exact_equality_or_equivalence_proved']
