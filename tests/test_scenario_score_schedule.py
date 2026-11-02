from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_scenario_confidence import schedule,RULES


def test_shards_cover_all_conditions_and_fixed_models_without_inflating_independence():
    whole=schedule(1,96900001,'integration')
    shards=[schedule(1,96900001,'integration',i,4) for i in range(4)]
    flat=[case for part in shards for case in part]
    assert len(whole)==len(flat)==len(set(flat))==80
    assert set(whole)==set(flat)
    assert all(len(part)==20 for part in shards)
    assert len({(r[0],r[1]) for r in flat})==5
    assert len(RULES)==10 and len(set(RULES))==10
