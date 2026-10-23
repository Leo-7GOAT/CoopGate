from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_interventions_v31 import MODELS, case, scope
from coopgate.shifted_sumo_v19 import PLANTS, SCENARIOS


def rows():
    return [dict(global_initial_index=32*j+i,scenario=scene,initial_index=i,
        seed=31000+32*j+i,plant_name=plant,controller=model,aborted=(plant=='lag_strong'))
        for j,scene in enumerate(SCENARIOS) for i in range(32) for plant in PLANTS for model in MODELS]


def test_scope_keeps_all_scenes_plants_models_and_all_existing_failures():
    values = rows()
    result = scope(values)
    assert len(result) == 1920 and sum(r['aborted'] for r in result) == 480
    assert [case(r) for r in result] == sorted(case(r) for r in values)


@pytest.mark.parametrize('change',['missing','duplicate','wrong_plant','wrong_index'])
def test_scope_cannot_replace_a_failure_or_silently_change_cell_coverage(change):
    values = rows()
    if change == 'missing': values.pop()
    elif change == 'duplicate': values[-1] = deepcopy(values[0])
    elif change == 'wrong_plant': values[-1]['plant_name'] = 'nominal'
    else: values[-1]['initial_index'] = 32
    with pytest.raises(ValueError,match='All 1920'):
        scope(values)
