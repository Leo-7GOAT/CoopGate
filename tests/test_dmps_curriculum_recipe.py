from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import train_dmps_curriculum as recipe
import train_dmps_observable_long_run as previous


def test_training_seed_stream_and_budget_preserved_but_new_test_initials_disjoint():
    assert recipe.STEPS==32768 and recipe.training_cases()==previous.training_cases()
    tests={c['seed'] for c in recipe.evaluation_cases()}
    assert tests.isdisjoint(c['seed'] for c in recipe.training_cases())
    assert tests.isdisjoint(c['seed'] for c in previous.evaluation_cases())
    assert len(recipe.evaluation_cases())==10


@pytest.mark.parametrize('collision,violations',[(True,0),(False,1)])
def test_training_physical_failure_stops_before_automatic_next_episode(monkeypatch,collision,violations):
    monkeypatch.setattr(recipe.curriculum.CurriculumExecution,'execute',lambda *args:(None,0.,True,{}))
    env=object.__new__(recipe.CurriculumProgress)
    env.raw=SimpleNamespace(collision=collision,violations=violations)
    with pytest.raises(RuntimeError,match='preserve prefix without replacement'):
        env.execute(None,None)
