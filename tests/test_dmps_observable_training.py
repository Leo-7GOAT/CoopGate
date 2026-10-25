from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_dmps_observable_development import evaluation_cases,install_bridges,training_cases
from test_dmps_execution_bridge import Execution,ForecastPlanner


def test_training_evaluation_initials_disjoint_and_schedule_fixed():
    train=training_cases();test=evaluation_cases()
    assert len(train)==257 and len({c['seed'] for c in train})==257
    assert {c['seed'] for c in train}.isdisjoint(c['seed'] for c in test)
    assert len(test)==10 and [c['index'] for c in test]==list(range(10))


def test_explicit_finite_horizon_end_is_absorbing_for_unchanged_author_run_condition():
    train=Execution();train.num_steps=50
    trainer=SimpleNamespace(env=train,test_env=Execution(),model=SimpleNamespace(critic1=None))
    install_bridges(trainer,ForecastPlanner)
    done=True;ep_len=50
    assert done and ep_len<trainer.max_ep_len
    assert trainer.max_ep_len==51
