"""Scientific grid and equivalence checks must reject changes in outcomes."""
import copy
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import evaluate_scalar_nominal_sumo as study


def test_fresh_grid_covers_every_scene_plant_mode_without_reusing_initials():
    cases=study.schedule()
    assert len(cases)==20 and len({c[0] for c in cases})==20
    assert Counter(c[1] for c in cases)==dict.fromkeys(study.SCENES,4)
    assert Counter(c[2] for c in cases)==dict.fromkeys(study.PROFILES,5)
    assert Counter(c[3] for c in cases)==dict.fromkeys(study.MODES,5)
    assert len(cases)*len(study.METHODS)==140


def test_behavior_comparison_excludes_timing_but_preserves_outcomes_and_terminal_errors():
    r=dict(row=dict(seed=1,completed=True,team_cost=12.,method_id='original_4703',setup_ms=1.,controller_ms=[2.]),
        trace=[dict(command=[0.,1.,0.],gate_ms=2.,actual_learning_used=True)],calls=[dict(wall_ms=2.,error=None)])
    other=copy.deepcopy(r);other['row'].update(method_id='scalar_4703',setup_ms=8.,controller_ms=[9.]);other['trace'][0]['gate_ms']=9.
    assert study.behavior(r)==study.behavior(other)
    for where,key,value in (('row','seed',2),('row','team_cost',13.),('trace','command',[1.,1.,0.]),
                            ('trace','actual_learning_used',False),('calls','error',dict(message='failure'))):
        changed=copy.deepcopy(other)
        (changed[where] if where=='row' else changed[where][0])[key]=value
        assert study.behavior(r)!=study.behavior(changed)
