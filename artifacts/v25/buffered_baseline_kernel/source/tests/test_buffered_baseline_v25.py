import ast
import inspect

import numpy as np
import pytest

from coopgate import buffered_baseline_v25 as fast
from coopgate.deduplicated_planner_v19 import deduplicated_joint_rollout
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import team_rule_tail
from coopgate.marl_v12 import project_joint
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate


def inputs():
    return np.array([[-80.,12.],[-85.,13.],[-120.,11.]]),np.zeros(3),parameters(MergeConfig())


@pytest.mark.parametrize('reference',[project_joint,team_rule_tail,deduplicated_joint_rollout])
def test_downstream_candidate_order_objectives_and_operations_are_identical(reference):
    counterpart=getattr(fast,reference.py_func.__name__)
    first=ast.parse(inspect.getsource(reference.py_func)).body[0]
    second=ast.parse(inspect.getsource(counterpart.py_func)).body[0]
    assert ast.dump(first,include_attributes=False)==ast.dump(second,include_attributes=False)
    assert fast.robust_certificate.targetoptions['fastmath'] is False


@pytest.mark.parametrize('radius',[0.,.1,.5,1.])
def test_certificate_matches_random_and_boundary_inputs_without_mutation(radius):
    state,previous,c=inputs()
    rng=np.random.default_rng(251022)
    tests=[]
    for _ in range(600):
        x=np.column_stack((rng.uniform(-150,80,3),rng.uniform(0,20,3)))
        prev=rng.uniform(-5,2.5,3)
        command=rng.uniform(-5.1,2.6,3)
        tests.append((x,prev,command))
    for position in (-30.,0.,30.,60.):
        for v in (0.,1e-12,10.,20.):
            x=state.copy();x[:,0]+=(position-x[0,0]);x[:,1]=v
            for acceleration in (-5.,-2.5,0.,2.5):
                prev=np.full(3,acceleration)
                for extra in (-1e-9,0.,1e-9):
                    tests.append((x,prev,np.full(3,acceleration+extra)))
    for x,prev,command in tests:
        before=[v.copy() for v in (x,prev,command,c)]
        assert fast.robust_certificate(x,prev,command,c,radius)==robust_certificate(x,prev,command,c,radius)
        for value,old in zip((x,prev,command,c),before): np.testing.assert_array_equal(value,old)


@pytest.mark.parametrize('backup_steps',[0,1,3,20])
@pytest.mark.parametrize('substeps',[1,3,5])
def test_backup_stage_and_stopping_boundaries(backup_steps,substeps):
    x,prev,c=inputs();c[10]=backup_steps;c[1]=substeps*c[0]
    for velocity in (0.,.001,3.,19.):
        x[:,1]=velocity
        for a in (-5.,-2.5,0.,2.5):
            prev[:]=a
            command=np.full(3,a)
            assert fast.robust_certificate(x,prev,command,c,.5)==robust_certificate(x,prev,command,c,.5)


def test_joint_projection_preserves_vertex_ties_and_fallback():
    x,prev,c=inputs()
    rng=np.random.default_rng(2501)
    for k in range(160):
        state=x.copy();state[:,0]+=rng.uniform(-60,50,3);state[:,1]=rng.uniform(0,20,3)
        previous=np.full(3,2.5) if k%3==0 else rng.choice([-5.,-2.5,0.,2.5],3)
        proposal=previous.copy() if k%2==0 else rng.choice([-5.,-2.5,0.,2.5],3)
        first=project_joint(state,previous,proposal,c,.5)
        second=fast.project_joint(state,previous,proposal,c,.5)
        np.testing.assert_array_equal(first[0],second[0])
        assert first[1:]==second[1:]


def test_full_baseline_cost_commands_and_work_match():
    state,prev,c=inputs()
    for k in range(12):
        x=state.copy();x[:,0]+=k*3.;x[:,1]+=k/12.
        for step in (0,15,48):
            first=deduplicated_joint_rollout(x,prev,step,c)
            second=fast.deduplicated_joint_rollout(x,prev,step,c)
            np.testing.assert_array_equal(first[0],second[0])
            assert first[1:]==second[1:]
