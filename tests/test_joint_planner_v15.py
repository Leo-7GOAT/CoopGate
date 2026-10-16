import numpy as np
from coopgate.fast_v8 import parameters,advance_one
from coopgate.merge_v4 import MergeConfig
from coopgate.robust_v9 import robust_certificate
from coopgate.joint_planner_v15 import joint_rollout,team_rule_tail


def test_nominal_closed_loop_planning_does_not_regress_from_its_rule_continuation():
    c=parameters(MergeConfig())
    for speeds in ((12.,13.,11.),(15.,16.,14.)):
        state=np.column_stack(([-80.,-85.,-120.],speeds));previous=np.zeros(3)
        baseline,_=team_rule_tail(state,previous,0,c);total=0.
        for step in range(50):
            if np.all(state[:,0]>=60.):break
            command,value,rule,count=joint_rollout(state,previous,step,c)
            assert value<=rule+1e-8 and count>=0
            assert robust_certificate(state,previous,command,c,.5)
            assert np.max(np.abs(command-previous))<=2.5+1e-8
            total+=np.sum((state[:,0]<60.)*(.5+.05*np.abs(command-previous)))/3.
            assert advance_one(state,command,c);previous=command
        assert total<=baseline+1e-8 and np.all(state[:,0]>=60.)


def test_arrived_team_has_zero_remaining_cost_despite_clearance_commands():
    c=parameters(MergeConfig());state=np.array([[200.,10.],[160.,10.],[120.,10.]])
    _,value,rule,count=joint_rollout(state,np.zeros(3),30,c)
    assert value==rule==0. and count==0
