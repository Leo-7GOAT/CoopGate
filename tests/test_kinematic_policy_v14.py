import numpy as np
import torch
from coopgate.kinematic_policy_v14 import arrival_feature,candidates,choose_compiled,KinematicPolicy
from coopgate.certificate_policy_v14 import candidates as old_candidates
from coopgate.ego_v13 import EgoVector


def test_analytic_arrival_feature_has_correct_stop_and_cap_cases():
    assert arrival_feature(-1.,10.,1.)==0.
    assert arrival_feature(100.,10.,0.)==10.
    assert arrival_feature(101.,10.,-.5)==25.
    # Accelerate from 10 to 20 for 5 s (75 m), then cruise another 25 m.
    np.testing.assert_allclose(arrival_feature(100.,10.,2.),6.25)
    np.testing.assert_allclose(arrival_feature(12.,10.,-2.),(10.-np.sqrt(52.))/2.)


def test_added_features_preserve_commands_and_compiled_actor():
    torch.set_num_threads(1);torch.manual_seed(233)
    model=KinematicPolicy();env=EgoVector(3,733)
    for _ in range(8):
        actions=[]
        for i in range(env.n):
            x,prev,step=env.x[i],env.previous[i],int(env.steps[i])
            fe,commands,mask=candidates(x,prev,step,env.packed)
            _,old_commands,old_mask=old_candidates(x,prev,step,env.packed)
            np.testing.assert_array_equal(commands,old_commands);np.testing.assert_array_equal(mask,old_mask)
            selected=int(model.choose(fe[None],mask[None])[0])
            compiled,index=choose_compiled(x,prev,step,env.packed,model.compiled_weights())
            if index!=selected:
                np.testing.assert_array_equal(fe[index],fe[selected])
            np.testing.assert_array_equal(compiled,commands[selected]);actions.append(selected)
        env.step(np.array(actions)[:,None])
