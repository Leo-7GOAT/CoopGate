import numpy as np
import pytest

from coopgate.actuator_prediction_roundoff_v31 import plant_arrays
from coopgate.actuator_recovery_v34 import class_bounds
from coopgate.admissible_recovery_pruning import optimistic_values,pruned_forecast
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.fast_v8 import parameters
from coopgate.memoized_recovery_v37 import forecast_kernel,make_cache,CAPACITY
from coopgate.merge_v4 import MergeConfig
from coopgate.online_actuator_gate_v32 import reference_parameters

STATE=np.array([[-140.,12.],[-220.,10.],[-190.,12.]])


@pytest.mark.parametrize('profile',('nominal','gain_drag','lag_mild','lag_strong'))
def test_outward_fast_arrival_is_a_lower_bound_for_random_slew_admissible_commands(profile):
    rng=np.random.default_rng(181);c=parameters(MergeConfig())
    gain,drag,rho=plant_arrays(reference_parameters(profile),.1)
    for case in range(32):
        x=np.column_stack((rng.uniform(-150,30,3),rng.uniform(0,20,3)))
        prev=rng.uniform(-5,2.5,3);memory=rng.uniform(-5,2.5,3)
        tape=disturbance_tape(88100+case,'uniform',.1,250)
        root=int(rng.integers(0,10));step=root+int(rng.integers(0,8))
        finish=np.full(3,25.-root*.5);variation=rng.uniform(0,15,3)
        lower,_=optimistic_values(x,prev,memory,step,root,tape,gain,rho,c,finish,variation)
        for strategy in range(3):
            state=x.copy();point=memory.copy();previous=prev.copy();v=variation.copy();f=finish.copy()
            for k in range(step,50):
                active=state[:,0]<60
                if not active.any():break
                command=(np.minimum(2.5,previous+2.5) if strategy==0 else
                         rng.uniform(np.maximum(-5,previous-2.5),np.minimum(2.5,previous+2.5)))
                v+=active*np.abs(command-previous)
                for sub in range(5):
                    point=rho*point+(1-rho)*command
                    state[:,1]=np.clip(state[:,1]+.1*(gain*point-drag*state[:,1]**2+tape[k*5+sub]),0,20)
                    state[:,0]+=.1*state[:,1]
                f[active&(state[:,0]>=60)]=(k-root+1)*.5;previous=command
            assert np.all(lower[:,1]<=f)
            assert np.all(lower[:,0]<=f+.05*v)


@pytest.mark.parametrize('profile',('nominal','gain_drag','lag_mild','lag_strong'))
def test_disabled_pruning_retains_every_original_forecast_result(profile):
    c=parameters(MergeConfig());gain,drag,rho=plant_arrays(reference_parameters(profile),.1)
    args=(STATE,np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),0,
          disturbance_tape(991,'uniform',.1,250),gain,drag,rho,class_bounds(profile),c,np.zeros(3),True)
    cache,stats=make_cache();expected=forecast_kernel(*args,cache,stats,CAPACITY)
    cache,stats=make_cache();actual=pruned_forecast(*args,cache,stats,CAPACITY,np.zeros((3,2)),False)
    for a,b in zip(actual[:22],expected):np.testing.assert_equal(a,b)


def test_pruned_prefix_corresponds_to_an_inadmissible_full_forecast():
    c=parameters(MergeConfig());gain,drag,rho=plant_arrays(reference_parameters('nominal'),.1)
    common=(STATE,np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),0,
            np.zeros((250,3)),gain,drag,rho,class_bounds('nominal'),c)
    cache,stats=make_cache();baseline=forecast_kernel(*common,np.zeros(3),False,cache,stats,CAPACITY)
    assert baseline[2]
    cache,stats=make_cache();full=forecast_kernel(*common,np.full(3,-2.5),True,cache,stats,CAPACITY)
    cache,stats=make_cache();short=pruned_forecast(*common,np.full(3,-2.5),True,cache,stats,CAPACITY,baseline[0])
    assert short[3]==4 and not short[2]
    assert not full[2] or np.any(full[0]>baseline[0]+1e-10)
    assert short[5]<full[5]
