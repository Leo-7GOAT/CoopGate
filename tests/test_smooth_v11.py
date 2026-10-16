import itertools
import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('numba')
from coopgate.fast_v8 import advance_one, parameters
from coopgate.merge_v4 import MergeConfig
from coopgate.planning_v5 import portfolio
from coopgate.robust_v9 import robust_certificate, feedback
from coopgate.rollout_v10 import active_count, earliest_ticks
from coopgate.smooth_v11 import (SmoothGate, base_command, ternary_feedback, binary_feedback_ordered,
    tail_value, improve_smooth, variation_units, less)

ROOT=Path(__file__).resolve().parents[1]


def old_states():
    """Already published states, rounded memories only when recertified."""
    records=[]
    for path in sorted((ROOT/'artifacts/v9/study').glob('trace_*robust_learned_pair.json')):
        trace=json.loads(path.read_text())
        for k in range(1,len(trace),3):
            records.append((np.array(trace[k-1]['x']),np.array(trace[k-1]['command']),k))
    return records


def exhaustive_ternary(x,prev,k,c,weights,first,coast):
    """Independent Python product, all 27 certificates, original tie order."""
    proposal=np.minimum(np.maximum(np.where((k<coast)&(np.arange(3)!=first)&(x[:,0]<c[11]),0.,c[5]),
        np.maximum(-c[4],prev-c[6]*c[1])),np.minimum(c[5],prev+c[6]*c[1]))
    backup=np.maximum(-c[4],prev-c[6]*c[1])
    best=np.inf;chosen=None
    for choices in itertools.product(range(3),repeat=3):
        u=np.array([backup[i] if d==0 else prev[i] if d==1 else proposal[i] for i,d in enumerate(choices)])
        if robust_certificate(x,prev,u,c,.5):
            score=sum(weights[i]*(u[i]-proposal[i])**2-1e-8*u[i] for i in range(3))
            if score<best:best,chosen=score,u
    assert chosen is not None
    return chosen


def test_sorted_projection_matches_all_27_joint_certificates():
    c=parameters(MergeConfig());records=old_states()
    assert len(records)>40
    for index,(x,prev,k) in enumerate(records):
        p=portfolio()[(index*7)%37];weights=np.array(p['weights'])
        actual=ternary_feedback(x,prev,c,.5,weights,p['first'],p['coast_steps'],k)
        expected=exhaustive_ternary(x,prev,k,c,weights,p['first'],p['coast_steps'])
        np.testing.assert_array_equal(actual,expected)
        assert robust_certificate(x,prev,actual,c,.5)
        np.testing.assert_array_equal(binary_feedback_ordered(x,prev,c,.5,weights,p['first'],p['coast_steps'],k),
            feedback(x,prev,c,.5,weights,p['first'],p['coast_steps'],k))


def reference_value(x,prev,k,c,p,ternary):
    x,prev=x.copy(),prev.copy();ticks,units=0,0
    for step in range(k,50):
        count=int(sum(x[:,0]<60))
        if not count:break
        u=base_command(x,prev,step,c,.5,np.array(p['weights']),p['first'],p['coast_steps'],ternary)
        ticks+=count
        units+=int(np.rint(np.abs(u-prev)/.625).sum())
        assert advance_one(x,u,c)
        prev=u
    return ticks,units


def test_two_component_tail_matches_independent_reference_and_deadline():
    c=parameters(MergeConfig())
    for index,(x,prev,k) in enumerate(old_states()[::3]):
        p=portfolio()[(index*11)%37]
        for family in (False,True):
            reference=reference_value(x,prev,k,c,p,family)
            actual=tail_value(x,prev,k,c,.5,np.array(p['weights']),p['first'],p['coast_steps'],
                .625,family,True,10**9,10**9,False)
            assert actual[:2]==reference and not actual[3]
            assert earliest_ticks(x,prev,k,c)<=reference[0]
    assert tail_value(x,prev,50,c,.5,np.ones(3),0,0,.625,True,True,10**9,10**9,False)==(0,0,0,False)


def test_lexicographic_pruning_keeps_time_ties_with_lower_variation():
    c=parameters(MergeConfig());saved=0
    for index,(x,prev,k) in enumerate(old_states()):
        p=portfolio()[(index*5)%37];weights=np.array(p['weights']);family=bool(index%2)
        t,v,_,_=tail_value(x,prev,k,c,.5,weights,p['first'],p['coast_steps'],.625,family,True,10**9,10**9,False)
        args=(x,prev,k,c,.5,weights,p['first'],p['coast_steps'],.625,family,True,
            np.argsort(-weights,kind='stable'),t,v)
        fast,ref=improve_smooth(*args,True),improve_smooth(*args,False)
        np.testing.assert_array_equal(fast[0],ref[0]);assert fast[1:3]==ref[1:3]
        assert fast[1:3]<=(t,v) and robust_certificate(x,prev,fast[0],c,.5)
        saved+=ref[3]-fast[3]
    assert saved>0
    assert less(10,9,10,10,True) and not less(10,9,10,10,False)
    assert less(9,100,10,1,True)
    assert not less(11,0,10,100,True)


@pytest.mark.parametrize('families',[False,True])
def test_nominal_lexicographic_improvement_and_global_time_recursion(families):
    config=MergeConfig();c=parameters(config)
    items=json.loads((ROOT/'artifacts/v9/study/initial_states.json').read_text())
    for index,item in enumerate(items[::12]):
        x=np.array(item['state']);prev=np.zeros(3)
        gate=SmoothGate(x,prev,config,.5,'pair',(index*11)%37,ternary=True,family_rollouts=families)
        total_t,total_v=0,0
        for k in range(50):
            if not active_count(x,c):break
            u,info=gate.select(x,prev,None)
            base=info['base_ticks'],info['base_variation_units']
            selected=info['selected_ticks'],info['selected_variation_units']
            assert selected<=base
            if k==0:initial_bound=base
            else:assert base<=remaining_bound
            stage=active_count(x,c);variation=variation_units(u,prev,.625)
            remaining_bound=(selected[0]-stage,selected[1]-variation)
            total_t+=stage;total_v+=variation
            assert advance_one(x,u,c);prev=u
        assert (total_t,total_v)<=initial_bound
        assert total_t*config.decision_dt/3<=gate.predicted_cost+1e-10


def test_rejects_nonlattice_memory_parameters_and_exhausted_horizon():
    x=np.array([[-80.,12.],[-85.,13.],[-120.,11.]])
    for q in (0.,-.1,np.nan,.3):
        with pytest.raises(ValueError):SmoothGate(x,np.zeros(3),quantum=q)
    with pytest.raises(ValueError):SmoothGate(x,np.array([.1,0,0]))
    gate=SmoothGate(x,np.zeros(3))
    with pytest.raises(ValueError,match='lattice'):gate.select(x,np.array([.1,0,0]),None)
    gate.step=50
    with pytest.raises(ValueError,match='horizon'):gate.select(x,np.zeros(3),None)
