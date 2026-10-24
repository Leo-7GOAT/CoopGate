import numpy as np
import pytest

from coopgate import memoized_recovery_v37 as original
from coopgate import reuse_prediction_certificate as revised


def compare(profile,forced,path,capacity=original.CAPACITY,state=None,command=None):
    gate=original.MemoizedRecoveryGate(None,profile,'baseline')
    state=np.array([[-140.,12.],[-220.,10.],[-190.,12.]]) if state is None else state
    previous=np.zeros(3)
    command=np.zeros(3) if command is None else command
    args=(state,previous,np.zeros(3),np.zeros(3),np.zeros(3),0,gate.tapes[path],
          gate.gain,gate.drag,gate.retention,gate.bounds,gate.packed,command,forced)
    saved=[v.copy() if isinstance(v,np.ndarray) else v for v in args]
    cache1,stats1=original.make_cache();cache2,stats2=original.make_cache()
    proofs=np.zeros(4,dtype=np.int64)
    before=original.forecast_kernel(*args,cache1,stats1,capacity,True)
    after=revised.forecast_kernel(*args,cache2,stats2,capacity,proofs,True)
    assert len(before)==len(after)==22
    for a,b in zip(before,after):
        if isinstance(a,np.ndarray):
            assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
        else: assert a==b
    np.testing.assert_array_equal(stats1,stats2)
    for a,b in zip(args,saved):
        if isinstance(a,np.ndarray): assert a.tobytes()==b.tobytes()
    assert proofs[2]==after[6] and proofs[3]<=proofs[2]
    return before,proofs


@pytest.mark.parametrize('profile',['nominal','gain_drag','lag_mild','lag_strong'])
@pytest.mark.parametrize('forced',[False,True])
@pytest.mark.parametrize('path',[0,32])
def test_all_prediction_outputs_and_planning_work_are_bit_exact(profile,forced,path):
    result,proofs=compare(profile,forced,path)
    assert result[2] and proofs[1]>0 and proofs[3]<proofs[2]


def test_forced_first_command_is_still_checked_and_rejected():
    result,proofs=compare('lag_strong',True,0,command=np.full(3,10.))
    assert result[3]==3 and not result[2]
    assert proofs[0]==1 and proofs[1]==0


def test_failed_recovery_does_not_reuse_a_failed_certificate():
    result,proofs=compare('lag_strong',False,0,state=np.array([[-1.,12.],[-1.,12.],[-2.,12.]]))
    assert result[3]==2 and not result[2] and proofs[1]==0


def test_zero_planning_cache_capacity_preserves_certificate_equivalence():
    result,proofs=compare('gain_drag',False,32,capacity=0)
    assert result[2] and proofs[1]>0
