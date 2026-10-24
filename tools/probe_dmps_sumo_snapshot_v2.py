"""Additive audit-type correction; preserve the failed first probe and sources."""
import argparse
from pathlib import Path
from types import FunctionType

import numpy as np

import probe_dmps_sumo_snapshot as original

SOURCES=(*original.SOURCES,'tools/probe_dmps_sumo_snapshot_v2.py',
         'tests/test_dmps_snapshot_comparison.py','docs/dmps_sumo_snapshot_audit_correction.md')


def trajectory_equivalent(a,b,path='root'):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:trajectory_equivalent(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert isinstance(b,(list,tuple)) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):trajectory_equivalent(x,y,path+f'[{i}]')
    elif isinstance(a,bool):
        assert isinstance(b,bool) and a==b,path
    elif isinstance(a,int):
        # NumPy may coerce PCG64's 128-bit integers to object and isfinite then
        # fails. RNG state and all counters must be exact integers anyway.
        assert isinstance(b,int) and not isinstance(b,bool) and a==b,path
    elif isinstance(a,float):
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=0,err_msg=path)
    else:
        assert a==b,path


restore_checks=FunctionType(original.restore_checks.__code__,
    {**original.restore_checks.__globals__,'trajectory_equivalent':trajectory_equivalent})
run=FunctionType(original.run.__code__,{**original.run.__globals__,
    'restore_checks':restore_checks,'SOURCES':SOURCES})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
