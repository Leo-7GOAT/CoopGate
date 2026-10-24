"""Original fixed DMPS interface scope with isolated native restores."""
import argparse
from pathlib import Path
from types import FunctionType

from coopgate.dmps_sumo_process import DMPSProcessEnv
import probe_dmps_sumo_snapshot as original
import probe_dmps_sumo_snapshot_v2 as corrected_audit
import probe_dmps_sumo_clock as clocked

SOURCES=(*clocked.SOURCES,'src/coopgate/dmps_sumo_process.py','tools/probe_dmps_sumo_process.py',
         'tests/test_dmps_sumo_process.py','docs/dmps_sumo_process_protocol.md')
run=FunctionType(original.run.__code__,{**original.run.__globals__,'SOURCES':SOURCES,
    'restore_checks':corrected_audit.restore_checks,'DMPSSumoSnapshotEnv':DMPSProcessEnv})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
