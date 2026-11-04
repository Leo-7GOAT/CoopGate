"""Observable, physically realized reset states for task-completion curriculum.

No rewards, dynamics or evaluation states are changed. These are training/reset
utilities; every native placement is before the experiment time origin.
"""
import numpy as np

from .merge_v4 import BRANCH, backup, certify

PHASES=('near_goal','post_merge','single_approach','original_merge')


def phase_at(actual_decisions):
    if isinstance(actual_decisions,(bool,np.bool_)) or not isinstance(actual_decisions,(int,np.integer)) or actual_decisions<0:
        raise ValueError('Nonnegative integer real-decision count required')
    return min(int(actual_decisions)//8192,3)


def target_state(seed,phase):
    if phase not in range(4):raise ValueError('Unknown fixed curriculum phase')
    if phase==3:return None
    rng=np.random.default_rng(np.random.SeedSequence([seed,240925]))
    positions=((39.,57.,21.),(15.,45.,-15.),(-40.,35.,-75.))[phase]
    spread=(1.,3.,5.)[phase];low,high=((19.5,20.),(13.,16.),(11.,14.))[phase]
    shift=rng.uniform(-spread,spread);speed=rng.uniform(low,high)
    return np.column_stack((np.asarray(positions)+shift,np.full(3,speed)))


def initialize_native(raw,seed,phase):
    """Place native SUMO vehicles, then verify their true route coordinates."""
    if raw.t!=0 or raw.substep!=0 or raw.trace:raise RuntimeError('Curriculum placement is reset-only')
    np.testing.assert_array_equal(raw.prev,np.zeros(3))
    target=target_state(seed,phase)
    if target is not None:
        for i,ident in enumerate(raw.ids):
            before=target[i,0]-raw.c.dt*target[i,1]
            if before>=0:lane='out_0';position=before
            else:
                lane=('a' if BRANCH[i]==0 else 'b')+'_0'
                position=raw.conn.lane.getLength(lane)+before
            raw.conn.vehicle.moveTo(ident,lane,float(position))
            raw.conn.vehicle.setSpeed(ident,float(target[i,1]))
        raw.conn.simulationStep();offsets=[]
        for ident in raw.ids:
            lane=raw.conn.vehicle.getLaneID(ident)
            position=raw.conn.vehicle.getLanePosition(ident)
            if lane!='out_0':position-=raw.conn.lane.getLength(lane)
            offsets.append(position-raw.conn.vehicle.getDistance(ident))
        raw.offsets=offsets;raw.x=raw.read();raw.initial=raw.x.copy()
        np.testing.assert_allclose(raw.x,target,atol=1e-8,rtol=0)
        if raw.conn.simulation.getCollidingVehiclesIDList():raise RuntimeError('Native curriculum reset collision')
        raw.scenario='curriculum_'+PHASES[phase]
    if not certify(raw.x,np.zeros(3),backup(np.zeros(3),raw.c),raw.c)[0]:
        raise RuntimeError('Curriculum initial is outside nominal backup recovery; no seed replacement')
    return dict(seed=seed,phase=phase,phase_name=PHASES[phase],
        target_state=None if target is None else target.tolist(),actual_initial=raw.initial.tolist(),
        simulation_origin=raw.conn.simulation.getTime(),placement_after_time_origin=False)
