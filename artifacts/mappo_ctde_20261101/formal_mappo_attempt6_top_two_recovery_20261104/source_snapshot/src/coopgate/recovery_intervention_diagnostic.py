"""Post-hoc V34 baseline suffixes after an exactly replayed physical prefix.

Known parameters/future innovations belong only to offline forecasts. The native
controller owns no plant truth or future tape and copies only the archived prefix.
"""
import time

import numpy as np

from .actuator_intervention_v31 import prefix_values
from .actuator_recovery_v34 import ActuatorRecoveryBaseline,certificate
from .performance_identity_v31 import audit_identity
from .recovery_forecast_v35 import forecast

ROLES=('baseline_suffix','candidate_then_baseline')


def adoption_state(row,trace,step):
    if type(step) is not int or not 0<=step<len(trace) or not trace[step]['actual_learning_used']:
        raise ValueError('An actual archived learning adoption is required')
    x=np.asarray(row['initial'] if step==0 else trace[step-1]['x'],dtype=float)
    previous=np.zeros(3) if step==0 else np.asarray(trace[step-1]['command'],dtype=float)
    entry=trace[step];active=x[:,0]<60
    if not np.any(np.abs(np.asarray(entry['command'])[active]-np.asarray(entry['baseline_command'])[active])>1e-9):
        raise ValueError('Adoption flag must correspond to a changed active command')
    return x,previous


def boundary(row,trace,step):
    x,previous=adoption_state(row,trace,step)
    memory=np.zeros(3) if step==0 else np.asarray(trace[step-1]['actuator_states'][-1],dtype=float)
    entry=trace[step]
    return x,previous,memory,np.asarray(entry['actuator_memory_lower']),np.asarray(entry['actuator_memory_upper'])


def predictions(row,trace,step,tape,plant=None,memory=None,record_trace=False):
    x,previous,actual,lo,hi=boundary(row,trace,step)
    point=actual if memory is None else np.asarray(memory)
    parameters=row['plant'] if plant is None else plant
    bounds=np.asarray(trace[step]['actuator_bounds'])
    return {role:forecast(x,previous,point,lo,hi,step,tape,parameters,bounds,
        first_command=np.asarray(trace[step]['command']) if role=='candidate_then_baseline' else None,
        record_trace=record_trace) for role in ROLES}


def pair(branches):
    base,candidate=(branches[r] for r in ROLES);complete=base['complete'] and candidate['complete']
    return dict(both_complete=bool(complete),baseline_complete=bool(base['complete']),candidate_complete=bool(candidate['complete']),
        difference=(np.asarray(candidate['values'])-base['values']).tolist() if complete else None,
        phase_difference=(np.asarray(candidate['crossing'])-base['crossing']).tolist() if complete else None)


class RecordedRecoveryBranch:
    def __init__(self,row,trace,step,role):
        if role not in ROLES:raise ValueError('Unknown fixed branch')
        adoption_state(row,trace,step)
        self.initial=np.asarray(row['initial']).copy()
        self.prefix=[dict(x=np.asarray(t['x']).copy(),command=np.asarray(t['command']).copy()) for t in trace[:step+1]]
        self.recorded_base=np.asarray(trace[step]['baseline_command']).copy()
        self.recorded_lo=np.asarray(trace[step]['actuator_memory_lower']).copy()
        self.recorded_hi=np.asarray(trace[step]['actuator_memory_upper']).copy()
        self.baseline=ActuatorRecoveryBaseline(row['plant_name']);self.target=step;self.role=role;self.step=0

    def select(self,state,previous,unused=None):
        started=time.perf_counter();base,info=self.baseline.select(state,previous)
        if self.step<=self.target:
            np.testing.assert_allclose(state,self.initial if self.step==0 else self.prefix[self.step-1]['x'],atol=1e-8,rtol=0)
            np.testing.assert_array_equal(previous,np.zeros(3) if self.step==0 else self.prefix[self.step-1]['command'])
        at_target=self.step==self.target
        if at_target:
            np.testing.assert_array_equal(base,self.recorded_base)
            np.testing.assert_allclose(info['actuator_memory_lower'],self.recorded_lo,atol=1e-12,rtol=0)
            np.testing.assert_allclose(info['actuator_memory_upper'],self.recorded_hi,atol=1e-12,rtol=0)
        archived=self.step<self.target or (at_target and self.role=='candidate_then_baseline')
        command=self.prefix[self.step]['command'].copy() if archived else base
        check=certificate(state,previous,command,self.baseline.memory_lower,self.baseline.memory_upper,self.baseline.bounds)
        if not check['certified']:raise RuntimeError('Diagnostic command lacks the unchanged stopping certificate')
        active=np.asarray(state)[:,0]<60;used=bool(np.any(np.abs(command[active]-base[active])>1e-9))
        self.step+=1
        info.update(baseline_command=base.tolist(),actual_learning_used=used,
            diagnostic_archived_prefix=self.step-1<self.target,
            diagnostic_intervention=at_target and self.role=='candidate_then_baseline',
            diagnostic_role=self.role,diagnostic_only=True,production_controller_changed=False,
            recovery_applied=bool(not used and info['recovery_applied']),
            gate_ms=1000*(time.perf_counter()-started))
        return command,info


def compare_prediction(prediction,record,source,source_trace,step):
    row,trace=record['row'],record['trace'];suffix=trace[step:];expected=prediction['trace']
    assert len(trace)>=step and len(suffix)==len(expected)
    for a,b in zip(trace[:step],source_trace[:step]):
        np.testing.assert_array_equal(a['command'],b['command'])
        np.testing.assert_allclose(a['x'],b['x'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(a['actuator_states'],b['actuator_states'],atol=1e-9,rtol=0)
    errors={name:0. for name in ('x','actuator_states','disturbance')}
    for predicted,actual in zip(expected,suffix):
        np.testing.assert_array_equal(predicted['command'],actual['command'])
        for name,tolerance in [('x',1e-8),('actuator_states',1e-9),('disturbance',1e-9)]:
            error=float(np.max(np.abs(np.asarray(predicted[name])-actual[name])))
            errors[name]=max(errors[name],error);assert error<=tolerance
        for name in ('actuator_memory_lower','actuator_memory_upper'):
            np.testing.assert_allclose(predicted[name],actual[name],atol=1e-12,rtol=0)
    whole=np.column_stack((row['per_vehicle_cost'],row['per_vehicle_finish']))
    offset=prefix_values(source['initial'],source_trace[:step])
    metric_error=float(np.max(np.abs(np.asarray(prediction['values'])+offset-whole)))
    assert metric_error<=1e-9 and prediction['complete']==row['completed']
    assert bool(prediction['error_code'])==bool(row['aborted'])
    return dict(prefix_exact=True,commands_exact=True,physical_errors=errors,metric_error=metric_error,
        completion_matches=True,suffix_commands=len(suffix))


def identity(source,baseline,local_pairs):
    return audit_identity(source,baseline,local_pairs)
