"""Shared additive-SUMO collection and independent candidate review.

All policies and physical dynamics are existing frozen implementations. This
module supports execution integration; it does not authorize fresh calibration.
"""
import numpy as np
import torch

from coopgate.advantage_gate_v18 import action_then_baseline, baseline_suffix
from coopgate.arrival_phase_v20 import arrival_phase_suffix
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.fast_v8 import parameters
from coopgate.joint_planner_v15 import joint_rollout
from coopgate.merge_v4 import MergeConfig
from coopgate.prospective_validation_v21 import FAMILIES, profile_names
from coopgate.temporal_portfolio_v20 import temporal_stress_bank
from experiment_ego_v13 import run_episode
from probe_disturbed_gate_v19 import digest
from resume_frozen_risk_v19 import build_record
from review_repair_closed_loop_v20 import check_forecast_record, check_selected, reconstruct_candidates
from study_gate_v19 import regime_for
from study_matched_transfer_v21 import ROOT, factories, methods as all_methods
from study_phase_repair_v21 import phase_record_check
from study_repair_v20 import trace_metadata, verify_trace
from verify_development_v14 import trace_audit


def method_definitions(family=None):
    available = all_methods()
    if family is None:
        names = ['baseline', *[name for name in available if name!='baseline' and not name.startswith('portfolio0_')]]
    else:
        names = ['baseline', *profile_names(family)]
    return {name:available[name] for name in names}


def record(row, trace, initial, case, spec):
    index, mode, name = case
    if row['seed'] != spec['seed_base']+index or row['disturbance'] != mode or row['scenario'] != regime_for(spec,index):
        raise ValueError('Executed a different initial, mode or regime')
    result, _, _, _ = build_record(row,trace,initial,index,name,spec['methods'][name],spec)
    result.update(trace_metadata(trace))
    return result


def execute(env, case, spec, factory):
    index, mode, _ = case
    row,trace,initial = run_episode(env,spec['seed_base']+index,regime_for(spec,index),mode,factory,True)
    return record(row,trace,initial,case,spec),trace


def verify_record(row, trace, spec):
    name = row['controller']
    method = spec['methods'][name]
    rebuilt,adopted,checks,failures = build_record(dict(row),trace,np.asarray(row['initial']),
        row['initial_index'],name,method,spec)
    rebuilt.update(trace_metadata(trace))
    np.testing.assert_equal(rebuilt,row)
    trace_audit(trace,row)
    if method['kind']=='repair':
        verify_trace(trace,row,method,adopted,checks,failures)
        for item in trace:
            phase_record_check(item,method)


class IndependentReview:
    """Reconstruct physical traces, actors, joint candidates and every admission."""
    def __init__(self, protocol):
        torch.set_num_threads(1)
        self.protocol=protocol
        self.models={}
        for seed,entry in protocol['models'].items():
            if digest(ROOT/entry['path'])!=entry['sha256']:
                raise ValueError('Frozen training model changed')
            self.models[seed]=load_coupled(ROOT/entry['path'])[0]
        self.packed=parameters(MergeConfig())
        _,self.tapes=temporal_stress_bank('sampled')
        self.cache={}
        self.counts=dict(physical_traces=0,raw_actor_checks=0,baseline_checks=0,nominal_checks=0,
            candidate_families=0,admitted_forecasts=0,distinct_admitted_forecasts=0,error_case_replays=0)
        self.error_cases=[]
        self.factories=None

    def check(self,row,trace):
        spec=self.protocol['spec']
        method=spec['methods'][row['controller']]
        verify_record(row,trace,spec)
        self.counts['physical_traces']+=1
        kind=method['kind']
        actor=JointPolicyGate(self.models[str(method['seed'])]) if kind!='baseline' else None
        state,previous=np.asarray(row['initial']),np.zeros(3)
        for step,item in enumerate(trace):
            command=np.asarray(item['command'])
            if kind=='baseline':
                np.testing.assert_array_equal(command,joint_rollout(state,previous,step,self.packed)[0])
                self.counts['baseline_checks']+=1
            else:
                actor.step=step
                raw_command,actor_info=actor.select(state,previous)
                if kind=='raw':
                    np.testing.assert_array_equal(command,raw_command)
                    self.counts['raw_actor_checks']+=1
                elif kind=='nominal':
                    if item.get('nominal_joint_action_admitted',False):
                        np.testing.assert_array_equal(command,raw_command)
                        delta=action_then_baseline(state,previous,step,command,self.packed)[0]-baseline_suffix(state,previous,step,self.packed)[0]
                        if np.any(delta>1e-10) or delta[:,0].mean()>=-1e-9:
                            raise ValueError('Independently reconstructed nominal admission fails')
                    else:
                        np.testing.assert_array_equal(command,joint_rollout(state,previous,step,self.packed)[0])
                    self.counts['nominal_checks']+=1
                elif kind=='repair':
                    baseline=joint_rollout(state,previous,step,self.packed)[0]
                    np.testing.assert_array_equal(item['actor_proposal'],actor_info['proposal'])
                    np.testing.assert_array_equal(item['baseline_command'],baseline)
                    if item['forecast_errors'] or item['generation_errors']:
                        if self.factories is None:
                            self.factories=factories(self.protocol)
                        controller=self.factories[row['controller']](state)
                        controller.step=step
                        replay,info=controller.select(state,previous)
                        np.testing.assert_array_equal(command,replay)
                        for field in ('forecast_errors','generation_errors','composition','repair_weight',
                                      'selected_bank_differences','portfolio_joint_action_admitted'):
                            np.testing.assert_equal(item[field],info[field])
                        self.error_cases.append(dict(initial_index=row['initial_index'],mode=row['disturbance'],
                            method=row['controller'],step=step,forecast_errors=item['forecast_errors'],generation_errors=item['generation_errors']))
                        self.counts['error_case_replays']+=1
                    else:
                        candidates,original_count=reconstruct_candidates(state,previous,np.asarray(actor_info['proposal']),baseline,self.packed,True)
                        check_selected(item,baseline,candidates,original_count)
                        self.counts['candidate_families']+=1
                else:
                    raise ValueError('Unregistered common-profile controller')
            if item.get('portfolio_joint_action_admitted',False):
                key=(step,state.tobytes(),previous.tobytes(),command.tobytes())
                if key not in self.cache:
                    differences,phases,complete=[],[],True
                    for tape in self.tapes:
                        base,bt,bc=arrival_phase_suffix(state,previous,step,tape,self.packed,command,False)
                        value,vt,vc=arrival_phase_suffix(state,previous,step,tape,self.packed,command,True)
                        differences.append(value-base)
                        phases.append(vt-bt)
                        complete=complete and bc and vc
                    self.cache[key]=np.asarray(differences),np.asarray(phases),complete
                    self.counts['distinct_admitted_forecasts']+=1
                differences,phases,complete=self.cache[key]
                check_forecast_record(differences,item['selected_bank_differences'])
                tolerance=method.get('phase_tolerance_seconds')
                if tolerance is not None:
                    if not complete or np.any(phases>tolerance+1e-10):
                        raise ValueError('Independent complete phase forecast fails')
                    np.testing.assert_allclose(phases,item['selected_phase_differences'],atol=1e-10,rtol=0)
                self.counts['admitted_forecasts']+=1
            state,previous=np.asarray(item['x']),command


def validate_common_family(protocol,family):
    if family not in FAMILIES or protocol['spec']['methods']!=method_definitions(family):
        raise ValueError('A whole predeclared three-model family is required')
