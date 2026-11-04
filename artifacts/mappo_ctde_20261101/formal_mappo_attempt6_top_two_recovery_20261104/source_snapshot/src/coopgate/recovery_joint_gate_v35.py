"""Common actuator-aware baseline/oracle for cooperative performance admission.

Fixed trained actors are shared by both performance-gate ablations and the
explicit MPS/MAMPS physical-switch adapters. Finite-path margins are not
probabilities and do not inherit any earlier statistical certificate.
"""
import time

import numpy as np
from numba import njit

from .actuator_prediction_roundoff_v31 import plant_arrays
from .actuator_recovery_v34 import ActuatorRecoveryBaseline, interval_substep, stopping_certificate, validate_inputs
from .ego_v13 import ego_observations
from .marl_v12 import LEVELS, action_mask
from .online_actuator_gate_v32 import reference_parameters
from .physical_switch_v27 import interval_transition_flags, recovery_search
from .recovery_forecast_v35 import forecast_kernel
from .temporal_portfolio_v20 import BANK_STREAM, temporal_stress_bank


RULES = ('baseline','mps_shared','mamps_shared','nominal_performance','bank_performance')


@njit(cache=True, fastmath=False)
def actuator_certificate_flags(state,previous,command,memory_lower,memory_upper,bounds,packed):
    """Incident-agent flags with the same joint terminal set as V34.

    As in the existing shared-oracle adaptation, flags do not latch success
    at individual stopping times. Their conjunction must match the full check.
    """
    valid = np.ones(3,np.bool_)
    for i in range(3):
        if (command[i] < max(-packed[4],previous[i]-packed[6]*packed[1])-1e-10
                or command[i] > min(packed[5],previous[i]+packed[6]*packed[1])+1e-10):
            valid[i] = False
    lo,hi,mlo,mhi = state.copy(),state.copy(),memory_lower.copy(),memory_upper.copy()
    tail = command.copy()
    for stage in range(1+int(packed[10])):
        if stage:
            tail = np.maximum(tail-packed[6]*packed[1],-packed[4])
        for _ in range(round(packed[1]/packed[0])):
            qlo,qhi,nmlo,nmhi = interval_substep(lo,hi,mlo,mhi,tail,bounds,packed)
            valid &= interval_transition_flags(lo,hi,qlo,qhi,packed)
            lo,hi,mlo,mhi = qlo,qhi,nmlo,nmhi
        invariant = stage > 0 and np.all(hi[:,1] == 0.)
        for i in range(3):
            limit = -bounds[i,6]/bounds[i,0]
            invariant = invariant and mhi[i] <= limit and tail[i] <= limit
        if invariant:
            return valid
    return np.zeros(3,np.bool_)


@njit(cache=True, fastmath=False)
def project_command(state,previous,proposal,baseline,mlo,mhi,bounds,packed):
    """Finite nearest-vertex projection, with the common baseline as fallback.

    Cleared vehicles keep their baseline commands. Thus changing an irrelevant
    actor coordinate cannot manufacture a learning-adoption event.
    """
    commands = np.empty((27,3))
    scores = np.empty(27)
    for mode in range(27):
        code,score = mode,0.
        for i in range(2,-1,-1):
            choice = code%3
            code //= 3
            if state[i,0] >= packed[11]:
                value = baseline[i]
            elif choice == 0:
                value = max(previous[i]-packed[6]*packed[1],-packed[4])
            elif choice == 1:
                value = previous[i]
            else:
                value = proposal[i]
            commands[mode,i] = value
            score += (value-proposal[i])**2
        scores[mode] = score
    for index in np.argsort(scores,kind='mergesort'):
        if stopping_certificate(state,previous,commands[index],mlo,mhi,bounds,packed)[0]:
            return commands[index].copy(),int(index)
    if not stopping_certificate(state,previous,baseline,mlo,mhi,bounds,packed)[0]:
        raise RuntimeError('Shared recovery baseline is not certified')
    return baseline.copy(),-1


def compositions(state,previous,proposal,baseline,mlo,mhi,bounds,packed):
    originals,seen = [],[baseline.copy()]
    for code in range(1,8):
        mask = np.array([bool(code & (1 << i)) for i in range(3)])
        target = np.where(mask,proposal,baseline)
        command,vertex = project_command(state,previous,target,baseline,mlo,mhi,bounds,packed)
        if any(np.array_equal(command,prior) for prior in seen):
            continue
        seen.append(command.copy())
        originals.append((code,1.,command.copy(),vertex))
    candidates = list(originals)
    for code,_,command,_ in originals:
        for weight in (.75,.5,.25):
            target = baseline+weight*(command-baseline)
            repaired,vertex = project_command(state,previous,target,baseline,mlo,mhi,bounds,packed)
            if any(np.array_equal(repaired,prior) for prior in seen):
                continue
            seen.append(repaired.copy())
            candidates.append((code,weight,repaired.copy(),vertex))
    return candidates,len(originals)


class RecoveryJointGate:
    def __init__(self,model,profile,rule='bank_performance'):
        if rule not in RULES:
            raise ValueError('Unknown shared-recovery decision rule')
        if rule != 'baseline' and model is None:
            raise ValueError('A fixed learned policy is required')
        self.model,self.profile,self.rule = model,profile,rule
        self.baseline = ActuatorRecoveryBaseline(profile)
        self.c,self.packed,self.bounds = self.baseline.c,self.baseline.packed,self.baseline.bounds
        self.reference = reference_parameters(profile)
        self.gain,self.drag,self.retention = plant_arrays(self.reference,self.c.dt)
        self.names,tapes = temporal_stress_bank('sampled')
        self.tapes = np.ascontiguousarray(tapes*(.1/.5))
        self.memory,self.step,self.learning_used_ever = np.zeros(3),0,False

    def predict(self,state,previous,mlo,mhi,command,path):
        first = np.zeros(3) if command is None else command
        return forecast_kernel(state,previous,self.memory,mlo,mhi,self.step,self.tapes[path],
            self.gain,self.drag,self.retention,self.bounds,self.packed,first,command is not None)

    def select(self,state,previous,unused=None):
        started = time.perf_counter()
        x,prev,_,_,_ = validate_inputs(state,previous,self.baseline.memory_lower,
                                       self.baseline.memory_upper,self.bounds,self.c)
        if self.baseline.step != self.step:
            raise RuntimeError('Baseline and gate clocks differ')
        if self.step:
            for _ in range(round(self.c.decision_dt/self.c.dt)):
                self.memory = self.retention*self.memory+(1.-self.retention)*prev
        base,base_info = self.baseline.select(x,prev)
        mlo,mhi = np.array(base_info['actuator_memory_lower']),np.array(base_info['actuator_memory_upper'])
        if np.any(self.memory < mlo-1e-12) or np.any(self.memory > mhi+1e-12):
            raise RuntimeError('Reference actuator point escaped the command-history interval')
        proposal = base.copy()
        indices = None
        active = x[:,0] < self.c.goal
        if self.rule != 'baseline':
            mask = action_mask(prev,self.c)
            indices = np.asarray(self.model.proposal(ego_observations(x,prev,self.step,self.c),mask))
            if (indices.shape != (3,) or not np.issubdtype(indices.dtype,np.integer)
                    or np.any(indices < 0) or np.any(indices >= len(LEVELS))
                    or not mask[np.arange(3),indices].all()):
                raise ValueError('Learned proposal violates the common action mask')
            proposal = LEVELS[indices].copy()
            proposal[~active] = base[~active]
        chosen = base.copy()
        candidate_evidence,nominal_evidence,masks,flags,winner,phases = [],[],[],[],[],[]
        candidate_count = original_count = calls = transitions = planning = checks = 0
        planning_complete = True
        selected_code,selected_weight,margin,baseline_complete = 0,None,None,None
        if self.rule in ('mps_shared','mamps_shared'):
            checker = lambda command:actuator_certificate_flags(x,prev,command,mlo,mhi,self.bounds,self.packed)
            chosen,_,masks,flags = recovery_search(proposal,base,checker,self.rule)
        elif self.rule in ('nominal_performance','bank_performance'):
            candidates,original_count = compositions(x,prev,proposal,base,mlo,mhi,self.bounds,self.packed)
            candidate_count = len(candidates)
            baseline_values,candidate_values = {},{}
            def predict(command,path):
                nonlocal calls,transitions,planning,checks,planning_complete
                value = self.predict(x,prev,mlo,mhi,command,path)
                calls += 1
                transitions += int(value[4]);planning += int(value[5]);checks += int(value[6])
                planning_complete = planning_complete and bool(value[21])
                return value
            baseline_values[0] = predict(None,0)
            baseline_complete = bool(baseline_values[0][2])
            ranked = []
            for index,(code,weight,command,_) in enumerate(candidates):
                after = predict(command,0)
                candidate_values[index] = after
                if not after[2] or not baseline_complete:
                    nominal_evidence.append(dict(candidate=index,baseline_complete=baseline_complete,
                        candidate_complete=bool(after[2]),passed=False,differences=None,phase_differences=None))
                    continue
                delta = after[0]-baseline_values[0][0]
                phase_delta = after[1]-baseline_values[0][1]
                admissible = bool(np.all(delta <= 1e-10) and delta[:,0].mean() < -1e-9
                                  and np.all(phase_delta <= .05+1e-10))
                nominal_evidence.append(dict(candidate=index,baseline_complete=True,candidate_complete=True,
                    passed=admissible,differences=delta.tolist(),phase_differences=phase_delta.tolist()))
                if admissible:
                    ranked.append((float(delta[:,0].mean()),-weight,code,index))
            ranked.sort()
            path_count = 1 if self.rule == 'nominal_performance' else len(self.tapes)
            for _,_,code,index in ranked:
                candidate = candidates[index][2]
                differences,phase_differences = [],[]
                failure = None
                passed = True
                for path in range(path_count):
                    if path not in baseline_values:
                        baseline_values[path] = predict(None,path)
                    before = baseline_values[path]
                    after = candidate_values[index] if path == 0 else predict(candidate,path)
                    if not before[2] or not after[2]:
                        passed = False
                        failure = dict(path=path,baseline_complete=bool(before[2]),candidate_complete=bool(after[2]),
                                       baseline_error=int(before[3]),candidate_error=int(after[3]))
                        break
                    delta = after[0]-before[0]
                    phase_delta = after[1]-before[1]
                    differences.append(delta.tolist());phase_differences.append(phase_delta.tolist())
                    if np.any(delta > 1e-10) or np.any(phase_delta > .05+1e-10):
                        passed = False
                        break
                candidate_evidence.append(dict(candidate=index,composition=code,passed=passed,failure=failure,
                    checked_paths=len(differences),differences=differences,phase_differences=phase_differences))
                if passed:
                    if len(differences) != path_count:
                        raise RuntimeError('Incomplete performance evidence')
                    chosen,selected_code,selected_weight = candidate.copy(),code,candidates[index][1]
                    winner,phases = differences,phase_differences
                    margin = min(-float(np.max(differences)),.05-float(np.max(phase_differences)))
                    break
        physical = stopping_certificate(x,prev,chosen,mlo,mhi,self.bounds,self.packed)
        if not physical[0]:
            raise RuntimeError('Selected joint command lacks the common stopping certificate')
        if not np.array_equal(chosen[~active],base[~active]):
            raise RuntimeError('Cleared vehicles must retain their common baseline commands')
        used = bool(np.any(np.abs(chosen[active]-base[active]) > 1e-9))
        self.learning_used_ever |= used
        self.step += 1
        return chosen,dict(in_set=True,certified=True,intervention=float(not used),
            gate_ms=1000*(time.perf_counter()-started),rule=self.rule,actuator_class=self.profile,
            certificate_kind='bounded_actuator_memory_stopping',actuator_bounds=self.bounds.tolist(),
            actuator_memory_lower=mlo.tolist(),actuator_memory_upper=mhi.tolist(),
            baseline_command=base.tolist(),baseline_recovery_proposed=base_info['recovery_applied'],
            baseline_evidence={k:v for k,v in base_info.items() if k != 'gate_ms'},
            recovery_applied=bool(not used and base_info['recovery_applied']),
            agent_actions=None if indices is None else indices.tolist(),actor_proposal=proposal.tolist(),
            actual_learning_used=used,actual_learning_used_ever=self.learning_used_ever,
            candidate_count=candidate_count,original_candidate_count=original_count,
            composition=selected_code,repair_weight=selected_weight,candidate_evidence=candidate_evidence,
            nominal_evidence=nominal_evidence,
            selected_bank_differences=winner,selected_phase_differences=phases,
            mode_search_masks=masks,mode_search_flags=flags,forecast_calls=calls,
            performance_forecast_transitions=transitions,completed_planning_model_transitions=planning,
            planning_work_complete=planning_complete,recovery_certificate_substeps=checks,
            prediction_parameters=self.reference,reconstructed_actuator=self.memory.tolist(),
            forecast_bank_stream=BANK_STREAM,forecast_bank_size=len(self.tapes),external_bank_radius=.1,
            phase_tolerance_seconds=.05,finite_bank_performance_margin=margin,margin_is_probability=False,
            reference_baseline_complete=baseline_complete,future_realization_used=False,
            true_randomized_plant_used=False,conditional_performance_risk_certified=False)
