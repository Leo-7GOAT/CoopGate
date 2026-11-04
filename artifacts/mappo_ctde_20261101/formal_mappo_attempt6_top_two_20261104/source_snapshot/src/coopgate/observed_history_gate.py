"""Rejection-only observed-history and strict crossing-phase development gate.

The frozen compact gate still supplies every candidate and physical baseline.
Only its selected learned command is additionally checked. No actor retrains,
realized plant coefficients, or future innovation tape enter this selector.
"""
import time
import numpy as np
from .compact_coupled_gate import CompactCoupledGate
from .observed_lag_bias_fit import fit_observed_history
from .memoized_recovery_v37 import make_cache,CAPACITY
from .scalar_nominal_recovery import forecast_kernel


class ObservedHistoryGate(CompactCoupledGate):
    def __init__(self,model,profile,bank_path):
        super().__init__(model,profile,bank_path)
        self.measured_states=[];self.executed_commands=[]

    def select(self,state,previous,unused=None):
        started=time.perf_counter();step=self.step;ever=self.learning_used_ever
        x=np.asarray(state,dtype=float);prev=np.asarray(previous,dtype=float)
        self.measured_states.append(x.copy())
        if step:self.executed_commands.append(prev.copy())
        chosen,info=super().select(x,prev,unused)
        used=info['actual_learning_used'];reason=None;fit_evidence=None;extra_calls=0
        if used and np.any(np.asarray(info['selected_phase_differences'])>1e-10):
            reason='Positive continuous crossing delay in an existing forecast path'
        if used and reason is None and self.profile in ('nominal','lag_mild','lag_strong'):
            fit=fit_observed_history(self.measured_states,np.asarray(self.executed_commands).reshape((-1,3)),self.profile)
            if fit is not None:
                memory=fit['memory'];mlo=np.array(info['actuator_memory_lower']);mhi=np.array(info['actuator_memory_upper'])
                if np.any(memory<mlo-1e-10) or np.any(memory>mhi+1e-10):raise ValueError('Fitted witness memory outside declared interval')
                args=(x,prev,memory,mlo,mhi,step,np.tile(fit['bias'],(250,1)),np.ones(3),np.zeros(3),fit['retention'],self.bounds,self.packed)
                cache,stats=make_cache();before=forecast_kernel(*args,np.zeros(3),False,cache,stats,CAPACITY)
                cache,stats=make_cache();after=forecast_kernel(*args,chosen,True,cache,stats,CAPACITY)
                delta=after[0]-before[0];phase=after[1]-before[1];valid=bool(before[2] and after[2]);extra_calls=2
                score=float(-delta[:,0].mean()/before[0][:,0].mean()) if valid else None
                fit_evidence=dict(retention=fit['retention'].tolist(),bias=fit['bias'].tolist(),memory=memory.tolist(),
                    fit_squared_error=fit['fit_squared_error'].tolist(),usable_windows=fit['usable_windows'],
                    complete_pair=valid,differences=delta.tolist(),phase_differences=phase.tolist(),relative_gain=score,
                    future_bias_is_hypothesis=True,full_parameter_box_certified=False)
                if not valid or np.any(delta>1e-10) or np.any(phase>1e-10) or score<self.gain_floor:
                    reason='Observed-history forecast did not pass unchanged per-car/gain tests and strict phase check'
        if used and reason is not None:
            info['upstream_selection_rejected']=dict(command=chosen.tolist(),composition=info['composition'],repair_weight=info['repair_weight'],
                gain_margin_evidence=info['gain_margin_evidence'],score=info['selected_gain_confidence_score'],
                baseline_team_costs=info['selected_baseline_forecast_team_costs'],selected_bank_differences=info['selected_bank_differences'],
                selected_phase_differences=info['selected_phase_differences'])
            chosen=np.asarray(info['baseline_command']).copy();self.learning_used_ever=ever
            info.update(actual_learning_used=False,actual_learning_used_ever=ever,intervention=1.,composition=0,repair_weight=None,
                selected_gain_confidence_score=None,selected_baseline_forecast_team_costs=[],gain_margin_evidence=[],
                selected_bank_differences=[],selected_phase_differences=[],finite_bank_performance_margin=None,scenario_pass_fraction=None,
                recovery_applied=bool(info['baseline_recovery_proposed']))
        info.update(development_controller='observed_history_strict_phase',extra_rejection_reason=reason,
            observed_history_forecast=fit_evidence,observed_history_forecast_calls=extra_calls,
            strict_crossing_phase_tolerance_s=0.,history_intervals_observed=step,
            no_privileged_prediction_inputs=True,conditional_performance_risk_certified=False,
            gate_ms=1000*(time.perf_counter()-started))
        return chosen,info
