"""Mechanism ablation: compact gate plus strict phase rejection,without fitting."""
import time
import numpy as np
from .compact_coupled_gate import CompactCoupledGate


class StrictPhaseOnlyGate(CompactCoupledGate):
    def select(self,state,previous,unused=None):
        started=time.perf_counter();ever=self.learning_used_ever
        chosen,info=super().select(state,previous,unused)
        rejected=bool(info['actual_learning_used'] and np.any(np.asarray(info['selected_phase_differences'])>1e-10))
        if rejected:
            info['upstream_selection_rejected']=dict(command=chosen.tolist(),composition=info['composition'],repair_weight=info['repair_weight'],
                gain_margin_evidence=info['gain_margin_evidence'],score=info['selected_gain_confidence_score'],
                baseline_team_costs=info['selected_baseline_forecast_team_costs'],selected_bank_differences=info['selected_bank_differences'],
                selected_phase_differences=info['selected_phase_differences'])
            chosen=np.asarray(info['baseline_command']).copy();self.learning_used_ever=ever
            info.update(actual_learning_used=False,actual_learning_used_ever=ever,intervention=1.,composition=0,repair_weight=None,
                selected_gain_confidence_score=None,selected_baseline_forecast_team_costs=[],gain_margin_evidence=[],
                selected_bank_differences=[],selected_phase_differences=[],finite_bank_performance_margin=None,scenario_pass_fraction=None,
                recovery_applied=bool(info['baseline_recovery_proposed']))
        info.update(development_controller='strict_phase_only_ablation',strict_phase_rejected=rejected,
            strict_crossing_phase_tolerance_s=0.,history_fit_enabled=False,gate_ms=1000*(time.perf_counter()-started))
        return chosen,info
