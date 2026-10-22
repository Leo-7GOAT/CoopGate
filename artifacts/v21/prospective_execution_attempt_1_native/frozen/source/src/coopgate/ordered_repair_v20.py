"""Check temporal paths first while preserving the exact finite-bank conjunction."""
import time

import numpy as np

from .fast_repair_v20 import FastRepairPortfolioGate


class TemporalFirstRepairGate(FastRepairPortfolioGate):
    def __init__(self, model, bounded=False):
        super().__init__(model, bounded, 'sampled')
        if len(self.tapes) != 33:
            raise ValueError('This order is defined only for the frozen sampled 33-path bank')
        # Keep nominal path zero first. The 16 extra time-varying paths precede
        # the 16 original fixed-bias/pulse paths. No path is added or removed.
        self.path_order = (0, *range(17, 33), *range(1, 17))
        self.canonical_order = np.argsort(self.path_order)
        self.names = tuple(self.names[i] for i in self.path_order)
        self.tapes = np.ascontiguousarray(self.tapes[list(self.path_order)])

    def select(self, state, previous, unused=None):
        started = time.perf_counter()
        command, info = super().select(state, previous, unused)
        # Rejection traces explicitly identify their evaluation order. The
        # complete accepted certificate remains in the original canonical order.
        for candidate in info['candidate_evidence']:
            candidate['evaluated_original_path_indices'] = list(self.path_order[:candidate['checked_scenarios']])
        if info['portfolio_joint_action_admitted']:
            if len(info['selected_bank_differences']) != 33:
                raise RuntimeError('A reordered incomplete bank cannot authorize an action')
            info['selected_bank_differences'] = np.asarray(info['selected_bank_differences'])[self.canonical_order].tolist()
        info['forecast_path_order'] = list(self.path_order)
        info['gate_ms'] = (time.perf_counter() - started) * 1000.
        return command, info
