"""Fixed temporal disturbance extensions for joint baseline-relative admission.

Every extension retains the original 17 paths. Its random stream is a property
of the controller, independent of episode seeds, observations and future noise.
An enlarged finite bank still requires fresh trajectory-level risk calibration.
"""
from itertools import product

import numpy as np

from .portfolio_gate_v19 import JointPortfolioGate
from .stress_prediction_v17 import fixed_stress_bank


BANKS = ('alternating', 'sampled', 'combined')
BANK_STREAM = 20019


def temporal_stress_bank(kind):
    if kind not in BANKS:
        raise ValueError('Unknown frozen temporal bank')
    old_names, old_tapes = fixed_stress_bank()
    names, tapes = list(old_names), list(old_tapes.copy())
    if kind in ('alternating', 'combined'):
        for pattern in ('alternating_bias', 'alternating_pulse'):
            for signs in product((-1., 1.), repeat=3):
                error = .5 * np.asarray(signs)
                tape = np.zeros((250, 3))
                if pattern == 'alternating_bias':
                    for block, start in enumerate(range(0, 250, 10)):
                        tape[start:start + 10] = (-1.) ** block * error
                else:
                    for burst, start in enumerate(range(10, 250, 20)):
                        tape[start:start + 10] = (-1.) ** burst * error
                names.append(pattern + ':' + ','.join(str(int(s)) for s in signs))
                tapes.append(tape)
    if kind in ('sampled', 'combined'):
        for mode_index, mode in enumerate(('uniform', 'pulse')):
            for sample in range(4):
                rng = np.random.default_rng(np.random.SeedSequence([BANK_STREAM, mode_index, sample]))
                if mode == 'uniform':
                    tape = rng.uniform(-.5, .5, (250, 3))
                else:
                    tape = np.zeros((250, 3))
                    for start in range(10, 250, 20):
                        tape[start:start + 10] = rng.choice([-.5, .5], 3)
                for sign in (1., -1.):
                    names.append(f'sampled_{mode}:{sample}:{int(sign)}')
                    tapes.append(sign * tape)
    return tuple(names), np.asarray(tapes)


class TemporalPortfolioGate(JointPortfolioGate):
    """Same actors, projection, ranking and AND rule, with a larger fixed bank."""

    def __init__(self, model, bank='combined'):
        if bank not in BANKS:
            raise ValueError('Unknown frozen temporal bank')
        super().__init__(model, max_failed_scenarios=0)
        self.names, self.tapes = temporal_stress_bank(bank)
        self.temporal_bank = bank

    def select(self, state, previous, unused=None):
        command, info = super().select(state, previous, unused)
        info.update(temporal_bank=self.temporal_bank, forecast_bank_size=len(self.tapes),
                    forecast_bank_stream=BANK_STREAM,
                    out_of_bank_risk_certified=False)
        return command, info
