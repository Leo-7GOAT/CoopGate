"""Declared-class parameter vertices crossed with persistent bias vertices.

Development witness library only. Finite vertices do not certify the nonlinear
parameter box or arbitrary disturbance sequences. No realized plant is read.
"""
from itertools import product
import numpy as np
from .actuator_recovery_v34 import class_bounds


def coupled_library(profile):
    bounds = class_bounds(profile)
    # Vary each independent uncertain coefficient; do not pair gain with drag.
    choices = [tuple(sorted(set((float(row[lo]), float(row[lo+1])))))
               for lo in (0, 2, 4) for row in bounds]
    for parameters in product(*choices):
        gain, drag, rho = (np.asarray(parameters[i:i+3]) for i in (0, 3, 6))
        for signs in product((-1., 1.), repeat=3):
            bias = np.asarray(signs) * bounds[:, 6]
            yield gain.copy(), drag.copy(), rho.copy(), bias


def history_memory(commands, rho, substeps=5):
    """Exact scalar recurrence in executed-command order, starting at reset."""
    memory = np.zeros(3)
    for command in commands:
        for _ in range(substeps):
            memory = rho * memory + (1. - rho) * np.asarray(command)
    return memory
