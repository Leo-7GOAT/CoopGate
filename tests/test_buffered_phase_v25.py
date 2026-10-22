import ast
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np
import pytest

from coopgate import buffered_phase_v25 as buffered
from coopgate import cached_portfolio_v19 as cached
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.fast_repair_v20 import FastRepairPortfolioGate
from coopgate.phase_repair_v21 import PhaseRepairGate,cached_arrival_phase

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('first,second',[
    *[(getattr(cached,n).py_func,getattr(buffered,n).py_func) for n in
      ('cached_command','cached_nominal_suffix','cached_action_then_baseline','cached_disturbed_suffix')],
    (cached_arrival_phase.py_func,buffered.cached_arrival_phase.py_func),
    (FastRepairPortfolioGate._future,buffered.BufferedPhaseGate._future),
    (PhaseRepairGate._phase,buffered.BufferedPhaseGate._phase),
    (PhaseRepairGate.select,buffered.BufferedPhaseGate.select)])
def test_full_candidate_order_paths_checks_and_accounting_unchanged(first,second):
    assert ast.dump(ast.parse(textwrap.dedent(inspect.getsource(first))),include_attributes=False)==ast.dump(
        ast.parse(textwrap.dedent(inspect.getsource(second))),include_attributes=False)


def test_complete_gate_matches_all_previously_frozen_slow_and_adopted_calls():
    plan=json.loads((ROOT/'artifacts/v24/phase_runtime_attribution/plan.json').read_text())
    models={seed:load_coupled(ROOT/meta['path'])[0] for seed,meta in plan['models'].items()}
    for case in plan['cases']:
        gate=buffered.BufferedPhaseGate(models[case['model']],.05)
        gate.step=case['step']
        command,info=gate.select(np.array(case['state']),np.array(case['previous']))
        np.testing.assert_array_equal(command,case['expected_command'])
        actual={k:v for k,v in info.items() if k!='gate_ms'}
        np.testing.assert_equal(actual,case['expected_info'])
        assert len(gate.tapes)==33 and gate.bounded is False
