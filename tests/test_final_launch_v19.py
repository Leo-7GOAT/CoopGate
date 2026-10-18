import copy
import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('final_launch_v19', TOOLS / 'start_final_v19.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def records():
    methods = {'baseline': {'kind': 'baseline'}, 'portfolio0_1802': {'kind': 'portfolio', 'seed': 1802}}
    design = dict(seed_base=200, n_initials=10, sampling='iid_mixture', modes=['clean', 'uniform'])
    plan = dict(profile_definitions=methods, profiles=['portfolio0_1802'], seeds=['1802'], final_test=design)
    model = {'1802': {'sha256': 'model_a'}}
    risk = dict(models=model, spec=dict(seed_base=100, n_initials=10, sampling='iid_mixture', modes=design['modes'],
                                       data_role='risk_validation', methods=methods), source_sha256={'src/gate.py': 'source_a'})
    final = dict(models=model, spec=dict(**design, data_role='final_confirmation', methods=methods),
                 source_sha256={'src/gate.py': 'source_a'})
    calibration = dict(models=model, names=plan['profiles'], ltt_selection={'1802': 'portfolio0_1802'},
                       mean_spi_selection={'1802': 'baseline'})
    return tuple(copy.deepcopy(value) for value in (plan, calibration, risk, final))


def test_calibration_weights_alone_do_not_authorize_changed_controller_code():
    plan, calibration, risk, final = records()
    assert launcher.validate_identity(plan, calibration, risk, final)['complete_policy_identity_preserved']
    final['source_sha256']['src/gate.py'] = 'new_online_decision_rule'
    with pytest.raises(ValueError, match='source changed'):
        launcher.validate_identity(plan, calibration, risk, final)


def test_overlapping_initials_cannot_be_called_a_final_confirmation():
    plan, calibration, risk, final = records()
    final['spec']['seed_base'] = plan['final_test']['seed_base'] = 105
    with pytest.raises(ValueError, match='overlap'):
        launcher.validate_identity(plan, calibration, risk, final)


def test_statistical_selection_cannot_replace_a_training_replicate():
    plan, calibration, risk, final = records()
    calibration['ltt_selection']['1802'] = 'portfolio0_1812'
    with pytest.raises(ValueError, match='crosses a training replicate'):
        launcher.validate_identity(plan, calibration, risk, final)


def test_changed_final_population_or_model_is_rejected():
    plan, calibration, risk, final = records()
    final['spec']['modes'] = ['clean']
    with pytest.raises(ValueError, match='design changed'):
        launcher.validate_identity(plan, calibration, risk, final)
    plan, calibration, risk, final = records()
    final['models']['1802']['sha256'] = 'new_model'
    with pytest.raises(ValueError, match='model identity'):
        launcher.validate_identity(plan, calibration, risk, final)
