import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
_spec = importlib.util.spec_from_file_location('resume_frozen_risk_v19', TOOLS / 'resume_frozen_risk_v19.py')
recovery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recovery)


def fixture():
    design = dict(seed_base=61000001, n_initials=8, shards=2, sampling='iid_mixture', modes=['clean', 'uniform'],
                  methods={'baseline': {'kind': 'baseline'}, 'portfolio0_1802': {'kind': 'portfolio', 'max_failed_scenarios': 0}})
    cases = recovery.schedule(design, 0)
    rows = [dict(initial_index=i, disturbance=mode, controller=name, seed=design['seed_base'] + i,
                 scenario=recovery.regime_for(design, i), completed=False, opaque_result=[-100., 3.5]) for i, mode, name in cases]
    return design, rows


def test_schedule_is_a_disjoint_complete_partition_with_frozen_permutations():
    design, rows = fixture()
    zero, one = recovery.schedule(design, 0), recovery.schedule(design, 1)
    assert len(zero) == len(one) == 16 and not set(zero).intersection(one)
    assert set(zero + one) == {(i, mode, name) for i in range(8) for mode in design['modes'] for name in design['methods']}
    expected_first = [(0, 'clean', str(name)) for name in np.random.default_rng(61000001 + 1919).permutation(list(design['methods']))]
    assert zero[:2] == expected_first
    assert recovery.validate_prefix(rows[:7], design, 0) == zero


@pytest.mark.parametrize('mutation', ['drop', 'duplicate', 'reorder', 'seed', 'regime'])
def test_recovery_rejects_censoring_reordering_and_changed_initials(mutation):
    design, rows = fixture()
    part = copy.deepcopy(rows[:7])
    if mutation == 'drop': del part[2]
    elif mutation == 'duplicate': part[3] = copy.deepcopy(part[2])
    elif mutation == 'reorder': part[0], part[1] = part[1], part[0]
    elif mutation == 'seed': part[-1]['seed'] += 1
    else: part[-1]['scenario'] = 'shifted'
    with pytest.raises(ValueError):
        recovery.validate_prefix(part, design, 0)


@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_atomic_append_preserves_the_entire_prefix_including_failed_records(tmp_path, newline):
    design, rows = fixture()
    path = tmp_path / 'episodes.json'
    before = recovery.encoded_rows(rows[:7], newline)
    path.write_bytes(before)
    loaded = json.loads(path.read_bytes())
    loaded.extend(rows[7:])
    recovery.atomic_bytes(path, recovery.encoded_rows(loaded, newline))
    after = json.loads(path.read_bytes())
    assert recovery.encoded_rows(after[:7], newline) == before
    assert all(not row['completed'] for row in after)
    assert not list(tmp_path.glob('*.tmp'))
    assert len(recovery.validate_prefix(after, design, 0)) == len(after)


def test_boundary_timing_changes_are_allowed_but_physical_or_admission_changes_are_not():
    original = dict(controller_ms=[1., 2.], setup_ms=3., per_vehicle_finish=[5., 6., 7.],
                    completed=True, learned_action_steps=2, chosen_compositions=[1, 3])
    actual = copy.deepcopy(original)
    actual['controller_ms'], actual['setup_ms'] = [8., 9.], 100.
    recovery.compare_boundary(actual, original)
    for key, value in [('completed', False), ('learned_action_steps', 1), ('chosen_compositions', [1, 2]),
                       ('per_vehicle_finish', [5., 6., 7.5])]:
        changed = copy.deepcopy(actual)
        changed[key] = value
        with pytest.raises(AssertionError):
            recovery.compare_boundary(changed, original)


def test_failed_checkpoint_replacement_cannot_destroy_the_old_prefix(tmp_path, monkeypatch):
    _, rows = fixture()
    path = tmp_path / 'episodes.json'
    prefix = recovery.encoded_rows(rows[:7], '\r\n')
    path.write_bytes(prefix)

    def fail_replace(source, destination):
        raise OSError('Simulated replacement failure')

    monkeypatch.setattr(recovery.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='replacement failure'):
        recovery.atomic_bytes(path, recovery.encoded_rows(rows, '\r\n'))
    assert path.read_bytes() == prefix
    pending = list(tmp_path.glob('*.tmp'))
    assert len(pending) == 1 and json.loads(pending[0].read_bytes()) == rows
