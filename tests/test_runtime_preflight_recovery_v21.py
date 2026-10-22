from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from coopgate.episode_store_v21 import EpisodeStore
from probe_disturbed_gate_v19 import digest
from recover_runtime_preflight_v21 import completed_pair_prefix, read_prefix, role_schedule


def test_mixed_role_prefix_preserves_the_next_exact_complete_case():
    cases = [(0, 'clean', 'a'), (0, 'clean', 'b'), (0, 'uniform', 'a')]
    present = role_schedule(cases[:1])
    assert completed_pair_prefix(cases, present) == (cases[1:], cases[0])
    assert completed_pair_prefix(cases, role_schedule(cases)) == ([], None)


@pytest.mark.parametrize('mutation', ['hole', 'duplicate', 'partial', 'relabel'])
def test_a_partial_or_modified_prefix_cannot_be_silently_resumed(mutation):
    cases = [(0, 'clean', 'a'), (0, 'clean', 'b')]
    roles = role_schedule(cases)
    bad = {'hole':roles[3:], 'duplicate':roles[:3]*2, 'partial':roles[:2],
           'relabel':[(0, 'clean', 'changed', r[-1]) for r in roles[:3]]}[mutation]
    with pytest.raises(ValueError):
        completed_pair_prefix(cases, bad)


def test_explicit_recovery_reads_unsealed_commits_without_modifying_them(tmp_path):
    path = tmp_path / 'interrupted.sqlite'
    metadata, case = {'role':'prior'}, (0, 'clean', 'a', 'collection')
    store = EpisodeStore(path, metadata)
    store.append(case, {'cost':3.5}, [{'x':[1, 2]}])
    store.close()
    checksum = digest(path)
    assert read_prefix(path, metadata, [case], checksum) == [(case, {'cost':3.5}, [{'x':[1, 2]}])]
    assert digest(path) == checksum
    with pytest.raises(ValueError):
        read_prefix(path, {'role':'other'}, [case], checksum)
    with pytest.raises(ValueError):
        read_prefix(path, metadata, [], checksum)
    with pytest.raises(ValueError):
        read_prefix(path, metadata, [case], '0'*64)
