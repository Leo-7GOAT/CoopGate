from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prospective_runtime_v21 import method_definitions,validate_common_family
from preflight_prospective_runtime_v21 import cases


@pytest.mark.parametrize('family',['phase05','phase10','repair'])
def test_every_common_family_keeps_all_three_models_and_both_comparators(family):
    methods=method_definitions(family)
    assert len(methods)==10
    assert set(methods)=={'baseline',*[f'{prefix}_{seed}' for seed in (1802,1812,1822) for prefix in ('raw','nominal',family)]}
    protocol=dict(spec=dict(methods=methods))
    validate_common_family(protocol,family)
    altered=deepcopy(protocol)
    altered['spec']['methods'].pop(f'raw_1812')
    with pytest.raises(ValueError):validate_common_family(altered,family)


def test_integration_superset_covers_every_family_without_fresh_risk_samples():
    methods=method_definitions()
    assert len(methods)==16
    plan=dict(protocol=dict(spec=dict(methods=methods,modes=['clean','uniform','bias','pulse'])))
    assigned=[case for shard in range(3) for case in cases(plan,shard)]
    assert len(assigned)==len(set(assigned))==64
    assert set(assigned)=={(0,mode,name) for mode in plan['protocol']['spec']['modes'] for name in methods}
    assert sum(2+(case[1]=='clean') for case in assigned)==144


def test_unknown_or_mixed_gate_family_is_rejected():
    with pytest.raises(ValueError):method_definitions('changed_after_calibration')
    protocol=dict(spec=dict(methods=method_definitions('phase05')))
    with pytest.raises(ValueError):validate_common_family(protocol,'phase10')
