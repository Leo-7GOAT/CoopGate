"""Host-isolated identity tests do not bypass production runtime rejection."""
from copy import deepcopy

import pytest
import execute_prospective_v23 as execution


def test_historical_scientific_load_still_rejects_incompatible_runtime(monkeypatch):
    frozen=execution.read(execution.OUTPUTS['prospective']/'frozen/plan.json')
    incompatible=deepcopy(frozen['runtime'])
    incompatible['python']='deliberate-incompatible-unit-fixture'
    monkeypatch.setattr(execution,'runtime_identity',lambda:incompatible)
    with pytest.raises(ValueError,match='Frozen Python/package/SUMO runtime changed'):
        execution.load('prospective')
