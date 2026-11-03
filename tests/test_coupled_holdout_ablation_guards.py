"""Secondary experiment cannot run against a live primary or changed actors."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_coupled_holdout_ablation as a


@pytest.fixture
def host_runtime_fixture(monkeypatch):
    """Isolate registration identity from host versions,without editing records.

    Production validation still compares the actual frozen candidate runtime.
    These unit tests target case/model identity; the separate mismatch test
    below checks that the production runtime rejection remains effective.
    """
    original=a.e.design.validate_candidate
    runtime=dict(python=a.e.platform.python_version(),numpy=a.e.np.__version__,
                 torch=a.e.torch.__version__,numba=a.e.numba.__version__)
    def candidate():
        value=deepcopy(original());value['runtime']=runtime.copy();return value
    monkeypatch.setattr(a.e.design,'validate_candidate',candidate)


def test_registered_scope_is_all_original_confirmation_initials(host_runtime_fixture):
    p=a.guard()
    assert len(p['cases'])==1024 and p['methods']==['4703','4704','4705']
    assert p['expected_additional_sumo_executions']==3072 and not p['new_error_allocated']
    assert not p['primary_confirmation_started_at_registration']


def test_no_secondary_execution_directory_is_created_while_primary_live(tmp_path,monkeypatch):
    def refuse(require_complete=False):
        assert require_complete is True
        raise FileNotFoundError('Root terminal witness absent')
    monkeypatch.setattr(a,'OUT',tmp_path/'secondary')
    monkeypatch.setattr(a,'bound',refuse)
    with pytest.raises(FileNotFoundError):a.run()
    assert not (a.OUT/'execution').exists()


def test_actor_or_case_change_is_not_a_supplementary_ablation(monkeypatch,host_runtime_fixture):
    p=a.e.read(a.OUT/'plan.json');original=a.e.read
    bad=deepcopy(p);bad['methods']=['4703']
    monkeypatch.setattr(a.e,'read',lambda path:bad if Path(path)==a.OUT/'plan.json' else original(path))
    with pytest.raises(ValueError,match='every original'):a.guard()


def test_runtime_mismatch_still_rejects_registration(monkeypatch,host_runtime_fixture):
    original=a.e.design.validate_candidate
    def mismatch():
        value=original();value['runtime']['python']='deliberate-incompatible-fixture';return value
    monkeypatch.setattr(a.e.design,'validate_candidate',mismatch)
    with pytest.raises(ValueError,match='Frozen actual runtime'):a.guard()
