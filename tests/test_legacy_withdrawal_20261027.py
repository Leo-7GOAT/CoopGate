from copy import deepcopy
from pathlib import Path
import sys
from types import FunctionType
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import legacy_withdrawal_contract_20261027 as contract
import execute_joint_risk_after_withdrawal_20261027 as route


def decision():
    return [contract.read(contract.PROPOSAL),contract.read(contract.JOINT),
        contract.read(contract.OUT/'authorization.json'),contract.read(contract.OUT/'process_stop.json')]


def test_original_schedule_contract_and_all_models_match_approved_proposal():
    args=decision();contract.validate_decision(*args)
    assert args[1]['contract']['previous_attempt_terminal_required'] is True
    assert len(args[1]['stages']['calibration'])==3072
    assert len(args[1]['stages']['confirmation'])==1024
    assert set(args[1]['models'])=={'1802','1812','1822'}


def test_approval_chain_and_both_original_frozen_attempts_are_exact():
    contract.decision_inputs()


@pytest.mark.parametrize('key,value',[
    ('human_approval_received',False),('partial_effects_read',True),
    ('old_alpha_reassigned',True),('new_alpha',.025),('new_sumo_runs',1024)])
def test_reject_unapproved_outcome_based_or_reduced_scope_withdrawal(key,value):
    args=decision();args[2][key]=value
    with pytest.raises(ValueError):contract.validate_decision(*args)


@pytest.mark.parametrize('key',[
    'legacy_completed','legacy_certificate_obtained','data_deleted','unknown_historical_exits_rewritten'])
def test_stop_is_never_a_completed_success_or_rewritten_historical_exit(key):
    args=decision();args[3][key]=True
    with pytest.raises(ValueError):contract.validate_decision(*args)


@pytest.mark.parametrize('section',['stages','models','contract'])
def test_original_scientific_bytes_cannot_be_changed_under_amendment(section):
    args=decision();args[1][section]=deepcopy(args[1][section])
    if section=='stages':args[1][section]['confirmation'][0]['seed']+=1
    elif section=='models':args[1][section]['1802']['sha256']='0'*64
    else:args[1][section]['risk_limit']=.10
    with pytest.raises(ValueError,match='changed'):contract.validate_decision(*args)


def test_only_prerequisite_and_entry_path_change_in_private_namespace():
    original=route.original
    before=original.verify_previous_attempt
    namespace=route.routed_functions()
    for name,function in vars(original).items():
        if isinstance(function,FunctionType) and function.__module__==original.__name__ and name!='verify_previous_attempt':
            assert namespace[name].__code__ is function.__code__
    assert namespace['verify_previous_attempt'] is contract.prerequisite
    assert namespace['SCRIPT']==contract.ROUTE
    assert original.verify_previous_attempt is before
    assert original.SCRIPT=='tools/execute_joint_risk_v42.py'


def test_archive_and_original_byte_tampering_fail_closed(tmp_path,monkeypatch):
    root=tmp_path;out=root/'withdrawal';out.mkdir()
    cal=root/'cal';cal.mkdir();raw=cal/'observations.sqlite';raw.write_bytes(b'original durable bytes')
    archive=out/'part.zip';archive.write_bytes(b'preserved archive')
    inventory=out/'inventory.json'
    monkeypatch.setattr(contract,'ROOT',root);monkeypatch.setattr(contract,'OUT',out)
    monkeypatch.setattr(contract,'CAL',cal);monkeypatch.setattr(contract,'INVENTORY',inventory)
    monkeypatch.setattr(contract,'OPERATIONAL',())
    contract.exclusive(inventory,dict(files=contract.original_files(),archives=[dict(path='withdrawal/part.zip',sha256=contract.digest(archive))],
        every_archived_entry_sha256_verified=True,source_bytes_unchanged_after_archive=True,
        partial_efficacy_outcomes_read=False,legacy_completion_claimed=False,data_deleted=False))
    contract.verify_inventory(full=True)
    raw.write_bytes(b'changed original')
    with pytest.raises(ValueError,match='data bytes changed'):contract.verify_inventory(full=True)
    archive.write_bytes(b'changed archive')
    with pytest.raises(ValueError,match='archive changed'):contract.verify_inventory()
