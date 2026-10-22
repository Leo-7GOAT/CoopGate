from copy import deepcopy
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import audit_precise_deadline_v26 as audit


def sample():
    return dict(case=["clean","1802","normal","coarse"],
        row=dict(disturbance="clean",model="1802",variant="normal",clock="coarse",aborted=False,
                 ego_cost=10.,initial_admitted=True,setup_ms=4.,controller_ms=[1.]),
        trace=[dict(command=[0.,0.,0.],performance_admitted=True,deadline_primary_used=True,
                    fault_fallback=False,gate_ms=1.,primary_process_pid=123)],
        calls=[dict(wall_ms=1.1,error=None)])


def test_pair_ignores_only_clock_telemetry():
    old, new = sample(), sample()
    new["row"].update(clock="precise", setup_ms=90.,controller_ms=[2.])
    new["trace"][0].update(gate_ms=2.,primary_process_pid=456)
    assert audit.compare_pair(old,new)["non_timing_equal"]


@pytest.mark.parametrize("field,value", [("command",[1.,0.,0.]),("performance_admitted",False),
    ("deadline_primary_used",False),("fault_fallback",True),("fault_reason","proposal_deadline")])
def test_pair_keeps_admission_fallback_and_action_differences(field,value):
    old, new = sample(), sample()
    new["trace"][0][field] = value
    result = audit.compare_pair(old,new)
    assert not result["non_timing_equal"] and field in result["trace_differences"][0]["fields"]


def test_pair_keeps_outcome_and_length_differences():
    old, new = sample(), sample()
    new["row"]["ego_cost"] += 1.
    new["trace"].append(deepcopy(new["trace"][0]))
    result = audit.compare_pair(old,new)
    assert result["row_differences"] == ["ego_cost"]
    assert result["trace_differences"] == [dict(step=1,fields=["missing_step"])]


def test_incomplete_case_set_or_changed_identity_is_refused():
    record = sample()
    plan = dict(actual_sumo_runs=1,cases=[record["case"]])
    audit.validate_records([record],plan)
    with pytest.raises(ValueError,match="complete frozen"):
        audit.validate_records([],plan)
    record["row"]["model"] = "1812"
    with pytest.raises(ValueError,match="identity"):
        audit.validate_records([record],plan)


def test_terminal_failure_timing_cannot_be_omitted():
    record = sample()
    record["row"]["aborted"] = True
    with pytest.raises(ValueError,match="terminal failures"):
        audit.validate_records([record],dict(actual_sumo_runs=1,cases=[record["case"]]))


@pytest.mark.parametrize("duration", [-1.,float("nan"),float("inf")])
def test_invalid_callback_times_are_refused(duration):
    record = sample()
    record["calls"][0]["wall_ms"] = duration
    with pytest.raises(ValueError,match="callback time"):
        audit.validate_records([record],dict(actual_sumo_runs=1,cases=[record["case"]]))
