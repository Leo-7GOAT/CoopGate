import copy
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import audit_ood_validation_v39 as auditor
import design_ood_validation_v39 as design
import execute_ood_validation_v39 as execute
from coopgate.episode_store_v21 import inventory, observations, seal_path
from coopgate.disturbance_v9 import disturbance_tape
from coopgate.merge_v4 import MergeConfig, transition_safe
from coopgate.ood_mismatch_v39 import (
    PerceptionStream,
    compound_substep,
    delayed_command,
    validate_plant,
)


def dummy_models():
    return {model: object() for model in design.MODELS}


def one_group():
    return {
        "block_index": 0,
        "initial_id": "compressed_simultaneous:73900001",
        "seed": 73_900_001,
        "regime": "compressed_simultaneous",
        "plant": design.PLANTS[0],
        "perception": design.PERCEPTIONS[0],
        "plant_realization_key": f"73900001:{design.PLANTS[0]}",
        "perception_realization_key": f"73900001:{design.PERCEPTIONS[0]}",
        "innovation_stream_seed": 173_900_001,
        "method_order": design.ordered_methods(0),
    }


def test_exact_13_method_schedule_identity_and_rotation():
    group = one_group()
    schedule = execute.schedule(group)
    assert len(schedule) == len({tuple(case) for case in schedule}) == 13
    assert {(case[-2], case[-1]) for case in schedule} == {
        (item["rule"], item["model"]) for item in design.method_keys()
    }
    assert all(case[:6] == [0, group["initial_id"], group["seed"], group["regime"],
                            group["plant"], group["perception"]] for case in schedule)
    reversed_group = {**group, "block_index": 1, "method_order": design.ordered_methods(1)}
    assert execute.schedule(reversed_group) != schedule


class FakeGate:
    def __init__(self, model, profile, rule, command=(0.0, 0.0, 0.0)):
        self.args = (model, profile, rule)
        self.command = np.asarray(command, dtype=float)

    def select(self, state, previous):
        return self.command.copy(), {"gate_ms": 9.0, "certified": True}


def test_method_adapter_routes_baseline_and_all_learned_models():
    models = dummy_models()
    made = []

    class Capture(FakeGate):
        def __init__(self, model, profile, rule):
            super().__init__(model, profile, rule)
            made.append(self.args)

    for method in design.method_keys():
        wrapped = execute.route_gate(method, models, gate_cls=Capture)
        command, _ = wrapped.select(np.zeros((3, 2)), np.zeros(3))
        np.testing.assert_array_equal(command, np.zeros(3))
    assert made[0] == (None, "lag_strong", "baseline")
    assert {(rule, model_id) for model, _, rule in made[1:]
            for model_id in design.MODELS if model is models[model_id]} == {
        (rule, model_id) for rule in design.RULES[1:] for model_id in design.MODELS
    }


def test_adapter_records_illegal_return_as_terminal_error_not_success():
    wrapped = execute.TimedGate(FakeGate(None, "lag_strong", "baseline", command=(9.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="Invalid executed command"):
        wrapped.select(np.zeros((3, 2)), np.zeros(3))
    assert len(wrapped.calls) == 1
    assert wrapped.calls[0]["error"]["type"] == "ValueError"
    auditor.check_calls(
        {"aborted": True, "abort_reason": "Invalid executed command or slew"},
        [],
        wrapped.calls,
    )


@pytest.mark.parametrize("name", design.PERCEPTIONS)
def test_wrapper_and_independent_auditor_share_exact_frozen_perception_tape(name):
    seed = 73_900_001
    realization = design.perception_realization(seed, name)
    original = design._perception_tape(seed, name)
    independent = auditor.independent_perception_tape(seed, name)
    assert independent == original
    assert auditor.value_digest(independent) == realization["stream_sha256"]
    stream = PerceptionStream(realization, original)
    history = [np.array([[-100.0, 10.0], [-105.0, 11.0], [-110.0, 9.0]])]
    observed, evidence = stream.observe(history)
    assert observed.shape == (3, 2) and np.isfinite(observed).all()
    assert evidence["observed_sha256"] == auditor.array_digest(observed)
    assert evidence["profile"] == name
    if name == "stale_dropout":
        assert len(evidence["source_decision"]) == 3


def test_compound_plant_has_fifo_delay_and_matches_independent_recurrence():
    spec = design.plant_realization(73_900_001, "delay_gain_lag_heterogeneous")
    plant = validate_plant(spec)
    history = [np.array([1.0, 2.0, 0.5]), np.array([-1.0, 1.5, 0.0])]
    delayed = delayed_command(history, plant["delay"])
    expected_delayed = np.array([
        history[len(history) - 1 - plant["delay"][i]][i]
        if len(history) - 1 - plant["delay"][i] >= 0 else 0.0
        for i in range(3)
    ])
    np.testing.assert_array_equal(delayed, expected_delayed)
    velocity, memory, innovation = np.array([8.0, 9.0, 10.0]), np.zeros(3), np.array([0.1, -0.1, 0.0])
    acceleration, next_memory = compound_substep(delayed, velocity, memory, innovation, plant)
    retention = np.exp(-0.1 / np.asarray(spec["lag_tau_seconds"]))
    independent_memory = retention * memory + (1 - retention) * delayed
    independent_acceleration = (np.asarray(spec["gain"]) * independent_memory
                                - np.asarray(spec["quadratic_drag"]) * velocity**2 + innovation)
    np.testing.assert_allclose(next_memory, independent_memory, atol=0, rtol=0)
    np.testing.assert_allclose(acceleration, independent_acceleration, atol=0, rtol=0)
    assert spec["outside_declared_controller_class"] is True


def test_independent_auditor_reconstructs_synthetic_true_physics_and_clean_observation(monkeypatch):
    group = one_group()
    initial_state = design.initial_state(group["seed"], group["regime"])
    plant = design.plant_realization(group["seed"], group["plant"])
    perception = design.perception_realization(group["seed"], group["perception"])
    plan = {
        "initials": [{"initial_id": group["initial_id"], "seed": group["seed"],
                      "regime": group["regime"], "state": initial_state}],
        "plant_contract": {"realizations": {group["plant_realization_key"]: plant}},
        "perception_contract": {"realizations": {group["perception_realization_key"]: perception}},
    }
    info = {
        "gate_ms": 123.0,
        "in_set": True,
        "certified": True,
        "intervention": 1.0,
        "actual_learning_used": False,
        "baseline_command": [0.0, 0.0, 0.0],
    }

    class ReplayGate:
        def __init__(self, model, profile, rule):
            assert model is None and profile == "lag_strong" and rule == "baseline"

        def select(self, observed, previous):
            return np.zeros(3), copy.deepcopy(info)

    monkeypatch.setattr(auditor, "RecoveryJointGate", ReplayGate)
    state = np.asarray(initial_state, dtype=float)
    before_state = state.copy()
    innovation_tape = disturbance_tape(group["innovation_stream_seed"], "uniform", 0.1, 250)
    gain = np.asarray(plant["gain"])
    drag = np.asarray(plant["quadratic_drag"])
    tau = np.asarray(plant["lag_tau_seconds"])
    retention = np.exp(-0.1 / tau)
    memory = np.zeros(3)
    delayed = np.zeros(3)
    innovations, memories, accelerations, errors = [], [], [], []
    violations = 0
    maximum = 0.0
    exceeded = 0
    for substep in range(5):
        innovation = innovation_tape[substep]
        memory = retention * memory + (1 - retention) * delayed
        acceleration = gain * memory - drag * state[:, 1] ** 2 + innovation
        error = acceleration
        innovations.append(innovation.tolist())
        memories.append(memory.tolist())
        accelerations.append(acceleration.tolist())
        errors.append(error.tolist())
        maximum = max(maximum, float(np.max(np.abs(error))))
        exceeded += int(np.max(np.abs(error)) > 0.5 + 1e-9)
        old = state[:, 0].copy()
        state[:, 1] = np.clip(state[:, 1] + acceleration * 0.1, 0, 20)
        state[:, 0] += state[:, 1] * 0.1
        violations += not bool(transition_safe(old[None], state[:, 0][None], MergeConfig())[0])
    stream = PerceptionStream(perception, [])
    observed, perception_evidence = stream.observe([before_state])
    trace = [{
        "t": 0.5,
        "true_state_before": before_state.tolist(),
        "observed_state": observed.tolist(),
        "perception": perception_evidence,
        "x": state.tolist(),
        "command": [0.0, 0.0, 0.0],
        "delayed_command": [0.0, 0.0, 0.0],
        "innovation": innovations,
        "actuator_states": memories,
        "applied_accelerations": accelerations,
        "disturbance": errors,
        "realized_error_in_nominal_half_radius": bool(np.max(np.abs(errors)) <= 0.5 + 1e-9),
        **info,
    }]
    finish = [25.0] * 3
    cost = [25.0] * 3
    row = {
        "seed": group["seed"],
        "scenario": group["regime"],
        "initial": initial_state,
        "rule": "baseline",
        "model": "none",
        "plant": plant,
        "plant_name": group["plant"],
        "perception_name": group["perception"],
        "perception_stream_sha256": perception["stream_sha256"],
        "innovation_stream_seed": group["innovation_stream_seed"],
        "innovation_tape_sha256": auditor.array_digest(innovation_tape),
        "compound_mismatch_outside_declared_class": True,
        "exact_state_observation_assumption_met": True,
        "physical_certificate_applicable": False,
        "collision": True,
        "aborted": False,
        "abort_reason": "",
        "terminal_attempt": None,
        "completed": False,
        "contract_violations": violations,
        "error_contract_exceeded_substeps": exceeded,
        "max_actual_error": maximum,
        "max_command_slew": 0.0,
        "elapsed": 0.5,
        "control_steps": 1,
        "controller_ms": [123.0],
        "disturbance": "uniform",
        "radius": 0.1,
        "per_vehicle_finish": finish,
        "per_vehicle_variation": [0.0] * 3,
        "per_vehicle_cost": cost,
        "team_cost": 25.0,
        "failure": True,
        "failure_reasons": ["collision"] + (["route_geometry_violation"] if violations else [])
        + ["incomplete_or_timeout"],
        "failure_penalized_loss": 50.0,
        "learned_action_steps": 0,
        "fallback_steps": 1,
        "adopted": False,
    }
    record = {"row": row, "calls": [{"wall_ms": 1.0, "error": None}]}
    review = auditor.audit_episode(
        plan, group, {"rule": "baseline", "model": "none"}, record, trace, None
    )
    assert review["compound_dynamics_replayed"] is True
    assert review["perception_replayed"] is True
    assert review["fallback_identity_exact"] is True
    truncated = copy.deepcopy(record)
    truncated["row"]["collision"] = False
    truncated["row"]["failure_reasons"].remove("collision")
    with pytest.raises(ValueError, match="all 50 decisions"):
        auditor.audit_episode(
            plan, group, {"rule": "baseline", "model": "none"}, truncated, trace, None
        )


def test_one_block_synthetic_smoke_persists_before_audit_and_seals(tmp_path):
    group = one_group()
    directory = tmp_path / "00000"
    count = 0

    def run_fn(plan, actual_group, method, models, env):
        nonlocal count
        count += 1
        record = {
            "row": {"ordinal": count, "aborted": False},
            "calls": [{"wall_ms": 1.0, "error": None}],
            "episode_wall_seconds": 0.01,
        }
        return record, [{"ordinal": count}]

    def audit_fn(plan, actual_group, method, record, trace, model):
        # A separate read-only connection must already see this committed run.
        state = inventory(directory / "observations.sqlite")
        assert len(state["cases"]) == record["row"]["ordinal"]
        return {"persisted_before_audit": True, "fallback_identity_exact": True}

    reviews = execute.collect_block(
        {}, group, directory, dummy_models(), object(), "plan", "binding", run_fn, audit_fn
    )
    assert len(reviews) == count == 13
    assert seal_path(directory / "observations.sqlite").exists()
    assert (directory / "complete.json").exists()
    rows = list(
        observations(
            directory / "observations.sqlite",
            execute.metadata(group, "plan", "binding"),
            execute.schedule(group),
        )
    )
    assert len(rows) == 13 and all(review["persisted_before_audit"] for review in reviews)


def test_failed_post_persistence_audit_leaves_immutable_unsealed_partial_block(tmp_path):
    group = one_group()
    directory = tmp_path / "00000"

    def run_fn(*args):
        return ({"row": {"aborted": False}, "calls": [], "episode_wall_seconds": 0.0}, [])

    def audit_fn(*args):
        assert len(inventory(directory / "observations.sqlite")["cases"]) == 1
        raise ValueError("synthetic independent audit failed")

    with pytest.raises(ValueError, match="synthetic independent audit failed"):
        execute.collect_block(
            {}, group, directory, dummy_models(), object(), "plan", "binding", run_fn, audit_fn
        )
    assert (directory / "observations.sqlite").exists()
    assert not seal_path(directory / "observations.sqlite").exists()
    assert not (directory / "complete.json").exists()
    with pytest.raises(FileExistsError):
        execute.collect_block(
            {}, group, directory, dummy_models(), object(), "plan", "binding", run_fn, audit_fn
        )


def completion_fixtures():
    plan = {"budget": {"actual_sumo_runs": 22_464, "paired_condition_blocks": 1_728}}
    launch = {
        "launcher_pid": 10,
        "native_pid": 11,
        "parent_pid": 10,
        "native_creation_date": "20261024120000.000000+480",
        "command_line": "python worker",
        "watcher_pid": 9,
        "plan_sha256": "plan",
        "execution_binding_sha256": "binding",
    }
    receipt = {**launch, "exit_code": 0}
    receipt["collection_complete_sha256"] = "collection"
    receipt["blocks_manifest_sha256"] = "blocks"
    spawn = {
        "launcher_pid": 10,
        "watcher_pid": 9,
        "command": [sys.executable, "-X", "utf8", execute.SCRIPT, "worker"],
        "expected_runs": 22_464,
        "plan_sha256": "plan",
        "execution_binding_sha256": "binding",
    }
    done = {
        "native_pid": 11,
        "plan_sha256": "plan",
        "execution_binding_sha256": "binding",
        "actual_sumo_runs": 22_464,
        "blocks": 1_728,
        "blocks_manifest_sha256": "blocks",
    }
    return plan, spawn, receipt, launch, done


@pytest.mark.parametrize(
    "target,field,value",
    (
        ("receipt", "exit_code", 1),
        ("receipt", "native_pid", 12),
        ("spawn", "expected_runs", 22_463),
        ("launch", "plan_sha256", "other"),
        ("done", "execution_binding_sha256", "other"),
        ("done", "actual_sumo_runs", 22_463),
        ("done", "blocks", 1_727),
    ),
)
def test_zero_exit_acknowledged_identity_and_exact_coverage_are_mandatory(target, field, value):
    plan, spawn, receipt, launch, done = completion_fixtures()
    execute.validate_completion(
        plan, "plan", "binding", spawn, receipt, launch, done, "collection", "blocks"
    )
    {"spawn": spawn, "receipt": receipt, "launch": launch, "done": done}[target][field] = value
    with pytest.raises(ValueError):
        execute.validate_completion(
            plan, "plan", "binding", spawn, receipt, launch, done, "collection", "blocks"
        )


def test_source_hash_merge_conflicts_fail_closed(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    relative = source.relative_to(tmp_path).as_posix()
    plan = {"source_sha256": {relative: "0" * 64}}
    old_root = execute.ROOT
    execute.ROOT = tmp_path
    try:
        with pytest.raises(ValueError, match="hash conflict"):
            execute.merge_source_hashes(plan, [source])
    finally:
        execute.ROOT = old_root


def test_frozen_binding_ignores_unrelated_future_files_but_keeps_every_bound_dependency():
    plan = {"source_sha256": {"tools/design_ood_validation_v39.py": "a" * 64}}
    sources = {name: "b" * 64 for name in execute.NEW_EXECUTION_SOURCES}
    sources["tools/design_ood_validation_v39.py"] = "a" * 64
    sources["src/coopgate/ood_mismatch_v39.py"] = "c" * 64
    sources["src/coopgate/existing_at_bind.py"] = "d" * 64
    inventory = sorted(name for name in sources if name.startswith("src/coopgate/") and name.endswith(".py"))
    binding = {
        "source_sha256": sources,
        "bound_src_inventory": inventory,
        "source_set_policy": (
            "All coopgate Python sources present at bind are immutable; unrelated files added later are outside this binding."
        ),
    }
    execute.validate_bound_source_set(plan, binding)
    # A later unrelated file is intentionally absent from the immutable binding
    # and therefore does not alter validation of the frozen dependency set.
    changed = copy.deepcopy(binding)
    changed["source_sha256"].pop("src/coopgate/existing_at_bind.py")
    with pytest.raises(ValueError, match="inventory"):
        execute.validate_bound_source_set(plan, changed)
    changed = copy.deepcopy(binding)
    changed["source_sha256"].pop("tools/audit_ood_validation_v39.py")
    with pytest.raises(ValueError, match="required"):
        execute.validate_bound_source_set(plan, changed)


def test_worker_block_manifest_binds_complete_database_and_seal_bytes(tmp_path, monkeypatch):
    out = tmp_path / "study"
    directory = out / "blocks" / "00000"
    directory.mkdir(parents=True)
    (directory / "complete.json").write_text("{}\n", encoding="utf-8")
    (directory / "observations.sqlite").write_bytes(b"database")
    (directory / "observations.sqlite.seal.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(execute, "OUT", out)
    entry = execute.block_manifest_entry(directory, 0)
    plan = {"budget": {"paired_condition_blocks": 1, "actual_sumo_runs": 13}}
    manifest = {
        "plan_sha256": "plan",
        "execution_binding_sha256": "binding",
        "blocks": 1,
        "actual_sumo_runs": 13,
        "entries": [entry],
    }
    execute.validate_blocks_manifest(plan, "plan", "binding", manifest)
    (directory / "observations.sqlite").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="bytes changed"):
        execute.validate_blocks_manifest(plan, "plan", "binding", manifest)


def test_holm_adjustment_is_monotone_and_never_smaller_than_raw_p():
    raw = {"1802": 0.01, "1812": 0.03, "1822": 0.02}
    adjusted = execute._holm_adjust(raw)
    assert all(adjusted[key] >= raw[key] for key in raw)
    ordered = sorted(raw, key=raw.get)
    assert [adjusted[key] for key in ordered] == sorted(adjusted.values())


def primary_fixture(failure_pairs, loss_gains):
    initials = []
    rows = []
    methods = design.method_keys()
    for regime in design.REGIMES:
        for offset in range(64):
            initial_id = f"{regime}:{offset}"
            initials.append({"initial_id": initial_id, "regime": regime})
            for plant in design.PLANTS:
                for perception in design.PERCEPTIONS:
                    for method in methods:
                        if method == {"rule": "baseline", "model": "none"}:
                            failure = failure_pairs[regime][0]
                            loss = 25.0
                        elif method["rule"] == "bank_performance":
                            failure = failure_pairs[regime][1]
                            loss = 25.0 - loss_gains[regime]
                        else:
                            failure = False
                            loss = 25.0
                        rows.append({
                            "initial_id": initial_id,
                            "plant_name": plant,
                            "perception_name": perception,
                            "rule": method["rule"],
                            "model": method["model"],
                            "failure": failure,
                            "failure_penalized_loss": loss,
                        })
    plan = {
        "initials": initials,
        "inference": {"bootstrap_draws": 100, "bootstrap_seed": 39},
    }
    return execute.primary_analysis(plan, rows)


def test_zero_discordant_failures_and_constant_losses_are_untestable_not_certified():
    result = primary_fixture(
        {regime: (False, False) for regime in design.REGIMES},
        {regime: 0.0 for regime in design.REGIMES},
    )
    assert result["all_three_fixed_models_pass"] is False
    for model in design.MODELS:
        summary = result["models"][model]
        assert summary["failure_risk_difference"] == 0.0
        assert summary["failure_nonzero_cluster_contrasts"] == 0
        assert summary["failure_margin_testable"] is False
        assert summary["loss_margin_testable"] is False
        assert summary["failure_holm_adjusted_p"] == 1.0
        assert summary["loss_holm_adjusted_p"] == 1.0
        assert summary["frozen_thresholds_pass"] is False


def test_constant_within_each_regime_is_untestable_even_when_pooled_variance_is_positive():
    failure_pairs = {
        design.REGIMES[0]: (False, True),
        design.REGIMES[1]: (False, False),
        design.REGIMES[2]: (True, False),
    }
    loss_gains = {
        design.REGIMES[0]: 1.0,
        design.REGIMES[1]: 2.0,
        design.REGIMES[2]: 3.0,
    }
    result = primary_fixture(failure_pairs, loss_gains)
    for model in design.MODELS:
        summary = result["models"][model]
        assert summary["failure_cluster_variance"] > 0
        assert summary["loss_cluster_variance"] > 0
        assert sum(summary["failure_within_stratum_variance"].values()) == 0
        assert sum(summary["loss_within_stratum_variance"].values()) == 0
        assert summary["failure_stratified_bootstrap_variance"] < 1e-15
        assert summary["loss_stratified_bootstrap_variance"] < 1e-15
        assert summary["failure_margin_testable"] is False
        assert summary["loss_margin_testable"] is False
        assert summary["frozen_thresholds_pass"] is False
