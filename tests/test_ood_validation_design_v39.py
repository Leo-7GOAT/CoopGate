import copy
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import design_ood_validation_v39 as design


def models():
    return {
        model: {
            "path": f"models/{model}.pt",
            "sha256": model.zfill(64),
            "ego_seed": int(model) - 901,
            "training_seed": int(model),
            "updates": 6,
            "target_updates": 6,
        }
        for model in design.MODELS
    }


@pytest.fixture(scope="module")
def plan():
    return design.build_plan(models())


def test_complete_independent_paired_budget_and_determinism(plan):
    assert plan == design.build_plan(models())
    assert plan["budget"] == {
        "independent_initials": 192,
        "factor_cells_per_initial": 9,
        "paired_condition_blocks": 1728,
        "methods_per_block": 13,
        "primary_runs": 6912,
        "secondary_comparator_runs": 15552,
        "actual_sumo_runs": 22464,
        "all_runs_required_before_any_efficacy_summary": True,
    }
    seeds = [item["seed"] for item in plan["initials"]]
    assert len(seeds) == len(set(seeds)) == 192
    assert min(seeds) == 73_900_001 and max(seeds) == 73_902_064
    for regime in design.REGIMES:
        records = [item for item in plan["initials"] if item["regime"] == regime]
        assert len(records) == 64
        assert [item["seed"] for item in records] == list(
            range(design.SEED_BASES[regime], design.SEED_BASES[regime] + 64)
        )


def test_all_ood_factors_and_methods_are_fully_crossed(plan):
    expected_methods = {(item["rule"], item["model"]) for item in design.method_keys()}
    assert expected_methods == {("baseline", "none")} | {
        (rule, model) for rule in design.RULES[1:] for model in design.MODELS
    }
    cells = {}
    per_initial = {}
    execution_keys = set()
    for group in plan["groups"]:
        cell = (group["regime"], group["plant"], group["perception"])
        cells[cell] = cells.get(cell, 0) + 1
        per_initial[group["initial_id"]] = per_initial.get(group["initial_id"], 0) + 1
        assert {(item["rule"], item["model"]) for item in group["method_order"]} == expected_methods
        for method in group["method_order"]:
            identity = (group["initial_id"], group["plant"], group["perception"],
                        method["rule"], method["model"])
            assert identity not in execution_keys
            execution_keys.add(identity)
    assert set(cells) == set((r, p, o) for r in design.REGIMES for p in design.PLANTS for o in design.PERCEPTIONS)
    assert set(cells.values()) == {64}
    assert set(per_initial.values()) == {9}
    assert len(execution_keys) == 22_464


def test_initials_are_outside_declared_training_axes_without_seed_rejection(plan):
    by_regime = {regime: [] for regime in design.REGIMES}
    for item in plan["initials"]:
        state = item["state"]
        assert state == design.initial_state(item["seed"], item["regime"])
        assert state[0][0] - state[2][0] >= design.SEPARATION
        by_regime[item["regime"]].append(state)
    assert all(8.5 <= state[0][0] - state[2][0] <= 12.0 for state in by_regime["compressed_simultaneous"])
    assert all((-25 - state[1][0]) / state[1][1] < (-25 - state[0][0]) / state[0][1]
               for state in by_regime["cross_route_priority_flip"])
    assert all(state[2][1] - state[0][1] >= 9.0 for state in by_regime["rear_end_merge_wave"])


def test_all_frozen_initials_initialize_the_unchanged_common_baseline(plan):
    assert design.verify_initial_baseline_domain(plan) == {
        "kind": "deterministic_design_feasibility_not_rollout_efficacy",
        "controller_profile": "lag_strong",
        "checked_initials": 192,
        "failures": 0,
        "seed_rejection_or_replacement": False,
    }


def test_every_plant_is_a_true_compound_out_of_class_mismatch(plan):
    seen = set()
    for item in plan["plant_contract"]["realizations"].values():
        seen.add(item["name"])
        assert all(delay >= 1 for delay in item["command_delay_decisions"])
        assert all(abs(gain - 1.0) >= 0.035 for gain in item["gain"])
        assert all(drag > 0 for drag in item["quadratic_drag"])
        assert all(tau > 0 for tau in item["lag_tau_seconds"])
        assert item["outside_declared_controller_class"] is True
        assert item["controller_declared_profile"] == "lag_strong"
    assert seen == set(design.PLANTS)
    assert "inherit no V34/V35 physical certificate" in plan["plant_contract"]["guarantee_scope"]


def test_perception_truth_firewall_and_nontrivial_disturbances(plan):
    realizations = plan["perception_contract"]["realizations"]
    for key, item in realizations.items():
        assert item["shared_by_all_methods"] is True
        assert item["truth_reserved_for_physics_and_post_run_audit"] is True
        assert len(item["stream_sha256"]) == 64
        if key.endswith(":bounded_bias_jitter"):
            assert any(abs(value) >= 0.5 for value in item["static_position_bias_m"])
            assert item["position_jitter_bound_m"] == 0.25
        if key.endswith(":stale_dropout"):
            assert item["observation_delay_decisions"] == 1
            assert item["realized_dropout_count"] > 0
            assert item["maximum_consecutive_dropouts"] == 2
    assert "unavailable to all controllers" in plan["perception_contract"]["truth_firewall"]


def test_estimands_failure_accounting_scope_and_inference_are_locked(plan):
    assert "not a full external MPS reproduction" in plan["comparator_scope"]["mps_shared"]
    assert "not a full external MAMPS reproduction" in plan["comparator_scope"]["mamps_shared"]
    assert plan["primary_rule"] == "bank_performance"
    assert set(plan["primary_estimands"]) == {
        "population", "methods", "failure_risk_difference",
        "failure_penalized_cost_gain_seconds", "interpretation",
    }
    assert plan["inference"]["independent_unit"].startswith("initial_id")
    assert plan["inference"]["bootstrap_draws"] == 20_000
    assert plan["inference"]["no_optional_stopping"] is True
    assert plan["episode_failure"]["failed_runs_retained"] is True
    assert plan["episode_failure"]["no_complete_case_primary_analysis"] is True
    assert plan["validity_and_acceptance"]["failed_acceptance_is_reported_not_repaired"] is True
    assert plan["scope"]["does_not_establish"]
    assert plan["active_or_partial_efficacy_outcomes_read"] is False
    assert plan["outcome_input_paths"] == []


@pytest.mark.parametrize(
    "mutation",
    ("drop_initial", "drop_group", "duplicate_group", "drop_method", "alter_order",
     "alter_plant", "alter_perception", "outcome_read", "budget"),
)
def test_validation_rejects_scope_or_schedule_tampering(plan, mutation):
    changed = copy.deepcopy(plan)
    if mutation == "drop_initial":
        changed["initials"].pop()
    elif mutation == "drop_group":
        changed["groups"].pop()
    elif mutation == "duplicate_group":
        changed["groups"][-1] = copy.deepcopy(changed["groups"][0])
    elif mutation == "drop_method":
        changed["groups"][0]["method_order"].pop()
    elif mutation == "alter_order":
        changed["groups"][0]["method_order"].reverse()
    elif mutation == "alter_plant":
        next(iter(changed["plant_contract"]["realizations"].values()))["gain"][0] = 1.0
    elif mutation == "alter_perception":
        next(iter(changed["perception_contract"]["realizations"].values()))["stream_sha256"] = "0" * 64
    elif mutation == "outcome_read":
        changed["active_or_partial_efficacy_outcomes_read"] = True
    elif mutation == "budget":
        changed["budget"]["actual_sumo_runs"] -= 1
    with pytest.raises(ValueError):
        design.validate_plan(changed)


def test_real_frozen_model_manifest_is_the_only_design_artifact_input():
    loaded = design.load_models()
    assert tuple(loaded) == design.MODELS
    for model, item in loaded.items():
        assert item["training_seed"] == int(model)
        assert design.digest(design.ROOT / item["path"]) == item["sha256"]
    assert design.MODEL_PLAN.as_posix().endswith("artifacts/v32/online_actuator_preflight/plan.json")

