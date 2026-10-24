"""Freeze-before-run V39 design for OOD interactions and compound mismatch.

This module is deliberately a plan builder, not an experiment launcher.  It
reads only the committed V32 model manifest and fixed model checkpoints.  It
does not inspect V39 outputs or any running/partial efficacy observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v39/ood_mismatch_validation"
MODEL_PLAN = ROOT / "artifacts/v32/online_actuator_preflight/plan.json"

VERSION = "v39_ood_compound_mismatch_1"
MODELS = ("1802", "1812", "1822")
RULES = (
    "baseline",
    "mps_shared",
    "mamps_shared",
    "nominal_performance",
    "bank_performance",
)
PRIMARY_RULE = "bank_performance"
REGIMES = (
    "compressed_simultaneous",
    "cross_route_priority_flip",
    "rear_end_merge_wave",
)
PLANTS = (
    "delay_gain_lag_mild",
    "delay_gain_lag_heterogeneous",
    "delay_gain_lag_severe",
)
PERCEPTIONS = (
    "clean_reference",
    "bounded_bias_jitter",
    "stale_dropout",
)
SEED_BASES = {
    "compressed_simultaneous": 73_900_001,
    "cross_route_priority_flip": 73_901_001,
    "rear_end_merge_wave": 73_902_001,
}
INITIALS_PER_REGIME = 64
DECISION_DT = 0.5
PHYSICS_DT = 0.1
HORIZON = 25.0
DECISIONS = int(HORIZON / DECISION_DT)
SEPARATION = 6.8
BOOTSTRAP_SEED = 39_003_939
BOOTSTRAP_DRAWS = 20_000

SOURCES = (
    "pyproject.toml",
    "src/coopgate/merge_v4.py",
    "src/coopgate/disturbance_v9.py",
    "src/coopgate/shifted_sumo_v19.py",
    "src/coopgate/matched_shift_v21.py",
    "src/coopgate/cooperative_iteration_v16.py",
    "src/coopgate/actuator_recovery_v34.py",
    "src/coopgate/recovery_forecast_v35.py",
    "src/coopgate/recovery_joint_gate_v35.py",
    "tools/experiment_ego_v13.py",
    "tools/study_recovery_gate_v35.py",
    "tools/design_ood_validation_v39.py",
    "tests/test_ood_validation_design_v39.py",
    "docs/v39_ood_validation_protocol.md",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _unit(seed: int, stream: str, index: int) -> float:
    """Stable U(0,1) design value, independent of NumPy RNG versions."""
    raw = f"{VERSION}|{seed}|{stream}|{index}".encode("ascii")
    integer = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return (integer + 0.5) / 2**64


def _uniform(seed: int, stream: str, index: int, low: float, high: float) -> float:
    return low + (high - low) * _unit(seed, stream, index)


def _signed(seed: int, stream: str, index: int, low: float, high: float) -> float:
    magnitude = _uniform(seed, stream + ":magnitude", index, low, high)
    return magnitude if _unit(seed, stream + ":sign", index) >= 0.5 else -magnitude


def initial_state(seed: int, regime: str) -> list[list[float]]:
    """Exact three-actor OOD initial state in route coordinates [position, speed]."""
    if regime == "compressed_simultaneous":
        # Keep every frozen state inside the unchanged recovery baseline's
        # initial domain while shifting density and arrival structure.
        v0 = _uniform(seed, regime, 0, 9.0, 11.0)
        v1 = _uniform(seed, regime, 1, 9.0, 11.0)
        v2 = _uniform(seed, regime, 2, 8.0, 10.0)
        p0 = -_uniform(seed, regime, 3, 100.0, 115.0)
        p2 = p0 - _uniform(seed, regime, 4, 8.5, 12.0)
        arrival0 = (-25.0 - p0) / v0
        arrival1 = arrival0 + _uniform(seed, regime, 5, -0.12, 0.12)
        p1 = -25.0 - v1 * arrival1
    elif regime == "cross_route_priority_flip":
        # The cross-route actor is deliberately first; the same-route pair is
        # closer than the 30--45 m training support.
        v0 = _uniform(seed, regime, 0, 8.0, 10.5)
        v1 = _uniform(seed, regime, 1, 15.5, 18.5)
        v2 = _uniform(seed, regime, 2, 9.0, 12.0)
        p0 = -_uniform(seed, regime, 3, 88.0, 101.0)
        p1 = -_uniform(seed, regime, 4, 47.0, 56.0)
        p2 = p0 - _uniform(seed, regime, 5, 18.0, 25.0)
    elif regime == "rear_end_merge_wave":
        # A fast follower closes on a slow lead while cross-route traffic reaches
        # the merge; the closing-speed structure is absent from training.
        v0 = _uniform(seed, regime, 0, 4.5, 7.0)
        v1 = _uniform(seed, regime, 1, 12.0, 15.0)
        v2 = _uniform(seed, regime, 2, 16.0, 19.0)
        p0 = -_uniform(seed, regime, 3, 52.0, 60.0)
        p1 = -_uniform(seed, regime, 4, 69.0, 78.0)
        p2 = p0 - _uniform(seed, regime, 5, 50.0, 60.0)
    else:
        raise ValueError("Unknown V39 OOD regime")
    state = [[p0, v0], [p1, v1], [p2, v2]]
    if not all(math.isfinite(x) for row in state for x in row):
        raise ValueError("Non-finite generated initial state")
    if not all(-150.0 < row[0] < -25.0 and 0.0 < row[1] < 20.0 for row in state):
        raise ValueError("Generated initial outside fixed-route insertion bounds")
    if state[0][0] - state[2][0] < SEPARATION:
        raise ValueError("Generated same-route pair violates initial separation")
    return state


def plant_realization(seed: int, name: str) -> dict[str, Any]:
    """Exact delayed first-order actuator parameters for one paired initial."""
    if name == "delay_gain_lag_mild":
        delays = [1, 1, 1]
        magnitude = (0.035, 0.065)
        drag_range = (0.00010, 0.00030)
        tau_range = (0.12, 0.24)
    elif name == "delay_gain_lag_heterogeneous":
        base = [1, 2, 1]
        shift = seed % 3
        delays = base[shift:] + base[:shift]
        magnitude = (0.065, 0.12)
        drag_range = (0.00025, 0.00065)
        tau_range = (0.30, 0.65)
    elif name == "delay_gain_lag_severe":
        base = [2, 3, 2]
        shift = seed % 3
        delays = base[shift:] + base[:shift]
        magnitude = (0.12, 0.22)
        drag_range = (0.00055, 0.00100)
        tau_range = (0.75, 1.25)
    else:
        raise ValueError("Unknown V39 compound plant")
    signs = (-1.0, 1.0, -1.0) if seed % 2 else (1.0, -1.0, 1.0)
    gain = [
        1.0 + signs[i] * _uniform(seed, name + ":gain", i, *magnitude)
        for i in range(3)
    ]
    drag = [_uniform(seed, name + ":drag", i, *drag_range) for i in range(3)]
    tau = [_uniform(seed, name + ":tau", i, *tau_range) for i in range(3)]
    result = {
        "name": name,
        "command_delay_decisions": delays,
        "gain": gain,
        "quadratic_drag": drag,
        "lag_tau_seconds": tau,
        "external_innovation_radius": 0.1,
        "controller_declared_profile": "lag_strong",
        "outside_declared_controller_class": True,
        "violated_assumptions": [
            "pure_command_delay_not_in_existing_model",
            "gain_and_drag_not_in_lag_strong_class",
            "exact_state_observation_not_available_under_two_perception_profiles",
        ],
    }
    if (
        not all(delay >= 1 for delay in delays)
        or not all(abs(value - 1.0) > 0.0 for value in gain)
        or not all(value > 0.0 for value in drag)
        or not all(value > 0.0 for value in tau)
    ):
        raise ValueError("Every V39 plant must combine delay, gain, drag and lag")
    return result


def _perception_tape(seed: int, name: str) -> list[Any]:
    if name == "clean_reference":
        return []
    if name == "bounded_bias_jitter":
        return [
            [
                [
                    _signed(seed, name + ":position_jitter", step * 3 + actor, 0.0, 0.25),
                    _signed(seed, name + ":speed_jitter", step * 3 + actor, 0.0, 0.15),
                ]
                for actor in range(3)
            ]
            for step in range(DECISIONS)
        ]
    if name == "stale_dropout":
        holds = [0, 0, 0]
        tape: list[list[bool]] = []
        for step in range(DECISIONS):
            row = []
            for actor in range(3):
                dropped = _unit(seed, name + ":dropout", step * 3 + actor) < 0.20
                if holds[actor] >= 2:
                    dropped = False
                holds[actor] = holds[actor] + 1 if dropped else 0
                row.append(dropped)
            tape.append(row)
        return tape
    raise ValueError("Unknown V39 perception profile")


def perception_realization(seed: int, name: str) -> dict[str, Any]:
    tape = _perception_tape(seed, name)
    result: dict[str, Any] = {
        "name": name,
        "shared_by_all_methods": True,
        "truth_reserved_for_physics_and_post_run_audit": True,
        "stream_sha256": value_digest(tape),
        "decisions": DECISIONS,
    }
    if name == "clean_reference":
        result.update(
            observation_delay_decisions=0,
            static_position_bias_m=[0.0] * 3,
            static_speed_bias_mps=[0.0] * 3,
            dropout_probability=0.0,
        )
    elif name == "bounded_bias_jitter":
        result.update(
            observation_delay_decisions=0,
            static_position_bias_m=[
                _signed(seed, name + ":position_bias", i, 0.50, 1.50) for i in range(3)
            ],
            static_speed_bias_mps=[
                _signed(seed, name + ":speed_bias", i, 0.20, 0.60) for i in range(3)
            ],
            position_jitter_bound_m=0.25,
            speed_jitter_bound_mps=0.15,
            dropout_probability=0.0,
        )
    elif name == "stale_dropout":
        result.update(
            observation_delay_decisions=1,
            initial_observation_rule="repeat_exact_initial_measurement_until_delayed_sample_exists",
            dropout_probability=0.20,
            maximum_consecutive_dropouts=2,
            realized_dropout_count=sum(sum(row) for row in tape),
            dropout_rule="hold_last_delivered_delayed_measurement_per_actor",
        )
    else:
        raise ValueError("Unknown V39 perception profile")
    return result


def method_keys() -> list[dict[str, str]]:
    result = [{"rule": "baseline", "model": "none"}]
    result.extend(
        {"rule": rule, "model": model}
        for model in MODELS
        for rule in RULES[1:]
    )
    return result


def ordered_methods(block_index: int) -> list[dict[str, str]]:
    methods = method_keys()
    offset = block_index % len(methods)
    methods = methods[offset:] + methods[:offset]
    if block_index % 2:
        methods = methods[::-1]
    return methods


def load_models() -> dict[str, dict[str, Any]]:
    """Read only the frozen model manifest; never read an efficacy result."""
    manifest = read_json(MODEL_PLAN)
    models = manifest.get("models")
    if not isinstance(models, dict) or tuple(models) != MODELS:
        raise ValueError("The three fixed V18/V32 model sets are required in frozen order")
    allowed = ("path", "sha256", "ego_seed", "training_seed", "updates", "target_updates")
    result = {model: {key: models[model][key] for key in allowed} for model in MODELS}
    for model, item in result.items():
        if item["training_seed"] != int(model) or digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Frozen learned model changed: " + model)
    return result


def _initial_records() -> list[dict[str, Any]]:
    records = []
    for regime in REGIMES:
        for offset in range(INITIALS_PER_REGIME):
            seed = SEED_BASES[regime] + offset
            records.append(
                {
                    "initial_id": f"{regime}:{seed}",
                    "seed": seed,
                    "regime": regime,
                    "state": initial_state(seed, regime),
                }
            )
    return records


def build_plan(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build and validate the complete result-independent V39 schedule."""
    if tuple(models) != MODELS:
        raise ValueError("All fixed models are required in canonical order")
    initials = _initial_records()
    plants: dict[str, dict[str, Any]] = {}
    perceptions: dict[str, dict[str, Any]] = {}
    groups = []
    for initial in initials:
        seed = initial["seed"]
        for plant in PLANTS:
            plants[f"{seed}:{plant}"] = plant_realization(seed, plant)
        for perception in PERCEPTIONS:
            perceptions[f"{seed}:{perception}"] = perception_realization(seed, perception)
        for plant in PLANTS:
            for perception in PERCEPTIONS:
                block_index = len(groups)
                groups.append(
                    {
                        "block_index": block_index,
                        "initial_id": initial["initial_id"],
                        "seed": seed,
                        "regime": initial["regime"],
                        "plant": plant,
                        "perception": perception,
                        "innovation_stream_seed": seed + 100_000_000,
                        "plant_realization_key": f"{seed}:{plant}",
                        "perception_realization_key": f"{seed}:{perception}",
                        "method_order": ordered_methods(block_index),
                    }
                )
    plan: dict[str, Any] = {
        "protocol_version": VERSION,
        "purpose": (
            "Prospective paired SUMO validation of the unchanged recovery gate under "
            "out-of-training-support interactions, compound actuator mismatch and perception error."
        ),
        "design_only": True,
        "launch_implemented": False,
        "must_freeze_before_execution": True,
        "active_or_partial_efficacy_outcomes_read": False,
        "outcome_input_paths": [],
        "permitted_design_input": "artifacts/v32/online_actuator_preflight/plan.json model manifest only",
        "models": json.loads(json.dumps(models)),
        "fixed_rules": list(RULES),
        "comparator_scope": {
            "baseline": "unchanged common recovery controller",
            "mps_shared": "V35 behavior-matched MPS adapter, not a full external MPS reproduction",
            "mamps_shared": "V35 behavior-matched MAMPS adapter, not a full external MAMPS reproduction",
            "nominal_performance": "existing unbuffered confidence-gate mechanism control",
            "bank_performance": "fixed proposed buffered recovery gate and sole primary learned method",
        },
        "primary_rule": PRIMARY_RULE,
        "seed_design": {
            "sampling": "three disjoint deterministic hash streams fixed before execution",
            "ranges": {
                regime: [SEED_BASES[regime], SEED_BASES[regime] + INITIALS_PER_REGIME - 1]
                for regime in REGIMES
            },
            "independent_initials_per_regime": INITIALS_PER_REGIME,
            "independent_initials_total": len(initials),
            "pairing_unit": "initial_id",
            "pairing": "Each initial is crossed with all 3 plants, all 3 perceptions and all 13 methods.",
            "no_seed_rejection_or_replacement": True,
        },
        "initials": initials,
        "scenario_contract": {
            "sumo_topology": "existing fixed-route three-actor two-to-one merge; branches [0,1,0]",
            "training_support_reference": (
                "Existing nominal/dense training used speeds 8--17 m/s, positions -70--110 m, "
                "and a 30--45 m same-route lead gap."
            ),
            "ood_axes": {
                "compressed_simultaneous": "8.5--12 m same-route gap and nearly simultaneous merge arrival",
                "cross_route_priority_flip": "cross-route actor deterministically reaches the conflict first",
                "rear_end_merge_wave": "large follower/leader closing-speed interaction concurrent with merge conflict",
            },
            "preflight_rule": (
                "Every frozen state must insert exactly and initialize the common baseline. Any failure "
                "invalidates the protocol; the seed may not be replaced."
            ),
        },
        "plant_contract": {
            "profiles": list(PLANTS),
            "realizations": plants,
            "substep_equation": (
                "q[k]=FIFO(command,delay); m[k+1]=exp(-dt/tau)*m[k]+"
                "(1-exp(-dt/tau))*q[k]; a[k]=gain*m[k]-drag*v[k]^2+innovation[k]"
            ),
            "physics_dt_seconds": PHYSICS_DT,
            "decision_dt_seconds": DECISION_DT,
            "initial_fifo_and_actuator_memory": "all zeros",
            "innovation": "same bounded uniform radius-0.1 tape for every paired method and factor cell of an initial",
            "guarantee_scope": (
                "All three compound plants violate the unchanged gate's declared lag_strong class. "
                "Results are empirical robustness evidence and inherit no V34/V35 physical certificate."
            ),
        },
        "perception_contract": {
            "profiles": list(PERCEPTIONS),
            "realizations": perceptions,
            "fairness": (
                "Within a condition, baseline and every learned gate receive the identical perturbed "
                "position/speed stream and exact executed previous command."
            ),
            "truth_firewall": (
                "True SUMO state is used only by physics, collision/geometry checks and post-run audit; "
                "it is unavailable to all controllers."
            ),
            "certificate_scope": (
                "The existing exact-state stopping certificate is not claimed under biased, stale or "
                "dropped observations."
            ),
        },
        "groups": groups,
        "budget": {
            "independent_initials": len(initials),
            "factor_cells_per_initial": len(PLANTS) * len(PERCEPTIONS),
            "paired_condition_blocks": len(groups),
            "methods_per_block": len(method_keys()),
            "primary_runs": len(groups) * (1 + len(MODELS)),
            "secondary_comparator_runs": len(groups) * len(MODELS) * 3,
            "actual_sumo_runs": len(groups) * len(method_keys()),
            "all_runs_required_before_any_efficacy_summary": True,
        },
        "primary_estimands": {
            "population": (
                "Equal-weight mixture of the three frozen OOD regime generators; within each initial, "
                "equal weight over the nine plant/perception cells."
            ),
            "methods": "bank_performance for each fixed model versus the one paired recovery baseline",
            "failure_risk_difference": (
                "theta_F(model)=E[F_bank(model)-F_baseline], paired and aggregated first within initial_id"
            ),
            "failure_penalized_cost_gain_seconds": (
                "theta_L(model)=E[L_baseline-L_bank(model)], L=mean_i(min(finish_i,25)+"
                "0.05*variation_i)+25*F"
            ),
            "interpretation": (
                "Positive theta_L favors the gate; nonpositive theta_F favors the gate. "
                "These are deployment-mixture estimands, not a conditional-on-completion analysis."
            ),
        },
        "secondary_estimands": [
            "completion and failure components by regime, plant and perception cell",
            "adoption rate, fallback fraction, switch count and longest consecutive learned segment",
            "descriptive conditional harm among adopted episodes with denominator reported",
            "per-vehicle capped finish, command variation, jerk and route-geometry violations",
            "controller call p50/p95/max, calls above 500 ms and total controller time",
            "effect modification relative to clean perception and mild compound mismatch",
            "exact fallback-command identity against the common recovery baseline",
        ],
        "inference": {
            "independent_unit": "initial_id; the 9 factor cells and 13 methods are repeated paired measures",
            "aggregation": "first average paired contrasts within initial, then equal-weight the three regimes",
            "intervals": "stratified paired cluster bootstrap resampling initials within regime",
            "primary_tests": (
                "One-sided centered cluster-bootstrap margin tests at failure difference 0.01 and "
                "failure-penalized cost gain 0, with Holm familywise adjustment across three models."
            ),
            "displayed_intervals": "Unadjusted one-sided percentile intervals, descriptive only",
            "degenerate_rule": (
                "Zero within-stratum/stratified-bootstrap variance, zero cluster variance, or no nonzero "
                "failure contrast is untestable and fails closed; "
                "zero observed discordance is not a rare-event population certificate."
            ),
            "finite_sample_population_risk_guarantee": False,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "multiplicity": "Holm familywise alpha 0.05 across the three fixed models, separately per primary estimand",
            "no_pseudoreplication": "22,464 SUMO runs are not treated as independent samples",
            "no_optional_stopping": True,
        },
        "episode_failure": {
            "indicator_is_one_if_any": [
                "SUMO collision report",
                "route-geometry transition violation",
                "controller exception or abort",
                "timeout or any vehicle incomplete at the 25 s horizon",
                "non-finite command, state, trace or registered metric",
                "executed command violates fixed acceleration or slew limits",
            ],
            "failed_runs_retained": True,
            "failed_run_cost_rule": "Use the prespecified capped finish/variation plus 25 s failure penalty.",
            "no_complete_case_primary_analysis": True,
        },
        "validity_and_acceptance": {
            "study_invalid_if_any": [
                "missing/duplicate run or incomplete exact 22,464-run coverage",
                "source, model, plan, tape, pairing, replay or receipt hash mismatch",
                "controller access to true rather than frozen perturbed observation",
                "initial state cannot be inserted exactly or common baseline cannot initialize",
                "method-specific physical innovation or perception tape",
            ],
            "fallback_integrity_required": (
                "Every recorded fallback decision must exactly equal an independently replayed common "
                "recovery-baseline command under the same observation and command history."
            ),
            "claim_passes_only_if": [
                "all validity and fallback-integrity checks pass",
                "for every fixed model, the nondegenerate one-sided theta_F < 0.01 margin test passes after Holm adjustment",
                "for every fixed model, the nondegenerate one-sided theta_L > 0 margin test passes after Holm adjustment",
            ],
            "failed_acceptance_is_reported_not_repaired": True,
        },
        "scope": {
            "applies_to": (
                "Three longitudinal actors on the existing fixed-route SUMO merge, the three frozen "
                "learned model sets, the unchanged V35 rules and exactly the declared generators."
            ),
            "does_not_establish": [
                "lane-changing, vision, arbitrary traffic density or real-vehicle validity",
                "formal safety under compound mismatch or perception error",
                "population risk certification or superiority outside the frozen mixture",
                "a new MARL training algorithm or equilibrium guarantee",
            ],
        },
        "execution_contract": {
            "implementation_status": "A separate additive executor and independent auditor are required before launch.",
            "required_before_launch": [
                "commit this plan builder, tests and protocol document",
                "freeze plan.json exclusively before executing any case",
                "bind complete transitive source, runtime, SUMO and model hashes",
                "use acknowledged native worker identities and durable per-block stores",
                "persist a run before checking it; summarize only after successful full exit and replay",
            ],
            "resume_rule": "Never overwrite or outcome-selectively restart; use a separately frozen additive recovery plan.",
            "partial_efficacy_reporting_forbidden": True,
        },
    }
    plan["schedule_sha256"] = value_digest(groups)
    plan["initials_sha256"] = value_digest(initials)
    plan["plant_realizations_sha256"] = value_digest(plants)
    plan["perception_realizations_sha256"] = value_digest(perceptions)
    validate_plan(plan)
    return plan


def verify_initial_baseline_domain(plan: dict[str, Any]) -> dict[str, Any]:
    """Outcome-free check that all frozen initials can start the common baseline.

    This is a deterministic design feasibility check, not a rollout and not an
    efficacy result.  A fresh baseline instance is used for every initial.
    """
    import numpy as np

    from coopgate.actuator_recovery_v34 import ActuatorRecoveryBaseline

    failures = []
    for record in plan["initials"]:
        try:
            command, info = ActuatorRecoveryBaseline("lag_strong").select(
                np.asarray(record["state"], dtype=float), np.zeros(3)
            )
            if not info["certified"] or not np.isfinite(command).all():
                failures.append({"initial_id": record["initial_id"], "reason": "uncertified"})
        except (RuntimeError, ValueError) as exc:
            failures.append({"initial_id": record["initial_id"], "reason": str(exc)})
    if failures:
        raise ValueError(
            f"{len(failures)} frozen initials cannot initialize the common baseline; "
            "change the generator before freezing, never reject seeds"
        )
    return {
        "kind": "deterministic_design_feasibility_not_rollout_efficacy",
        "controller_profile": "lag_strong",
        "checked_initials": len(plan["initials"]),
        "failures": 0,
        "seed_rejection_or_replacement": False,
    }


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("active_or_partial_efficacy_outcomes_read") is not False or plan.get("outcome_input_paths") != []:
        raise ValueError("V39 design must not use prior or partial efficacy outcomes")
    if tuple(plan["models"]) != MODELS or tuple(plan["fixed_rules"]) != RULES:
        raise ValueError("Fixed model or rule family changed")
    initials = plan["initials"]
    expected_initials = len(REGIMES) * INITIALS_PER_REGIME
    if len(initials) != expected_initials or len({r["seed"] for r in initials}) != expected_initials:
        raise ValueError("Exactly 192 independent, non-reused initial seeds are required")
    for regime in REGIMES:
        subset = [r for r in initials if r["regime"] == regime]
        expected = list(range(SEED_BASES[regime], SEED_BASES[regime] + INITIALS_PER_REGIME))
        if [r["seed"] for r in subset] != expected:
            raise ValueError("Changed or rejected initial seed in " + regime)
        for record in subset:
            if record["state"] != initial_state(record["seed"], regime):
                raise ValueError("Initial generator mismatch")
    expected_methods = {(item["rule"], item["model"]) for item in method_keys()}
    groups = plan["groups"]
    expected_groups = expected_initials * len(PLANTS) * len(PERCEPTIONS)
    if len(groups) != expected_groups or [g["block_index"] for g in groups] != list(range(expected_groups)):
        raise ValueError("Incomplete or reordered paired condition grid")
    identities = set()
    repetitions: dict[str, int] = {r["initial_id"]: 0 for r in initials}
    for group in groups:
        key = (group["initial_id"], group["plant"], group["perception"])
        if key in identities:
            raise ValueError("Duplicate V39 factor condition")
        identities.add(key)
        repetitions[group["initial_id"]] += 1
        methods = {(item["rule"], item["model"]) for item in group["method_order"]}
        if methods != expected_methods or len(group["method_order"]) != len(expected_methods):
            raise ValueError("Every condition requires exactly the same 13 paired methods")
        if group["method_order"] != ordered_methods(group["block_index"]):
            raise ValueError("Changed result-independent method order")
    if set(repetitions.values()) != {len(PLANTS) * len(PERCEPTIONS)}:
        raise ValueError("Each initial must appear in all nine factor cells")
    plants = plan["plant_contract"]["realizations"]
    perceptions = plan["perception_contract"]["realizations"]
    if len(plants) != expected_initials * len(PLANTS) or len(perceptions) != expected_initials * len(PERCEPTIONS):
        raise ValueError("Incomplete plant or perception realization map")
    for key, value in plants.items():
        seed_text, name = key.split(":", 1)
        if value != plant_realization(int(seed_text), name):
            raise ValueError("Changed compound plant realization")
    for key, value in perceptions.items():
        seed_text, name = key.split(":", 1)
        if value != perception_realization(int(seed_text), name):
            raise ValueError("Changed perception realization")
    budget = plan["budget"]
    if budget != {
        "independent_initials": 192,
        "factor_cells_per_initial": 9,
        "paired_condition_blocks": 1_728,
        "methods_per_block": 13,
        "primary_runs": 6_912,
        "secondary_comparator_runs": 15_552,
        "actual_sumo_runs": 22_464,
        "all_runs_required_before_any_efficacy_summary": True,
    }:
        raise ValueError("V39 paired budget changed")
    if plan["schedule_sha256"] != value_digest(groups):
        raise ValueError("Schedule hash mismatch")
    if plan["initials_sha256"] != value_digest(initials):
        raise ValueError("Initial hash mismatch")
    if plan["plant_realizations_sha256"] != value_digest(plants):
        raise ValueError("Plant hash mismatch")
    if plan["perception_realizations_sha256"] != value_digest(perceptions):
        raise ValueError("Perception hash mismatch")


def _ensure_committed(paths: list[Path]) -> str:
    """Require exact tracked HEAD bytes for every design source/input."""
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if Path(root).resolve() != ROOT.resolve():
        raise ValueError("Unexpected Git root")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True
        )
        if tracked.returncode:
            raise ValueError("Commit design dependency before freezing: " + relative)
        blob = subprocess.run(
            ["git", "show", "HEAD:" + relative], cwd=ROOT, check=True, capture_output=True
        ).stdout
        if blob != path.read_bytes():
            raise ValueError("Working bytes differ from committed dependency: " + relative)
    return commit


def _runtime_identity() -> dict[str, Any]:
    binaries = {}
    for name in ("sumo", "netconvert"):
        located = shutil.which(name)
        if not located:
            raise FileNotFoundError("Required SUMO binary not found: " + name)
        path = Path(located).resolve()
        binaries[name] = {"path": str(path), "sha256": digest(path)}
    packages = {}
    for name in ("numpy", "scipy", "torch", "numba", "traci", "sumolib"):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": packages,
        "binaries": binaries,
    }


def _exclusive_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def freeze() -> None:
    if OUT.exists():
        raise FileExistsError("Preserve the first V39 frozen design")
    models = load_models()
    source_paths = [ROOT / name for name in SOURCES]
    commit = _ensure_committed([*source_paths, MODEL_PLAN])
    plan = build_plan(models)
    plan["offline_initial_baseline_preflight"] = verify_initial_baseline_domain(plan)
    plan.update(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,
        source_sha256={name: digest(ROOT / name) for name in SOURCES},
        input_sha256={
            MODEL_PLAN.relative_to(ROOT).as_posix(): digest(MODEL_PLAN),
            **{item["path"]: item["sha256"] for item in models.values()},
        },
        runtime=_runtime_identity(),
    )
    validate_plan(plan)
    OUT.mkdir(parents=True, exist_ok=False)
    _exclusive_json(OUT / "plan.json", plan)
    print(
        json.dumps(
            {
                "plan": str((OUT / "plan.json").relative_to(ROOT)),
                "independent_initials": plan["budget"]["independent_initials"],
                "actual_sumo_runs": plan["budget"]["actual_sumo_runs"],
                "plan_sha256": digest(OUT / "plan.json"),
            }
        ),
        flush=True,
    )


def check() -> None:
    plan = build_plan(load_models())
    print(
        json.dumps(
            {
                "protocol_version": plan["protocol_version"],
                "independent_initials": plan["budget"]["independent_initials"],
                "paired_condition_blocks": plan["budget"]["paired_condition_blocks"],
                "actual_sumo_runs": plan["budget"]["actual_sumo_runs"],
                "schedule_sha256": plan["schedule_sha256"],
                "artifact_created": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "freeze"))
    globals()[parser.parse_args().command]()
