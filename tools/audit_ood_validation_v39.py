"""Independent replay audit for complete V39 OOD episodes.

The auditor reconstructs perception, FIFO delay, actuator memory, physical
substeps, gate decisions and failure accounting without calling the V39 SUMO
wrapper's implementation helpers.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from coopgate.disturbance_v9 import disturbance_tape
from coopgate.merge_v4 import MergeConfig, transition_safe
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate


VERSION = "v39_ood_compound_mismatch_1"


def value_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def array_digest(value: Any) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


def _unit(seed: int, stream: str, index: int) -> float:
    raw = f"{VERSION}|{seed}|{stream}|{index}".encode("ascii")
    integer = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return (integer + 0.5) / 2**64


def _uniform(seed: int, stream: str, index: int, low: float, high: float) -> float:
    return low + (high - low) * _unit(seed, stream, index)


def _signed(seed: int, stream: str, index: int, low: float, high: float) -> float:
    magnitude = _uniform(seed, stream + ":magnitude", index, low, high)
    return magnitude if _unit(seed, stream + ":sign", index) >= 0.5 else -magnitude


def independent_perception_tape(seed: int, name: str, decisions: int = 50) -> list[Any]:
    """Independent implementation of the frozen V39 observation streams."""
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
            for step in range(decisions)
        ]
    if name == "stale_dropout":
        holds = [0, 0, 0]
        tape = []
        for step in range(decisions):
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


def _expected_observation(
    realization: dict[str, Any],
    tape: list[Any],
    history: list[np.ndarray],
    step: int,
    last_delivered: np.ndarray,
    last_delivered_decision: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    name = realization["name"]
    current = history[-1]
    if name == "clean_reference":
        observed = current.copy()
        evidence = dict(
            profile=name,
            decision=step,
            source_decision=step,
            dropped=[False] * 3,
            jitter=[[0.0, 0.0]] * 3,
        )
    elif name == "bounded_bias_jitter":
        jitter = np.asarray(tape[step], dtype=float)
        bias = np.column_stack(
            (
                np.asarray(realization["static_position_bias_m"], dtype=float),
                np.asarray(realization["static_speed_bias_mps"], dtype=float),
            )
        )
        observed = current + bias + jitter
        evidence = dict(
            profile=name,
            decision=step,
            source_decision=step,
            dropped=[False] * 3,
            jitter=jitter.tolist(),
        )
    elif name == "stale_dropout":
        source_decision = max(0, step - int(realization["observation_delay_decisions"]))
        source = history[source_decision]
        dropped = np.asarray(tape[step], dtype=bool)
        observed = source.copy()
        observed[dropped] = last_delivered[dropped]
        last_delivered = last_delivered.copy()
        last_delivered_decision = last_delivered_decision.copy()
        effective_source = np.full(3, source_decision, dtype=int)
        effective_source[dropped] = last_delivered_decision[dropped]
        last_delivered[~dropped] = source[~dropped]
        last_delivered_decision[~dropped] = source_decision
        evidence = dict(
            profile=name,
            decision=step,
            source_decision=effective_source.tolist(),
            dropped=dropped.tolist(),
            jitter=[[0.0, 0.0]] * 3,
        )
    else:
        raise ValueError("Unknown V39 perception profile")
    evidence["observed_sha256"] = array_digest(observed)
    return observed, evidence, last_delivered, last_delivered_decision


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items() if key != "gate_ms"}
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _compare_json(actual: Any, expected: Any, message: str) -> None:
    if _without_timing(actual) != _without_timing(expected):
        raise ValueError(message)


def _validate_command(command: np.ndarray, previous: np.ndarray) -> None:
    if (
        command.shape != (3,)
        or not np.isfinite(command).all()
        or np.any(command < -5.0 - 1e-10)
        or np.any(command > 2.5 + 1e-10)
        or np.any(np.abs(command - previous) > 2.5 + 1e-10)
    ):
        raise ValueError("Invalid executed command or slew")


def check_calls(row: dict[str, Any], trace: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
    successes = [call for call in calls if call["error"] is None]
    failures = [call for call in calls if call["error"] is not None]
    if (
        len(successes) != len(trace)
        or len(failures) != int(row["aborted"])
        or (failures and calls[-1] is not failures[-1])
        or any(
            not math.isfinite(float(call["wall_ms"])) or float(call["wall_ms"]) < 0.0
            for call in calls
        )
    ):
        raise ValueError("Incomplete successful/aborted V39 select timing")
    if failures and failures[0]["error"]["message"] != row["abort_reason"]:
        raise ValueError("Terminal select error differs from recorded V39 abort")
    if (
        failures
        and row.get("terminal_attempt") is not None
        and failures[0]["error"] != row["terminal_attempt"].get("error")
    ):
        raise ValueError("Terminal select error type/message differs from attempted-call evidence")


def _failure_reasons(row: dict[str, Any], completed: bool, violations: int) -> list[str]:
    reasons = []
    if row["collision"]:
        reasons.append("collision")
    if violations:
        reasons.append("route_geometry_violation")
    if row["aborted"]:
        reasons.append("controller_abort")
    if not completed:
        reasons.append("incomplete_or_timeout")
    return reasons


def audit_episode(
    plan: dict[str, Any],
    group: dict[str, Any],
    method: dict[str, str],
    record: dict[str, Any],
    trace: list[dict[str, Any]],
    model: Any,
) -> dict[str, Any]:
    """Replay one already-persisted episode and return non-effect audit facts."""
    row = record["row"]
    check_calls(row, trace, record["calls"])
    initial_record = next(item for item in plan["initials"] if item["initial_id"] == group["initial_id"])
    initial = np.asarray(initial_record["state"], dtype=float)
    if (
        row["initial"] != initial_record["state"]
        or row["seed"] != group["seed"]
        or row["scenario"] != group["regime"]
        or row["rule"] != method["rule"]
        or row["model"] != method["model"]
        or row["innovation_stream_seed"] != group["innovation_stream_seed"]
    ):
        raise ValueError("Recorded initial differs from frozen V39 initial")
    plant = plan["plant_contract"]["realizations"][group["plant_realization_key"]]
    perception = plan["perception_contract"]["realizations"][group["perception_realization_key"]]
    if (
        row["plant"] != plant
        or row["plant_name"] != group["plant"]
        or row["perception_name"] != perception["name"]
        or row["perception_stream_sha256"] != perception["stream_sha256"]
        or row["compound_mismatch_outside_declared_class"] is not True
        or row["physical_certificate_applicable"] is not False
        or row["exact_state_observation_assumption_met"]
        != (group["perception"] == "clean_reference")
    ):
        raise ValueError("Recorded V39 factor realization changed")
    perception_tape = independent_perception_tape(row["seed"], perception["name"], perception["decisions"])
    if value_digest(perception_tape) != perception["stream_sha256"]:
        raise ValueError("Independent perception tape differs from frozen hash")
    innovation_tape = disturbance_tape(
        group["innovation_stream_seed"], "uniform", 0.1, int(25.0 / 0.1)
    )
    if array_digest(innovation_tape) != row["innovation_tape_sha256"]:
        raise ValueError("Independent innovation tape differs")

    delay = np.asarray(plant["command_delay_decisions"], dtype=int)
    gain = np.asarray(plant["gain"], dtype=float)
    drag = np.asarray(plant["quadratic_drag"], dtype=float)
    tau = np.asarray(plant["lag_tau_seconds"], dtype=float)
    retention = np.exp(-0.1 / tau)
    state = initial.copy()
    history = [state.copy()]
    last_delivered = initial.copy()
    last_delivered_decision = np.zeros(3, dtype=int)
    previous = np.zeros(3)
    memory = np.zeros(3)
    commands: list[np.ndarray] = []
    active = np.ones(3, dtype=bool)
    finish = np.full(3, 25.0)
    variation = np.zeros(3)
    cursor = 0
    violations = 0
    maximum_error = 0.0
    maximum_slew = 0.0
    exceeded = 0
    learned_steps = 0
    fallback_steps = 0
    gate = RecoveryJointGate(model, "lag_strong", method["rule"])

    for step, item in enumerate(trace):
        np.testing.assert_allclose(item["t"], (step + 1) * 0.5, atol=1e-10, rtol=0)
        np.testing.assert_allclose(item["true_state_before"], state, atol=1e-8, rtol=0)
        observed, evidence, last_delivered, last_delivered_decision = _expected_observation(
            perception, perception_tape, history, step, last_delivered, last_delivered_decision
        )
        recorded_observed = np.asarray(item["observed_state"], dtype=float)
        np.testing.assert_allclose(recorded_observed, observed, atol=1e-8, rtol=0)
        # SUMO's exact readback may differ from the requested analytical initial
        # by roundoff.  First validate it numerically against the independent
        # observation reconstruction, then verify the evidence hash against the
        # exact bytes that the gate actually received and replay on those bytes.
        evidence["observed_sha256"] = array_digest(recorded_observed)
        _compare_json(item["perception"], evidence, "Perception evidence differs")
        command, info = gate.select(recorded_observed, previous)
        command = np.asarray(command, dtype=float)
        _validate_command(command, previous)
        np.testing.assert_allclose(item["command"], command, atol=0.0, rtol=0.0)
        for key, value in info.items():
            if key != "gate_ms":
                _compare_json(item.get(key), value, "Gate replay evidence differs: " + key)
        if not bool(item["actual_learning_used"]):
            np.testing.assert_allclose(item["command"], item["baseline_command"], atol=0.0, rtol=0.0)
            fallback_steps += 1
        else:
            learned_steps += 1
        variation += active * np.abs(command - previous)
        maximum_slew = max(maximum_slew, float(np.max(np.abs(command - previous))) / 0.5)
        commands.append(command.copy())
        delayed = np.zeros(3)
        for actor in range(3):
            index = len(commands) - 1 - int(delay[actor])
            if index >= 0:
                delayed[actor] = commands[index][actor]
        np.testing.assert_allclose(item["delayed_command"], delayed, atol=0.0, rtol=0.0)
        for substep in range(5):
            innovation = innovation_tape[cursor]
            cursor += 1
            np.testing.assert_allclose(item["innovation"][substep], innovation, atol=0.0, rtol=0.0)
            memory = retention * memory + (1.0 - retention) * delayed
            acceleration = gain * memory - drag * state[:, 1] ** 2 + innovation
            error = acceleration - command
            np.testing.assert_allclose(item["actuator_states"][substep], memory, atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                item["applied_accelerations"][substep], acceleration, atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(item["disturbance"][substep], error, atol=1e-10, rtol=0)
            size = float(np.max(np.abs(error)))
            maximum_error = max(maximum_error, size)
            exceeded += int(size > 0.5 + 1e-9)
            before = state[:, 0].copy()
            state[:, 1] = np.clip(state[:, 1] + acceleration * 0.1, 0.0, 20.0)
            state[:, 0] += state[:, 1] * 0.1
            violations += not bool(transition_safe(before[None], state[:, 0][None], MergeConfig())[0])
        if item["realized_error_in_nominal_half_radius"] != bool(
            np.max(np.abs(item["disturbance"])) <= 0.5 + 1e-9
        ):
            raise ValueError("Recorded nominal-half-radius diagnostic differs")
        np.testing.assert_allclose(item["x"], state, atol=1e-8, rtol=0)
        newly = active & (state[:, 0] >= 60.0)
        finish[newly] = item["t"]
        active[newly] = False
        previous = command
        history.append(state.copy())

    if row["aborted"]:
        terminal = row["terminal_attempt"]
        if terminal is None:
            raise ValueError("Aborted V39 row lacks terminal attempted observation")
        observed, evidence, _, _ = _expected_observation(
            perception,
            perception_tape,
            history,
            len(trace),
            last_delivered,
            last_delivered_decision,
        )
        np.testing.assert_allclose(terminal["true_state"], state, atol=1e-8, rtol=0)
        recorded_observed = np.asarray(terminal["observed_state"], dtype=float)
        np.testing.assert_allclose(recorded_observed, observed, atol=1e-8, rtol=0)
        evidence["observed_sha256"] = array_digest(recorded_observed)
        np.testing.assert_allclose(terminal["previous_command"], previous, atol=0.0, rtol=0.0)
        _compare_json(terminal["perception"], evidence, "Terminal perception evidence differs")
        replay_error = None
        try:
            command, _ = gate.select(recorded_observed, previous)
            _validate_command(np.asarray(command, dtype=float), previous)
        except Exception as exc:
            replay_error = {"type": type(exc).__name__, "message": str(exc)}
        if replay_error != terminal["error"]:
            raise ValueError("Terminal gate replay error differs")
    elif row["terminal_attempt"] is not None:
        raise ValueError("Successful V39 row retains a terminal failure attempt")

    completed = bool(not active.any() and not row["collision"] and not row["aborted"])
    if (
        not completed
        and not row["aborted"]
        and not row["collision"]
        and len(trace) != round(25.0 / 0.5)
    ):
        raise ValueError("Non-aborted, non-collision timeout must retain all 50 decisions")
    reasons = _failure_reasons(row, completed, violations)
    cost = finish + 0.05 * variation
    failure = bool(reasons)
    for actual, expected, name in (
        (row["per_vehicle_finish"], finish, "finish"),
        (row["per_vehicle_variation"], variation, "variation"),
        (row["per_vehicle_cost"], cost, "cost"),
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0, err_msg=name)
    np.testing.assert_allclose(row["team_cost"], cost.mean(), atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        row["failure_penalized_loss"], cost.mean() + 25.0 * failure, atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(row["max_actual_error"], maximum_error, atol=1e-9, rtol=0)
    np.testing.assert_allclose(row["max_command_slew"], maximum_slew, atol=1e-9, rtol=0)
    np.testing.assert_allclose(row["elapsed"], 0.5 * len(trace), atol=1e-10, rtol=0)
    if (
        row["completed"] != completed
        or row["failure"] != failure
        or row["failure_reasons"] != reasons
        or row["contract_violations"] != violations
        or row["error_contract_exceeded_substeps"] != exceeded
        or row["learned_action_steps"] != learned_steps
        or row["fallback_steps"] != fallback_steps
        or row["adopted"] != bool(learned_steps)
        or row["control_steps"] != len(trace)
        or row["controller_ms"] != [item["gate_ms"] for item in trace]
        or row["disturbance"] != "uniform"
        or row["radius"] != 0.1
    ):
        raise ValueError("V39 outcome/failure accounting differs")
    return {
        "physical_substeps": cursor,
        "gate_decisions": len(trace),
        "terminal_error_replayed": bool(row["aborted"]),
        "fallback_steps": fallback_steps,
        "learned_action_steps": learned_steps,
        "fallback_identity_exact": True,
        "perception_replayed": True,
        "compound_dynamics_replayed": True,
        "failure": failure,
    }
