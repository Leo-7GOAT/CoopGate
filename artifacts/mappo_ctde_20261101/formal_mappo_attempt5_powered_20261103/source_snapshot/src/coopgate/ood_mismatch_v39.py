"""V39 SUMO wrapper for compound actuator mismatch and perturbed perception.

The controller receives an explicitly generated observation while SUMO evolves
the hidden true state.  The wrapper deliberately makes no safety-certificate
claim: every V39 plant and both noisy observation profiles violate assumptions
of the unchanged V35 gate.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .disturbance_v9 import DisturbedMergeEnv, disturbance_tape
from .merge_v4 import BRANCH, advance, transition_safe


def value_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def array_digest(value: Any) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


def validate_plant(plant: dict[str, Any]) -> dict[str, np.ndarray]:
    required = {
        "command_delay_decisions",
        "gain",
        "quadratic_drag",
        "lag_tau_seconds",
        "external_innovation_radius",
        "outside_declared_controller_class",
    }
    if not required <= set(plant):
        raise ValueError("Incomplete V39 compound plant")
    delay = np.asarray(plant["command_delay_decisions"])
    gain = np.asarray(plant["gain"], dtype=float)
    drag = np.asarray(plant["quadratic_drag"], dtype=float)
    tau = np.asarray(plant["lag_tau_seconds"], dtype=float)
    if (
        delay.shape != (3,)
        or not np.issubdtype(delay.dtype, np.integer)
        or np.any(delay < 1)
        or gain.shape != (3,)
        or drag.shape != (3,)
        or tau.shape != (3,)
        or not np.isfinite(np.concatenate((gain, drag, tau))).all()
        or np.any(np.abs(gain - 1.0) <= 0.0)
        or np.any(drag <= 0.0)
        or np.any(tau <= 0.0)
        or plant["external_innovation_radius"] != 0.1
        or plant["outside_declared_controller_class"] is not True
    ):
        raise ValueError("Invalid V39 compound plant")
    return {"delay": delay.astype(int), "gain": gain, "drag": drag, "tau": tau}


def delayed_command(command_history: list[np.ndarray], delay: np.ndarray) -> np.ndarray:
    """Per-actor FIFO output after the current command has been appended."""
    if delay.shape != (3,) or not command_history:
        raise ValueError("Command history and three delays required")
    result = np.zeros(3)
    for actor in range(3):
        index = len(command_history) - 1 - int(delay[actor])
        if index >= 0:
            result[actor] = command_history[index][actor]
    return result


def compound_substep(
    delayed: np.ndarray,
    velocity: np.ndarray,
    memory: np.ndarray,
    innovation: np.ndarray,
    plant: dict[str, np.ndarray],
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """One substep of the predeclared delayed gain/drag/lag dynamics."""
    delayed = np.asarray(delayed, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    memory = np.asarray(memory, dtype=float)
    innovation = np.asarray(innovation, dtype=float)
    if any(value.shape != (3,) for value in (delayed, velocity, memory, innovation)):
        raise ValueError("Three-actor actuator values required")
    retention = np.exp(-dt / plant["tau"])
    next_memory = retention * memory + (1.0 - retention) * delayed
    acceleration = plant["gain"] * next_memory - plant["drag"] * velocity**2 + innovation
    if not np.isfinite(np.concatenate((next_memory, acceleration))).all():
        raise ValueError("Non-finite compound actuator state")
    return acceleration, next_memory


class PerceptionStream:
    """Stateful, deterministic controller observation stream."""

    def __init__(self, realization: dict[str, Any], tape: list[Any]):
        self.realization = json.loads(json.dumps(realization))
        self.name = realization["name"]
        if self.name not in {"clean_reference", "bounded_bias_jitter", "stale_dropout"}:
            raise ValueError("Unknown V39 perception profile")
        if realization.get("shared_by_all_methods") is not True:
            raise ValueError("V39 perception must be shared by every paired method")
        if value_digest(tape) != realization["stream_sha256"]:
            raise ValueError("Perception tape differs from frozen realization")
        self.tape = tape
        self.step = 0
        self.last_delivered: np.ndarray | None = None
        self.last_delivered_decision = np.zeros(3, dtype=int)

    def observe(self, true_history: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
        if not true_history or any(np.asarray(item).shape != (3, 2) for item in true_history):
            raise ValueError("Complete three-actor true-state history required")
        current = np.asarray(true_history[-1], dtype=float)
        if not np.isfinite(current).all():
            raise ValueError("Non-finite true state")
        if self.step >= self.realization["decisions"]:
            raise ValueError("Perception horizon exhausted")
        if self.name == "clean_reference":
            observed = current.copy()
            evidence = {
                "profile": self.name,
                "decision": self.step,
                "source_decision": self.step,
                "dropped": [False, False, False],
                "jitter": [[0.0, 0.0]] * 3,
            }
        elif self.name == "bounded_bias_jitter":
            jitter = np.asarray(self.tape[self.step], dtype=float)
            if jitter.shape != (3, 2) or not np.isfinite(jitter).all():
                raise ValueError("Invalid bounded perception jitter")
            bias = np.column_stack(
                (
                    np.asarray(self.realization["static_position_bias_m"], dtype=float),
                    np.asarray(self.realization["static_speed_bias_mps"], dtype=float),
                )
            )
            observed = current + bias + jitter
            evidence = {
                "profile": self.name,
                "decision": self.step,
                "source_decision": self.step,
                "dropped": [False, False, False],
                "jitter": jitter.tolist(),
            }
        else:
            source_decision = max(
                0, self.step - int(self.realization["observation_delay_decisions"])
            )
            source = np.asarray(true_history[source_decision], dtype=float)
            if self.last_delivered is None:
                self.last_delivered = np.asarray(true_history[0], dtype=float).copy()
            dropped = np.asarray(self.tape[self.step], dtype=bool)
            if dropped.shape != (3,):
                raise ValueError("Invalid stale/dropout mask")
            observed = source.copy()
            observed[dropped] = self.last_delivered[dropped]
            effective_source = np.full(3, source_decision, dtype=int)
            effective_source[dropped] = self.last_delivered_decision[dropped]
            self.last_delivered[~dropped] = source[~dropped]
            self.last_delivered_decision[~dropped] = source_decision
            evidence = {
                "profile": self.name,
                "decision": self.step,
                "source_decision": effective_source.tolist(),
                "dropped": dropped.tolist(),
                "jitter": [[0.0, 0.0]] * 3,
            }
        self.step += 1
        if observed.shape != (3, 2) or not np.isfinite(observed).all():
            raise ValueError("Invalid controller observation")
        evidence["observed_sha256"] = array_digest(observed)
        return observed, evidence


class OODMismatchMergeEnv(DisturbedMergeEnv):
    """Existing SUMO merge with V39 hidden plant and observation processes."""

    def reset_v39(
        self,
        *,
        seed: int,
        regime: str,
        initial: list[list[float]],
        plant: dict[str, Any],
        perception: dict[str, Any],
        perception_tape: list[Any],
        innovation_stream_seed: int,
    ) -> np.ndarray:
        # Start the unchanged topology/lifecycle, then place the exact frozen
        # state before the bookkeeping time origin, as in shifted_sumo_v19.
        super().reset_disturbed(seed, "dense", "uniform", 0.1)
        target = np.asarray(initial, dtype=float)
        if target.shape != (3, 2) or not np.isfinite(target).all():
            raise ValueError("Invalid frozen V39 initial state")
        for actor, ident in enumerate(self.ids):
            route = "a" if BRANCH[actor] == 0 else "b"
            length = self.conn.lane.getLength(route + "_0")
            self.conn.vehicle.moveTo(
                ident,
                route + "_0",
                float(length + target[actor, 0] - target[actor, 1] * self.c.dt),
            )
            self.conn.vehicle.setSpeed(ident, float(target[actor, 1]))
        self.conn.simulationStep()
        self.offsets = []
        for actor, ident in enumerate(self.ids):
            route = "a" if BRANCH[actor] == 0 else "b"
            position = self.conn.vehicle.getLanePosition(ident) - self.conn.lane.getLength(route + "_0")
            self.offsets.append(position - self.conn.vehicle.getDistance(ident))
        self.x = self.read()
        np.testing.assert_allclose(self.x, target, atol=1e-8, rtol=0)
        if self.conn.simulation.getCollidingVehiclesIDList():
            raise RuntimeError("Frozen V39 initial insertion collides")
        self.initial = self.x.copy()
        self.seed = int(seed)
        self.scenario = regime
        self.disturbance_mode = "uniform"
        self.disturbance_radius = 0.1
        self.plant_spec = json.loads(json.dumps(plant))
        self.plant = validate_plant(plant)
        self.perception_spec = json.loads(json.dumps(perception))
        self.perception = PerceptionStream(perception, perception_tape)
        self.tape = disturbance_tape(
            int(innovation_stream_seed), "uniform", 0.1, round(self.c.horizon / self.c.dt)
        )
        self.innovation_stream_seed = int(innovation_stream_seed)
        self.innovation_tape_sha256 = array_digest(self.tape)
        self.substep = 0
        self.actuator_memory = np.zeros(3)
        self.command_history: list[np.ndarray] = []
        self.true_history = [self.x.copy()]
        self.abort_reason = ""
        self.terminal_attempt: dict[str, Any] | None = None
        self.max_command_slew = 0.0
        self.max_actual_error = 0.0
        self.error_contract_exceeded_substeps = 0
        self.collision = False
        self.violations = 0
        self.outside = 0
        self.uncertified = 0
        self.error = 0.0
        self.finish = np.full(3, np.nan)
        self.interventions = []
        self.jerks = []
        self.trace = []
        self.t = 0.0
        self.prev = np.zeros(3)
        return self.obs()

    def step(self, actions):
        del actions  # V39 gates produce their own joint command.
        c = self.c
        previous = self.prev.copy()
        observed = None
        perception_evidence = None
        try:
            observed, perception_evidence = self.perception.observe(self.true_history)
            self.terminal_attempt = {
                "true_state": self.x.tolist(),
                "observed_state": observed.tolist(),
                "perception": perception_evidence,
                "previous_command": previous.tolist(),
            }
            command, info = self.gate.select(observed, previous)
            command = np.asarray(command, dtype=float)
            if (
                command.shape != (3,)
                or not np.isfinite(command).all()
                or np.any(command < -5.0 - 1e-10)
                or np.any(command > 2.5 + 1e-10)
                or np.any(np.abs(command - previous) > 2.5 + 1e-10)
            ):
                raise ValueError("Invalid executed command or slew")
        except Exception as exc:
            self.abort_reason = str(exc)
            if self.terminal_attempt is not None:
                self.terminal_attempt["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            return self.obs(), 0.0, True, False, self.metrics()

        self.terminal_attempt = None
        predicted = self.x[None].copy()
        advance(predicted, command[None], c)
        self.command_history.append(command.copy())
        delayed = delayed_command(self.command_history, self.plant["delay"])
        errors, innovations, actuator_states, applied_accelerations = [], [], [], []
        true_before = self.x.copy()
        for _ in range(round(c.decision_dt / c.dt)):
            before = self.x.copy()
            innovation = self.tape[self.substep]
            self.substep += 1
            applied, self.actuator_memory = compound_substep(
                delayed, self.x[:, 1], self.actuator_memory, innovation, self.plant, c.dt
            )
            error = applied - command
            innovations.append(innovation.tolist())
            actuator_states.append(self.actuator_memory.tolist())
            applied_accelerations.append(applied.tolist())
            errors.append(error.tolist())
            size = float(np.max(np.abs(error)))
            self.max_actual_error = max(self.max_actual_error, size)
            self.error_contract_exceeded_substeps += int(size > 0.5 + 1e-9)
            for actor, ident in enumerate(self.ids):
                speed = float(np.clip(self.x[actor, 1] + applied[actor] * c.dt, 0.0, c.vmax))
                self.conn.vehicle.setSpeed(ident, speed)
            self.conn.simulationStep()
            self.x = self.read()
            self.collision |= bool(self.conn.simulation.getCollidingVehiclesIDList())
            self.violations += not bool(
                transition_safe(before[:, 0][None], self.x[:, 0][None], c)[0]
            )
        self.t += c.decision_dt
        self.prev = command
        self.true_history.append(self.x.copy())
        self.error = max(self.error, float(np.max(np.abs(predicted[0] - self.x))))
        self.outside += not bool(info["in_set"])
        self.uncertified += not bool(info["certified"])
        self.interventions.append(float(info["intervention"]))
        self.jerks.append(float(np.mean(np.abs(command - previous))) / c.decision_dt)
        self.max_command_slew = max(
            self.max_command_slew, float(np.max(np.abs(command - previous))) / c.decision_dt
        )
        newly = (self.x[:, 0] >= c.goal) & np.isnan(self.finish)
        self.finish[newly] = self.t
        self.trace.append(
            {
                "t": self.t,
                "true_state_before": true_before.tolist(),
                "observed_state": observed.tolist(),
                "perception": perception_evidence,
                "x": self.x.tolist(),
                "command": command.tolist(),
                "delayed_command": delayed.tolist(),
                "innovation": innovations,
                "actuator_states": actuator_states,
                "applied_accelerations": applied_accelerations,
                "disturbance": errors,
                "realized_error_in_nominal_half_radius": bool(
                    np.max(np.abs(errors)) <= 0.5 + 1e-9
                ),
                **info,
            }
        )
        terminal = self.collision or bool(np.all(np.isfinite(self.finish)))
        truncated = self.t >= c.horizon and not terminal
        return self.obs(), 0.0, terminal, truncated, self.metrics()

    def metrics(self):
        result = super().metrics()
        if hasattr(self, "plant_spec"):
            result.update(
                plant=self.plant_spec,
                plant_name=self.plant_spec["name"],
                perception_name=self.perception_spec["name"],
                perception_stream_sha256=self.perception_spec["stream_sha256"],
                innovation_stream_seed=self.innovation_stream_seed,
                innovation_tape_sha256=self.innovation_tape_sha256,
                max_actual_error=self.max_actual_error,
                error_contract_exceeded_substeps=self.error_contract_exceeded_substeps,
                compound_mismatch_outside_declared_class=True,
                exact_state_observation_assumption_met=self.perception_spec["name"]
                == "clean_reference",
                physical_certificate_applicable=False,
                terminal_attempt=self.terminal_attempt,
            )
        return result
