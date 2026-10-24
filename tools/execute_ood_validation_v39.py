"""Bind, execute and summarize the frozen V39 OOD validation plan.

The design plan and this additive execution implementation are frozen in two
separate steps.  ``bind`` records the exact implementation/runtime/model bytes;
the binding itself must then be committed before ``watch`` can launch SUMO.
Interrupted blocks are immutable and require a separately frozen recovery.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.episode_store_v21 import EpisodeStore, encode, observations
from coopgate.ood_mismatch_v39 import OODMismatchMergeEnv
from coopgate.recovery_joint_gate_v35 import RecoveryJointGate
import audit_ood_validation_v39 as auditor
import design_ood_validation_v39 as design


ROOT = Path(__file__).resolve().parents[1]
OUT = design.OUT
PLAN = OUT / "plan.json"
BINDING = OUT / "execution_binding/manifest.json"
SCRIPT = "tools/execute_ood_validation_v39.py"
TEST = "tests/test_ood_validation_executor_v39.py"
NEW_EXECUTION_SOURCES = (
    "tools/design_ood_validation_v39.py",
    "tools/audit_ood_validation_v39.py",
    SCRIPT,
    "src/coopgate/ood_mismatch_v39.py",
    "tests/test_ood_validation_design_v39.py",
    TEST,
    "tests/test_ood_sumo_integration_v39.py",
    "docs/v39_ood_validation_protocol.md",
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def exclusive_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _retry_replace(source: Path, destination: Path, attempts: int = 41, delay: float = 0.25) -> None:
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            transient = (
                getattr(error, "winerror", None) in (5, 32, 33)
                or isinstance(error, PermissionError) and error.errno in (errno.EACCES, errno.EPERM)
            )
            if not transient or attempt + 1 == attempts:
                raise
            time.sleep(delay)


def atomic_bytes(path: Path, raw: bytes) -> None:
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _retry_replace(temporary, path)


def exclusive_bytes(path: Path, raw: bytes) -> None:
    with Path(path).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def committed(path: Path) -> None:
    relative = Path(path).relative_to(ROOT).as_posix()
    blob = subprocess.run(
        ["git", "show", "HEAD:" + relative], cwd=ROOT, check=True, capture_output=True
    ).stdout
    if hashlib.sha256(blob).hexdigest() != digest(path):
        raise ValueError("Commit exact V39 file before use: " + relative)


def sources_committed(paths: list[Path]) -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("utf-8").split("\0")
    wanted = {path.relative_to(ROOT).as_posix() for path in paths}
    changed = subprocess.run(
        ["git", "diff", "--name-only", "-z", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("utf-8").split("\0")
    if not wanted <= set(tracked) or wanted & set(changed):
        raise ValueError("All V39 execution sources must be committed and unchanged")
    for name in sorted(wanted):
        blob = subprocess.run(
            ["git", "show", "HEAD:" + name], cwd=ROOT, check=True, capture_output=True
        ).stdout
        if blob != (ROOT / name).read_bytes():
            raise ValueError("Runtime source differs from committed V39 bytes: " + name)


def runtime_identity() -> dict[str, Any]:
    return design._runtime_identity()


def native_snapshot() -> list[dict[str, Any]]:
    command = (
        "$OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|sumo)' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Depth 3"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        text=False,
        capture_output=True,
        check=True,
    )
    raw = result.stdout.decode("utf-8-sig")
    data = json.loads(raw) if raw.strip() else []
    data = data if isinstance(data, list) else [data]
    if any(
        not isinstance(item, dict)
        or type(item.get("ProcessId")) is not int
        or type(item.get("ParentProcessId")) is not int
        or not isinstance(item.get("Name"), str)
        for item in data
    ):
        raise ValueError("Incomplete native process snapshot")
    return data


def acknowledge_child(process: subprocess.Popen, execution: Path, role: str) -> dict[str, Any]:
    path = execution / f"{role}.native.json"
    deadline = time.monotonic() + 180.0
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError("V39 child terminated before publishing native identity")
        if time.monotonic() > deadline:
            raise TimeoutError("V39 native identity not observed")
        time.sleep(0.2)
    started = read(path)
    matches = [item for item in native_snapshot() if item["ProcessId"] == started["native_pid"]]
    if (
        len(matches) != 1
        or matches[0]["ParentProcessId"] != started["parent_pid"]
        or (
            started["native_pid"] != process.pid
            and started["parent_pid"] != process.pid
        )
    ):
        raise ValueError("V39 native worker is not the observed child of the live handle")
    observed = matches[0]
    identity = {
        "launcher_pid": process.pid,
        "native_pid": started["native_pid"],
        "parent_pid": started["parent_pid"],
        "native_creation_date": observed.get("CreationDate"),
        "command_line": observed.get("CommandLine"),
    }
    proof = {**identity, "native_start_sha256": digest(path), "utc": utc()}
    acknowledgment = execution / f"{role}.acknowledged.json"
    exclusive_bytes(acknowledgment, encode(proof))
    return identity


def check_acknowledgment(
    execution: Path, role: str, launcher_pid: int, native_pid: int
) -> None:
    started = read(execution / f"{role}.native.json")
    proof = read(execution / f"{role}.acknowledged.json")
    if (
        proof["launcher_pid"] != launcher_pid
        or proof["native_pid"] != native_pid
        or started["native_pid"] != native_pid
        or proof["parent_pid"] != started["parent_pid"]
        or (native_pid != launcher_pid and proof["parent_pid"] != launcher_pid)
        or proof["native_start_sha256"] != digest(execution / f"{role}.native.json")
    ):
        raise ValueError("Original V39 native acknowledgment changed")


def read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan_inputs() -> dict[str, Any]:
    """Validate only the committed frozen design and immutable design inputs."""
    committed(PLAN)
    plan = read(PLAN)
    design.validate_plan(plan)
    for name, expected in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen V39 design source/input changed: " + name)
    sources_committed([ROOT / name for name in plan["source_sha256"]])
    if runtime_identity() != plan["runtime"]:
        raise ValueError("Frozen V39 design runtime changed")
    if plan.get("offline_initial_baseline_preflight") != design.verify_initial_baseline_domain(plan):
        raise ValueError("Frozen outcome-free initial feasibility preflight changed")
    for item in plan["models"].values():
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Frozen V39 learned model changed")
    return plan


def execution_source_paths(plan: dict[str, Any]) -> list[Path]:
    names = set(plan["source_sha256"]) | set(NEW_EXECUTION_SOURCES)
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "src/coopgate").glob("*.py"))
    return [ROOT / name for name in sorted(names)]


def merge_source_hashes(plan: dict[str, Any], paths: list[Path]) -> dict[str, str]:
    result = dict(plan["source_sha256"])
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        actual = digest(path)
        if name in result and result[name] != actual:
            raise ValueError("Execution/design source hash conflict: " + name)
        result[name] = actual
    return result


def validate_bound_source_set(plan: dict[str, Any], binding: dict[str, Any]) -> None:
    """Validate the frozen set without re-enumerating unrelated future files."""
    bound = binding["source_sha256"]
    for name, expected in plan["source_sha256"].items():
        if bound.get(name) != expected:
            raise ValueError("Bound execution lost or changed a design source: " + name)
    required = set(NEW_EXECUTION_SOURCES) | {"src/coopgate/ood_mismatch_v39.py"}
    if not required <= set(bound):
        raise ValueError("Bound execution is missing a required V39 source")
    package_inventory = sorted(name for name in bound if name.startswith("src/coopgate/") and name.endswith(".py"))
    if (
        binding.get("bound_src_inventory") != package_inventory
        or binding.get("source_set_policy")
        != "All coopgate Python sources present at bind are immutable; unrelated files added later are outside this binding."
    ):
        raise ValueError("Bound coopgate source inventory or policy changed")


def bind() -> None:
    """Freeze additive execution bytes only; never launch or read outcomes."""
    if BINDING.parent.exists():
        raise FileExistsError("Preserve the first V39 execution binding")
    plan = plan_inputs()
    paths = execution_source_paths(plan)
    sources_committed(paths)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    source_hashes = merge_source_hashes(plan, paths)
    manifest = {
        "purpose": "Additive execution binding for the already frozen V39 scientific design",
        "plan_sha256": digest(PLAN),
        "git_commit_before_binding_artifact": commit,
        "source_sha256": source_hashes,
        "bound_src_inventory": sorted(
            name for name in source_hashes if name.startswith("src/coopgate/") and name.endswith(".py")
        ),
        "source_set_policy": (
            "All coopgate Python sources present at bind are immutable; unrelated files added later are outside this binding."
        ),
        "runtime": runtime_identity(),
        "model_sha256": {model: item["sha256"] for model, item in plan["models"].items()},
        "exact_blocks": plan["budget"]["paired_condition_blocks"],
        "exact_methods_per_block": plan["budget"]["methods_per_block"],
        "exact_sumo_runs": plan["budget"]["actual_sumo_runs"],
        "controller_routing": "Unchanged RecoveryJointGate(rule, fixed model, lag_strong profile)",
        "observation_semantics": (
            "Every paired method uses the same frozen bias/jitter/dropout error tape. Observed "
            "values combine that tape with each method's own evolving true trajectory; true state is hidden."
        ),
        "perception_speed_transform": "additive without clipping; invalid observed states become retained aborts",
        "plant_semantics": "decision FIFO delay then substep lag, gain, quadratic drag and shared innovation tape",
        "certificate_scope": "Physical certificate inapplicable for all compound plants, including clean perception",
        "exception_scope": (
            "Every ordinary Exception raised by controller routing or command validation is a retained "
            "episode abort and must replay exactly; invalid hidden perception/physics evidence invalidates "
            "the study, while KeyboardInterrupt/SystemExit remain process failures."
        ),
        "nonfinite_scope": (
            "A nonfinite controller return is retained as a validated-command abort. Nonfinite hidden physics, "
            "trace or stored metrics cannot be physically replayed and therefore invalidate the worker/study; "
            "they are never summarized as efficacy evidence."
        ),
        "persistence": "Commit each run to SQLite before its independent replay audit; seal only after all 13 pass",
        "interruption": "No overwrite/resume. Preserve partial block and use a separately frozen recovery protocol.",
        "reporting": "No block effects are read until acknowledged native zero exit and complete 22,464-run receipt",
        "warmup": "Disposable instance of all 13 routes on the first frozen clean observation before timing",
        "active_or_partial_efficacy_outcomes_read": False,
        "outcome_input_paths": [],
        "frozen_utc": utc(),
    }
    BINDING.parent.mkdir(parents=True, exist_ok=False)
    exclusive_json(BINDING, manifest)
    print(
        json.dumps(
            {
                "binding": BINDING.relative_to(ROOT).as_posix(),
                "plan_sha256": manifest["plan_sha256"],
                "source_files": len(manifest["source_sha256"]),
                "actual_sumo_runs": manifest["exact_sumo_runs"],
                "binding_sha256": digest(BINDING),
                "launched": False,
            }
        ),
        flush=True,
    )


def binding_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = plan_inputs()
    committed(BINDING)
    binding = read(BINDING)
    if (
        binding["plan_sha256"] != digest(PLAN)
        or binding["exact_blocks"] != plan["budget"]["paired_condition_blocks"]
        or binding["exact_methods_per_block"] != plan["budget"]["methods_per_block"]
        or binding["exact_sumo_runs"] != plan["budget"]["actual_sumo_runs"]
        or binding["active_or_partial_efficacy_outcomes_read"] is not False
        or binding["outcome_input_paths"] != []
    ):
        raise ValueError("V39 execution binding differs from frozen design")
    validate_bound_source_set(plan, binding)
    for name, expected in binding["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Bound V39 execution source changed: " + name)
    sources_committed([ROOT / name for name in binding["source_sha256"]])
    if binding["runtime"] != runtime_identity():
        raise ValueError("Bound V39 execution runtime changed")
    if binding["model_sha256"] != {
        model: item["sha256"] for model, item in plan["models"].items()
    }:
        raise ValueError("Bound V39 model family changed")
    return plan, binding


def case_key(group: dict[str, Any], method: dict[str, str]) -> list[Any]:
    return [
        group["block_index"],
        group["initial_id"],
        group["seed"],
        group["regime"],
        group["plant"],
        group["perception"],
        method["rule"],
        method["model"],
    ]


def schedule(group: dict[str, Any]) -> list[list[Any]]:
    return [case_key(group, method) for method in group["method_order"]]


def metadata(group: dict[str, Any], plan_sha: str, binding_sha: str) -> dict[str, Any]:
    return {
        "stage": "v39_ood_mismatch_validation",
        "plan_sha256": plan_sha,
        "execution_binding_sha256": binding_sha,
        "block_index": group["block_index"],
        "initial_id": group["initial_id"],
    }


def _validate_returned_command(command: Any, previous: Any) -> None:
    command = np.asarray(command, dtype=float)
    previous = np.asarray(previous, dtype=float)
    if (
        command.shape != (3,)
        or previous.shape != (3,)
        or not np.isfinite(command).all()
        or np.any(command < -5.0 - 1e-10)
        or np.any(command > 2.5 + 1e-10)
        or np.any(np.abs(command - previous) > 2.5 + 1e-10)
    ):
        raise ValueError("Invalid executed command or slew")


class TimedGate:
    """Record complete calls, including adapter-level command validation."""

    def __init__(self, gate: Any):
        self.gate = gate
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gate, name)

    def select(self, *args, **kwargs):
        started = time.perf_counter()
        error = None
        try:
            result = self.gate.select(*args, **kwargs)
            if len(args) < 2:
                raise ValueError("V39 gate adapter requires observed state and previous command")
            _validate_returned_command(result[0], args[1])
            return result
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            self.calls.append(
                {"wall_ms": 1000.0 * (time.perf_counter() - started), "error": error}
            )


def route_gate(
    method: dict[str, str], models: dict[str, Any], gate_cls: type = RecoveryJointGate
) -> TimedGate:
    if method not in design.method_keys():
        raise ValueError("Method is not in the frozen 13-route family")
    rule, model_id = method["rule"], method["model"]
    model = None if model_id == "none" else models[model_id]
    if (rule == "baseline") != (model is None):
        raise ValueError("Only the common baseline may omit a fixed learned model")
    return TimedGate(gate_cls(model, "lag_strong", rule))


def _initial(plan: dict[str, Any], initial_id: str) -> dict[str, Any]:
    matches = [item for item in plan["initials"] if item["initial_id"] == initial_id]
    if len(matches) != 1:
        raise ValueError("Exactly one frozen V39 initial required")
    return matches[0]


def _finite_trace(trace: list[dict[str, Any]]) -> bool:
    numeric = []
    for item in trace:
        for key in (
            "true_state_before",
            "observed_state",
            "x",
            "command",
            "delayed_command",
            "innovation",
            "actuator_states",
            "applied_accelerations",
            "disturbance",
        ):
            numeric.extend(np.asarray(item[key], dtype=float).ravel().tolist())
    return bool(np.isfinite(numeric).all()) if numeric else True


def finalize_row(
    env: OODMismatchMergeEnv,
    initial: dict[str, Any],
    method: dict[str, str],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    row = env.metrics()
    previous = np.zeros(3)
    active = np.ones(3, dtype=bool)
    variation = np.zeros(3)
    for item in trace:
        command = np.asarray(item["command"], dtype=float)
        variation += active * np.abs(command - previous)
        active &= np.asarray(item["x"], dtype=float)[:, 0] < env.c.goal
        previous = command
    finish = np.nan_to_num(env.finish, nan=env.c.horizon)
    cost = finish + 0.05 * variation
    completed = bool(np.all(np.isfinite(env.finish)) and not row["collision"] and not row["aborted"])
    reasons = []
    if row["collision"]:
        reasons.append("collision")
    if row["contract_violations"]:
        reasons.append("route_geometry_violation")
    if row["aborted"]:
        reasons.append("controller_abort")
    if not completed:
        reasons.append("incomplete_or_timeout")
    if not _finite_trace(trace) or not np.isfinite(np.concatenate((finish, variation, cost))).all():
        reasons.append("non_finite_record")
    failure = bool(reasons)
    learned_steps = sum(bool(item["actual_learning_used"]) for item in trace)
    fallback_steps = len(trace) - learned_steps
    row.update(
        initial=initial["state"],
        rule=method["rule"],
        model=method["model"],
        controller=method["rule"]
        if method["model"] == "none"
        else method["rule"] + "_" + method["model"],
        certificate_contract="empirical_out_of_assumption_stress_only",
        per_vehicle_finish=finish.tolist(),
        per_vehicle_variation=variation.tolist(),
        per_vehicle_cost=cost.tolist(),
        team_cost=float(cost.mean()),
        completed=completed,
        failure=failure,
        failure_reasons=reasons,
        failure_penalized_loss=float(cost.mean() + env.c.horizon * failure),
        control_steps=len(trace),
        learned_action_steps=learned_steps,
        fallback_steps=fallback_steps,
        adopted=bool(learned_steps),
        controller_ms=[float(item["gate_ms"]) for item in trace],
    )
    return row


def run_case(
    plan: dict[str, Any],
    group: dict[str, Any],
    method: dict[str, str],
    models: dict[str, Any],
    env: OODMismatchMergeEnv,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    initial = _initial(plan, group["initial_id"])
    plant = plan["plant_contract"]["realizations"][group["plant_realization_key"]]
    perception = plan["perception_contract"]["realizations"][
        group["perception_realization_key"]
    ]
    tape = design._perception_tape(group["seed"], group["perception"])
    env.reset_v39(
        seed=group["seed"],
        regime=group["regime"],
        initial=initial["state"],
        plant=plant,
        perception=perception,
        perception_tape=tape,
        innovation_stream_seed=group["innovation_stream_seed"],
    )
    started = time.perf_counter()
    gate = route_gate(method, models)
    setup_ms = 1000.0 * (time.perf_counter() - started)
    env.gate = gate
    episode_started = time.perf_counter()
    while True:
        _, _, done, truncated, _ = env.step(None)
        if done or truncated:
            break
    trace = env.trace
    row = finalize_row(env, initial, method, trace)
    row["setup_ms"] = setup_ms
    record = {
        "row": row,
        "calls": gate.calls,
        "episode_wall_seconds": time.perf_counter() - episode_started,
    }
    return record, trace


def collect_block(
    plan: dict[str, Any],
    group: dict[str, Any],
    directory: Path,
    models: dict[str, Any],
    env: Any,
    plan_sha: str,
    binding_sha: str,
    run_fn: Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]] = run_case,
    audit_fn: Callable[..., dict[str, Any]] = auditor.audit_episode,
) -> list[dict[str, Any]]:
    """Persist each case before audit; a failed audit leaves an unsealed block."""
    directory.mkdir()
    store = EpisodeStore(directory / "observations.sqlite", metadata(group, plan_sha, binding_sha))
    reviews = []
    try:
        for method in group["method_order"]:
            record, trace = run_fn(plan, group, method, models, env)
            key = case_key(group, method)
            store.append(key, record, trace)
            # Required ordering: the durable transaction precedes every check.
            model = None if method["model"] == "none" else models[method["model"]]
            review = audit_fn(plan, group, method, record, trace, model)
            reviews.append({"case": key, **review})
        store.seal(schedule(group))
    finally:
        store.close()
    exclusive_json(
        directory / "complete.json",
        {
            **metadata(group, plan_sha, binding_sha),
            "actual_sumo_runs": len(reviews),
            "reviews": reviews,
            "database_sha256": digest(directory / "observations.sqlite"),
            "seal_sha256": digest(directory / "observations.sqlite.seal.json"),
        },
    )
    return reviews


def block_manifest_entry(directory: Path, block_index: int) -> dict[str, Any]:
    return {
        "block_index": block_index,
        "complete_sha256": digest(directory / "complete.json"),
        "database_sha256": digest(directory / "observations.sqlite"),
        "seal_sha256": digest(directory / "observations.sqlite.seal.json"),
    }


def validate_blocks_manifest(
    plan: dict[str, Any], plan_sha: str, binding_sha: str, manifest: dict[str, Any]
) -> None:
    expected_blocks = plan["budget"]["paired_condition_blocks"]
    if (
        manifest["plan_sha256"] != plan_sha
        or manifest["execution_binding_sha256"] != binding_sha
        or manifest["blocks"] != expected_blocks
        or manifest["actual_sumo_runs"] != plan["budget"]["actual_sumo_runs"]
        or len(manifest["entries"]) != expected_blocks
        or [entry["block_index"] for entry in manifest["entries"]] != list(range(expected_blocks))
    ):
        raise ValueError("Incomplete or changed V39 block manifest scope")
    for entry in manifest["entries"]:
        directory = OUT / "blocks" / f"{entry['block_index']:05d}"
        expected = block_manifest_entry(directory, entry["block_index"])
        if entry != expected:
            raise ValueError("V39 block bytes changed after worker completion")


def warm_gates(plan: dict[str, Any], models: dict[str, Any]) -> None:
    initial = np.asarray(plan["initials"][0]["state"], dtype=float)
    for method in design.method_keys():
        route_gate(method, models).select(initial, np.zeros(3))


def worker() -> None:
    execution = OUT / "execution"
    startup = execution / "worker.native.json"
    exclusive_bytes(
        startup,
        encode({"native_pid": os.getpid(), "parent_pid": os.getppid(), "utc": utc()}),
    )
    acknowledgment = execution / "worker.acknowledged.json"
    deadline = time.monotonic() + 180.0
    while not acknowledgment.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("No V39 parent acknowledgment; no simulation started")
        time.sleep(0.2)
    proof = read(acknowledgment)
    if (
        proof["native_pid"] != os.getpid()
        or proof["parent_pid"] != os.getppid()
        or proof["native_start_sha256"] != digest(startup)
    ):
        raise ValueError("V39 native process acknowledgment mismatch")
    plan, _ = binding_inputs()
    plan_sha, binding_sha = digest(PLAN), digest(BINDING)
    torch.set_num_threads(1)
    models = {model: load_coupled(ROOT / item["path"])[0] for model, item in plan["models"].items()}
    warm_gates(plan, models)
    blocks = OUT / "blocks"
    blocks.mkdir(exist_ok=False)
    completed = 0
    block_entries = []
    with ExitStack() as stack:
        env = stack.enter_context(OODMismatchMergeEnv())
        for group in plan["groups"]:
            directory = blocks / f"{group['block_index']:05d}"
            reviews = collect_block(
                plan, group, directory, models, env, plan_sha, binding_sha
            )
            if len(reviews) != plan["budget"]["methods_per_block"]:
                raise ValueError("Incomplete V39 block")
            block_entries.append(block_manifest_entry(directory, group["block_index"]))
            completed += len(reviews)
            atomic_bytes(
                execution / "progress.json",
                encode(
                    {
                        "utc": utc(),
                        "native_pid": os.getpid(),
                        "completed_blocks": group["block_index"] + 1,
                        "expected_blocks": plan["budget"]["paired_condition_blocks"],
                        "completed_runs": completed,
                        "expected_runs": plan["budget"]["actual_sumo_runs"],
                    }
                ),
            )
            print(
                json.dumps(
                    {
                        "block": group["block_index"],
                        "completed_runs": completed,
                        "total_runs": plan["budget"]["actual_sumo_runs"],
                    }
                ),
                flush=True,
            )
    if completed != plan["budget"]["actual_sumo_runs"]:
        raise ValueError("Incomplete V39 exact schedule")
    blocks_manifest = execution / "blocks_manifest.json"
    exclusive_json(
        blocks_manifest,
        {
            "plan_sha256": plan_sha,
            "execution_binding_sha256": binding_sha,
            "blocks": plan["budget"]["paired_condition_blocks"],
            "actual_sumo_runs": completed,
            "entries": block_entries,
            "utc": utc(),
        },
    )
    exclusive_json(
        execution / "collection_complete.json",
        {
            "plan_sha256": plan_sha,
            "execution_binding_sha256": binding_sha,
            "native_pid": os.getpid(),
            "actual_sumo_runs": completed,
            "blocks": plan["budget"]["paired_condition_blocks"],
            "blocks_manifest_sha256": digest(blocks_manifest),
            "utc": utc(),
        },
    )


def watch() -> None:
    plan, _ = binding_inputs()
    execution = OUT / "execution"
    execution.mkdir(exist_ok=False)
    with (execution / "worker.stdout.log").open("x") as stdout, (
        execution / "worker.stderr.log"
    ).open("x") as stderr:
        command = [sys.executable, "-X", "utf8", SCRIPT, "worker"]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        exclusive_json(
            execution / "spawn.json",
            {
                "watcher_pid": os.getpid(),
                "launcher_pid": process.pid,
                "command": command,
                "plan_sha256": digest(PLAN),
                "execution_binding_sha256": digest(BINDING),
                "expected_runs": plan["budget"]["actual_sumo_runs"],
                "utc": utc(),
            },
        )
        try:
            native = acknowledge_child(process, execution, "worker")
            exclusive_json(
                execution / "launch.json",
                {
                    **native,
                    "watcher_pid": os.getpid(),
                    "plan_sha256": digest(PLAN),
                    "execution_binding_sha256": digest(BINDING),
                    "utc": utc(),
                },
            )
        except Exception as error:
            exclusive_json(
                execution / "observation_error.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "launcher_pid": process.pid,
                    "utc": utc(),
                },
            )
            code = process.wait()
            exclusive_json(
                execution / "exit.json",
                {
                    "launcher_pid": process.pid,
                    "exit_code": code,
                    "plan_sha256": digest(PLAN),
                    "execution_binding_sha256": digest(BINDING),
                    "utc": utc(),
                },
            )
            raise
        code = process.wait()
        collection = execution / "collection_complete.json"
        blocks_manifest = execution / "blocks_manifest.json"
        exclusive_json(
            execution / "exit.json",
            {
                **native,
                "exit_code": code,
                "plan_sha256": digest(PLAN),
                "execution_binding_sha256": digest(BINDING),
                "collection_complete_sha256": digest(collection) if collection.exists() else None,
                "blocks_manifest_sha256": digest(blocks_manifest) if blocks_manifest.exists() else None,
                "utc": utc(),
            },
        )
    if code:
        raise SystemExit(code)


def validate_completion(
    plan: dict[str, Any],
    plan_sha: str,
    binding_sha: str,
    spawn: dict[str, Any],
    receipt: dict[str, Any],
    launch: dict[str, Any],
    done: dict[str, Any],
    collection_sha: str,
    blocks_manifest_sha: str,
) -> None:
    if (
        receipt["exit_code"] != 0
        or any(
            receipt.get(key) != launch.get(key)
            for key in (
                "launcher_pid",
                "native_pid",
                "parent_pid",
                "native_creation_date",
                "command_line",
            )
        )
        or spawn["launcher_pid"] != launch["launcher_pid"]
        or spawn["watcher_pid"] != launch["watcher_pid"]
        or spawn["command"] != [sys.executable, "-X", "utf8", SCRIPT, "worker"]
        or spawn["expected_runs"] != plan["budget"]["actual_sumo_runs"]
        or receipt["plan_sha256"] != plan_sha
        or launch["plan_sha256"] != plan_sha
        or spawn["plan_sha256"] != plan_sha
        or done["plan_sha256"] != plan_sha
        or receipt["execution_binding_sha256"] != binding_sha
        or launch["execution_binding_sha256"] != binding_sha
        or spawn["execution_binding_sha256"] != binding_sha
        or done["execution_binding_sha256"] != binding_sha
        or done["native_pid"] != launch["native_pid"]
        or done["actual_sumo_runs"] != plan["budget"]["actual_sumo_runs"]
        or done["blocks"] != plan["budget"]["paired_condition_blocks"]
        or receipt["collection_complete_sha256"] != collection_sha
        or receipt["blocks_manifest_sha256"] != blocks_manifest_sha
        or done["blocks_manifest_sha256"] != receipt["blocks_manifest_sha256"]
    ):
        raise ValueError("Successful acknowledged zero exit and complete V39 scope required")


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda key: raw[key])
    result = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * raw[key]))
        result[key] = running
    return result


def _bootstrap(values: dict[str, np.ndarray], seed: int, draws: int) -> tuple[float, np.ndarray]:
    if set(values) != set(design.REGIMES) or any(len(value) != 64 for value in values.values()):
        raise ValueError("Exactly 64 independent initials per V39 regime required")
    point = float(np.mean([np.mean(values[regime]) for regime in design.REGIMES]))
    rng = np.random.default_rng(seed)
    samples = np.zeros(draws)
    for regime in design.REGIMES:
        value = np.asarray(values[regime], dtype=float)
        indices = rng.integers(0, len(value), size=(draws, len(value)))
        samples += value[indices].mean(axis=1) / len(design.REGIMES)
    return point, samples


def primary_analysis(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        condition = (row["initial_id"], row["plant_name"], row["perception_name"])
        grouped[condition][(row["rule"], row["model"])] = row
    values = {
        model: {"failure": defaultdict(list), "loss": defaultdict(list)} for model in design.MODELS
    }
    initial_regime = {item["initial_id"]: item["regime"] for item in plan["initials"]}
    by_initial = {
        model: defaultdict(lambda: {"failure": [], "loss": []}) for model in design.MODELS
    }
    for (initial_id, _, _), methods in grouped.items():
        if len(methods) != 13:
            raise ValueError("Incomplete 13-method V39 condition")
        baseline = methods[("baseline", "none")]
        for model in design.MODELS:
            gate = methods[("bank_performance", model)]
            by_initial[model][initial_id]["failure"].append(
                float(gate["failure"]) - float(baseline["failure"])
            )
            by_initial[model][initial_id]["loss"].append(
                float(baseline["failure_penalized_loss"])
                - float(gate["failure_penalized_loss"])
            )
    for model in design.MODELS:
        if len(by_initial[model]) != 192:
            raise ValueError("Incomplete V39 independent initial coverage")
        for initial_id, metrics in by_initial[model].items():
            if len(metrics["failure"]) != 9 or len(metrics["loss"]) != 9:
                raise ValueError("Each V39 initial requires all nine repeated factor cells")
            regime = initial_regime[initial_id]
            values[model]["failure"][regime].append(np.mean(metrics["failure"]))
            values[model]["loss"][regime].append(np.mean(metrics["loss"]))
    summaries, failure_p, loss_p = {}, {}, {}
    draws = plan["inference"]["bootstrap_draws"]
    base_seed = plan["inference"]["bootstrap_seed"]
    for index, model in enumerate(design.MODELS):
        failure_values = {key: np.asarray(value) for key, value in values[model]["failure"].items()}
        loss_values = {key: np.asarray(value) for key, value in values[model]["loss"].items()}
        failure_point, failure_draws = _bootstrap(failure_values, base_seed + 10 * index, draws)
        loss_point, loss_draws = _bootstrap(loss_values, base_seed + 10 * index + 1, draws)
        failure_clusters = np.concatenate([failure_values[regime] for regime in design.REGIMES])
        loss_clusters = np.concatenate([loss_values[regime] for regime in design.REGIMES])
        failure_within_stratum_variance = {
            regime: float(np.var(failure_values[regime], ddof=1)) for regime in design.REGIMES
        }
        loss_within_stratum_variance = {
            regime: float(np.var(loss_values[regime], ddof=1)) for regime in design.REGIMES
        }
        failure_variance = float(np.var(failure_clusters, ddof=1))
        loss_variance = float(np.var(loss_clusters, ddof=1))
        failure_bootstrap_variance = float(np.var(failure_draws, ddof=1))
        loss_bootstrap_variance = float(np.var(loss_draws, ddof=1))
        failure_testable = bool(
            np.isfinite(failure_variance)
            and failure_variance > 1e-15
            and sum(failure_within_stratum_variance.values()) > 1e-15
            and np.isfinite(failure_bootstrap_variance)
            and failure_bootstrap_variance > 1e-15
            and np.count_nonzero(failure_clusters) > 0
        )
        loss_testable = bool(
            np.isfinite(loss_variance)
            and loss_variance > 1e-15
            and sum(loss_within_stratum_variance.values()) > 1e-15
            and np.isfinite(loss_bootstrap_variance)
            and loss_bootstrap_variance > 1e-15
        )
        # Invert the two predeclared one-sided margin tests. Holm adjustment is
        # applied below across the three fixed models within each estimand.
        failure_p[model] = (
            float((1 + np.sum(failure_draws - failure_point <= failure_point - 0.01)) / (draws + 1))
            if failure_testable
            else 1.0
        )
        loss_p[model] = (
            float((1 + np.sum(loss_draws - loss_point >= loss_point)) / (draws + 1))
            if loss_testable
            else 1.0
        )
        summaries[model] = {
            "failure_risk_difference": failure_point,
            "failure_descriptive_unadjusted_one_sided_95pct_upper": float(
                np.quantile(failure_draws, 0.95)
            ),
            "failure_margin": 0.01,
            "failure_raw_margin_test_p": failure_p[model],
            "failure_margin_testable": failure_testable,
            "failure_cluster_variance": failure_variance,
            "failure_within_stratum_variance": failure_within_stratum_variance,
            "failure_stratified_bootstrap_variance": failure_bootstrap_variance,
            "failure_nonzero_cluster_contrasts": int(np.count_nonzero(failure_clusters)),
            "failure_penalized_cost_gain_seconds": loss_point,
            "loss_descriptive_unadjusted_one_sided_95pct_lower": float(
                np.quantile(loss_draws, 0.05)
            ),
            "loss_margin_seconds": 0.0,
            "loss_raw_margin_test_p": loss_p[model],
            "loss_margin_testable": loss_testable,
            "loss_cluster_variance": loss_variance,
            "loss_within_stratum_variance": loss_within_stratum_variance,
            "loss_stratified_bootstrap_variance": loss_bootstrap_variance,
            "independent_initials": 192,
            "repeated_cells_per_initial": 9,
        }
    failure_adjusted, loss_adjusted = _holm_adjust(failure_p), _holm_adjust(loss_p)
    for model in design.MODELS:
        summaries[model]["failure_holm_adjusted_p"] = failure_adjusted[model]
        summaries[model]["loss_holm_adjusted_p"] = loss_adjusted[model]
        summaries[model]["frozen_thresholds_pass"] = bool(
            summaries[model]["failure_margin_testable"]
            and summaries[model]["loss_margin_testable"]
            and summaries[model]["failure_risk_difference"] <= 0.01
            and failure_adjusted[model] <= 0.05
            and summaries[model]["failure_penalized_cost_gain_seconds"] > 0.0
            and loss_adjusted[model] <= 0.05
        )
    return {
        "models": summaries,
        "all_three_fixed_models_pass": all(
            summaries[model]["frozen_thresholds_pass"] for model in design.MODELS
        ),
        "inference": plan["inference"],
        "margin_test_note": (
            "Holm-adjusted one-sided centered cluster-bootstrap margin tests are the frozen decisions. "
            "Displayed percentile bounds are explicitly unadjusted and descriptive. Degenerate cluster "
            "contrasts are untestable and fail closed; no finite-sample population-risk guarantee is made."
        ),
    }


def summarize() -> None:
    plan, _ = binding_inputs()
    plan_sha, binding_sha = digest(PLAN), digest(BINDING)
    execution = OUT / "execution"
    spawn = read(execution / "spawn.json")
    receipt = read(execution / "exit.json")
    launch = read(execution / "launch.json")
    done = read(execution / "collection_complete.json")
    validate_completion(
        plan,
        plan_sha,
        binding_sha,
        spawn,
        receipt,
        launch,
        done,
        digest(execution / "collection_complete.json"),
        digest(execution / "blocks_manifest.json"),
    )
    check_acknowledgment(execution, "worker", launch["launcher_pid"], launch["native_pid"])
    blocks_manifest = read(execution / "blocks_manifest.json")
    validate_blocks_manifest(plan, plan_sha, binding_sha, blocks_manifest)
    torch.set_num_threads(1)
    models = {model: load_coupled(ROOT / item["path"])[0] for model, item in plan["models"].items()}
    rows = []
    timing: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    total = 0
    for group in plan["groups"]:
        directory = OUT / "blocks" / f"{group['block_index']:05d}"
        complete = read(directory / "complete.json")
        expected_metadata = metadata(group, plan_sha, binding_sha)
        if (
            any(complete[key] != value for key, value in expected_metadata.items())
            or complete["actual_sumo_runs"] != 13
            or complete["database_sha256"] != digest(directory / "observations.sqlite")
            or complete["seal_sha256"] != digest(directory / "observations.sqlite.seal.json")
            or len(complete["reviews"]) != 13
        ):
            raise ValueError("Complete unchanged V39 block receipt required")
        reviews = []
        for key, record, trace in observations(
            directory / "observations.sqlite", expected_metadata, schedule(group)
        ):
            method = {"rule": key[-2], "model": key[-1]}
            model = None if method["model"] == "none" else models[method["model"]]
            review = {"case": list(key), **auditor.audit_episode(plan, group, method, record, trace, model)}
            reviews.append(review)
            row = record["row"]
            row["initial_id"] = group["initial_id"]
            rows.append(row)
            timing[(row["rule"], row["model"], row["scenario"], row["plant_name"],
                    row["perception_name"])].extend(call["wall_ms"] for call in record["calls"])
            total += 1
        if reviews != complete["reviews"]:
            raise ValueError("Collection and final independent V39 audits differ")
    if total != plan["budget"]["actual_sumo_runs"]:
        raise ValueError("Incomplete V39 exact coverage")
    cells = []
    for key, calls in sorted(timing.items()):
        selected = [
            row
            for row in rows
            if (row["rule"], row["model"], row["scenario"], row["plant_name"],
                row["perception_name"]) == key
        ]
        cells.append(
            {
                "rule": key[0],
                "model": key[1],
                "regime": key[2],
                "plant": key[3],
                "perception": key[4],
                "episodes": len(selected),
                "failures": sum(row["failure"] for row in selected),
                "adopted": sum(row["adopted"] for row in selected),
                "mean_failure_penalized_loss": float(
                    np.mean([row["failure_penalized_loss"] for row in selected])
                ),
                "controller_calls": len(calls),
                "controller_p50_p95_max_ms": np.percentile(calls, [50, 95, 100]).tolist(),
                "controller_calls_over_500ms": sum(value > 500.0 for value in calls),
            }
        )
    primary = primary_analysis(plan, rows)
    analysis = OUT / "analysis"
    analysis.mkdir(exist_ok=False)
    report = {
        "plan_sha256": plan_sha,
        "execution_binding_sha256": binding_sha,
        "exit_sha256": digest(execution / "exit.json"),
        "actual_sumo_runs": total,
        "paired_condition_blocks": plan["budget"]["paired_condition_blocks"],
        "independent_initials": plan["budget"]["independent_initials"],
        "all_episodes_independently_replayed": True,
        "all_fallback_identities_exact": True,
        "primary": primary,
        "cells": cells,
        "scope": plan["scope"],
        "new_formal_safety_certificate": False,
        "partial_efficacy_outcomes_read": False,
    }
    exclusive_json(analysis / "report.json", report)
    exclusive_json(analysis / "sha256.json", {"report.json": digest(analysis / "report.json")})
    print(
        json.dumps(
            {
                "completed": True,
                "actual_sumo_runs": total,
                "all_three_fixed_models_pass": primary["all_three_fixed_models_pass"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "worker", "watch", "summarize"))
    globals()[parser.parse_args().command]()
