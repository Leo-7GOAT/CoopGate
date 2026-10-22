"""Frozen fault-injection study of process-isolated SUMO control callbacks.

Existing learned actors are forced eligible to exercise the software mechanism;
this is not performance-gate calibration or a physical actuation-delay model.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.deadline_profile_v26 import DeadlineProfile
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.joint_planner_v15 import JointRolloutGate
from execute_prospective_v23 import sources_committed
from experiment_ego_v13 import run_episode
from profile_phase_runtime_v24 import committed, digest, save
from study_gate_v19 import regime_for
from verify_development_v14 import trace_audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v26/deadline_sumo"
OLD = ROOT / "artifacts/v25/buffered_phase_sumo_audit/plan.json"
VARIANTS = ("baseline", "rejected", "normal", "slow", "hang", "exception", "nan")
NEW_SOURCES = ("src/coopgate/deadline_profile_v26.py", "tests/test_deadline_profile_v26.py",
               "tools/audit_deadline_sumo_v26.py")
WARM = np.array([[-80., 12.], [-85., 13.], [-120., 11.]])


class FaultingActor:
    def __init__(self, actor, mode, fault_step):
        self.actor, self.mode, self.fault_step = actor, mode, fault_step
        self.step = 0

    def select(self, state, previous):
        self.actor.step = self.step
        if self.step == self.fault_step:
            if self.mode == "slow":
                time.sleep(2.)
            elif self.mode == "hang":
                while True:
                    time.sleep(.1)
            elif self.mode == "exception":
                raise RuntimeError("Injected primary execution failure")
            elif self.mode == "nan":
                return np.full(3, np.nan), {}
        result = self.actor.select(state, previous)
        self.step += 1
        return result


@dataclass
class PrimaryFactory:
    model_path: str
    model_sha256: str
    mode: str
    fault_step: int

    def __call__(self):
        torch.set_num_threads(1)
        if digest(self.model_path) != self.model_sha256:
            raise ValueError("Frozen actor changed")
        actor = JointPolicyGate(load_coupled(self.model_path)[0])
        actor.select(WARM, np.zeros(3))
        actor.step = 0
        return FaultingActor(actor, self.mode, self.fault_step)


class ObservedGate:
    """Measure the full callback, including failed calls, without legacy adapters."""
    def __init__(self, gate):
        self.gate, self.calls = gate, []

    def select(self, *args):
        start = time.perf_counter()
        error = None
        try:
            return self.gate.select(*args)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.calls.append(dict(wall_ms=1000*(time.perf_counter()-start), error=error))


def freeze():
    old = json.loads(OLD.read_text())
    sources = dict(old["source_sha256"])
    sources.update({p: digest(ROOT/p) for p in NEW_SOURCES})
    sources_committed([ROOT/p for p in sources])
    spec = old["spec"]
    cases = [[mode, model, variant] for mode in ("clean", "uniform", "bias", "pulse")
             for model in ("1802", "1812", "1822") for variant in VARIANTS]
    plan = dict(purpose=__doc__, source_sha256=sources, old_plan_sha256=digest(OLD),
        models=old["models"], seed=spec["seed_base"], regime=regime_for(spec, 0),
        old_initials=1, new_independent_initials=0, cases=cases, actual_sumo_runs=84,
        primary_eligibility="Forced for normal and fault branches; not confidence calibration",
        fault_step=4, slow_seconds=2., proposal_budget_seconds=.45, decision_period_seconds=.5,
        timing="Complete parent callback, including snapshot, dispatch, reply decoding wait, command validation and fallback. Report startup and outside-callback child cleanup separately. Retain all early/late faults and actual timer overruns.",
        sim_time="SUMO advances after select; host waiting does not model physical evolution or actuator delay. Software timing only, no hard-real-time or road-safety guarantee.",
        runtime="One persistent process per admitted episode, no child for rejection; after first fault permanently use prechecked same-state baseline; late worker replies never applied; explicitly close every child after the episode",
        audit="Independently reconstruct every physical substep and each actually applied baseline/actor command; compare rejected trajectories to direct baseline; retain all failures and original admission labels",
        active_calibration_outcomes_read=False, calibration_controller_changed=False,
        new_risk_certificate=False, restart="No silent resume or replacement; preserve all outputs on failure")
    assert len(cases) == plan["actual_sumo_runs"]
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/"plan.json", plan)
    print(json.dumps(dict(plan_sha256=digest(OUT/"plan.json"), actual_sumo_runs=len(cases))), flush=True)


def review(record, model):
    row, trace = record["row"], record["trace"]
    trace_audit(trace, row)
    base, actor = JointRolloutGate(), JointPolicyGate(model)
    state, previous = np.array(row["initial"]), np.zeros(3)
    baseline_checks = actor_checks = 0
    latched = False
    for step, item in enumerate(trace):
        used = bool(item.get("deadline_primary_used", False))
        if used:
            assert not latched
            controller = actor
            actor_checks += 1
        else:
            controller = base
            baseline_checks += 1
        controller.step = step
        command, _ = controller.select(state, previous)
        np.testing.assert_array_equal(command, item["command"])
        if row["variant"] != "baseline":
            assert item["performance_admitted"] == (row["variant"] != "rejected")
            latched |= item["fault_fallback"]
            if latched:
                assert item["fault_fallback"] and not used
        state, previous = np.array(item["x"]), command
    assert len(record["calls"]) == len(trace) + int(row["aborted"])
    assert not record["shutdown"].get("worker_alive", False)
    return dict(physical_traces=1, baseline_commands=baseline_checks, actor_commands=actor_checks)


def run():
    committed(OUT/"plan.json")
    plan = json.loads((OUT/"plan.json").read_text())
    assert digest(OLD) == plan["old_plan_sha256"]
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT/name) != expected:
            raise ValueError("Frozen source differs: " + name)
    sources_committed([ROOT/p for p in plan["source_sha256"]])
    torch.set_num_threads(1)
    target = OUT/"results"
    target.mkdir(exist_ok=False)
    models = {}
    for seed, meta in plan["models"].items():
        assert digest(ROOT/meta["path"]) == meta["sha256"]
        models[seed] = load_coupled(ROOT/meta["path"])[0]
    JointRolloutGate().select(WARM, np.zeros(3))
    for model in models.values():
        JointPolicyGate(model).select(WARM, np.zeros(3))
    records, references = [], {}
    with DisturbedMergeEnv() as env:
        for index, (mode, model_id, variant) in enumerate(plan["cases"]):
            holder = {}
            meta = plan["models"][model_id]
            def factory(state):
                if variant == "baseline":
                    controller = JointRolloutGate()
                else:
                    controller = DeadlineProfile(
                        PrimaryFactory(str(ROOT/meta["path"]), meta["sha256"], variant, plan["fault_step"]),
                        JointRolloutGate(), admitted=variant != "rejected",
                        budget_seconds=plan["proposal_budget_seconds"]).prepare()
                holder["gate"] = ObservedGate(controller)
                return holder["gate"]
            started = time.perf_counter()
            shutdown, cleanup_ms = {}, 0.
            try:
                row, trace, initial = run_episode(env, plan["seed"], plan["regime"], mode, factory, True)
            finally:
                if "gate" in holder and isinstance(holder["gate"].gate, DeadlineProfile):
                    close_started = time.perf_counter()
                    shutdown = holder["gate"].gate.close()
                    cleanup_ms = 1000*(time.perf_counter()-close_started)
            row.update(initial=initial.tolist(), variant=variant, model=model_id,
                       initial_admitted=variant not in ("baseline", "rejected"))
            record = dict(case=[mode, model_id, variant], row=row, trace=trace,
                          calls=holder["gate"].calls, shutdown=shutdown,
                          cleanup_ms=cleanup_ms, total_wall_seconds=time.perf_counter()-started)
            save(target/f"case_{index:03d}.json", record)
            audit = review(record, models[model_id])
            save(target/f"case_{index:03d}.review.json", audit)
            if variant == "baseline":
                references[mode, model_id] = trace
            elif variant == "rejected":
                expected = references[mode, model_id]
                assert len(trace) == len(expected)
                for observed, old in zip(trace, expected, strict=True):
                    for key in ("t", "x", "command", "disturbance"):
                        np.testing.assert_array_equal(observed[key], old[key])
            records.append(record)
            print(json.dumps(dict(completed_runs=len(records), case=record["case"],
                                  completed=row["completed"], independently_checked=True)), flush=True)
    assert len(records) == 84
    groups = {}
    for variant in VARIANTS:
        group = [r for r in records if r["row"]["variant"] == variant]
        calls = [c["wall_ms"] for r in group for c in r["calls"]]
        trigger = [c["wall_ms"] for r in group for c,t in zip(r["calls"],r["trace"])
                   if t.get("deadline_triggered_this_step",False)]
        groups[variant] = dict(runs=len(group), completed=sum(r["row"]["completed"] for r in group),
            aborted=sum(r["row"]["aborted"] for r in group),
            collision=sum(r["row"]["collision"] for r in group),
            fallback_episodes=sum(any(t.get("fault_fallback",False) for t in r["trace"]) for r in group),
            primary_applied_steps=sum(t.get("deadline_primary_used",False) for r in group for t in r["trace"]),
            latched_before_fault_step=sum(any(t.get("fault_fallback",False) for t in r["trace"][:plan["fault_step"]]) for r in group),
            calls=len(calls), callback_p95_ms=float(np.percentile(calls,95)), callback_max_ms=max(calls),
            callback_over_500ms=sum(v>500 for v in calls), timeout_callbacks_ms=trigger,
            setup_ms=[r["row"]["setup_ms"] for r in group], cleanup_ms=[r["cleanup_ms"] for r in group])
    summary = dict(plan_sha256=digest(OUT/"plan.json"), actual_sumo_runs=84, new_independent_initials=0,
                   groups=groups, independently_reviewed_runs=84,
                   active_calibration_outcomes_read=False, new_risk_certificate=False,
                   hard_real_time_certified=False, physical_compute_delay_simulated=False)
    save(target/"summary.json", summary)
    save(target/"sha256.json", {p.name:digest(p) for p in sorted(target.iterdir()) if p.is_file()})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze":freeze, "run":run}[args.action]()
