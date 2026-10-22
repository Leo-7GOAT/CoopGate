"""Pair both deadline clocks over the entire existing fault-study grid.

All cases are old initials, with forced policy eligibility. Clock changes can
change decisions near a deadline; preserve and report such differences rather
than assuming equivalence or transferring a statistical risk certificate.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_deadline_sumo_v26 import PrimaryFactory, ObservedGate, VARIANTS, WARM, review
from coopgate.cooperative_iteration_v16 import load_coupled
from coopgate.cooperative_v15 import JointPolicyGate
from coopgate.deadline_profile_v26 import DeadlineProfile
from coopgate.disturbance_v9 import DisturbedMergeEnv
from coopgate.joint_planner_v15 import JointRolloutGate
from coopgate.precise_deadline_v26 import PreciseDeadlineProfile
from execute_prospective_v23 import sources_committed
from experiment_ego_v13 import run_episode
from profile_phase_runtime_v24 import committed, digest, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v26/precise_deadline_sumo"
OLD = ROOT / "artifacts/v26/deadline_sumo/plan.json"
CLASSES = {"coarse": DeadlineProfile, "precise": PreciseDeadlineProfile}
NEW_SOURCES = ("src/coopgate/precise_deadline_v26.py", "tests/test_precise_deadline_v26.py",
               "tools/audit_precise_deadline_v26.py", "tests/test_precise_deadline_audit_v26.py")
ROW_EXCLUDED = {"setup_ms", "controller_ms", "clock"}
TRACE_EXCLUDED = {"gate_ms", "baseline_preparation_ms", "primary_wait_ms", "deadline_setup_ms",
                  "primary_process_pid", "deadline_observed_overrun"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_pair(first, second):
    """Retain every physical, performance and admission field in the comparison."""
    a, b = first["row"], second["row"]
    row_diff = sorted(k for k in (a.keys() | b.keys()) - ROW_EXCLUDED if a.get(k) != b.get(k))
    x, y = first["trace"], second["trace"]
    trace_diff = []
    for step in range(max(len(x), len(y))):
        if step >= min(len(x), len(y)):
            trace_diff.append(dict(step=step, fields=["missing_step"]))
        else:
            fields = sorted(k for k in (x[step].keys() | y[step].keys()) - TRACE_EXCLUDED
                            if x[step].get(k) != y[step].get(k))
            if fields:
                trace_diff.append(dict(step=step, fields=fields))
    return dict(row_differences=row_diff, trace_differences=trace_diff,
                non_timing_equal=not row_diff and not trace_diff,
                paired_decisions=min(len(x), len(y)))


def freeze():
    old = read(OLD)
    sources = dict(old["source_sha256"])
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError("Frozen predecessor source changed: " + name)
    sources.update({name: digest(ROOT/name) for name in NEW_SOURCES})
    sources_committed([ROOT/name for name in sources])
    cases = []
    for i, case in enumerate(old["cases"]):
        for clock in (("coarse", "precise") if i % 2 == 0 else ("precise", "coarse")):
            cases.append([*case, clock])
    assert len(cases) == 168 and len({tuple(c) for c in cases}) == 168
    plan = dict(purpose=__doc__, old_plan_sha256=digest(OLD), source_sha256=sources,
        models=old["models"], seed=old["seed"], regime=old["regime"],
        cases=cases, actual_sumo_runs=168, paired_cases=84, old_independent_initials=1,
        new_independent_initials=0, fault_step=old["fault_step"], slow_seconds=old["slow_seconds"],
        proposal_budget_seconds=old["proposal_budget_seconds"],
        decision_period_seconds=old["decision_period_seconds"],
        order="Adjacent old/new pairs, alternating which clock executes first across all 84 cases",
        runtime="One serial experiment worker alongside the unchanged frozen calibration and full-scope audit",
        primary_eligibility=old["primary_eligibility"], timing=old["timing"], sim_time=old["sim_time"],
        scope="All three fixed models, four modes and seven branches from the completed 84-run study; no resampling or case selection",
        comparisons="Every row and trace field except explicitly named timing/process fields; retain all near-deadline differences and all early/late faults",
        row_excluded=sorted(ROW_EXCLUDED), trace_excluded=sorted(TRACE_EXCLUDED),
        audit="Independently check every trajectory and actual command using the unchanged predecessor reviewer; report complete-callback maxima, early timeouts, overshoots and process cleanup",
        active_calibration_outcomes_read=False, calibration_controller_changed=False,
        new_risk_certificate=False, hard_real_time_certified=False,
        restart="No implicit resume or replacement. Preserve outputs and record actual execution exit before analysis.")
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/"plan.json", plan)
    print(json.dumps(dict(actual_sumo_runs=168, plan_sha256=digest(OUT/"plan.json"))), flush=True)


def run():
    committed(OUT/"plan.json")
    plan = read(OUT/"plan.json")
    assert digest(OLD) == plan["old_plan_sha256"]
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT/name) != expected:
            raise ValueError("Frozen source changed: " + name)
    sources_committed([ROOT/name for name in plan["source_sha256"]])
    target = OUT/"results"
    target.mkdir(exist_ok=False)
    save(target/"clock_environment.json", {name: vars(time.get_clock_info(name))
                                         for name in ("monotonic", "perf_counter")})
    torch.set_num_threads(1)
    models = {}
    for model_id, meta in plan["models"].items():
        assert digest(ROOT/meta["path"]) == meta["sha256"]
        models[model_id] = load_coupled(ROOT/meta["path"])[0]
    JointRolloutGate().select(WARM, np.zeros(3))
    for model in models.values():
        JointPolicyGate(model).select(WARM, np.zeros(3))
    with DisturbedMergeEnv() as env:
        for index, (mode, model_id, variant, clock) in enumerate(plan["cases"]):
            holder = {}
            meta = plan["models"][model_id]
            def factory(state):
                if variant == "baseline":
                    controller = JointRolloutGate()
                else:
                    controller = CLASSES[clock](
                        PrimaryFactory(str(ROOT/meta["path"]), meta["sha256"], variant, plan["fault_step"]),
                        JointRolloutGate(), admitted=variant != "rejected",
                        budget_seconds=plan["proposal_budget_seconds"])
                    holder["profile"] = controller
                    controller.prepare()
                holder["gate"] = ObservedGate(controller)
                return holder["gate"]
            started = time.perf_counter()
            shutdown, cleanup_ms = {}, 0.
            try:
                row, trace, initial = run_episode(env, plan["seed"], plan["regime"], mode, factory, True)
            finally:
                if "profile" in holder:
                    close_started = time.perf_counter()
                    shutdown = holder["profile"].close()
                    cleanup_ms = 1000*(time.perf_counter()-close_started)
            row.update(initial=initial.tolist(), variant=variant, model=model_id, clock=clock,
                       initial_admitted=variant not in ("baseline", "rejected"))
            record = dict(case=[mode, model_id, variant, clock], row=row, trace=trace,
                          calls=holder["gate"].calls, shutdown=shutdown,
                          cleanup_ms=cleanup_ms, total_wall_seconds=time.perf_counter()-started)
            save(target/f"case_{index:03d}.json", record)
            save(target/f"case_{index:03d}.review.json", review(record, models[model_id]))
            print(json.dumps(dict(completed_runs=index+1, case=record["case"],
                                  completed=row["completed"], independently_checked=True)), flush=True)
    save(target/"run_finished.json", dict(actual_sumo_runs=len(plan["cases"]),
        plan_sha256=digest(OUT/"plan.json"), utc=datetime.now(timezone.utc).isoformat(),
        analysis_requires_observed_execution_exit=True))
    save(target/"sha256.json", {p.name: digest(p) for p in sorted(target.iterdir()) if p.is_file()})


def validate_records(records, plan):
    if len(records) != plan["actual_sumo_runs"] or [r["case"] for r in records] != plan["cases"]:
        raise ValueError("Exact complete frozen case order required")
    for record in records:
        row = record["row"]
        if [row["disturbance"], row["model"], row["variant"], row["clock"]] != record["case"]:
            raise ValueError("Record identity does not match case")
        if len(record["calls"]) != len(record["trace"]) + int(row["aborted"]):
            raise ValueError("Missing callback timing including terminal failures")
        if any(not np.isfinite(c["wall_ms"]) or c["wall_ms"] < 0 for c in record["calls"]):
            raise ValueError("Invalid callback time")


def summarize():
    receipt = read(OUT/"execution_observation.json")
    if receipt.get("exit_code") != 0 or receipt.get("plan_sha256") != digest(OUT/"plan.json"):
        raise ValueError("Actually observed successful execution exit required")
    plan, target = read(OUT/"plan.json"), OUT/"results"
    manifest = read(target/"sha256.json")
    expected = {f"case_{i:03d}{suffix}.json" for i in range(plan["actual_sumo_runs"])
                for suffix in ("", ".review")} | {"run_finished.json", "clock_environment.json"}
    if set(manifest) != expected:
        raise ValueError("Complete exact result and review manifest required")
    for name, expected_hash in manifest.items():
        assert digest(target/name) == expected_hash, name
    records = [read(target/f"case_{i:03d}.json") for i in range(plan["actual_sumo_runs"])]
    validate_records(records, plan)
    reviews = [read(target/f"case_{i:03d}.review.json") for i in range(plan["actual_sumo_runs"])]
    pairs = []
    for i in range(0, len(records), 2):
        a, b = records[i:i+2]
        assert a["case"][:3] == b["case"][:3] and {a["case"][3], b["case"][3]} == set(CLASSES)
        pairs.append(dict(case=a["case"][:3], **compare_pair(a, b)))
    groups = []
    for clock in CLASSES:
        for variant in VARIANTS:
            group = [r for r in records if r["row"]["clock"] == clock and r["row"]["variant"] == variant]
            assert len(group) == 12
            calls = [c["wall_ms"] for r in group for c in r["calls"]]
            triggers = [c["wall_ms"] for r in group for c,t in zip(r["calls"], r["trace"])
                        if t.get("deadline_triggered_this_step", False)]
            budget_ms = 1000*plan["proposal_budget_seconds"]
            error = [v-budget_ms for v in triggers]
            groups.append(dict(clock=clock, variant=variant, runs=12,
                completed=sum(r["row"]["completed"] for r in group),
                aborted=sum(r["row"]["aborted"] for r in group),
                collisions=sum(r["row"]["collision"] for r in group),
                fallback_steps=[next((s for s,t in enumerate(r["trace"]) if t.get("fault_fallback",False)), None) for r in group],
                primary_applied_steps=sum(t.get("deadline_primary_used",False) for r in group for t in r["trace"]),
                callback_count=len(calls), callback_p95_ms=float(np.percentile(calls,95)),
                callback_max_ms=max(calls), callback_over_500ms=sum(v>500 for v in calls),
                callback_over_budget=sum(v>budget_ms for v in calls),
                timeout_callbacks_ms=triggers, timeout_minus_budget_ms=error,
                early_timeouts=sum(v<0 for v in error),
                mean_absolute_timeout_budget_error_ms=None if not error else float(np.mean(np.abs(error))),
                setup_ms=[r["row"]["setup_ms"] for r in group], cleanup_ms=[r["cleanup_ms"] for r in group]))
    rejected_pairs = []
    for clock in CLASSES:
        for mode in ("clean", "uniform", "bias", "pulse"):
            for model_id in ("1802", "1812", "1822"):
                subset = {r["row"]["variant"]: r for r in records if r["case"][:2] == [mode,model_id] and r["case"][3] == clock}
                a, b = subset["baseline"]["trace"], subset["rejected"]["trace"]
                equal = len(a) == len(b) and all(all(x[k] == y[k] for k in ("t","x","command","disturbance")) for x,y in zip(a,b))
                rejected_pairs.append(dict(clock=clock, mode=mode, model=model_id, physical_equal=equal))
    shutdowns = [r["shutdown"] for r in records if r["shutdown"].get("worker_pid") is not None]
    assert len(shutdowns) == 120 and all(not s["worker_alive"] for s in shutdowns)
    audit = {key: sum(r[key] for r in reviews) for key in ("physical_traces","baseline_commands","actor_commands")}
    report = dict(plan_sha256=digest(OUT/"plan.json"), source_sha256=digest(__file__),
        execution_receipt_sha256=digest(OUT/"execution_observation.json"), result_manifest_sha256=digest(target/"sha256.json"),
        actual_sumo_runs=168, new_independent_initials=0, pairs=pairs, groups=groups,
        equal_pairs=sum(p["non_timing_equal"] for p in pairs),
        paired_decisions=sum(p["paired_decisions"] for p in pairs), rejected_baseline_pairs=rejected_pairs,
        independent_review=audit, reaped_workers=len(shutdowns),
        forced_terminations=sum(s["forced_termination"] for s in shutdowns),
        new_risk_certificate=False, hard_real_time_certified=False,
        physical_compute_delay_simulated=False, active_calibration_outcomes_read=False)
    presentation = OUT/"presentation"
    presentation.mkdir(exist_ok=False)
    save(presentation/"report.json", report)
    lines = ["# 统一高分辨率计时的完整配对检查", "",
        "重跑原有全部84种案例，每种分别使用原计时和高分辨率计时，交替先后顺序，共168次SUMO运行。两项正式研究同时运行；这些是一个旧初态的工程复测，不增加独立有效性样本。", "",
        "| 计时 | 分支 | 完成 | 调用数 | P95 ms | 最大 ms | >500 ms | 提前超时 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g['clock']} | {g['variant']} | {g['completed']}/12 | {g['callback_count']} | {g['callback_p95_ms']:.3f} | {g['callback_max_ms']:.3f} | {g['callback_over_500ms']} | {g['early_timeouts']} |")
    lines += ["", f"非计时字段完全一致：{report['equal_pairs']}/84对，配对决策{report['paired_decisions']}步。所有差异路径、逐步原始数据和完整命令复核均保留；没有删除边界结果。",
        "", f"独立检查{audit['physical_traces']}条物理轨迹、{audit['baseline_commands']}条基线命令、{audit['actor_commands']}条主策略命令。120个子进程均已回收，启动和清理耗时单独保存。",
        "", "统一时钟不保证操作系统在截止时刻立即调度，也不限制不可抢占的基线计算和主线程复核时间。实际调用仍可能超过450 ms预算或500 ms控制周期。SUMO等待期间不推进物理状态，未模拟计算延迟行驶；结果不构成硬实时、实车安全或新性能风险保证。近截止时刻的策略选择可能因修正而变化，旧认证不能自动转用。"]
    (presentation/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    save(presentation/"sha256.json", {p.name:digest(p) for p in sorted(presentation.iterdir()) if p.is_file()})
    print(json.dumps(dict(actual_sumo_runs=168, equal_pairs=report["equal_pairs"], independent_review=audit,
        callback_over_500ms=sum(g["callback_over_500ms"] for g in groups))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "summarize"))
    args = parser.parse_args()
    {"freeze":freeze, "run":run, "summarize":summarize}[args.action]()
