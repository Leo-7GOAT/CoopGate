"""Present the complete process-isolation study only after observed execution exit."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from profile_phase_runtime_v24 import digest, save

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/v26/deadline_sumo"
OUT = ROOT / "artifacts/v26/deadline_sumo_presentation"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    receipt = read(STUDY / "execution_observation.json")
    if receipt.get("exit_code") != 0 or receipt.get("plan_sha256") != digest(STUDY / "plan.json"):
        raise ValueError("A successful actually observed execution exit is required")
    plan = read(STUDY / "plan.json")
    result = STUDY / "results"
    for name, expected in read(result / "sha256.json").items():
        assert digest(result / name) == expected, name
    records = [read(result / f"case_{i:03d}.json") for i in range(plan["actual_sumo_runs"])]
    assert [r["case"] for r in records] == plan["cases"] and len(records) == 84
    assert len(list(result.glob("case_*.review.json"))) == 84
    reviews = [read(result / f"case_{i:03d}.review.json") for i in range(84)]
    original = read(result / "summary.json")
    groups = []
    for variant, saved in original["groups"].items():
        group = [r for r in records if r["row"]["variant"] == variant]
        assert len(group) == 12
        calls = [c["wall_ms"] for r in group for c in r["calls"]]
        fallback_steps = [next((k for k,t in enumerate(r["trace"]) if t.get("fault_fallback",False)), None) for r in group]
        triggers = [c["wall_ms"] for r in group for c,t in zip(r["calls"],r["trace"])
                    if t.get("deadline_triggered_this_step",False)]
        assert saved["calls"] == len(calls)
        assert saved["completed"] == sum(r["row"]["completed"] for r in group)
        assert saved["fallback_episodes"] == sum(s is not None for s in fallback_steps)
        assert saved["callback_max_ms"] == max(calls)
        assert saved["callback_over_500ms"] == sum(v > 500 for v in calls)
        assert saved["timeout_callbacks_ms"] == triggers
        groups.append(dict(variant=variant, runs=12, completed=saved["completed"],
            faults=saved["fallback_episodes"], fallback_steps=fallback_steps,
            parent_callbacks=len(calls), callback_median_ms=float(np.median(calls)),
            callback_p95_ms=float(np.percentile(calls,95)), callback_max_ms=max(calls),
            callback_over_budget=sum(v>1000*plan["proposal_budget_seconds"] for v in calls),
            callback_over_500ms=sum(v>500 for v in calls), timeout_callback_ms=triggers,
            startup_ms=[r["row"]["setup_ms"] for r in group], cleanup_ms=[r["cleanup_ms"] for r in group]))
    audit = {key:sum(r[key] for r in reviews) for key in ("physical_traces","baseline_commands","actor_commands")}
    shutdowns = [r["shutdown"] for r in records if r["shutdown"].get("worker_pid") is not None]
    assert len(shutdowns) == 60 and all(not r["worker_alive"] for r in shutdowns)
    assert all(t["performance_admitted"] for r in records if r["row"]["variant"] in ("slow","hang","exception","nan") for t in r["trace"])
    result_report = dict(utc=datetime.now(timezone.utc).isoformat(), plan_sha256=digest(STUDY/"plan.json"),
        execution_observation_sha256=digest(STUDY/"execution_observation.json"),
        result_manifest_sha256=digest(result/"sha256.json"), original_summary_sha256=digest(result/"summary.json"),
        source_sha256=digest(__file__), actual_sumo_runs=84, old_initials=1, new_independent_initials=0,
        variants=groups, independent_review=audit, spawned_workers=60,
        all_worker_handles_reaped=True, forced_worker_terminations=sum(r["forced_termination"] for r in shutdowns),
        original_fault_eligibility_retained=True,
        paired_rejected_baseline_trajectories=12, new_risk_certificate=False,
        hard_real_time_certified=False, physical_computation_delay_simulated=False,
        active_calibration_outcomes_read=False)
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/"report.json", result_report)
    lines = ["# 进程隔离与基线接管的完整仿真检查", "",
        "全部84次执行结束并取得实际成功退出记录后，汇总一个旧初态、三份固定模型、四种扰动及七种运行分支。学习控制器在相应分支被人为设为可用，实验不检验置信度门控的筛选效果，也不增加独立有效性样本。", "",
        "| 分支 | 完成 | 发生回退 | 调用数 | P95 ms | 最大 ms | 超过500 ms |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"baseline":"直接基线","rejected":"初始拒绝","normal":"正常主控制器","slow":"第4步延迟2秒","hang":"第4步永久卡住","exception":"第4步异常","nan":"第4步NaN"}
    for group in groups:
        lines.append(f"| {labels[group['variant']]} | {group['completed']}/12 | {group['faults']}/12 | {group['parent_callbacks']} | {group['callback_p95_ms']:.3f} | {group['callback_max_ms']:.3f} | {group['callback_over_500ms']} |")
    lines += ["", "内部预算为450 ms，控制周期为500 ms。完整父进程调用包含基线准备、派发、等待、命令复核与返回处理；模型装载/预热和回合结束后的子进程清理分别报告。报告保存超过450 ms和超过500 ms的计数，不能将定时器到期等同于回调恰好在预算时刻完成。", "",
        f"独立复核重建{audit['physical_traces']}条物理轨迹、{audit['baseline_commands']}条基线命令和{audit['actor_commands']}条实际采用的主控制器命令。12条初始拒绝轨迹与直接基线逐步一致。60个子进程全部完成回收，故障分支保留初始准入标签与实际采用历史。", "",
        "SUMO在控制回调返回后推进物理时间，本实验没有模拟等待期间车辆继续运动，也没有计算网络通信延迟。这是软件阻塞隔离和同状态基线接管证据，不是实车计算延迟下的安全定理、硬实时保证或新的性能风险证书。超时回退改变闭环策略，原控制器的统计认证不能直接转用。"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    save(OUT/"sha256.json", {p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    print(json.dumps(dict(actual_sumo_runs=84, independent_review=audit,
                         over_500ms=sum(g["callback_over_500ms"] for g in groups)), ensure_ascii=False))


if __name__ == "__main__":
    main()
