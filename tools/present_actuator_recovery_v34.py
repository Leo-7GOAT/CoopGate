"""Present the completed physical-recovery development without partial claims."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from coopgate.episode_store_v21 import observations
from execute_prospective_v23 import exclusive_json
from profile_phase_runtime_v24 import digest
import study_actuator_recovery_v34 as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v34/recovery_presentation'


def main():
    plan = study.inputs()
    report = study.read(study.OUT/'analysis/report.json')
    proof = study.read(study.OUT/'completion_verification.json')
    if (proof['summary_exit_code'] != 0 or proof['report_sha256'] != digest(study.OUT/'analysis/report.json')
            or report['actual_sumo_runs'] != plan['actual_sumo_runs']
            or not report['all_independent_physical_and_observer_reviews_pass']):
        raise ValueError('Complete successful independent review is required')
    cells = [r for r in report['cells'] if r['scene'] is None]
    if [r['plant'] for r in cells] != list(study.PROFILES) or any(r['runs'] != 160 for r in cells):
        raise ValueError('Every plant and initial condition is required')
    old = study.baseline_records()
    shared_fields = ('t','x','command','innovation','actuator_states','disturbance')
    paired_physics = []
    for key,record,trace in observations(study.OUT/'observations.sqlite',study.metadata(),plan['cases']):
        _,reference = old[(*key[:-1],'baseline')]
        exact = len(trace) == len(reference) and all(
            all(a[field] == b[field] for field in shared_fields) for a,b in zip(trace,reference))
        paired_physics.append(dict(case=list(key),shared_physical_trace_fields_exact=exact))
    summary = dict(actual_sumo_runs=report['actual_sumo_runs'],old_initials=160,
        commands=sum(r['review']['commands'] for r in report['cases']),
        recovered_old_incomplete=sum(not r['baseline']['completed'] and r['row']['completed'] for r in report['cases']),
        newly_incomplete=sum(r['baseline']['completed'] and not r['row']['completed'] for r in report['cases']),
        physical_violations=sum(r['review']['physical_violations'] for r in report['cases']),
        new_independent_initials=0,new_performance_risk_certificate=False,cells=cells,
        identical_physical_trajectories_by_plant={p:sum(r['case'][4] == p and r['shared_physical_trace_fields_exact']
            for r in paired_physics) for p in study.PROFILES},
        callbacks=report['callbacks'],report_sha256=digest(study.OUT/'analysis/report.json'))
    OUT.mkdir(exist_ok=False)
    exclusive_json(OUT/'summary.json',summary)
    exclusive_json(OUT/'paired_physical_trace_comparison.json',dict(fields=shared_fields,cases=paired_physics))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,2,figsize=(11,4.7),dpi=150)
    x = np.arange(4)
    for index,(key,label,color) in enumerate((('baseline_completed','Original nominal baseline','#737f8c'),
                                             ('recovery_completed','Actuator-aware recovery','#157a9b'))):
        values = [r[key] for r in cells]
        bars = axes[0].bar(x+(index-.5)*.36,values,.36,color=color,label=label)
        axes[0].bar_label(bars,padding=3,fontsize=9)
    axes[0].set_ylabel('Completed episodes / 160')
    axes[0].set_ylim(0,185)
    bars = axes[1].bar(x,[r['recovery_episodes'] for r in cells],.55,color='#b57b2a')
    axes[1].bar_label(bars,padding=3,fontsize=9)
    axes[1].set_ylabel('Episodes using stopping recovery / 160')
    axes[1].set_ylim(0,max(10,max(r['recovery_episodes'] for r in cells)*1.3))
    for ax in axes:
        ax.set_xticks(x,['Nominal','Gain / drag','Mild lag','Strong lag'])
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
    fig.suptitle('Actuator-memory recovery: all 640 old scene / plant conditions',y=.98,fontsize=13)
    fig.text(.5,.035,'160 old initials shared across four plants; all original failures retained.\n'
        'Declared plant-class bounds are additional controller information; no new MARL or performance-risk certificate.',
        ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.14,1,.83))
    for extension in ('png','svg'):
        fig.savefig(OUT/f'recovery_completion.{extension}')
    plt.close(fig)
    rows = ['| 动力学 | 原基线完成 /160 | 新恢复完成 /160 | 新中止 | 新超时 | 使用恢复的回合 | 双方完整配对数 | 完整配对子集成本改善 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    names = ['名义','增益/阻力','轻滞后','强滞后']
    for name,r in zip(names,cells):
        gain = r['complete_pair_descriptive_gain_percent']
        display = '未定义' if gain is None else f'{gain:.3f}%'
        rows.append(f"| {name} | {r['baseline_completed']} | {r['recovery_completed']} | {r['recovery_aborted']} | {r['recovery_timeouts']} | {r['recovery_episodes']} | {r['both_complete_pairs']} | {display} |")
    details = ['| 场景 | 动力学 | 原完成 /32 | 新完成 /32 | 使用恢复 | 完整配对子集成本改善 |',
               '|---|---|---:|---:|---:|---:|']
    for r in report['cells']:
        if r['scene'] is None: continue
        gain = r['complete_pair_descriptive_gain_percent']
        display = '未定义' if gain is None else f'{gain:.3f}%'
        details.append(f"| {r['scene']} | {r['plant']} | {r['baseline_completed']} | {r['recovery_completed']} | {r['recovery_episodes']} | {display} |")
    failures = ['| 原失败种子 | 场景 | 原中止时刻 / s | 新完成时刻 / s | 新恢复决策次数 |',
                '|---|---|---:|---:|---:|']
    for r in report['cases']:
        if r['baseline']['completed']: continue
        failures.append(f"| {r['baseline']['seed']} | {r['baseline']['scenario']} | {r['baseline']['elapsed']:.1f} | {r['row']['elapsed']:.1f} | {r['review']['recovery_steps']} |")
    callback = report['callbacks']
    text = f'''# 执行器状态恢复的完整开发结果

全部 {report['actual_sumo_runs']} 次真实 SUMO 执行和独立复核已完成，工作进程及汇总进程实际退出均为 0。范围为五场景、每场景 32 个旧初态及四类动力学，包含全部旧基线失败条件；新增独立初态为零。复核重建 {summary['commands']:,} 条命令、每个物理子步及在线内部状态区间。

{chr(10).join(rows)}

原未完成条件中，新恢复完成 {summary['recovered_old_incomplete']} 个；原完成条件中，新恢复未完成 {summary['newly_incomplete']} 个。全部实际执行的固定路线几何违反为 {summary['physical_violations']}。每个场景/动力学单元的完整数据都保留在报告中，不能只选择总表中较好的动力学。

旧/新运行的实际命令、位置、创新、内部执行器状态和实际误差也逐项比较。完全相同的轨迹数（各类分母 160）为：{json.dumps(summary['identical_physical_trajectories_by_plant'],ensure_ascii=False)}。它与控制器内部证书字段相同是不同命题；新控制器明确记录新的物理合同。

收益列仅描述双方均完成的配对子集；有任一失败时，它不构成全范围效能结论。原始截断成本、中止、超时和初始无法认证均保留。新控制器额外使用事先声明的参数类别范围，因此不是等信息的算法优势对照；与其他预测安全过滤器的全面比较尚未实现。安全恢复也不自动给出相对原基线的性能不退化保证。

全部完整调用共 {callback['calls']} 次，含 {callback['terminal_calls']} 次终止异常；p95 为 {callback['p50_p95_p99_max_ms'][1]:.3f} ms，最大值 {callback['p50_p95_p99_max_ms'][3]:.3f} ms，超过 500 ms 共 {callback['over_500ms']} 次。这是并发桌面负载下本控制器的测量，不是硬实时或原 MARL 门控的耗时证明。

数学条件是正确的类别界、|w|≤0.1、已知零初始执行器状态、准确观测和固定命令周期。区间停车论证及数值实现限制见[协议与推导](v34_actuator_recovery_protocol.md)。正在运行的正式性能风险校准及 V32 全网格仍使用原冻结代码；本研究不替换其控制器、不读取其中途效果、不继承其证书。

![全范围完成与恢复](../artifacts/v34/recovery_presentation/recovery_completion.png)

## 全部场景单元

{chr(10).join(details)}

## 全部原未完成条件

{chr(10).join(failures)}

证据：[全部条件与独立报告](../artifacts/v34/actuator_recovery_scope/analysis/report.json)、[真实完成核验](../artifacts/v34/actuator_recovery_scope/completion_verification.json)、[冻结计划](../artifacts/v34/actuator_recovery_scope/plan.json)。
'''
    with (ROOT/'docs/v34_complete_recovery_results.md').open('x',encoding='utf-8',newline='\n') as handle:
        handle.write(text)
    exclusive_json(OUT/'sha256.json',{p.name:digest(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
