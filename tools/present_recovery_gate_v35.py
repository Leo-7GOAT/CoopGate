"""Present only a complete, successfully replayed recovery-gate comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from coopgate.recovery_gate_audit_v35 import cell_summary
from execute_prospective_v23 import exclusive_json
from profile_phase_runtime_v24 import digest
import study_recovery_gate_v35 as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v35/recovery_gate_presentation'
LABELS = ['Recovery\nbaseline','MPS\nadapter','MAMPS\nadapter','Nominal\nperformance','33-path\nperformance']
COLORS = ['#788490','#c38736','#3182a5','#2b8c6b','#7957ac']


def table(rows, prefix):
    lines = [f'| {prefix} | 规则 | 完成 | 采用回合 | 采用后个体退化 | 全范围成本改善 | p95 / ms | >500 ms 调用 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        gain = row['all_scope_cost_gain_percent']
        gain = '不报告（有未完成配对）' if gain is None else f'{gain:.3f}%'
        risk = '未定义' if row['empirical_conditional_harm'] is None else f"{row['harmful_adopted']}/{row['adopted_with_valid_baseline']}"
        p95 = '无调用' if row['callback_p95_ms'] is None else f"{row['callback_p95_ms']:.3f}"
        lines.append(f"| {row['label']} | {row['rule']} | {row['completed']}/{row['runs']} | {row['adopted']}/{row['runs']} | {risk} | {gain} | {p95} | {row['callback_over_500ms']}/{row['callback_count']} |")
    return '\n'.join(lines)


def main():
    plan = study.inputs()
    report = study.read(study.OUT/'analysis/report.json')
    proof = study.read(study.OUT/'completion_verification.json')
    if (proof['summary_exit_code'] != 0 or proof['worker_exit_code'] != 0
            or proof['report_sha256'] != digest(study.OUT/'analysis/report.json')
            or report['actual_sumo_runs'] != plan['actual_sumo_runs']
            or not report['all_physical_and_full_gate_replays_match']):
        raise ValueError('Complete actual execution and full gate replay required')
    failure_conditions = {tuple(g['condition']) for g in plan['groups'] if g['old_baseline_failed']}
    aggregates, strata, models = [], [], []
    for rule in study.RULES:
        cases = [p for p in report['cases'] if p['row']['rule'] == rule]
        expected = 30 if rule == 'baseline' else 90
        if len(cases) != expected:
            raise ValueError('All three model sets and all paired conditions required')
        aggregates.append(dict(label='全部',rule=rule,**cell_summary(cases)))
        for failed in (False,True):
            group = [p for p in cases if (tuple(p['case'][:5]) in failure_conditions) == failed]
            strata.append(dict(label='旧失败十条件' if failed else '原二十覆盖条件',rule=rule,**cell_summary(group)))
        for model in (('none',) if rule == 'baseline' else study.MODELS):
            group = [p for p in cases if p['row']['model'] == model]
            models.append(dict(label=model,rule=rule,**cell_summary(group)))
    summary = dict(actual_sumo_runs=390,old_unique_initials=15,new_independent_initials=0,
        commands=sum(p['review']['commands'] for p in report['cases']),
        physical_violations=sum(p['review']['physical_violations'] for p in report['cases']),
        all_gate_replays_match=True,all_30_common_baselines_match_archived_physics=True,
        new_risk_certificate=False,aggregates=aggregates,strata=strata,models=models,
        report_sha256=digest(study.OUT/'analysis/report.json'))
    OUT.mkdir(exist_ok=False)
    exclusive_json(OUT/'summary.json',summary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,2,figsize=(12,8.2),dpi=160)
    x = np.arange(5)
    values = [r['all_scope_cost_gain_percent'] for r in aggregates]
    axes[0,0].bar(x,[np.nan if v is None else v for v in values],color=COLORS)
    axes[0,0].axhline(0,color='#666666',linewidth=.7)
    axes[0,0].set_ylabel('Mean team cost improvement (%)')
    axes[0,0].set_title('Same recovery baseline; incomplete pairs withhold gain')
    for i,v in enumerate(values):
        axes[0,0].annotate('withheld' if v is None else f'{v:.2f}%',(i,0 if v is None else v),
            xytext=(0,5 if v is None or v >= 0 else -12),textcoords='offset points',ha='center',fontsize=9)
    adoption = [100*r['adopted']/r['runs'] for r in aggregates]
    axes[0,1].bar(x,adoption,color=COLORS)
    axes[0,1].set_ylabel('Episodes using learning (%)')
    axes[0,1].set_title('Actual changed command on an active vehicle')
    axes[0,1].set_ylim(0,119)
    for i,(v,r) in enumerate(zip(adoption,aggregates)):
        axes[0,1].text(i,v+3,f"{r['adopted']}/{r['runs']}",ha='center',fontsize=9)
    risk = [r['empirical_conditional_harm'] for r in aggregates]
    axes[1,0].bar(x,[np.nan if v is None else 100*v for v in risk],color=COLORS)
    axes[1,0].set_ylabel('Individual harm among adopted episodes (%)')
    axes[1,0].set_title('Descriptive only; zero adoption gives no risk estimate')
    axes[1,0].set_ylim(0,119)
    for i,(v,r) in enumerate(zip(risk,aggregates)):
        label = 'n/a' if v is None else f"{r['harmful_adopted']}/{r['adopted_with_valid_baseline']}"
        axes[1,0].text(i,3 if v is None else 100*v+3,label,ha='center',fontsize=9)
    times = [r['callback_p95_ms'] for r in aggregates]
    axes[1,1].bar(x,times,color=COLORS)
    axes[1,1].axhline(500,color='#a14545',linestyle='--',linewidth=1,label='500 ms decision period')
    axes[1,1].set_ylabel('Full select callback p95 (ms; logarithmic)')
    axes[1,1].set_yscale('log')
    axes[1,1].set_ylim(max(.01,min(times)/2),max(900,max(times)*2))
    axes[1,1].set_title('Includes terminal calls; no equal-compute claim')
    axes[1,1].legend(frameon=False,fontsize=8)
    for i,v in enumerate(times):
        axes[1,1].text(i,v*1.16,f'{v:.2f}',ha='center',fontsize=9)
    axes[0,0].margins(y=.25)
    for ax in axes.flat:
        ax.set_xticks(x,LABELS)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Recovery-gate integration: all 390 matched SUMO executions',fontsize=14,y=.98)
    fig.text(.5,.025,'15 reused initials / 30 conditions, including all 10 old baseline failures.\n'
        'Three fixed actor sets share conditions; these are not independent replicates or a new risk certificate.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.085,1,.95))
    for ext in ('png','svg'):
        fig.savefig(OUT/f'recovery_gate_comparison.{ext}')
    plt.close(fig)
    text = f'''# 同一恢复基线下的门控集成结果

全部 390 次实际 SUMO 执行及完整门控重放结束，实际进程退出均为 0；核验 {summary['commands']:,} 条命令和全部物理子步。三十条共同基线的实际物理轨迹均复现对应的 V34 记录，固定路线几何违反共 {summary['physical_violations']} 次。

{table(aggregates,'范围')}

学习方法每行合并三个固定模型，共 90 次执行；基线执行 30 次。两者按条件配对，模型重复不增加独立初态。退化指任一车辆成本或到达时间变差，或者执行失败；它不同于碰撞。采用仅统计实际改变未到达车辆命令的回合。

该对照使用同一恢复基线、执行器类别信息、停车检查、真实参数与外生创新。真实参数与外生创新用于设置相同仿真环境，不作为在线门控输入。两种性能门控共享候选生成和排序。MPS/MAMPS 是共享判据的切换适配器，不是原论文的完整复现；各类规则的搜索机制和计算量不同。

结果来自 15 个旧初态的 30 个条件，特意保留全部十个旧基线失败条件；没有新增独立效能样本。以下同时拆分覆盖条件和旧失败条件，避免将这种混合分布视作一般交通分布。这里的经验退化率不是风险证书，新恢复门控没有继承旧正式校准结论。

![全部配对对照](../artifacts/v35/recovery_gate_presentation/recovery_gate_comparison.png)

## 两类预定来源条件

{table(strata,'来源')}

## 全部固定模型

{table(models,'模型')}

逐场景、动力学和模型的完整单元、原始截断成本、完整配对子集及全部终止调用保留在[完整报告](../artifacts/v35/recovery_gate_integration/analysis/report.json)。[协议与数学边界](v35_recovery_joint_gate_protocol.md)及[实际退出核验](../artifacts/v35/recovery_gate_integration/completion_verification.json)给出复核入口。正在运行的正式风险校准及 V32 全网格未改动。
'''
    with (ROOT/'docs/v35_complete_recovery_gate_results.md').open('x',encoding='utf-8',newline='\n') as handle:
        handle.write(text)
    exclusive_json(OUT/'sha256.json',{p.name:digest(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(dict(actual_sumo_runs=390,commands=summary['commands'],aggregates=aggregates)),flush=True)


if __name__ == '__main__':
    main()
