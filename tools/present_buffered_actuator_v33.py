"""Present only the complete, successfully reviewed actuator timing study."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import audit_buffered_actuator_v33 as audit
from coopgate.episode_store_v21 import observations
from execute_prospective_v23 import exclusive_json
from profile_phase_runtime_v24 import digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v33/buffered_actuator_presentation'


def main():
    plan=audit.inputs();report=audit.read(audit.OUT/'analysis/report.json')
    proof=audit.read(audit.OUT/'completion_verification.json')
    if (proof['report_sha256']!=digest(audit.OUT/'analysis/report.json')
            or proof['summary_exit_code']!=0 or not report['all_non_timing_fields_exact']
            or report['actual_sumo_runs']!=plan['actual_sumo_runs']):
        raise ValueError('Complete independently reviewed actual execution required')
    records={k:r for k,r,_ in observations(audit.OUT/'observations.sqlite',audit.metadata(),audit.schedule(plan))}
    groups=[]
    for label,origin in [('All pairs',None),('Old integration','preflight'),('Old baseline failures','all_old_baseline_failures')]:
        selected=[g for g in plan['groups'] if origin is None or g['source']==origin]
        group=dict(group=label,paired_cases=len(selected))
        for impl in audit.IMPLEMENTATIONS:
            calls=[c for g in selected for c in records[(*g['case'],impl)]['calls']]
            timing=audit.times(calls)
            group[impl]=dict(**timing,mean_ms=timing['sum_ms']/timing['calls'],
                over_500ms_percent=100*timing['over_500ms']/timing['calls'])
        group['time_reduction_percent']=100*(1-group['buffered']['sum_ms']/group['original']['sum_ms'])
        if group['original']['calls']!=group['buffered']['calls']:raise ValueError('Unequal paired callback counts')
        groups.append(group)
    for impl in audit.IMPLEMENTATIONS:
        for key in ('calls','over_500ms','terminal_calls'):
            if groups[0][impl][key]!=report['summary'][impl][key]:raise ValueError('Whole-study timing reconstruction differs')
        np.testing.assert_allclose(groups[0][impl]['sum_ms'],report['summary'][impl]['sum_ms'],rtol=0,atol=1e-6)
    unique={tuple(g['case'][:4]) for g in plan['groups']}
    summary=dict(report_sha256=digest(audit.OUT/'analysis/report.json'),actual_sumo_runs=report['actual_sumo_runs'],
        paired_cases=report['paired_cases'],commands=sum(p['commands'] for p in report['pairs']),
        completed_pairs=sum(p['completed'] for p in report['pairs']),aborted_pairs=sum(p['aborted'] for p in report['pairs']),
        out_of_physical_contract_pairs=sum(p['over_error_bound'] for p in report['pairs']),
        old_initials=len(unique),new_independent_initials=0,new_risk_certificate=False,groups=groups)
    OUT.mkdir(exist_ok=False)
    exclusive_json(OUT/'summary.json',summary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,axes=plt.subplots(1,3,figsize=(14,5.0));x=np.arange(3);width=.34
    colors=['#737f8c','#157a9b']
    specs=[('Mean full callback (ms)',lambda d:d['mean_ms']),
        ('Full callback p95 (ms)',lambda d:d['p50_p95_p99_max_ms'][1]),
        ('Callbacks exceeding 500 ms (%)',lambda d:d['over_500ms_percent'])]
    for ax,(ylabel,value) in zip(axes,specs):
        maximum=0.
        for ki,impl in enumerate(audit.IMPLEMENTATIONS):
            values=[value(g[impl]) for g in groups];maximum=max(maximum,*values)
            bars=ax.bar(x+(ki-.5)*width,values,width,color=colors[ki],label=impl.title())
            ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=8)
        ax.set_ylim(0,max(1,maximum*1.25));ax.set_xticks(x,['All pairs','Old integration','Old baseline\nfailures'])
        ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(f"Exact online actuator optimization: {report['paired_cases']} pairs / {report['actual_sumo_runs']} SUMO runs",y=.98,fontsize=13)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
    fig.text(.5,.035,f"All {summary['commands']:,} commands and non-timing fields match; {summary['aborted_pairs']} abort pairs retained. "
        f"{summary['old_initials']} old initials; no new efficacy samples.\nConcurrent desktop load, alternating execution order, terminal exceptions included; no hard-real-time guarantee.",ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.14,1,.83))
    for suffix in ('png','svg'):fig.savefig(OUT/f'buffered_actuator_timing.{suffix}')
    plt.close(fig)
    table=['| 范围 | 配对条件 | 每实现调用数 | 原均值 / ms | 优化均值 / ms | 原 p95 / ms | 优化 p95 / ms | 累计耗时减少 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    names=['全部','原集成条件','全部旧基线失败条件']
    for name,g in zip(names,groups):
        a,b=g['original'],g['buffered']
        table.append(f"| {name} | {g['paired_cases']} | {a['calls']} | {a['mean_ms']:.3f} | {b['mean_ms']:.3f} | {a['p50_p95_p99_max_ms'][1]:.3f} | {b['p50_p95_p99_max_ms'][1]:.3f} | {g['time_reduction_percent']:.3f}% |")
    a,b=groups[0]['original'],groups[0]['buffered']
    text=f'''# 在线执行器门控的完整等价优化结果

固定的 {report['actual_sumo_runs']} 次实际 SUMO 运行、{report['paired_cases']} 组成对条件已全部完成，工作进程实际退出为 0，完整独立汇总也实际退出为 0。优化前后的 {summary['commands']:,} 条命令和全部非计时字段一致，内部模型工作计数、有限扰动库证据及失败身份均保留。

范围包括原完整在线集成的全部 120 个条件，以及全部 10 个旧基线失败初态交叉三模型和两个预测版本所得的 60 个条件。所有 {summary['old_initials']} 个初态均来自旧研究；新增独立有效性样本为零。包含失败条件是预先声明的工程覆盖，不能当作总体交通分布。

{chr(10).join(table)}

计时包含全部成功与终止异常调用。原实现有 {a['over_500ms']} / {a['calls']} 次调用超过 500 ms，缓冲实现为 {b['over_500ms']} / {b['calls']}；优化后的最大调用仍为 {b['p50_p95_p99_max_ms'][3]:.3f} ms。均值或分位数改善不构成最坏时限保证。

保留 {summary['aborted_pairs']} 组控制器中止以及 {summary['out_of_physical_contract_pairs']} 组物理误差超出原合同的条件。本次优化保持控制器行为，因此没有修复或消除这些模型适用范围限制，也没有产生新的总体风险证书。正式风险校准和 V32 完整收益比较仍使用原冻结实现。

![完整配对测时](../artifacts/v33/buffered_actuator_presentation/buffered_actuator_timing.png)

来源：[完整独立报告](../artifacts/v33/buffered_actuator_audit/analysis/report.json)、[真实完成凭据](../artifacts/v33/buffered_actuator_audit/completion_verification.json)、[语法约束、等价性论证与实验协议](v33_buffered_actuator_protocol.md)。全部测量是在并发桌面负载下完成，配对执行先后交替；没有均质负载、硬实时或任意分布的认证结论。
'''
    with (ROOT/'docs/v33_complete_buffered_actuator_results.md').open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    exclusive_json(OUT/'sha256.json',{p.name:digest(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
