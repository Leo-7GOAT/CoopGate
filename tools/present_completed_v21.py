"""Render the complete V21 evidence after successful reviews and frozen selection.

No selection, pooling, significance threshold or risk estimate is changed here.
Every transfer cell, all fixed models, failures and zero-adoption denominators
remain visible. A successful execution is not itself a scientific acceptance.
"""
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import continue_v21_20261022 as completion
from coopgate.prospective_validation_v21 import FAMILIES, MODELS, select_development_family
from probe_disturbed_gate_v19 import digest, save
import recover_interrupted_v21 as prior
import select_completed_v21_scoped as selection
from start_final_v19 import require_committed, verify_manifest


ROOT = completion.ROOT
OUTPUT = ROOT/'artifacts/v21/completed_presentation_20261022'
CHOICE = selection.ATTEMPTS/'attempt_1/selection.json'
FAMILY_LABELS = {'phase05':'Phase 50 ms', 'phase10':'Phase 100 ms', 'repair':'Action repair'}


def load_complete():
    require_committed([ROOT/'tools/present_completed_v21.py', CHOICE])
    choice = prior.read(CHOICE)
    scope, plan, old, protocols, _ = selection.inputs()
    receipt = prior.read(completion.EXECUTION/'supervisor_exit.json')
    launch = prior.read(completion.EXECUTION/'watcher_launch.json')
    if (receipt['exit_code'] != 0 or receipt['supervisor_pid'] != launch['supervisor_pid']
            or receipt['plan_sha256'] != digest(completion.FROZEN/'plan.json')
            or choice['scoped_process_rules_sha256'] != digest(selection.FROZEN/'plan.json')):
        raise ValueError('Completed selection/exit identity differs')
    completion.require_stage('resume', sum(bool(g['missing_cases']) for g in plan['groups']))
    completion.require_stage('review', 2)
    finished = prior.read(completion.EXECUTION/'completed.json')
    summaries = {}
    for study, protocol in protocols.items():
        directory = completion.OUTPUT/'results'/study
        verify_manifest(directory)
        summary = prior.read(directory/'summary.json')
        if (digest(directory/'summary.json') != finished['summary_sha256'][study]
                or summary['actual_sumo_runs'] != protocol['spec']['expected_actual_sumo_runs']
                or summary['independent_review_counts']['physical_traces'] != summary['actual_sumo_runs']
                or summary['protocol_sha256'] != old['studies'][study]['protocol_sha256']):
            raise ValueError('Full independently reviewed study required')
        summaries[study] = summary
    if digest(completion.OUTPUT/'results/phase/summary.json') != choice['development_summary_sha256']:
        raise ValueError('Selected family refers to different development results')
    expected = select_development_family(summaries['phase']['results'], summaries['phase']['independent_initials'])
    for name, value in expected.items():
        np.testing.assert_equal(choice[name], value)
    return summaries, protocols, choice


def ratio(k, n):
    return f'{k}/{n} ({100*k/n:.2f}%)' if n else '0/0（未定义）'


def interval(values):
    return f'[{values[0]:.5f}, {values[1]:.5f}]'


def phase_figure(phase):
    colors = {'phase05':'#157f9c', 'phase10':'#d0781c', 'repair':'#7755a5'}
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 4.1), sharex=True, sharey=True, layout='constrained')
    for ax, seed in zip(axes, MODELS):
        for family in FAMILIES:
            row = phase['results'][f'{family}_{seed}']
            x, y = 100*row['actual_adoption_rate'], row['actual_mean_gain_percent']
            ax.scatter(x, y, s=65, color=colors[family], label=FAMILY_LABELS[family], zorder=3)
            ax.annotate(f"{row['actual_harmful_adopted']}/{row['actual_adopted']}", (x,y),
                        xytext=(5, 5+list(FAMILIES).index(family)*5), textcoords='offset points', fontsize=8)
        ax.axvline(20, color='#777777', linestyle='--', linewidth=.8)
        ax.axhline(.5, color='#777777', linestyle='--', linewidth=.8)
        ax.set(title=f'Fixed model {seed}', xlabel='Actual episode adoption (%)')
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Mean team-cost improvement (%)')
    axes[0].legend(fontsize=8, loc='best')
    figure.suptitle('Complete development: adoption and gain\nLabels: harmful adopted / adopted; no risk certification', fontsize=12)
    figure.savefig(OUTPUT/'phase_gain_adoption.png', dpi=200)
    figure.savefig(OUTPUT/'phase_gain_adoption.pdf')
    plt.close(figure)


def transfer_figure(transfer, spec):
    lookup = {(r['scenario'], r['plant'], r['method']):r for r in transfer['cells']}
    values = [r['mean_gain_percent'] for r in transfer['cells'] if any(r['method'].startswith(f+'_') for f in FAMILIES)]
    bound = max(1., max(abs(v) for v in values))
    figure, axes = plt.subplots(3, 3, figsize=(13, 10.5), layout='constrained')
    plant_labels = {'nominal':'Nominal', 'gain_drag':'Gain/drag', 'lag_mild':'Mild lag', 'lag_strong':'Strong lag'}
    scene_labels = {'nominal':'Ordinary', 'dense':'Dense', 'slow_yield':'Slow yield',
                    'asymmetric_speed':'Unequal speed', 'tight_headway':'Tight gap'}
    for fi, family in enumerate(FAMILIES):
        for si, seed in enumerate(MODELS):
            ax = axes[fi,si]
            cells = [[lookup[s,p,f'{family}_{seed}'] for p in spec['plants']] for s in spec['scenarios']]
            grid = np.asarray([[r['mean_gain_percent'] for r in line] for line in cells])
            picture = ax.imshow(grid, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
            for row, line in enumerate(cells):
                for col, cell in enumerate(line):
                    suffix = ('*' if cell['exceeded_error_runs'] else '')
                    suffix += ('†' if cell['baseline_failed_runs'] else '')
                    suffix += ('×' if cell['completed'] < cell['independent_initials'] else '')
                    ax.text(col, row, f"{cell['mean_gain_percent']:.2f}{suffix}", ha='center', va='center',
                            fontsize=8, color='white' if abs(grid[row,col])>.65*bound else '#111111')
            ax.set_xticks(range(len(spec['plants'])), [plant_labels[p] for p in spec['plants']], rotation=20, ha='right', fontsize=8)
            ax.set_yticks(range(len(spec['scenarios'])), [scene_labels[s] for s in spec['scenarios']], fontsize=8)
            ax.set_title(f'{FAMILY_LABELS[family]} / {seed}', fontsize=10)
    figure.colorbar(picture, ax=axes, label='Mean team-cost improvement over matched baseline (%)', shrink=.8)
    figure.suptitle('All phase/repair transfer cells; 32 paired initials per scenario\n'
                    '* physical error bound exceeded; † baseline failure; × incomplete controller; no transfer certificate', fontsize=12)
    figure.savefig(OUTPUT/'matched_transfer_all_families.png', dpi=200)
    figure.savefig(OUTPUT/'matched_transfer_all_families.pdf')
    plt.close(figure)


def main():
    summaries, protocols, choice = load_complete()
    if OUTPUT.exists():
        raise FileExistsError('Preserve completed presentation')
    OUTPUT.mkdir(parents=True)
    phase, transfer = summaries['phase'], summaries['transfer']
    n = phase['independent_initials']
    lines = ['# V21 完整开发与匹配动力学迁移结果', '',
        '本报告只展示原冻结汇总与选择，不重新筛选模型、改动阈值或合并独立样本。', '',
        f"时序门控研究包含 {n} 个独立初态和 {phase['actual_sumo_runs']} 次主执行；"
        f"迁移包含 {transfer['independent_initials']} 个独立初态和 {transfer['actual_sumo_runs']} 次主执行。"
        '扰动模式、训练模型与回放不增加独立样本量。三个模型是固定训练延续，并共享早期祖先。', '',
        '## 完整模型组选择', '',
        (f"冻结开发规则选中 `{choice['selected_family']}`；这不是风险认证，还须完成新的独立校准。"
         if choice['selected_family'] else '没有完整模型组通过冻结开发条件；没有合格的新校准候选。'), '',
        '| 门控 | 模型 | 成本改善 | 收益区间（秒） | 实际采用 | 实际有害/采用 | 组合有害/采用 | 四模式失败 | P95（ms） |',
        '|---|---:|---:|---|---:|---|---|---:|---:|']
    for family in FAMILIES:
        for seed in MODELS:
            r = phase['results'][f'{family}_{seed}']
            lines.append(f"| {family} | {seed} | {r['actual_mean_gain_percent']:.4f}% | "
                f"{interval(r['actual_mean_gain_pointwise_bootstrap95'])} | {r['actual_adopted']}/{n} | "
                f"{ratio(r['actual_harmful_adopted'],r['actual_adopted'])} | "
                f"{ratio(r['bundle_harmful_adopted'],r['bundle_adopted'])} | {r['failed_runs']}/{r['runs']} | {r['controller_p95_ms']:.2f} |")
    lines += ['', '采用表示整次运行至少使用一次学习候选。任一车辆成本/时间退化或执行失败均记为有害；'
              '经验比例及逐模型开发区间不是风险证书。计时混合采集阶段和并发桌面负载，不能作为独占部署基准。', '',
              '![采用与收益](phase_gain_adoption.png)', '', '## 同初态增量收益', '',
              '| 模型 | 新门控相对动作调整门控 | 平均增量（秒） | 逐模型 95% 配对区间（秒） |', '|---:|---|---:|---|']
    for seed, item in phase['paired_results'].items():
        for family, row in item['actual_incremental_gain'].items():
            lines.append(f"| {seed} | {family} − repair | {row['mean_seconds']:.6f} | {interval(row['pointwise_bootstrap95'])} |")
    lines += ['', '## 全部匹配迁移单元', '',
        '每格 32 个独立初态。四类动力学共用外生创新；不同控制器的实际执行器误差可以不同。'
        '下表保留所有控制器和失败。基线失败时较大的相对收益不能直接解释成控制器可靠。'
        '误差越出 ±0.5 m/s² 后，原物理前提不成立；仿真中止不表示车辆已安全停稳。'
        '这些逐格开发区间没有同时覆盖或迁移风险认证。', '',
        '![全部门控迁移单元](matched_transfer_all_families.png)', '',
        '| 场景 | 动力学 | 方法 | 改善 | 收益区间（秒） | 采用 | 有害/采用 | 未完成 | 中止 | 碰撞 | 约束事件 | 基线失败 | 误差越界运行 | 最大误差 |',
        '|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in transfer['cells']:
        count = r['independent_initials']
        lines.append(f"| {r['scenario']} | {r['plant']} | {r['method']} | {r['mean_gain_percent']:.4f}% | "
            f"{interval(r['gain_pointwise_bootstrap95'])} | {r['adopted']}/{count} | {ratio(r['harmful_adopted'],r['adopted'])} | "
            f"{count-r['completed']} | {r['aborted']} | {r['collisions']} | {r['contract_violations']} | "
            f"{r['baseline_failed_runs']} | {r['exceeded_error_runs']} | {r['maximum_actual_error']:.5f} |")
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    phase_figure(phase)
    transfer_figure(transfer, protocols['transfer']['spec'])
    save(OUTPUT/'display_data.json', dict(phase=phase, transfer=transfer, selection=choice))
    save(OUTPUT/'provenance.json', dict(source_sha256=digest(ROOT/'tools/present_completed_v21.py'),
        selection_sha256=digest(CHOICE), summaries={s:digest(completion.OUTPUT/'results'/s/'summary.json') for s in summaries},
        python=platform.python_version(), matplotlib=matplotlib.__version__, new_sumo_runs=0, new_risk_certified=False))
    save(OUTPUT/'sha256.json', prior.manifest(OUTPUT))
    print(json.dumps(dict(phase_methods=len(phase['results']), transfer_cells=len(transfer['cells']),
                          selected_family=choice['selected_family'], risk_certified=False)), flush=True)


if __name__ == '__main__':
    main()
