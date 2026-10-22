"""Present only the completed and independently audited frozen V19 study."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V19 = ROOT / 'artifacts/v19'
OUT = V19 / 'final_presentation'
DOC = ROOT / 'docs/v19_final_results.md'
METHODS = ('raw', 'nominal', 'portfolio0', 'conformal', 'ltt', 'mean_spi')
LABELS = ('Raw learning', 'Nominal gate', 'Strict joint gate', 'Conformal adapter', 'LTT adapter', 'Mean-SPI adapter')
SEEDS = ('1802', '1812', '1822')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists() or DOC.exists():
        raise FileExistsError('Preserve existing presentation artifacts')
    source = V19 / 'final_report/report.json'
    manifest = json.loads((V19 / 'final_report/sha256.json').read_text())
    if digest(source) != manifest['report.json']:
        raise ValueError('Final report changed')
    report = json.loads(source.read_text())
    if not report['original_report_values_reproduced'] or not report['real_whole_profile_replays_verified']:
        raise ValueError('Independent complete reporting and actual replay are required')
    for relative, expected in report['input_sha256'].items():
        if digest(V19 / relative) != expected:
            raise ValueError('Audited final input changed: ' + relative)
    results = report['results']
    rows, labels, colors = [], [], []
    palette = ('#5d636b', '#4776ab', '#16816e', '#8b789c', '#a57732', '#b45e61')
    for group, seed in enumerate(SEEDS):
        for offset, (method, label, color) in enumerate(zip(METHODS, LABELS, palette)):
            item = results[method]['per_training_seed'][seed]
            rows.append((group * 7 + offset, item))
            labels.append(f"{seed} | {label}" + (' *' if item['conditional_certificate']['certified'] else ''))
            colors.append(color)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 10.5), sharey=True, gridspec_kw={'width_ratios': [1.3, 1., 1.35]})
    for (y, item), color in zip(rows, colors):
        m = item['metrics']
        mean, (lo, hi) = m['mean_team_gain_seconds'], m['mean_team_gain_ci95']
        axes[0].errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt='o', color=color, capsize=3, markersize=5)
        adopted = m['actual_random_mode']
        axes[1].plot(100 * adopted['adoption_fraction'], y, 'o', color=color, markersize=5)
        if adopted['adopted']:
            rate = 100 * adopted['conditional_harm']
            axes[2].plot(rate, y, 'o', color=color, markersize=5)
            axes[2].annotate(f"{adopted['harmful_adopted']}/{adopted['adopted']}",
                             (rate, y), xytext=(7, 0), textcoords='offset points', va='center', fontsize=8)
        else:
            axes[2].text(8, y, 'undefined (0 adopted)', va='center', color='#777777', fontsize=8)
    ys = [r[0] for r in rows]
    axes[0].set_yticks(ys, labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color='#666666', lw=.8)
    axes[1].axvline(20, color='#999999', lw=.8, ls='--')
    axes[1].set_xlim(-4, 107)
    axes[2].axvline(5, color='#999999', lw=.8, ls='--')
    axes[2].set_xlim(-3, 95)
    for ax in axes:
        ax.grid(axis='x', alpha=.15)
        for divider in (6., 13.):
            ax.axhline(divider, color='#dddddd', lw=.8)
    axes[0].set_title('Mean team cost gain\npaired initial bootstrap 95% interval')
    axes[0].set_xlabel('Cost-seconds saved relative to baseline')
    axes[1].set_title('Episodes adopting learning\ndashed line: 20% target')
    axes[1].set_xlabel('Adoption (%)')
    axes[2].set_title('Observed harm among adopted episodes\ndashed line: 5% population risk target')
    axes[2].set_xlabel('Descriptive test frequency (%)')
    fig.suptitle('Independent V19 confirmation: all models and all matched adapters', fontsize=15, y=.978)
    fig.text(.02, .025, '512 independent initials; one preselected random disturbance mode per initial. Vehicles, models and modes are not extra independent samples.\n'
             '* Fixed profile passed both risk tests on separate calibration data. Test harm frequencies are not risk certificates.\n'
             'Conformal is a conservative marginal-coverage adaptation; LTT and Mean-SPI use the same fixed profile family.', fontsize=9, va='bottom')
    fig.subplots_adjust(left=.235, right=.985, top=.89, bottom=.13, wspace=.23)
    OUT.mkdir()
    fig.savefig(OUT / 'independent_comparison.png', dpi=160)
    fig.savefig(OUT / 'independent_comparison.pdf')
    plt.close(fig)
    lines = ['# V19 独立最终比较结果', '',
        '20,480 次实际 SUMO 主实验、156 次完整控制器部署回放和 10 次名义影子仿真均已完成。独立报告重建了校准计数、最终选择、收益区间和全部联合验收条件；回放中所有被选控制器与最终指标一致。回放及影子仿真不增加独立风险或有效性样本数。', '',
        '**严格门控在三份模型上保留了收益与采用率，但只有模型 1812 同时满足原定全部目标。** 另两份模型未通过既定校准中的两项风险认证，最终测试中较低的经验退化比例不能追溯改变认证结论。', '',
        '## 统计单位与原定条件', '',
        '主实验包含 512 个新 IID 初态，名义／密集场景等权混合、四种扰动、十个完整控制器。每个初态按预先冻结的随机流选一个实际扰动模式进行主统计，另报告四模式组合。三份模型均保留，不挑选最佳训练种子。', '',
        '联合目标是两项条件风险均获认证、实际学习采用率至少 20%、平均团队成本改善至少 0.5%，且按独立初态配对自举的 95% 收益区间下界为正。风险定义为任一车辆成本或到达时间退化或执行失败，不是碰撞率。', '',
        '## 严格联合门控', '',
        '| 模型 | 最终成本改善 | 采用／512 | 采用后退化／采用 | 四模式组合退化／采用 | 全部条件 |',
        '|---|---:|---:|---:|---:|---|']
    for seed in SEEDS:
        item = results['portfolio0']['per_training_seed'][seed]
        m, a, b = item['metrics'], item['metrics']['actual_random_mode'], item['metrics']['worst_mode_bundle']
        lines.append(f"| {seed} | {m['mean_team_gain_percent']:.4f}% | {a['adopted']}/512 | {a['harmful_adopted']}/{a['adopted']} | {b['harmful_adopted']}/{b['adopted']} | {'通过' if item['target_check']['all_conditions_met'] else '未通过风险认证'} |")
    lines += ['', '三个模型的实际采用率分别为 22.2656%、20.8984%、20.8984%；收益区间均为正。三个严格模型在主统计中均完成 512/512 次，四模式中也未出现执行失败。零采用运行均与同初态同扰动的基线结果一致。', '',
        '原校准中的实际／组合风险上界依次为 5.0340%／5.7895%、4.2897%／3.8787%、4.8672%／8.0785%。认证限定于预先声明的总体分布，不能解释成当前状态的置信度、逐模式保证或分布迁移后的保证。', '',
        '## 共同候选与预算下的对照', '',
        '| 方法 | 三固定模型平均成本收益 秒 | 95% 配对区间 秒 | 各模型成本改善百分比 1802／1812／1822 |',
        '|---|---:|---|---|']
    chinese = ('原始学习', '名义门控', '严格联合门控', 'Conformal 适配', 'LTT 适配', 'Mean SPI 适配')
    for method, label in zip(METHODS, chinese):
        f = results[method]['fixed_model_average']
        lo, hi = f['mean_team_gain_ci95']
        percents = '／'.join(f"{results[method]['per_training_seed'][s]['metrics']['mean_team_gain_percent']:.3f}" for s in SEEDS)
        lines.append(f"| {label} | {f['mean_team_gain_seconds']:.5f} | [{lo:.5f}, {hi:.5f}] | {percents} |")
    lines += ['',
        '名义门控保留较高收益，但采用后的经验逐车退化比例为 16.40%–17.78%。Mean SPI 选择名义模型 1802 与原始模型 1812、1822，保留均值收益，但没有采用后的条件风险认证。它是全信息仿真的均值检验适配，不是 CSPI-MT 原算法的完整复现。', '',
        'LTT 在模型 1812 中选择严格门控，在另外两份模型中回退基线；这与其原定认证规则一致。本文门控使用 LTT 认证，因此不能把该比较表述成“优于 LTT”。本次保守 Conformal 适配全部回退基线，产生零采用、零收益；条件风险在零采用时未定义，不能写成零条件风险，更不能据此断言所有共形门控都无效。', '',
        '原始学习模型 1802 的最终收益区间为 [−0.06533, 0.10459] 秒，跨越零，且主统计中 3 次未完成；1812 有 1 次未完成。上述结果不能删去，也不支持原始学习在每份模型上稳定优于强基线。', '',
        '![全部模型与匹配选择器的独立最终比较](../artifacts/v19/final_presentation/independent_comparison.png)', '',
        '## 工程边界与后续开发', '',
        '原冻结严格门控的主统计 P95 为 585.61–640.30 ms，P99 为 1731.80–1849.07 ms，仍未满足 500 ms 控制周期。并行桌面负载下的计时不能替代独占部署基准；此前等价加速回放也不能直接替换本研究的实测时间。Conformal 的完整名义预测准备开销须另行计入。', '',
        'V20 时变路径和完整动作调整策略已在查看本次最终结果前冻结，使用自己的新开发初态；本轮最终数据不用于调参。新策略仍需新的风险校准与独立确认，不能继承本轮证书。三份训练模型全部达到联合目标这一要求尚未满足。', '',
        '证据：[独立报告](../artifacts/v19/final_report/report.json)、[完整匹配比较](../artifacts/v19/final_comparison/comparison.json)、[实际部署回放](../artifacts/v19/comparison_replay/audit.json)。', '']
    DOC.write_text('\n'.join(lines), encoding='utf-8')
    record = dict(report_sha256=digest(source), source_sha256=digest(__file__), document_sha256=digest(DOC),
                  statistic_or_selection_rule_changed=False, final_outcomes_used_to_tune_v20=False,
                  strict_gate_all_replicates_meet_original_targets=report['strict_gate_all_replicates_meet_original_targets'])
    (OUT / 'presentation.json').write_text(json.dumps(record, indent=2) + '\n')
    (OUT / 'sha256.json').write_text(json.dumps({p.name:digest(p) for p in OUT.iterdir() if p.is_file()}, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
