"""Paired performance, selective-risk diagnostics and runtime from V6 evidence."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v6'
NAMES = {'constant': 'Constant + gate', 'fixed': 'Best fixed schedule', 'imitation': 'Distilled selector',
         'learned': 'Bandit selector', 'calibrated': 'Calibrated selector', 'teacher': 'Full planning teacher'}
ZH = {'constant': '传统基线', 'fixed': '训练集最优固定方案', 'imitation': '纯蒸馏选择器',
      'learned': '策略优化选择器', 'calibrated': '校准门控选择器', 'teacher': '完整规划教师'}


def main():
    protocol = json.loads((D/'protocol.json').read_text())
    calibration = json.loads((D/'calibration.json').read_text())
    rows = list(csv.DictReader((D/'study/episodes.csv').open()))
    assert len(rows) == protocol['executions']
    lookup = {(r['method'], r['scenario'], int(r['seed'])): r for r in rows}
    expected = {(m, s, seed) for m in protocol['methods'] for s, seeds in protocol['test_seeds'].items() for seed in seeds}
    assert set(lookup) == expected
    rng = np.random.default_rng(2026101508); summary = {}
    for split, seeds in protocol['test_seeds'].items():
        summary[split] = {}
        baseline = np.array([float(lookup['constant', split, seed]['mean_capped_finish_s']) for seed in seeds])
        teacher = np.array([float(lookup['teacher', split, seed]['mean_capped_finish_s']) for seed in seeds])
        opportunity = float((baseline-teacher).mean())
        for method in protocol['methods']:
            records = [lookup[method, split, seed] for seed in seeds]
            cost = np.array([float(r['mean_capped_finish_s']) for r in records])
            gains = baseline-cost
            ci = np.percentile(rng.choice(gains, size=(10000, len(gains)), replace=True).mean(axis=1), [2.5, 97.5])
            result = {'n': len(cost), 'mean_arrival_s': float(cost.mean()), 'mean_saved_s': float(gains.mean()),
                      'saved_ci95_s': ci.tolist(), 'gain_percent': float(100*gains.mean()/baseline.mean()),
                      'teacher_gain_retained_percent': float(100*gains.mean()/opportunity) if opportunity > 0 else None,
                      'mean_teacher_regret_s': float((cost-teacher).mean()),
                      'wins_ties_losses': [int(sum(gains > 1e-6)), int(sum(np.abs(gains) <= 1e-6)), int(sum(gains < -1e-6))],
                      'completed': sum(r['completed'] == 'True' for r in records),
                      'collisions': sum(r['collision'] == 'True' for r in records),
                      'violating_episodes': sum(int(r['contract_violations']) > 0 for r in records),
                      'uncertified_steps': sum(int(r['uncertified_steps']) for r in records),
                      'outside_steps': sum(int(r['outside_steps']) for r in records),
                      'mean_command_jerk': float(np.mean([float(r['mean_jerk']) for r in records])),
                      'max_prediction_error': max(float(r['max_prediction_error']) for r in records)}
            if method == 'calibrated':
                adopted = np.array([int(r['selected']) != 0 for r in records])
                upper = np.array([float(r['upper_excess_s']) for r in records])
                actual = np.array([float(r['proposal_actual_excess_s']) for r in records])
                result.update(nonbaseline_adoptions=int(adopted.sum()), adoption_rate=float(adopted.mean()),
                              proposal_coverage=float(np.mean(actual <= upper+1e-7)),
                              final_regression_rate=float(np.mean(gains < -1e-6)),
                              conditional_regression_rate=float(np.mean(gains[adopted] < -1e-6)) if adopted.any() else None)
            summary[split][method] = result
    latency = {}; raw_timing = json.loads((D/'study/timing.json').read_text())
    for method in protocol['methods']:
        group = [r for r in raw_timing if r['method'] == method]
        selection = [r['selection_ms'] for r in group]
        cycle = [v for r in group for v in r['cycle_ms']]
        first = [r['selection_ms']+r['cycle_ms'][0] for r in group]
        latency[method] = {'selection_p95_ms': float(np.percentile(selection, 95)), 'cycle_p95_ms': float(np.percentile(cycle, 95)),
                           'first_response_p95_ms': float(np.percentile(first, 95)), 'first_response_max_ms': max(first)}
    checkpoints = {}
    for seed in protocol['training']['seeds']:
        checkpoints[str(seed)] = {}
        for name in ('imitation', 'policy'):
            meta = torch.load(D/f'seed{seed}'/f'{name}.pt', map_location='cpu', weights_only=True)
            checkpoints[str(seed)][name] = {'epoch': meta['epoch'], 'validation_arrival_s': meta['validation_arrival_s']}
    contrasts = {}
    for split, seeds in protocol['test_seeds'].items():
        contrasts[split] = {}
        for reference, method in [('imitation', 'learned'), ('learned', 'calibrated')]:
            gains = np.array([float(lookup[reference, split, seed]['mean_capped_finish_s'])-float(lookup[method, split, seed]['mean_capped_finish_s']) for seed in seeds])
            ci = np.percentile(rng.choice(gains, size=(10000, len(gains)), replace=True).mean(axis=1), [2.5, 97.5])
            contrasts[split][f'{method}_vs_{reference}'] = {'mean_saved_s': float(gains.mean()), 'ci95_s': ci.tolist(),
                                                         'scope': 'Supplementary descriptive paired contrast, no multiplicity correction.'}
    result = {'executions': len(rows), 'independent_scenes': 120, 'summary': summary, 'timing': latency,
              'checkpoints': checkpoints, 'contrasts': contrasts, 'calibration_quantile_s': calibration['upper_residual_quantile_s'],
              'scope': protocol['scope'], 'calibration_scope': calibration['scope']}
    (D/'summary.json').write_text(json.dumps(result, indent=2)+'\n', newline='\n')
    colors = ['#77828d', '#aba292', '#68a5b1', '#4f799c', '#245d47', '#c28e65']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for i, (method, color) in enumerate(zip(protocol['methods'], colors)):
        axes[0].bar(np.arange(2)+(i-2.5)*.13, [summary[s][method]['mean_arrival_s'] for s in ('nominal', 'dense')],
                    width=.125, color=color, label=NAMES[method])
    axes[0].set(xticks=[0, 1], xticklabels=['Nominal (60)', 'Dense shift (60)'], ylabel='Mean capped arrival time (s)', ylim=(0, 16))
    axes[0].legend(fontsize=6.5, ncols=2, loc='upper right')
    names = ['imitation', 'learned', 'calibrated', 'teacher']
    for offset, split, color in [(-.17, 'nominal', '#245d47'), (.17, 'dense', '#c28e65')]:
        axes[1].bar(np.arange(4)+offset, [summary[split][m]['teacher_gain_retained_percent'] for m in names], width=.32, color=color, label=split)
    axes[1].set(xticks=np.arange(4), xticklabels=['Distilled', 'Bandit', 'Calibrated', 'Teacher'], ylabel='Teacher mean gain retained (%)')
    axes[1].axhline(0, color='#777777', linewidth=.5); axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.savefig(D/'results.png', dpi=200); fig.savefig(D/'results.pdf'); plt.close(fig)
    lines = ['# V6 学习与性能置信门控结果', '',
             '本轮新增768个训练、128个验证、256个校准初态，全部来自常规分布。训练标签由37方案模型展开生成，不等于42624次真实SUMO完整执行。主测试为120个独立新场景、6种方法、720次真实SUMO执行；三个模型组成一个固定集成，没有把模型数当作独立场景数。', '',
             '## 配对到达代价', '', '| 方法 | 常规 s | 高密度偏移 s | 常规教师收益保留 | 偏移教师收益保留 |', '|---|---:|---:|---:|---:|']
    for method in protocol['methods']:
        a, b = summary['nominal'][method], summary['dense'][method]
        lines.append(f"| {ZH[method]} | {a['mean_arrival_s']:.3f} | {b['mean_arrival_s']:.3f} | {a['teacher_gain_retained_percent']:.1f}% | {b['teacher_gain_retained_percent']:.1f}% |")
    lines += ['', '收益保留为均值之差的比值；负数表示比基线更慢。所有方法采用同一安全证书。已发表V5数据没有作为本轮训练或测试集。', '', '## 门控诊断', '']
    lines.append(f"常规校准集n=256，目标α=0.1，单侧残差分位数q={calibration['upper_residual_quantile_s']:.6f} s。接受条件为预测相对基线代价加q≤0。")
    for split in ('nominal', 'dense'):
        r = summary[split]['calibrated']; ci = r['saved_ci95_s']
        lines.append(f"- {split}：采用非基线方案 {r['nonbaseline_adoptions']}/60；提案上界覆盖率 {r['proposal_coverage']:.1%}；最终变慢率 {r['final_regression_rate']:.1%}；采用子集内变慢率 {r['conditional_regression_rate']}；平均节省 {r['mean_saved_s']:.3f} s，描述性95%配对区间 [{ci[0]:.3f}, {ci[1]:.3f}] s。")
    lines += ['', '理论是对校准集与新常规场景共同随机性的边际性能界，不是接受条件下的保证，不是碰撞概率，不适用于高密度分布偏移。测试观测比例不能证明所有场景均满足目标。', '', '## 训练检查点', '']
    lines.insert(-2, '本次常规测试的提案上界覆盖率83.3%低于名义90%，高密度仅70%。有限测试与固定校准集上的经验覆盖不等于边际理论覆盖；结果不能写成“已实证达到90%可靠度”。高密度门控仅保留28.4%教师收益，说明错误回退与错误接受都需要进一步诊断。')
    for seed, values in checkpoints.items():
        lines.append(f"- 种子{seed}：蒸馏轮次{values['imitation']['epoch']}，验证代价{values['imitation']['validation_arrival_s']:.4f} s；最终选择轮次{values['policy']['epoch']}，验证代价{values['policy']['validation_arrival_s']:.4f} s。")
    lines += ['', '补充的配对消融差异如下，仅作描述性比较，未进行多重比较校正。', '']
    for split, comparisons in contrasts.items():
        for name, value in comparisons.items():
            ci = value['ci95_s']
            lines.append(f"- {split} {name}：平均节省 {value['mean_saved_s']:.4f} s，95%区间 [{ci[0]:.4f}, {ci[1]:.4f}] s。")
    lines += ['', '## 运行时间与安全', '', '| 方法 | 高层选择P95 ms | 执行周期P95 ms | 首次响应P95 ms |', '|---|---:|---:|---:|']
    for method in protocol['methods']:
        t = latency[method]
        lines.append(f"| {ZH[method]} | {t['selection_p95_ms']:.2f} | {t['cycle_p95_ms']:.2f} | {t['first_response_p95_ms']:.2f} |")
    collisions = sum(v['collisions'] for split in summary.values() for v in split.values())
    violations = sum(v['violating_episodes'] for split in summary.values() for v in split.values())
    complete = sum(v['completed'] for split in summary.values() for v in split.values())
    lines += ['', f'720次执行中完成{complete}次，碰撞回合{collisions}次，路线合同违规回合{violations}次。所有实际选择方案的到达代价均逐回合与独立教师标签校验一致。', '',
              '学习器在线只做初态推断与反馈证书检查；完整教师的模型展开开销计入其首次响应。生成评测标签的额外工作不计作学习器控制成本，离线数据收集成本单独公开。单机计时不是WCET或计算延迟闭环保证。', '',
              '## 研究边界', '', '这是一项高层集中式离线bandit与门控实验，不是新增分布式MARL。物理证书仍依赖精确模型、固定路线与全受控车辆。Conformal切换已有相关工作，本轮不能据此宣称首次创新或达到顶会投稿成熟度。', '',
              '[理论与相关研究](theory_v6.md) · [冻结协议](../artifacts/v6/protocol.json) · [回合数据](../artifacts/v6/study/episodes.csv)', '',
              '```powershell', 'python tools/verify_release_v6.py --replay', 'python tools/study_v6.py --out runs/my_v6_reproduction', '```', '']
    (ROOT/'docs/v6_results.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    hashes = {p.relative_to(D).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(D.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
    (D/'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2)+'\n', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
