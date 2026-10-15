"""Derived statistics and scientific figures from the frozen V5 study."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v5'
NAMES = {'constant': 'Constant + gate', 'frozen_rl127': 'Frozen RL + gate',
         'short_portfolio': '2 s selection', 'no_coast': 'Full / no coasting', 'full_portfolio': 'Full portfolio'}
ZH = {'constant': '恒定加速＋门控', 'frozen_rl127': '冻结 RL127＋门控',
      'short_portfolio': '2秒目标选择', 'no_coast': '完整目标但无提前滑行', 'full_portfolio': '完整候选库规划'}


def main():
    protocol = json.loads((D/'protocol.json').read_text())
    rows = list(csv.DictReader((D/'study/episodes.csv').open()))
    assert len(rows) == protocol['executions']
    expected = {(m, s['scenario'], s['seed']) for m in protocol['methods'] for scenes in protocol['splits'].values() for s in scenes}
    assert {(r['method'], r['scenario'], int(r['seed'])) for r in rows} == expected
    lookup = {(r['method'], r['scenario'], int(r['seed'])): r for r in rows}
    summary = {}
    rng = np.random.default_rng(2026101506)
    for split, scenes in protocol['splits'].items():
        summary[split] = {}
        baseline = np.array([float(lookup['constant', split, s['seed']]['mean_capped_finish_s']) for s in scenes])
        for method in protocol['methods']:
            selected = [lookup[method, split, s['seed']] for s in scenes]
            cost = np.array([float(r['mean_capped_finish_s']) for r in selected])
            saved = baseline-cost
            ci = np.percentile(rng.choice(saved, size=(10000, len(saved)), replace=True).mean(axis=1), [2.5, 97.5])
            summary[split][method] = {
                'n': len(selected), 'mean_arrival_s': float(cost.mean()),
                'mean_saved_s': float(saved.mean()), 'saved_ci95_s': ci.tolist(),
                'reduction_percent': float(100*saved.mean()/baseline.mean()),
                'wins_ties_losses': [int(np.sum(saved > 1e-6)), int(np.sum(np.abs(saved) <= 1e-6)), int(np.sum(saved < -1e-6))],
                'completed': sum(r['completed'] == 'True' for r in selected),
                'collisions': sum(r['collision'] == 'True' for r in selected),
                'violating_episodes': sum(int(r['contract_violations']) > 0 for r in selected),
                'outside_steps': sum(int(r['outside_steps']) for r in selected),
                'uncertified_steps': sum(int(r['uncertified_steps']) for r in selected),
                'plan_fallbacks': sum(int(r['plan_fallbacks']) for r in selected),
                'mean_command_jerk': float(np.mean([float(r['mean_jerk']) for r in selected])),
                'max_prediction_error': max(float(r['max_prediction_error']) for r in selected)}
            for row in selected:
                if row['predicted_baseline']:
                    assert abs(float(row['predicted_baseline'])-float(lookup['constant', split, int(row['seed'])]['mean_capped_finish_s'])) < 1e-6
                    assert abs(float(row['predicted_cost'])-float(row['mean_capped_finish_s'])) < 1e-6
            if method in ('no_coast', 'full_portfolio'):
                assert np.min(saved) >= -1e-6
    timing = json.loads((D/'study/timing.json').read_text())
    latency = {}
    for method in protocol['methods']:
        values = [r for r in timing if r['method'] == method]
        cycle = [v for r in values for v in r['cycle_ms']]
        first = [r['planning_ms']+r['cycle_ms'][0] for r in values]
        planning = [r['planning_ms'] for r in values]
        latency[method] = {'episodes': len(values), 'cycles': len(cycle),
                           'planning_mean_ms': float(np.mean(planning)), 'planning_p95_ms': float(np.percentile(planning, 95)),
                           'cycle_p95_ms': float(np.percentile(cycle, 95)), 'cycle_max_ms': max(cycle),
                           'first_response_p95_ms': float(np.percentile(first, 95)), 'first_response_max_ms': max(first),
                           'first_over_500ms': sum(v > 500 for v in first)}
    result = {'executions': len(rows), 'independent_scenes': 80, 'summary': summary, 'timing': latency,
              'bootstrap': '10000 within-split paired scene resamples; descriptive 95% intervals, no risk certification.',
              'scope': 'Deterministic full-horizon planning headroom; no newly trained RL or global-optimality claim.'}
    (D/'summary.json').write_text(json.dumps(result, indent=2)+'\n', newline='\n')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
    palette = ['#697786', '#a5adb8', '#b16a52', '#5c99ad', '#215e48']
    for i, (method, color) in enumerate(zip(protocol['methods'], palette)):
        axes[0].bar(np.arange(2)+(i-2)*.15, [summary[s][method]['mean_arrival_s'] for s in ('nominal', 'dense')],
                    width=.14, color=color, label=NAMES[method])
    axes[0].set(xticks=[0, 1], xticklabels=['Nominal (40)', 'Dense (40)'], ylabel='Mean capped arrival time (s)', ylim=(0, 16))
    axes[0].legend(fontsize=7, ncols=2, loc='upper right')
    for split, color in [('nominal', '#215e48'), ('dense', '#b16a52')]:
        scenes = protocol['splits'][split]
        gains = sorted(float(lookup['constant', split, s['seed']]['mean_capped_finish_s'])-float(lookup['full_portfolio', split, s['seed']]['mean_capped_finish_s']) for s in scenes)
        axes[1].step(gains, np.arange(1, len(gains)+1)/len(gains), where='post', label=split.title(), color=color)
    axes[1].set(xlabel='Paired arrival time saved by full portfolio (s)', ylabel='Fraction of scenes', ylim=(0, 1.03))
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.savefig(D/'results.png', dpi=200); fig.savefig(D/'results.pdf'); plt.close(fig)
    lines = ['# V5 完整轨迹规划实验结果', '',
             '算法和80个新场景在运行前冻结，5种方法完成400次真实SUMO执行。开发仅使用500至502号种子在两类分布上的6个场景；本轮测试没有复用V4测试种子。', '',
             '## 主结果', '', '| 方法 | 常规平均到达时间 s | 高密度平均到达时间 s |', '|---|---:|---:|']
    for method in protocol['methods']:
        lines.append(f"| {ZH[method]} | {summary['nominal'][method]['mean_arrival_s']:.3f} | {summary['dense'][method]['mean_arrival_s']:.3f} |")
    lines += ['', '每车到达s=60的时间取均值；未完成按25秒计，碰撞整回合计25秒。两类分布各40个独立场景；同场景的5种控制器执行是配对数据，不是400个独立场景。', '']
    for split in ('nominal', 'dense'):
        r = summary[split]['full_portfolio']; ci = r['saved_ci95_s']
        lines.append(f"- {split}：平均节省 {r['mean_saved_s']:.3f} s（{r['reduction_percent']:.2f}%），配对bootstrap 95%区间 [{ci[0]:.3f}, {ci[1]:.3f}] s；胜／平／负 {r['wins_ties_losses']}；完成 {r['completed']}/{r['n']}。")
    lines += ['', '## 安全与模型一致性', '']
    lines.append(f"400次执行共记录 {sum(s[m]['collisions'] for s in summary.values() for m in protocol['methods'])} 次碰撞回合、{sum(s[m]['violating_episodes'] for s in summary.values() for m in protocol['methods'])} 次路线合同违规回合。最大单步模型残差为 {max(s[m]['max_prediction_error'] for s in summary.values() for m in protocol['methods']):.3e}。所有规划器预测代价均与实际执行一致，预测基线代价也逐场景对应真实基线执行；没有计划失配回退。这只验证本次精确模型合同，不证明真实道路零风险。")
    lines += ['', '## 计算成本', '', '| 方法 | 初始规划P95 ms | 后续周期P95 ms | 首次响应P95 ms | 首次响应最大 ms |', '|---|---:|---:|---:|---:|']
    for method in protocol['methods']:
        t = latency[method]
        lines.append(f"| {ZH[method]} | {t['planning_p95_ms']:.2f} | {t['cycle_p95_ms']:.2f} | {t['first_response_p95_ms']:.2f} | {t['first_response_max_ms']:.2f} |")
    lines += ['', '执行周期包含策略推断（若有）、当前证书重检、TraCI步进和指标更新。首次响应额外包含完整初始规划。周期统计包含首步的执行部分，但不将初始规划摊薄到后续周期。单机、单工作进程测试，不是WCET；仿真在计算期间不推进，不是执行延迟鲁棒性验证。短期选择消融仍生成完整轨迹，不用于比较计算节约。', '',
              '## 结论边界', '',
              '完整规划器属于基于模型的有限方案比较。它展示当前合同下可实现的性能改善，为下一阶段学习器提供教师；不能把这部分收益称为新的MARL收益，也没有证明全局最优。无提前滑行与短期选择消融分别检验候选库和目标时域。', '',
              '安全、平均效率和舒适性必须分开。各方法逐场景jerk、完成率、计划回退和模型残差均保存在原始CSV，不能用平均时间改善宣称所有车辆、更低能耗或更平顺。', '',
              '[理论与相关工作](theory_v5.md) · [冻结协议](../artifacts/v5/protocol.json) · [逐回合数据](../artifacts/v5/study/episodes.csv) · [全部候选代价](../artifacts/v5/study/candidate_costs.json)', '',
              '```powershell', 'python tools/verify_release_v5.py --replay', 'python tools/study_v5.py --out runs/my_v5_reproduction', '```', '']
    (ROOT/'docs/v5_results.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    hashes = {p.relative_to(D).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(D.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
    (D/'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2)+'\n', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
