"""Evaluate predeclared improvement criteria, with compiler/ordering ablations."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/v8'
ZH = {'constant': '传统基线', 'reference_pair': '原双方案核验', 'fast_pair': '编译双方案核验',
      'fast_full': '编译完整搜索', 'bb_natural': '自然顺序剪枝', 'bb_learned': '学习排序剪枝', 'reference_full': '原完整搜索'}


def main():
    protocol = json.loads((D/'protocol.json').read_text()); methods = protocol['methods']
    rows = list(csv.DictReader((D/'study/episodes.csv').open()))
    lookup = {(r['method'], r['scenario'], int(r['seed'])): r for r in rows}
    expected = {(m, s, seed) for m in methods for s, seeds in protocol['test_seeds'].items() for seed in seeds}
    assert set(lookup) == expected and len(rows) == len(lookup) == protocol['executions']
    rng = np.random.default_rng(2026101529); summary = {}; contrasts = {}
    for regime, seeds in protocol['test_seeds'].items():
        summary[regime] = {}; contrasts[regime] = {}
        baseline = np.array([float(lookup['constant', regime, s]['mean_capped_finish_s']) for s in seeds])
        teacher = np.array([float(lookup['reference_full', regime, s]['mean_capped_finish_s']) for s in seeds])
        reference_pair = np.array([float(lookup['reference_pair', regime, s]['mean_capped_finish_s']) for s in seeds])
        for method in methods:
            records = [lookup[method, regime, s] for s in seeds]
            cost = np.array([float(r['mean_capped_finish_s']) for r in records]); gain = baseline-cost
            summary[regime][method] = {
                'n': len(seeds), 'mean_arrival_s': float(cost.mean()), 'mean_saved_s': float(gain.mean()),
                'gain_percent': float(100*gain.mean()/baseline.mean()),
                'saved_ci95_s': np.percentile(rng.choice(gain, (10000, len(gain)), replace=True).mean(1), [2.5, 97.5]).tolist(),
                'teacher_gain_retained_percent': float(100*gain.mean()/(baseline-teacher).mean()),
                'wins_ties_losses': [int(sum(gain > 1e-6)), int(sum(abs(gain) <= 1e-6)), int(sum(gain < -1e-6))],
                'completed': sum(r['completed'] == 'True' for r in records), 'collisions': sum(r['collision'] == 'True' for r in records),
                'contract_violations': sum(int(r['contract_violations']) for r in records),
                'outside_steps': sum(int(r['outside_steps']) for r in records), 'uncertified_steps': sum(int(r['uncertified_steps']) for r in records),
                'max_prediction_error': max(float(r['max_prediction_error']) for r in records),
                'plan_fallback_count': sum(int(r['plan_fallback_count']) for r in records)}
        additional = reference_pair-teacher
        contrasts[regime]['full_vs_pair'] = {'mean_saved_s': float(additional.mean()),
            'ci95_s': np.percentile(rng.choice(additional, (10000, len(additional)), replace=True).mean(1), [2.5, 97.5]).tolist(),
            'better_tied_worse': [int(sum(additional > 1e-6)), int(sum(abs(additional) <= 1e-6)), int(sum(additional < -1e-6))]}
    raw = json.loads((D/'study/timing.json').read_text()); timing = {}
    keys = [(s, seed) for s, seeds in protocol['test_seeds'].items() for seed in seeds]
    responses = {m: np.array([float(lookup[m, s, seed]['first_response_ms']) for s, seed in keys]) for m in methods}
    for method in methods:
        group = [r for r in raw if r['method'] == method]
        timing[method] = {'selection_p95_ms': float(np.percentile([r['selection_ms'] for r in group], 95)),
                          'cycle_p95_ms': float(np.percentile([v for r in group for v in r['cycle_ms']], 95)),
                          'first_response_p95_ms': float(np.percentile(responses[method], 95)),
                          'first_response_max_ms': float(responses[method].max()),
                          'first_response_median_ms': float(np.median(responses[method]))}
    compiled = ('fast_pair', 'fast_full', 'bb_natural', 'bb_learned')
    counts = {m: np.array([int(lookup[m, s, seed]['certificate_calls']) for s, seed in keys]) for m in compiled}
    mechanism = {m: {'mean_certificate_calls': float(counts[m].mean()),
                     'mean_pruned_profiles': float(np.mean([int(lookup[m, s, seed]['pruned_profiles']) for s, seed in keys]))}
                 for m in compiled}
    mechanism['learned_reduction_vs_full'] = float(1-counts['bb_learned'].mean()/counts['fast_full'].mean())
    mechanism['learned_reduction_vs_natural'] = float(1-counts['bb_learned'].mean()/counts['bb_natural'].mean())
    reduction = counts['bb_natural']-counts['bb_learned']
    mechanism['learned_saved_calls_vs_natural_ci95'] = np.percentile(rng.choice(reduction, (10000, len(keys)), replace=True).mean(1), [2.5, 97.5]).tolist()
    ratios = {}
    indices = rng.integers(0, len(keys), (5000, len(keys)))
    for method, reference in [('fast_pair', 'reference_pair'), ('bb_learned', 'reference_pair'),
                              ('bb_learned', 'fast_full'), ('bb_learned', 'bb_natural')]:
        bootstrap = np.percentile(responses[method][indices], 95, axis=1)/np.percentile(responses[reference][indices], 95, axis=1)
        ratios[f'{method}_vs_{reference}'] = {
            'p95_ratio': timing[method]['first_response_p95_ms']/timing[reference]['first_response_p95_ms'],
            'paired_bootstrap_ci95': np.percentile(bootstrap, [2.5, 97.5]).tolist()}
    criteria = {
        'zero_recorded_safety_violations': all(r['collision'] == 'False' and all(int(r[k]) == 0 for k in ('contract_violations', 'outside_steps', 'uncertified_steps')) for r in rows),
        'all_fast_pair_costs_equal_reference_pair': all(lookup['fast_pair', s, seed]['mean_capped_finish_s'] == lookup['reference_pair', s, seed]['mean_capped_finish_s'] for s, seed in keys),
        'all_full_and_pruned_selections_equal_reference_full': all(lookup[m, s, seed]['selected'] == lookup['reference_full', s, seed]['selected'] and lookup[m, s, seed]['mean_capped_finish_s'] == lookup['reference_full', s, seed]['mean_capped_finish_s'] for m in ('fast_full', 'bb_natural', 'bb_learned') for s, seed in keys),
        'fast_pair_first_response_p95_ratio_max': ratios['fast_pair_vs_reference_pair']['p95_ratio'] <= protocol['criteria']['fast_pair_first_response_p95_ratio_max'],
        'bb_learned_first_response_p95_vs_reference_pair_max': ratios['bb_learned_vs_reference_pair']['p95_ratio'] <= protocol['criteria']['bb_learned_first_response_p95_vs_reference_pair_max'],
        'bb_learned_mean_certificate_reduction_vs_fast_full_min': mechanism['learned_reduction_vs_full'] >= protocol['criteria']['bb_learned_mean_certificate_reduction_vs_fast_full_min']}
    result = {'summary': summary, 'contrasts': contrasts, 'timing': timing, 'mechanism': mechanism,
              'latency_ratios': ratios, 'criteria': criteria, 'all_criteria_passed': all(criteria.values()),
              'cold_start': json.loads((D/'cold_start.json').read_text()),
              'intervals': 'Descriptive paired scene bootstrap; separate from a hard-real-time or distributional guarantee.'}
    (D/'summary.json').write_text(json.dumps(result, indent=2)+'\n', newline='\n')
    lines = ['# V8 精确核验加速结果', '', '本轮采用200个全新场景、七方法共1400次真实SUMO执行。此前360个公开场景仅用于开发审计；冻结协议后没有再调算法。每个场景的执行方法顺序按冻结规则随机排列。', '',
             f"预先声明的六项改善标准：{'全部通过' if all(criteria.values()) else '未全部通过'}。逐项结果见summary.json。", '',
             '## 同场景到达代价', '', '| 方法 | 常规 s | 高密度 s | 常规改善 | 高密度改善 |', '|---|---:|---:|---:|---:|']
    for m in methods:
        a, b = summary['nominal'][m], summary['dense'][m]
        lines.append(f"| {ZH[m]} | {a['mean_arrival_s']:.3f} | {b['mean_arrival_s']:.3f} | {a['gain_percent']:.2f}% | {b['gain_percent']:.2f}% |")
    lines += ['', '编译完整搜索及两种剪枝逐场景选择、代价均与原完整搜索一致；编译双方案与原双方案一致。精确性来自保留完整候选库与合法下界，不来自预测置信度。', '']
    for regime, value in contrasts.items():
        v = value['full_vs_pair']; ci = v['ci95_s']
        lines.append(f"- {regime}完整方案相对双方案再节省{v['mean_saved_s']:.4f} s，描述性95%配对区间[{ci[0]:.4f}, {ci[1]:.4f}]；更优/相同/更差={v['better_tied_worse']}。")
    lines += ['', '## 暖机计算成本', '', '| 方法 | 选择P95 ms | 首次响应P95 ms | 周期P95 ms |', '|---|---:|---:|---:|']
    for m in methods:
        t = timing[m]
        lines.append(f"| {ZH[m]} | {t['selection_p95_ms']:.3f} | {t['first_response_p95_ms']:.3f} | {t['cycle_p95_ms']:.3f} |")
    lines += ['', '首次响应包含真实在线方案计算与第一个SUMO执行周期。完整参考方法在计时窗口中重新规划，没有借用提前生成的评测标签。', '']
    for name, v in ratios.items():
        ci = v['paired_bootstrap_ci95']
        lines.append(f"- {name}首次响应P95比值{v['p95_ratio']:.4f}，配对bootstrap区间[{ci[0]:.4f}, {ci[1]:.4f}]。")
    lines += ['', '## 编译与学习贡献分离', '', '| 同编译后端 | 平均证书调用 | 平均剪枝方案数 |', '|---|---:|---:|']
    for m in compiled:
        v = mechanism[m]; lines.append(f"| {ZH[m]} | {v['mean_certificate_calls']:.1f} | {v['mean_pruned_profiles']:.1f} |")
    lines += ['', f"学习排序剪枝相对完整编译搜索减少{mechanism['learned_reduction_vs_full']:.1%}证书调用；相对自然顺序剪枝额外减少{mechanism['learned_reduction_vs_natural']:.1%}。必须将后者与学习推断开销一起判断，不能把编译带来的提速归给学习器。", '',
              '学习排序与自然顺序都返回相同最优方案，因此学习在本轮只可能改善搜索成本，不产生额外通行收益。若不追求学习机制，编译完整搜索本身也是强而简单的工程基线。', '',
              '## 启动与适用边界', '',
              f"全新缓存测量：导入{result['cold_start']['imports_ms']:.1f} ms，首次编译内核调用{result['cold_start']['first_kernel_call_ms']:.1f} ms，随后同一开发初态调用{result['cold_start']['second_kernel_call_ms']:.3f} ms。必须先预热再接管控制，不能用暖机毫秒数描述冷启动。", '',
              '本轮保持原精确模型、三车全受控和固定路线的限制。没有模拟计算期间车辆继续运动，不是WCET或实车验证；统计门控在V7的失效仍然存在。算法实现、有限库精确性和论文新颖性是不同问题。', '',
              '[理论](theory_v8.md) · [冻结协议](../artifacts/v8/protocol.json) · [原始回合](../artifacts/v8/study/episodes.csv)', '',
              '```powershell', 'python -m pip install -e ".[dev,fast]"', 'python tools/verify_release_v8.py --replay', 'python tools/study_v8.py --out runs/my_v8_reproduction', '```', '']
    (ROOT/'docs/v8_results.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    x = np.arange(len(methods)); labels = ['Baseline', 'Python\npair', 'Fast\npair', 'Fast\nfull', 'Natural\nbound', 'Learned\nbound', 'Python\nfull']
    for regime in ('nominal', 'dense'):
        axes[0].plot(x, [summary[regime][m]['mean_arrival_s'] for m in methods], 'o-', label=regime)
    axes[0].set_ylabel('Mean capped arrival time (s)'); axes[0].legend(); axes[0].set_title('Same 200 held-out scenes')
    axes[1].bar(x, [timing[m]['first_response_p95_ms'] for m in methods], color='#346e73')
    axes[1].set_yscale('log'); axes[1].set_ylabel('First response P95 (ms, log scale)'); axes[1].set_title('Warm, single-worker execution')
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=8); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(D/'results.png', dpi=180); plt.close(fig)
    hashes = {p.relative_to(D).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(D.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
    (D/'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2)+'\n', newline='\n')
    print(json.dumps({'criteria': criteria, 'timing': timing, 'mechanism': mechanism, 'contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    main()
