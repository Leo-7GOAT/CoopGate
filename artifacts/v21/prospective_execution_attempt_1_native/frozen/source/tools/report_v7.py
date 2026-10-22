"""Report both predeclared studies, including all-rejection negative results."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
ZH = {'constant': '传统基线', 'learned': '冻结学习器', 'v6_conformal': '原V6门控',
      'robust_conformal': '两分布最大分位数', 'robust_pac': '同时PAC门控',
      'verified_pair': '双方案模型核验', 'teacher': '完整规划教师'}


def wilson(k, n):
    if n == 0:
        return None
    z = norm.ppf(.975); p = k/n; denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def summarize(version):
    d = ROOT/f'artifacts/{version}'; protocol = json.loads((d/'protocol.json').read_text())
    rows = list(csv.DictReader((d/'study/episodes.csv').open()))
    lookup = {(r['method'], r['scenario'], int(r['seed'])): r for r in rows}
    expected = {(m, s, seed) for m in protocol['methods'] for s, seeds in protocol['test_seeds'].items() for seed in seeds}
    assert len(rows) == len(lookup) == protocol['executions'] and set(lookup) == expected
    rng = np.random.default_rng(2026101519); summary = {}
    for regime, seeds in protocol['test_seeds'].items():
        summary[regime] = {}
        baseline = np.array([float(lookup['constant', regime, s]['mean_capped_finish_s']) for s in seeds])
        teacher = np.array([float(lookup['teacher', regime, s]['mean_capped_finish_s']) for s in seeds])
        for method in protocol['methods']:
            records = [lookup[method, regime, s] for s in seeds]
            cost = np.array([float(r['mean_capped_finish_s']) for r in records]); gain = baseline-cost
            accepted = np.array([int(r['selected']) != 0 for r in records]); n = int(accepted.sum())
            failures = int(np.sum((gain < -1e-6) & accepted))
            result = {'n': len(seeds), 'mean_arrival_s': float(cost.mean()), 'mean_saved_s': float(gain.mean()),
                      'saved_ci95_s': np.percentile(rng.choice(gain, (10000, len(gain)), replace=True).mean(1), [2.5, 97.5]).tolist(),
                      'gain_percent': float(100*gain.mean()/baseline.mean()),
                      'teacher_gain_retained_percent': float(100*gain.mean()/(baseline-teacher).mean()),
                      'accepted': n, 'accepted_failures': failures, 'conditional_risk': failures/n if n else None,
                      'conditional_wilson95': wilson(failures, n), 'unconditional_risk': float(np.mean(gain < -1e-6)),
                      'wins_ties_losses': [int(sum(gain > 1e-6)), int(sum(np.abs(gain) <= 1e-6)), int(sum(gain < -1e-6))],
                      'completed': sum(r['completed'] == 'True' for r in records),
                      'collisions': sum(r['collision'] == 'True' for r in records),
                      'contract_violations': sum(int(r['contract_violations']) for r in records),
                      'outside_steps': sum(int(r['outside_steps']) for r in records),
                      'uncertified_steps': sum(int(r['uncertified_steps']) for r in records),
                      'max_prediction_error': max(float(r['max_prediction_error']) for r in records),
                      'plan_fallback_count': sum(int(r.get('plan_fallback_count', 0)) for r in records)}
            if method in ('v6_conformal', 'robust_conformal'):
                coverage = [float(r['proposal_actual_excess_s']) <= float(r['estimated_excess_s'])-float(r['threshold_s'])+1e-7 for r in records]
                result['proposal_coverage'] = float(np.mean(coverage))
                result['proposal_coverage_wilson95'] = wilson(sum(coverage), len(coverage))
            summary[regime][method] = result
    raw_timing = json.loads((d/'study/timing.json').read_text()); timing = {}
    for method in protocol['methods']:
        group = [r for r in raw_timing if r['method'] == method]
        timing[method] = {'selection_p95_ms': float(np.percentile([r['selection_ms'] for r in group], 95)),
                          'cycle_p95_ms': float(np.percentile([v for r in group for v in r['cycle_ms']], 95)),
                          'first_response_p95_ms': float(np.percentile([r['selection_ms']+r['cycle_ms'][0] for r in group], 95)),
                          'first_response_max_ms': max(r['selection_ms']+r['cycle_ms'][0] for r in group)}
    output = {'summary': summary, 'timing': timing, 'intervals': 'Descriptive per-regime paired bootstrap and Wilson 95%; no multiplicity correction for test comparisons.'}
    (d/'summary.json').write_text(json.dumps(output, indent=2)+'\n', newline='\n')
    return output


def main():
    results = {v: summarize(v) for v in ('v7', 'v7_pair')}
    cal = json.loads((ROOT/'artifacts/v7/calibration.json').read_text())
    lines = ['# V7 风险校准与双方案核验', '',
             'V7固定原学习模型，使用1024个新校准初态和160个独立测试场景，六方法共960次真实SUMO执行。校准诊断发现无可获证阈值后，另行冻结双方案核验补充协议，使用80个额外独立场景、四方法共320次执行。两个测试集合互不重叠；不能跨表直接比较绝对均值。', '',
             '## 先保留失败的统计校准', '',
             f"同时风险门控选中阈值：{cal['pac']['threshold_s']}。None表示所有非基线方案回退，不表示接受子集的风险为0%。本轮无候选同时通过两分布、13阈值的Clopper–Pearson校正上界。", '',
             '高密度校准中，阈值−1.0 s接受53次、退化7次；阈值−1.25 s虽无退化但仅接受13次，证据不足。其严格上界分别约31.5%和38.2%，都未达到10%目标。常规分布中存在合格候选，瓶颈主要是高密度分布的评分区分能力和严格接受区样本量。', '',
             '两分布最大分位数对照使用与PAC门控相同的数据。它与原V6对照的差异同时包含校准数据覆盖、样本量及阈值改变；不能把所有改善归因于统计方法。', '']
    for version, title in [('v7', '主实验'), ('v7_pair', '独立补充实验')]:
        data = results[version]
        lines += [f'## {title}', '', '| 分布 | 方法 | 到达 s | 改善 | 采用数 | 采用后退化数 | 条件风险95%区间 |', '|---|---|---:|---:|---:|---:|---|']
        for regime, group in data['summary'].items():
            for method, r in group.items():
                ci = r['conditional_wilson95']; ci_text = '不定义' if ci is None else f'[{ci[0]:.1%}, {ci[1]:.1%}]'
                lines.append(f"| {regime} | {ZH[method]} | {r['mean_arrival_s']:.3f} | {r['gain_percent']:.2f}% | {r['accepted']}/{r['n']} | {r['accepted_failures']} | {ci_text} |")
        lines += ['', '区间为描述性Wilson区间，不是校准阶段的同时风险证书；原始统计和配对bootstrap区间见summary.json。', '']
        if version == 'v7':
            for regime in ('nominal', 'dense'):
                r = data['summary'][regime]['robust_conformal']; ci = r['proposal_coverage_wilson95']
                lines.append(f"- {regime}最大分位数门控的提案上界实测覆盖率{r['proposal_coverage']:.1%}，描述性95% Wilson区间[{ci[0]:.1%}, {ci[1]:.1%}]。")
            lines += ['', '高密度实测覆盖率仍低于90%，不能声称本轮已解决经验覆盖不足；校准理论边际性质不等于固定这份校准集的经验达标。', '']
        lines += ['| 方法 | 选择P95 ms | 首次响应P95 ms | 周期P95 ms |', '|---|---:|---:|---:|']
        for method, t in data['timing'].items():
            lines.append(f"| {ZH[method]} | {t['selection_p95_ms']:.2f} | {t['first_response_p95_ms']:.2f} | {t['cycle_p95_ms']:.2f} |")
        lines.append('')
    lines += ['## 双方案核验的代价与保证', '',
              '双方案核验不是免费的置信估计：在线展开传统基线与学习提案各一条完整反馈轨迹，选择代价更低者，同分回退基线；执行时复用规划并持续检查联合制动证书。它保留每个场景中学习提案的正收益并删除负收益，但不能找到学习器未提出的更优方案。', '',
              '在精确模型、轨迹匹配且规划有效的条件下，J执行=min(J基线,J提案)≤J基线。状态失配后原规划失效并回退，不能继续声称原始全回合性能不退化。当前冻结基准没有模型扰动；未验证含计算延迟的闭环或硬实时截止。', '',
              '## 结论范围', '',
              '本轮增加了校准数据覆盖，也表明原评分不能支持所要求的两分布条件风险证书。全回退负结果完整保留。双方案核验提供了另一种收益与计算成本权衡，但没有解决未知模型下的置信度，也不构成新的分布式MARL算法。', '',
              '[理论与文献](theory_v7.md) · [主协议](../artifacts/v7/protocol.json) · [补充协议](../artifacts/v7_pair/protocol.json)', '',
              '```powershell', 'python tools/verify_release_v7.py --replay', 'python tools/study_v7.py --out runs/my_v7_reproduction', 'python tools/study_v7_pair.py --out runs/my_v7_pair_reproduction', '```', '']
    (ROOT/'docs/v7_results.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, version in zip(axes, ('v7', 'v7_pair')):
        summary = results[version]['summary']; methods = list(summary['nominal']); x = np.arange(len(methods))
        for offset, regime in [(-.18, 'nominal'), (.18, 'dense')]:
            ax.bar(x+offset, [summary[regime][m]['gain_percent'] for m in methods], width=.36, label=regime)
        ax.set_xticks(x, [m.replace('_', '\n') for m in methods], fontsize=8)
        ax.set_ylabel('Mean arrival-time improvement (%)'); ax.set_title(f'{version}: separate held-out scenes'); ax.legend()
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(ROOT/'artifacts/v7/results.png', dpi=180); plt.close(fig)
    for version in results:
        d = ROOT/f'artifacts/{version}'
        hashes = {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(d.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
        (d/'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2)+'\n', newline='\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
