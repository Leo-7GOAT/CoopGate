"""Cluster-aware paired results and the six predeclared V9 criteria."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; D = ROOT/'artifacts/v9'
ZH = {'v8_cached':'V8缓存计划', 'nominal_feedback':'名义反馈', 'robust_constant':'鲁棒传统基线',
      'robust_fixed_pair':'固定提案双方案', 'robust_random_pair':'随机提案双方案',
      'robust_learned_pair':'学习提案双方案', 'robust_full':'鲁棒完整搜索'}
MODES = {'clean':'零误差', 'uniform':'随机误差', 'bias':'持续偏差', 'pulse':'间歇脉冲'}


def failed(r):
    return r['aborted']=='True' or r['collision']=='True' or int(r['contract_violations'])>0


def aggregates(rows):
    return {'n':len(rows), 'mean_cost_s':float(np.mean([float(r['mean_capped_finish_s']) for r in rows])),
            'completed':sum(r['completed']=='True' for r in rows), 'aborted':sum(r['aborted']=='True' for r in rows),
            'collision_episodes':sum(r['collision']=='True' for r in rows),
            'contract_violation_episodes':sum(int(r['contract_violations'])>0 for r in rows),
            'contract_violating_substeps':sum(int(r['contract_violations']) for r in rows),
            'failed_episodes':sum(failed(r) for r in rows),
            'timeouts':sum(r['completed']=='False' and r['aborted']=='False' and r['collision']=='False' for r in rows),
            'mean_fallback_steps':float(np.mean([int(r['fallback_steps']) for r in rows])),
            'warm_first_response_p95_ms':float(np.percentile([float(r['first_response_ms']) for r in rows],95)),
            'max_prediction_error':max(float(r['max_prediction_error']) for r in rows)}


def summarize():
    protocol = json.loads((D/'protocol.json').read_text()); rows = list(csv.DictReader((D/'study/episodes.csv').open()))
    methods, modes = protocol['methods'], protocol['disturbances']
    lookup = {(r['scenario'],int(r['seed']),r['disturbance'],r['method']):r for r in rows}
    expected = {(s,seed,mode,m) for s,seeds in protocol['test_seeds'].items() for seed in seeds for mode in modes for m in methods}
    assert len(rows)==len(lookup)==protocol['executions'] and set(lookup)==expected
    summary = {mode:{m:aggregates([r for r in rows if r['disturbance']==mode and r['method']==m]) for m in methods} for mode in modes}
    grouped = {s:{m:aggregates([r for r in rows if r['scenario']==s and r['method']==m and r['disturbance']!='clean']) for m in methods} for s in protocol['test_seeds']}
    overall = {m:aggregates([r for r in rows if r['method']==m]) for m in methods}
    perturbed = {m:aggregates([r for r in rows if r['method']==m and r['disturbance']!='clean']) for m in methods}
    contrasts = {}; rng = np.random.default_rng(2026101691)
    for regime in ('all','nominal','dense'):
        keys = [(s,seed) for s,seeds in protocol['test_seeds'].items() if regime in ('all',s) for seed in seeds]
        contrasts[regime] = {}
        for method in methods:
            if method == 'robust_learned_pair': continue
            # Each seed contributes one mean difference across its three error modes.
            differences = [np.mean([float(lookup[s,seed,mode,method]['mean_capped_finish_s'])-
                                   float(lookup[s,seed,mode,'robust_learned_pair']['mean_capped_finish_s']) for mode in modes[1:]]) for s,seed in keys]
            baseline = np.mean([float(lookup[s,seed,mode,method]['mean_capped_finish_s']) for s,seed in keys for mode in modes[1:]])
            differences = np.array(differences)
            contrasts[regime][method] = {'clusters':len(keys),'mean_saved_s':float(differences.mean()),
                'gain_percent':float(100*differences.mean()/baseline),
                'paired_cluster_ci95_s':np.percentile(rng.choice(differences,(10000,len(keys)),replace=True).mean(1),[2.5,97.5]).tolist(),
                'better_tied_worse_clusters':[int(sum(differences>1e-6)),int(sum(abs(differences)<=1e-6)),int(sum(differences < -1e-6))]}
    target = overall['robust_learned_pair']; old_fail = perturbed['v8_cached']['failed_episodes']
    failure_reduction = (old_fail-perturbed['robust_learned_pair']['failed_episodes'])/old_fail if old_fail else 0.
    criteria = {
        'robust_learned_pair_zero_collision_contract_violation_and_abort':target['failed_episodes']==0,
        'robust_learned_pair_completion_fraction_min':target['completed']/target['n']>=.98,
        'robust_learned_pair_perturbed_gain_vs_robust_constant_percent_min':contrasts['all']['robust_constant']['gain_percent']>=3.,
        'robust_learned_pair_perturbed_paired_ci_lower_gt_zero':contrasts['all']['robust_constant']['paired_cluster_ci95_s'][0]>0.,
        'robust_learned_pair_failure_reduction_vs_v8_cached_min':failure_reduction>=.8,
        'robust_learned_pair_warm_first_response_p95_ms_max':target['warm_first_response_p95_ms']<=20.}
    assert set(criteria)==set(protocol['criteria'])
    result = {'summary_by_error':summary,'perturbed_by_regime':grouped,'overall':overall,'perturbed':perturbed,
              'contrasts_perturbed':contrasts,'failure_reduction_vs_v8':failure_reduction,
              'criteria':criteria,'all_criteria_passed':all(criteria.values()),
              'independent_clusters':200,'executions':len(rows)}
    (D/'summary.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    text = ['# V9 有界执行误差实验结果','',
            '算法与六项标准在新实验前冻结。200个全新初态、四类误差、七种方法，共5600次真实SUMO执行。每类误差重复同一初态，统计区间按初态整簇重采样。',
            '',f"六项改善标准全部通过：{result['all_criteria_passed']}。完整真假值见summary.json。",'',
            '## 全部条件结果','', '| 方法 | 完成/800 | 中止 | 合同违规回合 | 碰撞回合 | 首响应P95 ms |', '|---|---:|---:|---:|---:|---:|']
    for m in methods:
        r=overall[m];text.append(f"| {ZH[m]} | {r['completed']} | {r['aborted']} | {r['contract_violation_episodes']} | {r['collision_episodes']} | {r['warm_first_response_p95_ms']:.2f} |")
    text += ['', '控制器中止、碰撞和合同违规分别记录，可能发生在同一回合，不能相加当独立故障。中止按25 s计入代价；超时也单列，未删除不利场景。旧控制器是在其精确模型保证范围之外接受本轮压力测试。', '',
             '## 每类误差的任务代价','', '| 方法 | 零误差 s | 随机误差 s | 持续偏差 s | 间歇脉冲 s |', '|---|---:|---:|---:|---:|']
    for m in methods:text.append('| '+ZH[m]+' | '+' | '.join(f"{summary[mode][m]['mean_cost_s']:.3f}" for mode in modes)+' |')
    text += ['', '## 学习提案的独立贡献','',
             '以下三类扰动合并，每个种子先对三个模式的配对差取平均，再按种子bootstrap。正数表示学习提案双方案更快。固定/随机/学习提案使用同一物理证书和最多两个候选；不是把鲁棒证书的收益都归于学习。','',
             '| 比较方法 | 平均节省 s | 改善 % | 描述性95%整簇配对区间 s |','|---|---:|---:|---|']
    for m,r in contrasts['all'].items():text.append(f"| {ZH[m]} | {r['mean_saved_s']:.3f} | {r['gain_percent']:.2f} | [{r['paired_cluster_ci95_s'][0]:.3f}, {r['paired_cluster_ci95_s'][1]:.3f}] |")
    text += ['',f"相对鲁棒传统基线，常规/高密度的扰动平均代价分别改善{contrasts['nominal']['robust_constant']['gain_percent']:.2f}%/{contrasts['dense']['robust_constant']['gain_percent']:.2f}%。按每个初态的三模式平均差计，更好/持平/更差为{contrasts['all']['robust_constant']['better_tied_worse_clusters']}；不是逐场景不退化保证。",
             '', '鲁棒完整搜索平均更快，但首次响应开销更大。学习双方案在任务性能与计算开销之间取舍，不能声称同时优于完整搜索的所有指标。']
    text += ['', '## 适用边界','',
             '扰动界为事先声明的±0.5 m/s²，测量仍然精确；指令变化率约束不等于带噪实际加速度的变化率约束。安全论证依赖区间包络、联合制动后缀和停止终端集合。实际扰动下没有逐场景通行性能不退化保证。',
             '', '训练模型没有更新，仍是集中式离线上下文bandit。当前结果不证明置信评分可靠，也没有覆盖感知误差、未知误差界、非合作车辆、延迟或实车。首次响应计时均已预热，不包括编译冷启动。',
             '', '[理论与证明](theory_v9.md) · [冻结协议](../artifacts/v9/protocol.json) · [原始回合](../artifacts/v9/study/episodes.csv)', '']
    (ROOT/'docs/v9_results.md').write_text('\n'.join(text),encoding='utf-8',newline='\n')
    fig, axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    selected=['v8_cached','nominal_feedback','robust_constant','robust_random_pair','robust_learned_pair','robust_full']
    names=['V8 cached','Nominal\nfeedback','Robust\nbaseline','Random\npair','Learned\npair','Robust\nfull']
    axes[0].bar(names,[perturbed[m]['failed_episodes'] for m in selected],color=['#999999','#bbbbbb','#61758b','#4d97a2','#1a785b','#6ba68f'])
    axes[0].set_ylabel('Episodes with abort / contract violation / collision');axes[0].set_title('600 perturbed episodes per method')
    axes[1].bar(names,[perturbed[m]['mean_cost_s'] for m in selected],color='#356e74');axes[1].set_ylabel('Mean capped arrival cost (s)');axes[1].set_title('Aborts counted as 25 s, same error tapes')
    for ax in axes:ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(D/'results.png',dpi=180);plt.close(fig)
    hashes={str(p.relative_to(D)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    (D/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n',newline='\n')
    print(json.dumps({'criteria':criteria,'overall':overall,'contrasts':contrasts['all']},indent=2))


if __name__=='__main__':summarize()
