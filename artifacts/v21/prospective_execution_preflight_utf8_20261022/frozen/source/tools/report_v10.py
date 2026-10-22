"""Recompute the frozen V10 criteria and paired seed-cluster comparisons."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v10';TARGET='rollout_learned'
ZH={'v9_constant':'V9传统基线','v9_learned_pair':'V9学习双方案','v9_full':'V9完整搜索',
    'rollout_constant':'无学习滚动优化','rollout_fixed_pair':'固定提案滚动优化',
    'rollout_random_pair':'随机提案滚动优化','rollout_learned_binary':'学习提案二值动作',
    'rollout_learned':'V10学习提案滚动优化'}
MODES={'clean':'零误差','uniform':'随机误差','bias':'持续偏差','pulse':'间歇脉冲'}


def failed(r):
    return r['aborted']=='True' or r['collision']=='True' or int(r['contract_violations'])>0


def aggregate(rows, timing):
    controller=[v for t in timing for v in t['controller_ms']]
    cycle=[v for t in timing for v in t['cycle_ms']]
    return {'n':len(rows),'mean_cost_s':float(np.mean([float(r['mean_capped_finish_s']) for r in rows])),
        'completed':sum(r['completed']=='True' for r in rows),'failed_episodes':sum(failed(r) for r in rows),
        'aborted':sum(r['aborted']=='True' for r in rows),'collision_episodes':sum(r['collision']=='True' for r in rows),
        'contract_violation_episodes':sum(int(r['contract_violations'])>0 for r in rows),
        'timeouts':sum(r['completed']=='False' and r['aborted']=='False' and r['collision']=='False' for r in rows),
        'mean_command_jerk':float(np.mean([float(r['mean_jerk']) for r in rows])),
        'mean_stopped_vehicle_seconds':float(np.mean([float(r['stopped_vehicle_seconds']) for r in rows])),
        'mean_accepted_updates':float(np.mean([int(r['accepted_updates']) for r in rows])),
        'mean_base_switches':float(np.mean([int(r['base_switches']) for r in rows])),
        'warm_first_response_p95_ms':float(np.percentile([float(r['first_response_ms']) for r in rows],95)),
        'warm_controller_cycle_p95_ms':float(np.percentile(controller,95)),
        'warm_controller_cycle_max_ms':float(max(controller)),
        'warm_end_to_end_cycle_p95_ms':float(np.percentile(cycle,95)),
        'cycle_count':len(controller),
        'tail_feedback_steps':sum(int(r['tail_feedback_steps']) for r in rows),
        'trial_checks':sum(int(r['trial_checks']) for r in rows),
        'pruned_trials':sum(int(r['pruned_trials']) for r in rows)}


def compute(protocol, rows, timings):
    methods,modes=protocol['methods'],protocol['disturbances']
    lookup={(r['scenario'],int(r['seed']),r['disturbance'],r['method']):r for r in rows}
    expected={(s,seed,mode,m) for s,seeds in protocol['test_seeds'].items() for seed in seeds for mode in modes for m in methods}
    assert len(rows)==len(lookup)==protocol['executions'] and set(lookup)==expected
    by_timing={(r['scenario'],int(r['seed']),r['disturbance'],r['method']):r for r in timings}
    assert len(timings)==len(by_timing) and set(by_timing)==expected
    def stats(method, scenario=None, mode=None, perturbed=False):
        group=[r for r in rows if r['method']==method and (scenario is None or r['scenario']==scenario)
               and (mode is None or r['disturbance']==mode) and (not perturbed or r['disturbance']!='clean')]
        return aggregate(group,[by_timing[r['scenario'],int(r['seed']),r['disturbance'],method] for r in group])
    overall={m:stats(m) for m in methods}; perturbed={m:stats(m,perturbed=True) for m in methods}
    by_error={mode:{m:stats(m,mode=mode) for m in methods} for mode in modes}
    regimes={s:{m:stats(m,scenario=s,perturbed=True) for m in methods} for s in protocol['test_seeds']}
    contrasts={};rng=np.random.default_rng(2026101611)
    for regime in ('all','nominal','dense'):
        keys=[(s,seed) for s,seeds in protocol['test_seeds'].items() if regime in ('all',s) for seed in seeds]
        contrasts[regime]={}
        for method in methods:
            if method==TARGET:continue
            differences=np.array([np.mean([float(lookup[s,seed,mode,method]['mean_capped_finish_s'])-
                float(lookup[s,seed,mode,TARGET]['mean_capped_finish_s']) for mode in modes[1:]]) for s,seed in keys])
            baseline=np.mean([float(lookup[s,seed,mode,method]['mean_capped_finish_s']) for s,seed in keys for mode in modes[1:]])
            contrasts[regime][method]={'clusters':len(keys),'mean_saved_s':float(differences.mean()),
                'gain_percent':float(100*differences.mean()/baseline),
                'paired_cluster_ci95_s':np.percentile(rng.choice(differences,(10000,len(keys)),replace=True).mean(1),[2.5,97.5]).tolist(),
                'better_tied_worse_clusters':[int(sum(differences>1e-6)),int(sum(abs(differences)<=1e-6)),int(sum(differences < -1e-6))]}
    clean_worse=sum(float(lookup[s,seed,'clean',TARGET]['mean_capped_finish_s'])>
                    float(lookup[s,seed,'clean','v9_learned_pair']['mean_capped_finish_s'])+1e-8
                    for s,seeds in protocol['test_seeds'].items() for seed in seeds)
    target=overall[TARGET];c=protocol['criteria'];primary=contrasts['all']['v9_learned_pair']
    criteria={
        'zero_collision_contract_violation_and_abort':target['failed_episodes']==0,
        'completion_fraction_min':target['completed']/target['n']>=c['completion_fraction_min'],
        'perturbed_gain_vs_v9_learned_percent_min':primary['gain_percent']>=c['perturbed_gain_vs_v9_learned_percent_min'],
        'primary_paired_cluster_ci_lower_gt_zero':primary['paired_cluster_ci95_s'][0]>0,
        'perturbed_gain_vs_v9_full_percent_min':contrasts['all']['v9_full']['gain_percent']>=c['perturbed_gain_vs_v9_full_percent_min'],
        'perturbed_dense_gain_vs_v9_learned_percent_min':contrasts['dense']['v9_learned_pair']['gain_percent']>=c['perturbed_dense_gain_vs_v9_learned_percent_min'],
        'clean_worse_than_v9_learned_count_max':clean_worse<=c['clean_worse_than_v9_learned_count_max'],
        'warm_controller_cycle_p95_ms_max':target['warm_controller_cycle_p95_ms']<=c['warm_controller_cycle_p95_ms_max'],
        'warm_first_response_p95_ms_max':target['warm_first_response_p95_ms']<=c['warm_first_response_p95_ms_max']}
    assert set(criteria)==set(c)
    return {'executions':len(rows),'independent_clusters':sum(len(s) for s in protocol['test_seeds'].values()),
        'overall':overall,'perturbed':perturbed,'summary_by_error':by_error,'perturbed_by_regime':regimes,
        'contrasts_perturbed':contrasts,'clean_worse_than_v9_learned':clean_worse,
        'criteria':criteria,'all_criteria_passed':all(criteria.values())}


def main():
    protocol=json.loads((D/'protocol.json').read_text()); rows=list(csv.DictReader((D/'study/episodes.csv').open()))
    r=compute(protocol,rows,json.loads((D/'study/timing.json').read_text()))
    (D/'summary.json').write_text(json.dumps(r,indent=2)+'\n',newline='\n')
    methods=protocol['methods'];target=r['overall'][TARGET];primary=r['contrasts_perturbed']['all']['v9_learned_pair']
    text=['# V10 滚动策略改进实验结果','',
        '在新实验前冻结代码、200个全新初态和九项验收标准。四类误差、八种方法，共6400次实际SUMO执行；每种方法800回合。三类非零误差共600回合，但只对应200个独立初态。',
        '',f"九项预先声明的标准全部通过：{r['all_criteria_passed']}。",'',
        f"主要比较相对V9学习双方案，扰动下平均代价减少{primary['mean_saved_s']:.3f} s（{primary['gain_percent']:.2f}%）；95%配对整簇区间为[{primary['paired_cluster_ci95_s'][0]:.3f}, {primary['paired_cluster_ci95_s'][1]:.3f}] s。",'',
        '## 完成情况与计算成本','',
        '| 方法 | 完成/800 | 故障回合 | 暖机首响应P95 ms | 暖机控制周期P95 ms |','|---|---:|---:|---:|---:|']
    for m in methods:
        a=r['overall'][m];text.append(f"| {ZH[m]} | {a['completed']} | {a['failed_episodes']} | {a['warm_first_response_p95_ms']:.2f} | {a['warm_controller_cycle_p95_ms']:.2f} |")
    text += ['', '故障回合取中止、路线合同违规或碰撞的并集；分项仍分别保存在summary.json。超时与失败均保留，控制器中止按25 s计代价。暖机首响应包括初始策略选择及第一次TraCI周期；控制周期计时仅包括控制算法。', '',
        '## 每类执行误差的任务代价','',
        '| 方法 | 零误差 s | 随机误差 s | 持续偏差 s | 间歇脉冲 s |','|---|---:|---:|---:|---:|']
    for m in methods:text.append('| '+ZH[m]+' | '+' | '.join(f"{r['summary_by_error'][mode][m]['mean_cost_s']:.3f}" for mode in protocol['disturbances'])+' |')
    text += ['', '## 配对比较与贡献拆分','',
        '各初态先对三类非零误差取平均配对差，再以初态为整簇进行10000次bootstrap。正数代表V10学习提案滚动优化更快。主要对照为V9学习双方案；其他区间为描述性结果，未声称多重比较显著性。','',
        '| 对照方法 | 平均节省 s | 改善 % | 95%整簇区间 s | 更好/持平/更差初态 |','|---|---:|---:|---|---|']
    for m,a in r['contrasts_perturbed']['all'].items():
        ci=a['paired_cluster_ci95_s'];text.append(f"| {ZH[m]} | {a['mean_saved_s']:.3f} | {a['gain_percent']:.2f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {'/'.join(map(str,a['better_tied_worse_clusters']))} |")
    text += ['',f"相对V9学习双方案，常规/高密度扰动场景分别改善{r['contrasts_perturbed']['nominal']['v9_learned_pair']['gain_percent']:.2f}%/{r['contrasts_perturbed']['dense']['v9_learned_pair']['gain_percent']:.2f}%。零误差下更差初态为{r['clean_worse_than_v9_learned']}/200；这是有限样本检查，理论依据见命题2。",
        '', '无学习、固定提案、随机提案与学习提案使用相同物理证书和滚动优化器。学习模块的额外收益必须单独看这些对照，不能把整个前瞻优化收益都称为强化学习收益。二值动作消融只使用每车两个端点，保留相同基策略比较与前瞻机制。','',
        '## 指令平顺与停车情况','',
        '| 方法 | 平均指令变化率 m/s³ | 平均停车车辆秒 |','|---|---:|---:|']
    for m in methods:
        a=r['perturbed'][m];text.append(f"| {ZH[m]} | {a['mean_command_jerk']:.3f} | {a['mean_stopped_vehicle_seconds']:.3f} |")
    text += ['', '停车车辆秒按每个0.5 s决策采样点累计未到达且速度低于0.2 m/s的车辆数，不等于连续时间精确停车时长。平顺指标针对指令，不对含执行误差的实际加速度变化率作同等声明。','',
        '## 结论边界','',
        '安全证书沿用V9，没有放松间距、误差界或变化率限制。名义性能不退化依赖完整基策略尾部和全局时间；实际扰动下仍有可能退化，表中保留全部不利初态。',
        '', '当前贡献是联合证书内的滚动动作改进及可复现实证；坐标rollout、基策略改进和预测安全过滤已有文献先例。冻结学习器仍是集中式上下文bandit，不是新的分布式MARL。',
        '', '仿真计算期间车辆暂停，暖机P95不等于最坏执行时间保证。本轮没有增加通信延迟统计，感知误差、未知误差界、非合作车辆和实车仍未验证。',
        '', '[理论与证明](theory_v10.md) · [冻结协议](../artifacts/v10/protocol.json) · [全部回合](../artifacts/v10/study/episodes.csv)', '']
    (ROOT/'docs/v10_results.md').write_text('\n'.join(text),encoding='utf-8',newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=['v9_constant','v9_learned_pair','v9_full','rollout_constant','rollout_learned_binary',TARGET]
    labels=['V9 baseline','V9 learned','V9 full','Rollout\nno learning','Rollout\nbinary','V10 rollout']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    colors=['#99a4ad','#7894a4','#5f7f94','#7caf9c','#4b9288','#196955']
    axes[0].bar(labels,[r['perturbed'][m]['mean_cost_s'] for m in selected],color=colors)
    axes[0].set_ylabel('Mean capped arrival cost (s)');axes[0].set_title('600 perturbed SUMO runs per method')
    axes[1].bar(labels,[r['overall'][m]['warm_controller_cycle_p95_ms'] for m in selected],color=colors)
    axes[1].set_ylabel('Warm controller cycle P95 (ms)');axes[1].set_title('Extra online work, unchanged safety certificate')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=8)
    fig.savefig(D/'results.png',dpi=180);plt.close(fig)
    hashes={str(p.relative_to(D)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    (D/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n',newline='\n')
    print(json.dumps({'criteria':r['criteria'],'primary':primary,'overall_target':target,'contrasts':r['contrasts_perturbed']['all']},indent=2))


if __name__=='__main__':main()
