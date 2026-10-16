"""Recompute complete paired accounting and the two prespecified endpoints."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v11';TARGET='smooth_learned'
ZH={'v9_learned_pair':'V9学习双方案','v10_rollout':'V10到达时间优化','smooth_constant':'无学习',
    'smooth_fixed_pair':'固定提案','smooth_random_pair':'随机提案','smooth_binary':'移除保持指令',
    'smooth_time_only':'移除平顺目标','smooth_single_family':'只优化较优基策略族','smooth_learned':'V11完整方法'}


def failed(r):
    return r['aborted']=='True' or r['collision']=='True' or int(r['contract_violations'])>0


def aggregate(rows,timing):
    control=[v for t in timing for v in t['controller_ms']]
    cycles=[v for t in timing for v in t['cycle_ms']]
    return {'n':len(rows),
        'mean_cost_s':float(np.mean([float(r['mean_capped_finish_s']) for r in rows])),
        'mean_total_variation':float(np.mean([float(r['total_command_variation']) for r in rows])),
        'mean_command_jerk':float(np.mean([float(r['mean_jerk']) for r in rows])),
        'completed':sum(r['completed']=='True' for r in rows),'failed_episodes':sum(failed(r) for r in rows),
        'aborted':sum(r['aborted']=='True' for r in rows),'collision_episodes':sum(r['collision']=='True' for r in rows),
        'contract_violation_episodes':sum(int(r['contract_violations'])>0 for r in rows),
        'timeouts':sum(r['completed']=='False' and not failed(r) for r in rows),
        'mean_stopped_vehicle_seconds':float(np.mean([float(r['stopped_vehicle_seconds']) for r in rows])),
        'max_command_change_rate':max(float(r['peak_command_change_rate']) for r in rows),
        'mean_tie_improvement_steps':float(np.mean([int(r['variation_tie_improvement_steps']) for r in rows])),
        'mean_family_switches':float(np.mean([int(r['family_switches']) for r in rows])),
        'warm_first_response_p95_ms':float(np.percentile([float(r['first_response_ms']) for r in rows],95)),
        'warm_controller_cycle_p95_ms':float(np.percentile(control,95)),
        'warm_controller_cycle_max_ms':float(max(control)),
        'warm_end_to_end_cycle_p95_ms':float(np.percentile(cycles,95)),
        'cycles':len(control),'tail_feedback_steps':sum(int(r['tail_feedback_steps']) for r in rows),
        'trial_checks':sum(int(r['trial_checks']) for r in rows),'pruned_trials':sum(int(r['pruned_trials']) for r in rows)}


def compute(protocol,rows,timings):
    methods,modes=protocol['methods'],protocol['disturbances']
    lookup={(r['scenario'],int(r['seed']),r['disturbance'],r['method']):r for r in rows}
    expected={(s,seed,mode,m) for s,seeds in protocol['test_seeds'].items() for seed in seeds for mode in modes for m in methods}
    assert len(rows)==len(lookup)==protocol['executions'] and set(lookup)==expected
    times={(r['scenario'],int(r['seed']),r['disturbance'],r['method']):r for r in timings}
    assert len(times)==len(timings) and set(times)==expected
    def stats(m,regime=None,mode=None,perturbed=False):
        group=[r for r in rows if r['method']==m and (regime is None or r['scenario']==regime)
            and (mode is None or r['disturbance']==mode) and (not perturbed or r['disturbance']!='clean')]
        return aggregate(group,[times[r['scenario'],int(r['seed']),r['disturbance'],m] for r in group])
    overall={m:stats(m) for m in methods}
    perturbed={m:stats(m,perturbed=True) for m in methods}
    by_mode={d:{m:stats(m,mode=d) for m in methods} for d in modes}
    by_regime={s:{m:stats(m,regime=s,perturbed=True) for m in methods} for s in protocol['test_seeds']}
    rng=np.random.default_rng(protocol['bootstrap_seed']);contrasts={}
    for regime in ('all','nominal','dense'):
        keys=[(s,seed) for s,seeds in protocol['test_seeds'].items() if regime in ('all',s) for seed in seeds]
        # Reuse the same resampled initial-state indices for both endpoints.
        bootstrap=rng.integers(0,len(keys),(protocol['bootstrap_draws'],len(keys)))
        contrasts[regime]={}
        for method in methods:
            if method==TARGET:continue
            contrasts[regime][method]={}
            for name,field in (('time','mean_capped_finish_s'),('variation','total_command_variation'),('jerk','mean_jerk')):
                difference=np.array([np.mean([float(lookup[s,seed,d,method][field])-float(lookup[s,seed,d,TARGET][field]) for d in modes[1:]]) for s,seed in keys])
                baseline=float(np.mean([float(lookup[s,seed,d,method][field]) for s,seed in keys for d in modes[1:]]))
                means=difference[bootstrap].mean(1)
                contrasts[regime][method][name]={'clusters':len(keys),'mean_reduction':float(difference.mean()),
                    'reduction_percent':float(100*difference.mean()/baseline),
                    'paired_cluster_ci95':np.percentile(means,[2.5,97.5]).tolist(),
                    'paired_cluster_ci97_5':np.percentile(means,[1.25,98.75]).tolist(),
                    'better_tied_worse_clusters':[int(sum(difference>1e-6)),int(sum(abs(difference)<=1e-6)),int(sum(difference< -1e-6))]}
    clean_worse=sum(float(lookup[s,seed,'clean',TARGET]['mean_capped_finish_s'])>
        float(lookup[s,seed,'clean','v9_learned_pair']['mean_capped_finish_s'])+1e-8
        for s,seeds in protocol['test_seeds'].items() for seed in seeds)
    a=overall[TARGET];primary=contrasts['all']['v10_rollout'];c=protocol['criteria']
    criteria={
        'zero_collision_contract_violation_and_abort':a['failed_episodes']==0,
        'completion_fraction_min':a['completed']/a['n']>=c['completion_fraction_min'],
        'perturbed_time_gain_vs_v10_percent_min':primary['time']['reduction_percent']>=c['perturbed_time_gain_vs_v10_percent_min'],
        'perturbed_variation_reduction_vs_v10_percent_min':primary['variation']['reduction_percent']>=c['perturbed_variation_reduction_vs_v10_percent_min'],
        'time_paired_cluster_ci97_5_lower_gt_zero':primary['time']['paired_cluster_ci97_5'][0]>0,
        'variation_paired_cluster_ci97_5_lower_gt_zero':primary['variation']['paired_cluster_ci97_5'][0]>0,
        'perturbed_jerk_reduction_vs_v10_percent_min':primary['jerk']['reduction_percent']>=c['perturbed_jerk_reduction_vs_v10_percent_min'],
        'clean_worse_than_v9_count_max':clean_worse<=c['clean_worse_than_v9_count_max'],
        'warm_controller_cycle_p95_ms_max':a['warm_controller_cycle_p95_ms']<=c['warm_controller_cycle_p95_ms_max'],
        'warm_first_response_p95_ms_max':a['warm_first_response_p95_ms']<=c['warm_first_response_p95_ms_max']}
    assert set(criteria)==set(c)
    return {'executions':len(rows),'independent_clusters':sum(map(len,protocol['test_seeds'].values())),
        'overall':overall,'perturbed':perturbed,'by_error':by_mode,'perturbed_by_regime':by_regime,
        'contrasts_perturbed':contrasts,'clean_worse_than_v9':clean_worse,
        'criteria':criteria,'all_criteria_passed':all(criteria.values())}


def main():
    protocol=json.loads((D/'protocol.json').read_text())
    rows=list(csv.DictReader((D/'study/episodes.csv').open()))
    r=compute(protocol,rows,json.loads((D/'study/timing.json').read_text()))
    (D/'summary.json').write_text(json.dumps(r,indent=2)+'\n',newline='\n')
    p=r['contrasts_perturbed']['all']['v10_rollout'];a=r['overall'][TARGET]
    lines=['# V11 通行效率与指令平顺性联合改进','',
        '代码、200个全新初态、九种方法和十项验收标准在测试前冻结。四类执行误差，共7200次实际SUMO执行，每种方法800回合。三类非零误差的600回合对应200个独立初态。','',
        f"全部预设标准通过：{r['all_criteria_passed']}。",'',
        f"相对冻结V10，扰动下到达时间减少{p['time']['reduction_percent']:.2f}%，累计指令变化量减少{p['variation']['reduction_percent']:.2f}%，平均指令变化率减少{p['jerk']['reduction_percent']:.2f}%。",'',
        '## 两个共同主要终点','',
        '主要比较预先指定为V11对V10。先对每个初态的三种非零误差取配对差，再整簇重采样10000次。时间和总变化量分别报告97.5%百分位区间，对两个主要终点作Bonferroni式置信水平分配；bootstrap区间为近似推断，不是有限样本分布无关保证。其余对照及95%区间仅作描述性分析。','',
        '| 指标 | V10 | V11 | 平均减少 | 97.5%配对整簇区间 | 更好/持平/更差初态 |','|---|---:|---:|---:|---|---|']
    for endpoint,label,field in [('time','截断平均到达时间 s','mean_cost_s'),('variation','三车累计指令变化量 m/s²','mean_total_variation')]:
        e=p[endpoint];ci=e['paired_cluster_ci97_5']
        lines.append(f"| {label} | {r['perturbed']['v10_rollout'][field]:.3f} | {r['perturbed'][TARGET][field]:.3f} | {e['mean_reduction']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {'/'.join(map(str,e['better_tied_worse_clusters']))} |")
    lines+=['','总变化量为从初始零指令到三车全部到达（或任务截止）的Σ|uₖ−uₖ₋₁|，包括已到达车辆继续受控阶段。它不除以回合长度，避免用较慢行驶稀释平顺指标。平均指令变化率另报，不能解释为执行误差下的真实车辆加加速度或乘坐舒适度实测。','',
        '## 同一初态与误差带上的全部方法','',
        '| 方法 | 扰动到达时间 s | 扰动总变化量 m/s² | 扰动指令变化率 m/s³ | 完成/800 | 故障回合 |','|---|---:|---:|---:|---:|---:|']
    for m in protocol['methods']:
        b=r['perturbed'][m];o=r['overall'][m]
        lines.append(f"| {ZH[m]} | {b['mean_cost_s']:.3f} | {b['mean_total_variation']:.3f} | {b['mean_command_jerk']:.3f} | {o['completed']} | {o['failed_episodes']} |")
    lines+=['','故障回合是碰撞、路线合同违规或控制器中止的并集。所有失败保留，中止按25 s计代价。零误差下V11比V9学习双方案更差的初态数为'+str(r['clean_worse_than_v9'])+'。','',
        '## 消融与学习提案的独立贡献','',
        '| 对照方法 | 时间减少 % | 总变化量减少 % | 时间95%区间 s |','|---|---:|---:|---|']
    for m,e in r['contrasts_perturbed']['all'].items():
        ci=e['time']['paired_cluster_ci95']
        lines.append(f"| {ZH[m]} | {e['time']['reduction_percent']:.2f} | {e['variation']['reduction_percent']:.2f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines+=['','移除平顺目标保留相同指令格点、两类尾部和逐类扫描，只按到达时间更新；移除保持指令只保留原二值尾部；只优化较优基策略族先选最好的原始尾部，再进行一次坐标扫描。无学习、固定与随机提案控制学习选择贡献，不能将整个在线优化收益归为强化学习收益。','',
        '## 运行成本','',
        '| 方法 | 暖机首响应P95 ms | 暖机控制周期P95 ms | 最大控制周期 ms |','|---|---:|---:|---:|']
    for m in protocol['methods']:
        b=r['overall'][m];lines.append(f"| {ZH[m]} | {b['warm_first_response_p95_ms']:.2f} | {b['warm_controller_cycle_p95_ms']:.2f} | {b['warm_controller_cycle_max_ms']:.2f} |")
    lines+=['','首响应包括初始网络提案、V9参考方案评价和第一次TraCI周期。控制周期只统计算法；完整TraCI周期另保存在原始数据。单工作进程、预热、随机方法顺序。SUMO在计算时暂停，时间分位数不是最坏执行时间保证，也没有增加通信延迟统计。','',
        '## 证明与工程边界','',
        'V9联合鲁棒备用证书与误差界保持不变。分层目标给出名义字典序性能不退化：时间相同时才保证更低的总变化量；时间更短时，不独立保证变化量也下降。实际误差下的双指标改善由配对实验支持，不能据此声称逐场景支配。',
        '', '学习网络仍是冻结的集中式上下文bandit。本轮贡献是固定指令格点上的分层前瞻、保持指令尾部、逐类候选改进与精确剪枝的实现和验证；rollout、安全过滤和字典序优化本身不是首次提出。当前仅覆盖三车合作固定路线、精确测量和已知执行误差界；未验证非合作交通、感知误差或实车。','',
        '[理论](theory_v11.md) · [冻结协议](../artifacts/v11/protocol.json) · [全部回合](../artifacts/v11/study/episodes.csv)','']
    (ROOT/'docs/v11_results.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=['v9_learned_pair','v10_rollout','smooth_time_only','smooth_binary','smooth_single_family',TARGET]
    labels=['V9 learned','V10 rollout','No variation\nobjective','No hold\ntail','One family\nsweep','V11 complete']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,field,ylabel in zip(axes,['mean_cost_s','mean_total_variation','mean_command_jerk'],
        ['Capped mean arrival time (s)','Command total variation (m/s²)','Mean command change rate (m/s³)']):
        values=[r['perturbed'][m][field] for m in selected]
        ax.bar(labels,values,color=['#99a4ad','#708d9b','#a9bfaf','#90b4a3','#6c9f88','#196955'])
        ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=8)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
        ax.set_ylim(0,max(values)*1.16)
    fig.savefig(D/'results.png',dpi=180);plt.close(fig)
    hashes={str(p.relative_to(D)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    (D/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n',newline='\n')
    print(json.dumps({'criteria':r['criteria'],'primary':r['contrasts_perturbed']['all']['v10_rollout'],'target':a},indent=2))


if __name__=='__main__':main()
