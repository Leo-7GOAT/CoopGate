"""Render the complete frozen verdict; never select models or change criteria."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v14/single';OUT=ROOT/'artifacts/v14/single_analysis'


def analyze():
    p=json.loads((D/'protocol.json').read_text());s=json.loads((D/'summary.json').read_text())
    rows=json.loads((D/'study/episodes.json').read_text());audit=json.loads((ROOT/'artifacts/v14/single_audit.json').read_text())
    assert len(rows)==p['expected_runs']==audit['validated_sumo_runs']
    assert audit['acceptance']==s['acceptance']
    for name,digest in json.loads((D/'sha256.json').read_text()).items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
    OUT.mkdir(parents=True,exist_ok=False)
    lookup={(r['seed'],r['disturbance'],r['controller']):r for r in rows}
    seeds=p['training_seeds'];base=s['methods']['rollout'];metrics=('ego_cost','ego_time','ego_variation')
    rng=np.random.default_rng(p['bootstrap']['seed']+50);n=len(p['initials']);ids=rng.integers(0,n,(10000,n))
    plot_data={}
    for seed in seeds:
        result={}
        for key in metrics:
            diffs=np.array([np.mean([lookup[item['seed'],m,'rollout'][key]-lookup[item['seed'],m,f'learned_{seed}'][key]
                                    for m in p['disturbances']]) for item in p['initials']])
            result[key]=dict(gain_percent=100*float(diffs.mean())/base[key],
                ci95_percent=(100*np.quantile(diffs[ids].mean(-1),[.025,.975])/base[key]).tolist())
        plot_data[str(seed)]=result
    vehicles={}
    for category in ('rollout','learned','last'):
        r=[row for row in rows if row['controller']==category or row['controller'].startswith(category+'_')]
        finish=np.array([row['per_vehicle_finish'] for row in r]);tv=np.array([row['per_vehicle_variation'] for row in r])
        vehicles[category]=dict(time=finish.mean(0).tolist(),variation=tv.mean(0).tolist(),cost=(finish+.05*tv).mean(0).tolist())
    evidence=dict(status='Complete frozen test; per-seed chart intervals are descriptive',plot_data=plot_data,
        per_vehicle=vehicles,source_protocol_sha256=audit['protocol_sha256'],source_episodes_sha256=audit['episodes_sha256'])
    (OUT/'analysis.json').write_text(json.dumps(evidence,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.8),layout='constrained')
    for ax,key,title in zip(axes,metrics,('Ego cost','Ego arrival time','Ego command variation')):
        for i,seed in enumerate(seeds):
            r=plot_data[str(seed)][key];v=r['gain_percent'];lo,hi=r['ci95_percent']
            ax.errorbar(i,v,yerr=[[max(0.,v-lo)],[max(0.,hi-v)]],fmt='o',color='#44667c',capsize=4)
        v=s['gain_percent']['learned'][key]
        lo,hi=np.array(s['paired_initial_cluster_intervals']['learned'][key]['ci95'])*100/base[key]
        ax.errorbar(3,v,yerr=[[max(0.,v-lo)],[max(0.,hi-v)]],fmt='D',color='#007f86',capsize=5)
        ax.axhline(0,color='#888888',linestyle='--',linewidth=.8)
        ax.set_xticks(range(4),[str(x) for x in seeds]+['Mean']);ax.set_ylabel('Gain over strong baseline (%)')
        ax.set_title(title);ax.grid(axis='y',alpha=.18)
    fig.suptitle(f"Frozen single-ego test: {n} initials, four disturbance modes\nPaired initial-cluster 95% intervals; positive values favor learning",fontsize=11)
    fig.savefig(OUT/'single_ego.png',dpi=180);fig.savefig(OUT/'single_ego.pdf');plt.close(fig)
    passed=s['individual_ego_verified'];checks=s['acceptance']
    lines=['# V14 冻结单车测试结果','',
        f"正式测试完成 {len(rows):,} 次实际 SUMO 执行，覆盖 {n} 个独立初态、四类扰动、三个训练种子的平均参数网络与最终未平均网络，以及同一强规划基线。",
        '',f"单车前置验收：**{'通过' if passed else '未通过'}**（{checks['passed']}/{checks['total']} 项）。这不是 MARL 成功结论。",'',
        '训练协议冻结于 `0d9e7b4`；最终模型、评测源码和全部初态冻结于 `e36a6fb`，之后才启动测试。未按测试分数选择中间检查点，未改变验收标准。','',
        '|方法|自车时间 s|自车指令变化 m/s²|自车综合代价 s|','|---|---:|---:|---:|']
    entries=[('强规划基线',base)]+[(f'平均参数网络 {seed}',s['methods'][f'learned_{seed}']) for seed in seeds]
    entries += [('三种子平均',s['seed_average']['learned']),('未平均最终网络（三种子）',s['seed_average']['last'])]
    for label,r in entries:lines.append(f"|{label}|{r['ego_time']:.6f}|{r['ego_variation']:.6f}|{r['ego_cost']:.6f}|")
    lines += ['','|主指标|平均改善|基线减学习的配对 95% 区间|','|---|---:|---:|']
    for key in metrics:
        lo,hi=s['paired_initial_cluster_intervals']['learned'][key]['ci95']
        lines.append(f"|{key}|{s['gain_percent']['learned'][key]:.3f}%|[{lo:.6f}, {hi:.6f}]|")
    lines += ['','上述区间以初态为簇，对四类扰动和三个固定训练模型求平均。它不等于对所有未来训练种子的精确推断。','',
        '|车辆|基线时间 s|学习时间 s|基线成本 s|学习成本 s|','|---|---:|---:|---:|---:|']
    for i in range(3):lines.append(f"|{i}|{vehicles['rollout']['time'][i]:.6f}|{vehicles['learned']['time'][i]:.6f}|{vehicles['rollout']['cost'][i]:.6f}|{vehicles['learned']['cost'][i]:.6f}|")
    lines += ['','逐车均值用于显示外部性；不能据此宣称每辆车均获益或非退化。','',
        '|预定冲突组|独立初态|自车成本改善|自车时间改善|','|---|---:|---:|---:|']
    for name,g in s['conflict_groups'].items():
        if not g['initials']:continue
        cost=100*(g['baseline']['ego_cost']-g['learned']['ego_cost'])/g['baseline']['ego_cost']
        arrival=100*(g['baseline']['ego_time']-g['learned']['ego_time'])/g['baseline']['ego_time']
        lines.append(f"|{name}|{g['initials']}|{cost:.3f}%|{arrival:.3f}%|")
    lines += ['','分组使用固定的初始冲突入口 ETA 分离阈值 1.25 秒；它不是普适道路复杂度定义，分组结果是描述性比较。','',
        '## 验收及复核','']
    lines += [f"- {name}: {'PASS' if value else 'FAIL'}" for name,value in checks['checks'].items()]
    lines += ['',f"复核覆盖 {audit['audited_traces']} 条完整轨迹、{audit['compiled_actor_comparisons']} 次 Torch/编译执行核对，以及 {audit['sumo_replays']} 次额外实际 SUMO 重放。物理结论限于所述三车纵向、精确状态与有界扰动模型。",'',
        '网络执行没有未来性能搜索；全部方法仍共享物理安全过滤器，安全不能归功于未经保护的网络本身。当前没有为 V14 网络验证新的性能门控，不能套用 V13 的门控收益数字。','',
        '下一步是执行已准备的真实三车联合训练和强协同规划对照。' if passed else '保持 MARL 训练锁定，保留本次负结果，在新的开发假设及预先冻结的独立验证下继续改进单车策略。','',
        '[完整统计](../artifacts/v14/single/summary.json) · [审计记录](../artifacts/v14/single_audit.json) · [测试协议](../artifacts/v14/single/protocol.json)','',
        '![三个训练种子的单车结果](../artifacts/v14/single_analysis/single_ego.png)','']
    (ROOT/'docs/v14_single_results.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(acceptance=checks,analysis=str(OUT),report='docs/v14_single_results.md')),flush=True)


if __name__=='__main__':analyze()
