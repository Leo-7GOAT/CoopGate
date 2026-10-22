"""Read-only analysis of frozen V3 outcomes plus reproducible release hashes."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/v3/final'
SEEDS=[83,97,109]
METHODS=['raw_mpc','raw_rl','joint_mpc','individual','independent_front','no_memory','joint_bc','joint_constant','joint_rl']


def read(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            if v in ('True','False'):r[k]=v=='True'
            elif k not in ('method','scenario','split'):r[k]=float(v)
    return rows


def main():
    rows=[];cert=[]
    for seed in SEEDS:
        folder=OUT/f'seed{seed}';group=read(folder/'episodes.csv')
        assert len(group)==(740 if seed==83 else 640)
        assert json.loads((folder/'manifest.json').read_text())['episodes']==len(group)
        rows+=group;cert+=json.loads((folder/'certification.json').read_text())
    summaries={};comparisons={};rng=np.random.default_rng(2026101505)
    for split in ('test','shift'):
        summaries[split]={};comparisons[split]={}
        for method in METHODS:
            output=[]
            for seed in SEEDS:
                group=[r for r in rows if r['split']==split and r['method']==method and r['training_seed']==seed]
                assert len(group)==30
                output.append({'seed':seed,'n':30,'collisions':sum(r['collision'] for r in group),
                    'gap_violation_episodes':sum(r['gap_violation_substeps']>0 for r in group),
                    'progress_m':float(np.mean([r['collision_penalized_progress_m'] for r in group])),
                    'retained_fraction':float(np.mean([r['rl_retained_fraction'] for r in group])),
                    'intervention_fraction':float(np.mean([r['intervention_fraction'] for r in group])),
                    'jerk':float(np.mean([r['mean_abs_jerk'] for r in group])),
                    'outside_set_steps':int(sum(r['outside_set_steps'] for r in group)),
                    'uncertified_action_steps':int(sum(r['uncertified_action_steps'] for r in group)),
                    'max_prediction_error':max(r['max_prediction_error'] for r in group)})
            summaries[split][method]=output
        for baseline in ('joint_mpc','joint_bc','joint_constant','independent_front'):
            differences=[]
            for seed in SEEDS:
                group=[r for r in rows if r['split']==split and r['training_seed']==seed]
                own={r['seed']:r for r in group if r['method']=='joint_rl'}
                base={r['seed']:r for r in group if r['method']==baseline}
                d=np.array([own[s]['collision_penalized_progress_m']-base[s]['collision_penalized_progress_m'] for s in own])
                boot=d[rng.integers(0,len(d),(5000,len(d)))].mean(axis=1)
                differences.append({'seed':seed,'mean_paired_gain_m':float(d.mean()),
                    'bootstrap95_descriptive':np.quantile(boot,[.025,.975]).tolist(),'wins':int(sum(d>1e-5)),'losses':int(sum(d< -1e-5))})
            comparisons[split][baseline]=differences
    result={'primary_episodes':len(rows),'summaries':summaries,'paired_comparisons':comparisons,'certification':cert,
            'comparison_scope':'Per-seed paired scenes; percentile bootstrap intervals are descriptive, not multiplicity-adjusted significance claims.'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),newline='\n')
    lines=['# V3 固定协议实验结果','',
      '三个独立策略，各12000步门控内MAPPO及30回合教师初始化。2020次主实验执行；每方法、每策略、每分布30回合，认证集另计100回合。'
      '不同方法和种子共享场景，不把重复场景当成独立额外样本。任务为固定25秒的单车道纵向行驶，不能与V2合流成功率直接比较。','']
    for split in ('test','shift'):
        lines+=['## '+('同分布' if split=='test' else '双次急刹扰动'),'',
                '|方法|种子83 碰撞/间距违规|种子97|种子109|平均进展 m|平均指令jerk m/s³|提案保留率|',
                '|---|---:|---:|---:|---:|---:|---:|']
        for m in METHODS:
            g=summaries[split][m]
            lines+=['|'+m+'|'+'|'.join(f'{r["collisions"]}/{r["gap_violation_episodes"]}' for r in g)
                    +f'|{np.mean([r["progress_m"] for r in g]):.2f}|{np.mean([r["jerk"] for r in g]):.3f}|{np.mean([r["retained_fraction"] for r in g]):.1%}|']
        lines+=['','碰撞与间距违规均为回合计数，每项分母30；净间距低于2米即为违规，即使尚未碰撞。发生碰撞的回合进展记0。MPC无RL提案保留率，表中记0。','',
                '|对照|种子83 平均配对进展差 m|种子97|种子109|','|---|---:|---:|---:|']
        for m,g in comparisons[split].items():lines+=['|'+m+'|'+'|'.join(f'{r["mean_paired_gain_m"]:.2f}' for r in g)+'|']
        lines+=['','正数表示联合门控RL优于对应对照。完整的逐种子描述性bootstrap区间在summary.json，未作多重比较显著性结论。','']
    lines+=['## 独立回合检验','',
        '|控制器|事件|失败/100|校正上界|通过5%目标|','|---|---|---:|---:|---|']
    for r in cert:lines+=[f'|{r["method"]}{r["seed"] if r["method"]=="joint_rl" else ""}|{r["endpoint"]}|{r["failures"]}|{r["risk_upper"]:.2%}|{r["certified"]}|']
    lines+=['','8项同时检验，整体置信水平95%。目标5%仅演示统计检验，不是道路安全标准；该统计结论不覆盖未测试动力学。理论条件及反例见 [V3理论](theory_v3.md)。']
    lines+=['','## 如何解读结果','',
        '联合门控RL的480次测试及认证执行均没有碰撞、间距违规、离开备用可恢复集或执行未认证指令。'
        '不同策略共享场景，因此480次执行不是480个独立安全样本。忽略指令记忆和直接拼接逐车结果都会破坏安全条件。','',
        '学习策略尚未超过强基线：同分布的三种子平均进展为416.41米，联合门控MPC为427.54米，'
        '门控恒定加速为433.36米；双次急刹分别为236.63、255.37、263.34米。'
        'RL比恒定加速的指令变化更小，但尚未证明相对于调优传统规控的Pareto优势。','',
        '初态由联合制动轨迹构造并显式检查属于可恢复集合。SUMO按声明的离散模型执行，'
        '测试了仿真闭环而非真实车辆动力学不确定性；不能把本阶段当作V2合流性能提升。']
    supplement=ROOT/'artifacts/v3/supplement/results.json'
    if supplement.exists():
        extra=json.loads(supplement.read_text())
        lines+=['','## 链式求解与工程补充','',
            '在固定车序、相邻间距约束、可独立生成的备用轨迹及可加代价下，模式选择可精确分解为链式动态规划。'
            '停止时域H固定时总复杂度O(nH)，模式求解O(n)。动态规划和组合式安全本身不是首次提出。','',
            '三个策略各重放一个预先确定的测试场景，共150个SUMO决策周期；指令和状态与完整穷举一致。','',
            '|实现|完整循环均值 ms|P95 ms|最大 ms|超过500 ms|','|---|---:|---:|---:|---:|']
        for m,r in extra['timing'].items():
            lines+=[f'|{m}|{r["mean_ms"]:.2f}|{r["p95_ms"]:.2f}|{r["max_ms"]:.2f}|{r["over_500ms"]}|']
        lines+=['','每方法5场景、250周期，本机单仿真进程，包含策略、门控与SUMO步进。样本耗时不是最坏执行时间保证。','',
                '|受控车辆数|兼容表项|完整模式数|链式总均值 ms|','|---:|---:|---:|---:|']
        for r in extra['scaling']:
            lines+=[f'|{r["vehicles"]}|{r["pair_entries"]}|{r["joint_modes"]}|{r["mean_total_ms"]:.3f}|']
        lines+=['','扩展性来自合成状态，包含轨迹预计算和模式求解，不是64车SUMO实验。'
                '完整数据在`artifacts/v3/supplement`；补充实验没有用于重选主实验策略。']
    (ROOT/'docs/v3_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for i,split in enumerate(('test','shift')):
        for j,key in enumerate(('progress_m','gap_violation_episodes')):
            ax=axes[i,j];v=np.array([[r[key] for r in summaries[split][m]] for m in METHODS])
            ax.bar(range(len(METHODS)),v.mean(axis=1),color=['#64748b']*8+['#0891b2'])
            for s in range(3):ax.scatter(np.arange(len(METHODS))+.08*(s-1),v[:,s],s=12,color='black')
            ax.set_xticks(range(len(METHODS)),[m.replace('_','\n') for m in METHODS],fontsize=8)
            ax.set_title(('ID' if split=='test' else 'Double braking')+' / '+('Penalized progress (m)' if key=='progress_m' else 'Gap violation episodes / 30'))
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('CoopGate V3 / fixed-lane subproblem / dots: three training seeds')
    fig.savefig(OUT/'results.png',dpi=190);fig.savefig(OUT/'results.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    for path in (ROOT/'artifacts/v3').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.csv') and path.name!='SHA256SUMS.json':
            path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    hashes={str(p.relative_to(ROOT/'artifacts/v3')).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'artifacts/v3').rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    (ROOT/'artifacts/v3/SHA256SUMS.json').write_text(json.dumps(hashes,indent=2),newline='\n')
    print(json.dumps({'episodes':len(rows),'comparisons':comparisons,'certification':cert},indent=2))


if __name__=='__main__':main()
