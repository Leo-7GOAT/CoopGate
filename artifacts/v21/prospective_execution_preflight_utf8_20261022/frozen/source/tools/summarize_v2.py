"""Generate tables and figures directly from the frozen-study measurements."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v2_final'
SEEDS=[31,47,59]
METHODS=['traditional','mappo','independent','joint_robust','paired_no_gain','paired_no_reserve','paired_no_memory','paired']


def read_rows(path):
    with path.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            if v in ('True','False'):row[k]=v=='True'
            elif k not in ('method','split','scenario'):
                row[k]=float(v)
    return rows


def main():
    rows=[];cert=[]
    for seed in SEEDS:
        folder=OUT/f'seed{seed}'
        manifest=json.loads((folder/'manifest.json').read_text())
        data=read_rows(folder/'episodes.csv')
        assert len(data)==manifest['episodes']==(410 if seed==31 else 310)
        rows+=data
        cert+=json.loads((folder/'certification.json').read_text())
        rows+=read_rows(OUT/f'memory_seed{seed}.csv')
    summaries={}
    for split in ('test','shift'):
        summaries[split]={}
        for method in METHODS:
            by_seed=[]
            for seed in SEEDS:
                group=[r for r in rows if r['split']==split and r['method']==method and r['training_seed']==seed]
                assert len(group)==15
                by_seed.append({'seed':seed,'n':len(group),'collisions':sum(r['collision'] for r in group),
                                'successes':sum(r['success'] for r in group),
                                'penalized_time_s':float(np.mean([r['elapsed_s'] if r['success'] else 30 for r in group])),
                                'rl_fraction':float(np.mean([r['rl_fraction'] for r in group])),
                                'backup_unverified_steps':float(np.mean([r['backup_unverified_steps'] for r in group])),
                                'gate_mean_ms':float(np.mean([r['gate_mean_ms'] for r in group]))})
            summaries[split][method]=by_seed
    result={'primary_episodes':1030,'supplementary_memory_episodes':90,'summaries':summaries,'certification':cert}
    (OUT/'study_summary.json').write_text(json.dumps(result,indent=2),newline='\n')
    lines=['# V2 冻结实验结果','',
           '主实验1030次执行，执行器记忆补充消融90次。主要对照每策略每方法每分布15回合；'
           '不同方法和训练种子共享场景，不能将重复场景视为额外独立样本。','']
    for split in ('test','shift'):
        lines+=['## '+('同分布新场景' if split=='test' else '更强前车制动分布变化'),'',
                '|方法|种子31 碰撞/成功|种子47 碰撞/成功|种子59 碰撞/成功|平均失败罚时 s|平均RL比例|',
                '|---|---:|---:|---:|---:|---:|']
        for m in METHODS:
            data=summaries[split][m]
            lines+=['|'+m+'|'+'|'.join(f'{r["collisions"]}/{r["successes"]}' for r in data)
                    +f'|{np.mean([r["penalized_time_s"] for r in data]):.3f}|{np.mean([r["rl_fraction"] for r in data]):.1%}|']
        lines+=['','每个“碰撞/成功”单元格给出两个计数，分别以15回合为分母。失败罚时将所有未完成回合记为30秒，避免给早期碰撞奖励。','']
    lines+=['## 独立认证集','',
            '四个冻结控制器、两个失败事件，Bonferroni校正后共8个检验，整体置信水平95%。'
            '每项100个IID混合模板场景。通过仅指该SUMO分布上的5%目标，不是道路安全标准或分布外保证。','',
            '|控制器|事件|失败/100|风险上界|通过5%目标|','|---|---|---:|---:|---|']
    for r in cert:
        name=r['method']+(str(r['training_seed']) if r['method']=='paired' else '')
        lines += [f'|{name}|{r["endpoint"]}|{r["failures"]}|{r["risk_upper"]:.2%}|{r["certified"]}|']
    lines+=['','## 理论与工程边界','',
            '运行时分数没有被解释为安全概率；制动余量命题只在指定纵向模型中成立。'
            '使用同一纵向QP降低了模型差异，但背景与横向仍近似。备用检查失败不会被记作获得了安全证书。',
            '', '公式、证明和适用条件见 [理论说明](theory_v2.md)，创新性边界见 [查新](novelty_v2.md)。']
    lines+=['','## 结果解释','',
            '完整门控在ID场景均成功15/15，在分布变化中均成功13/15，完成率与传统规控相同。ID平均用时21.94秒，传统22.43秒，独立门控20.69秒；完整门控不是ID效率最优方法。',
            '', '更强制动下，完整门控碰撞为0、0、1次，去制动余量后为8、7、5次。去收益筛选使种子31与59成功数从13降到5，种子47仍为13；这是两个模块在本测试集上的初步依据。',
            '', '执行器记忆消融在分布变化下的成功、碰撞及平均失败罚时与完整门控相同，ID时间差也只有约0.04秒。没有证据支持把记忆模块单独当作主要性能贡献。',
            '', '完整门控ID和变化场景的RL使用比例分别仅约4.7%和2.1%，说明它主要靠传统备用保持完成率。现阶段证据支持退化防护，尚不支持超过传统规控或充分发挥协同学习优势。',
            '', '八项风险检验全部通过，但传统控制器也通过；该结果提供指定分布上的有限样本上界，并不证明算法优越。更强制动中仍有碰撞，也说明不能把此上界推广到分布外。',
            '', '完整门控每回合平均约40个ID决策周期、50个变化场景决策周期处于未验证备用状态。这是当前方法的重要局限：经验完成率不是逐步安全覆盖，备用规则尚未获得可恢复性证书。原始逐回合日志保留该计数。']
    (ROOT/'docs/v2_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    labels=[m.replace('paired_no_','no_').replace('_','\n') for m in METHODS]
    for i,split in enumerate(('test','shift')):
        for j,key in enumerate(('successes','penalized_time_s')):
            ax=axes[i,j]
            values=np.array([[r[key]/15 if key=='successes' else r[key] for r in summaries[split][m]] for m in METHODS])
            ax.bar(range(len(METHODS)),values.mean(axis=1),color=['#64748b']*7+['#0891b2'])
            for k in range(3):ax.scatter(np.arange(len(METHODS))+(k-1)*.09,values[:,k],s=13,color='black',zorder=3)
            ax.set_xticks(range(len(METHODS)),labels,fontsize=8)
            ax.set_title(('ID' if split=='test' else 'Shift')+' / '+('Task success rate' if key=='successes' else 'Failure-penalized time (s)'))
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('CoopGate V2 / dots are three training seeds; not confidence intervals')
    fig.savefig(OUT/'results.png',dpi=190);fig.savefig(OUT/'results.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
    for seed in SEEDS:
        data=read_rows(OUT/f'seed{seed}'/'training.csv')
        ax.plot([r['steps'] for r in data],[r['mean_last20_return'] for r in data],label=str(seed),alpha=.8)
    ax.set(xlabel='MAPPO steps after BC initialization',ylabel='Mean last 20 episode return')
    ax.legend(title='Seed');ax.grid(alpha=.2)
    fig.savefig(OUT/'training.png',dpi=180);plt.close(fig)
    # Portable release bytes and hashes for the v2 subtree.
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in ('.csv','.json') and p.name!='SHA256SUMS.json':
            p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    hashes={str(p.relative_to(OUT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    (OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2),newline='\n')
    print(json.dumps({'rows':len(rows),'certification':cert},indent=2))


if __name__=='__main__':main()
