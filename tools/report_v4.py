"""Summarize all frozen V4 outcomes, including negative learning results."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v4'


def read(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            if v in ('True','False'):r[k]=v=='True'
            elif k not in ('method','scenario','split'):r[k]=float(v)
    return rows


def main():
    protocol=json.loads((D/'protocol.json').read_text());rows=[];summary={};paired={};rng=np.random.default_rng(2026101305)
    from coopgate.learning_v4 import load
    from coopgate.merge_v4 import BRANCH
    policy_rows=[]
    for seed in protocol['training_seeds']:
        model,_=load(D/f'seed{seed}/policy.pt')
        for split in protocol['splits']:
            trace=json.loads((D/f'seed{seed}/trace_{split}_joint_rl.json').read_text());actions=[]
            for r in trace[:-1]:
                x=np.array(r['x']);prev=np.array(r['command'])
                features=np.column_stack((x[:,0]/150,x[:,1]/20,prev/5,BRANCH))
                obs=np.array([features[[i]+[j for j in range(3) if j!=i]].flatten() for i in range(3)],dtype=np.float32)
                actions.append(model.act(obs,True)[0])
            policy_rows.append({'seed':seed,'split':split,'decisions':len(actions),'max_acceleration_proposal_fraction':float(np.mean(np.array(actions)==4))})
    (D/'policy_audit.json').write_text(json.dumps({'scope':'Post-hoc audit of first saved trace per split, excluding its initial state; not all test decisions','rows':policy_rows},indent=2),newline='\n')
    for seed in protocol['training_seeds']:
        group=read(D/f'seed{seed}/episodes.csv');assert len(group)==320;rows+=group
    for split in protocol['splits']:
        summary[split]={};paired[split]={}
        for method in protocol['methods']:
            entries=[]
            for seed in protocol['training_seeds']:
                g=[r for r in rows if r['split']==split and r['method']==method and r['training_seed']==seed];assert len(g)==20
                entries.append({'seed':seed,'n':20,'completed':sum(r['completed'] for r in g),'collisions':sum(r['collision'] for r in g),
                    'violating_episodes':sum(r['contract_violations']>0 for r in g),'mean_time':float(np.mean([r['mean_capped_finish_s'] for r in g])),
                    'mean_jerk':float(np.mean([r['mean_jerk'] for r in g])),'intervention':float(np.mean([r['intervention'] for r in g])),
                    'outside_steps':sum(r['outside_steps'] for r in g),'uncertified_steps':sum(r['uncertified_steps'] for r in g),
                    'max_prediction_error':max(r['max_prediction_error'] for r in g)})
            summary[split][method]=entries
        for base in ('constant','eta','search','bc'):
            entries=[]
            for seed in protocol['training_seeds']:
                g=[r for r in rows if r['split']==split and r['training_seed']==seed]
                own={r['seed']:r for r in g if r['method']=='joint_rl'};ref={r['seed']:r for r in g if r['method']==base}
                delta=np.array([ref[s]['mean_capped_finish_s']-own[s]['mean_capped_finish_s'] for s in own])
                samples=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(axis=1)
                entries.append({'seed':seed,'seconds_saved':float(delta.mean()),'descriptive95':np.quantile(samples,[.025,.975]).tolist(),
                                'wins':int(sum(delta>1e-6)),'losses':int(sum(delta< -1e-6))})
            paired[split][base]=entries
    result={'episodes':len(rows),'summary':summary,'paired':paired,'scope':'Paired pilot, no population safety certificate; repeated fixed baselines are not independent replications.'}
    (D/'summary.json').write_text(json.dumps(result,indent=2),newline='\n')
    lines=['# V4 双路合流：固定协议探索性实验','',
           '三个种子各6000步训练，八种方法、两类分布、每项20场景，共960次SUMO执行。'
           '原始数据、检查点和源代码哈希全部发布；结果不用于挑选检查点。理论与范围见[说明](theory_v4.md)。','']
    for split,methods in summary.items():
        lines+=['## '+split,'','|方法|种子127 完成/碰撞/违规|种子139|种子151|平均到达时间 s ↓|平均指令jerk|','|---|---:|---:|---:|---:|---:|']
        for m,g in methods.items():
            lines+=['|'+m+'|'+'|'.join(f'{r["completed"]}/{r["collisions"]}/{r["violating_episodes"]}' for r in g)+f'|{np.mean([r["mean_time"] for r in g]):.3f}|{np.mean([r["mean_jerk"] for r in g]):.3f}|']
        lines+=['','每个计数分母为20。未完成车辆到达时间记25秒，碰撞回合全部记25秒。'
               '低jerk不代表任务表现好，停车不动和提前碰撞均可能有低jerk。','',
               '|对照|种子127 RL节省秒数|种子139|种子151|','|---|---:|---:|---:|']
        for base,g in paired[split].items():lines+=['|'+base+'|'+'|'.join(f'{r["seconds_saved"]:.3f}' for r in g)+'|']
        lines+=['','正数为RL更快。逐种子配对bootstrap仅为描述性区间，见summary.json；未作多重比较显著性结论。','']
    lines+=['## 解释边界','',
      'V4检验路线冲突区互斥及路线间距，同时记录SUMO碰撞；固定路线、全受控、精确速度接口和无内部路口连接是实质限制。'
      '本阶段不是自由换道、高保真转向模型或混合人驾交通的安全证明。','',
      '恒定加速、ETA和搜索使用同一联合制动证书。125动作搜索是两秒局部进展启发式，不是全局最优控制器，'
      '也不是MAMPS/DMPS或双层优化MARL的完整复现。创新性仍须与这些方法作对等比较。','',
      '安全和完成必须同时判断：可恢复集合包含安全停车但永远不通过的状态。对训练后性能下降或与恒定策略持平的种子，'
      '同样保留数据，不能只展示最好的一组。']
    lines+=['','## 主要发现','',
      '联合门控RL的三个种子均在两类测试中20/20完成，且无合同违规；但种子127和139的到达时间与恒定加速门控逐场景相同，'
      '种子151反而略慢。三种子常规平均11.408秒，对照11.308秒；高密度10.489秒，对照10.467秒。'
      '不能据此声称MARL提高了合流效率。','',
      '局部搜索只有10/20常规、5/20高密度场景全部完成，尽管未违反合同。'
      '这是该2秒启发式存在任务推进不足的证据，不是证明RL胜过成熟的预测控制。']
    extra=D/'priority_audit/summary.json'
    if extra.exists():
        audit=json.loads(extra.read_text())
        lines+=['','## 事后开发诊断','',
          f'另用20个开发场景、4种固定优先处理方式完成80次执行。逐场景事后选择最好结果，在{audit["improved_cases"]}/20个场景有收益，'
          f'平均仅节省{audit["mean_seconds_saved"]:.3f}秒。该后验选择不是可部署策略，不能计入RL主结果；它只提示当前固定优先级变体的可利用空间有限。']
    timing=D/'timing/results.json'
    if timing.exists():
        lines+=['','## 单进程工程耗时','', '|方法|完整循环均值 ms|P95 ms|最大 ms|','|---|---:|---:|---:|']
        for m,r in json.loads(timing.read_text()).items():lines+=[f'|{m}|{r["mean_ms"]:.2f}|{r["p95_ms"]:.2f}|{r["max_ms"]:.2f}|']
        lines+=['','每方法重放五个预定场景，主实验结束后测量，包含策略前向、门控与SUMO步进。均未超过500毫秒；不是最坏执行时间保证。']
    (ROOT/'docs/v4_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for i,split in enumerate(summary):
        for j,key in enumerate(('mean_time','completed')):
            ax=axes[i,j];v=np.array([[r[key] for r in summary[split][m]] for m in protocol['methods']])
            ax.bar(range(len(v)),v.mean(axis=1),color=['#64748b']*7+['#0891b2'])
            for k in range(3):ax.scatter(np.arange(len(v))+.08*(k-1),v[:,k],s=13,color='black')
            ax.set_xticks(range(len(v)),[m.replace('_','\n') for m in protocol['methods']],fontsize=8)
            ax.set_title(split+' / '+('Capped arrival time (s), lower is better' if key=='mean_time' else 'Completed episodes / 20'))
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('CoopGate V4 / fixed-route merge / all three training seeds')
    fig.savefig(D/'results.png',dpi=180);fig.savefig(D/'results.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    for p in D.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.csv') and p.name!='SHA256SUMS.json':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    hashes={str(p.relative_to(D)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    (D/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2),newline='\n')
    print(json.dumps({'episodes':len(rows),'summary':summary,'paired':paired},indent=2))


if __name__=='__main__':main()
