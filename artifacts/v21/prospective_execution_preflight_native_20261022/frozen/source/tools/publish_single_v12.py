"""Publish complete single-agent results, plots and a plain-language report."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_single_v12 import summarize
from gate_sweep_single_v12 import run as sweep

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v12/single'


def publish(study):
    study=Path(study)
    if not (D/'study').exists():shutil.copytree(study,D/'study')
    protocol=json.loads((D/'protocol.json').read_text());rows=json.loads((D/'study/episodes.json').read_text())
    summary=summarize(protocol,rows);(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',newline='\n')
    sweep(D/'study',D/'gate_sweep.json')
    methods=summary['methods'];means=summary['seed_average'];labels=['Rule','Rollout','BC','PPO','Gate']
    entries=[methods['rule'],methods['rollout'],means['bc'],means['rl'],means['gate']]
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,key,label in zip(axes,['mean_capped_finish_s','variation','objective'],['Mean arrival time (s)','Command total variation (m/s²)','Cost T + 0.05 D']):
        values=[r[key] for r in entries];bars=ax.bar(labels,values,color=['#9ca3af','#64748b','#d8a24b','#0f766e','#2563eb'])
        ax.set_ylabel(label);ax.set_ylim(0,max(values)*1.18);ax.spines[['top','right']].set_visible(False)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+max(values)*.025,f'{v:.3f}',ha='center',fontsize=9)
    fig.suptitle('Single learned vehicle: independent paired SUMO evaluation',fontsize=13)
    fig.savefig(D/'comparison.png',dpi=180);plt.close(fig)
    sweeps=json.loads((D/'gate_sweep.json').read_text())['rows']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for group,color,title in [('all','#2563eb','All'),('low_conflict','#0f766e','Low conflict'),('high_conflict','#c2410c','High conflict')]:
        rr=[r for r in sweeps if r['group']==group]
        for ax,key in zip(axes,['harmful_rate','mean_cost_gain']):
            ax.scatter([r['acceptance_rate']*100 for r in rr],[r[key]*(100 if key=='harmful_rate' else 1) for r in rr],color=color,label=title)
            for r in rr:
                ax.annotate('q=0' if r['threshold']=='uncalibrated' else r['threshold'],
                    (r['acceptance_rate']*100,r[key]*(100 if key=='harmful_rate' else 1)),xytext=(3,4),textcoords='offset points',fontsize=7,color=color)
            ax.set_xlabel('Episodes admitted to learned policy (%)');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Harmful admission / all episodes (%)');axes[1].set_ylabel('Mean retained cost improvement')
    axes[1].legend(fontsize=8);fig.suptitle('Calibrated gate trade-off; labels are alpha values',fontsize=12)
    fig.savefig(D/'gate_risk.png',dpi=180);plt.close(fig)
    p=summary['performance'];gate=summary['gate_diagnostics'];acceptance=summary['acceptance']
    text=['# 单车学习收益与性能置信门控的独立测试','',
        '**研究前置条件尚未满足。** 本轮冻结的十项团队指标全部通过，但学习自车到达时间为11.9095秒，滚动规划基线为11.0010秒，慢约8.26%；门控自车为11.3239秒，仍慢约2.94%。团队目标改善不能证明自车优于基线，不能据此进入MARL扩展。保留原始协议和结果，下一阶段以自车时间和自车指令变化量重新开展独立实验。','',
        '本阶段只有一个学习控制车辆。其余两车提出固定规则动作；联合安全层仍可对三车制动。它是扩展到 MARL 之前的单车前置验证。',
        '',f"128 个独立初态、四类执行误差、11 种控制器，共 {summary['executions']} 次实际 SUMO 执行。另有 256 个独立校准初态的 1024 次 SUMO 运行。三个训练种子分别进行 1048576 个模型环境转移；模型训练步数不能计作 SUMO 运行次数。",'',
        '每个 actor 用本车局部观测逐周期输出动作；固定最终检查点，行为克隆检查点作消融。教师为单动作滚动规划器，数据是512条模型轨迹。',
        '', '## 主对照结果','', '|方法|平均到达时间 s|累计指令变化量 m/s²|综合代价 T＋0.05D|','|---|---:|---:|---:|']
    for label,r in zip(['固定规则','滚动规划基线','行为克隆 三种子均值','单车PPO 三种子均值','校准门控 三种子均值'],entries):
        text.append(f"|{label}|{r['mean_capped_finish_s']:.4f}|{r['variation']:.4f}|{r['objective']:.4f}|")
    text += ['',f"相对滚动规划基线，PPO综合代价降低 {p['rl_cost_gain_percent']:.3f}%，指令变化量降低 {p['rl_variation_gain_percent']:.3f}%；到达时间变化为 {p['rl_arrival_regression_percent']:+.3f}%（负值为更快）。门控综合代价降低 {p['gate_cost_gain_percent']:.3f}%。",'',
        'λ=0.05在测试前固定。综合代价提升不等于每个分量、每个场景都改善，指令变化量也不是实际乘坐舒适性。','',
        '## 门控是否保留收益并减少错误放行','',
        '|初态分组|策略回合数|学习放行率|有害放行占全部|放行后的退化比例|保留净收益|',
        '|---|---:|---:|---:|---:|---:|']
    for group,label in [('all','全部'),('low_conflict','低冲突'),('high_conflict','高冲突')]:
        r=gate[group];conditional='无放行' if r['conditional_harm_rate'] is None else f"{r['conditional_harm_rate']*100:.2f}%"
        retained='无正净收益' if r['retained_net_gain_percent'] is None else f"{r['retained_net_gain_percent']:.2f}%"
        text.append(f"|{label}|{r['n']}|{r['acceptance_rate']*100:.2f}%|{r['harmful_acceptance_rate']*100:.2f}%|{conditional}|{retained}|")
    text += ['', '分母包含三个固定模型与四种误差重复；独立统计单位仍为128个初态。低/高冲突按初态ETA差1.25秒预先划分，不按实际收益事后分组。','',
        '共形上界控制的是可交换条件下的边际有害放行风险，α=0.1不是碰撞概率，也不保证放行子集内退化比例低于10%。性能门控在每次合流事件开始时选择一次；物理安全证书每0.5秒复核。',
        '', '未校准门控与其他α的曲线使用已执行的成对完整轨迹重组，不计作额外SUMO运行。主结果阈值始终为预先冻结的0.1。','',
        '## 预先冻结的验收','',f"通过 {acceptance['passed']}/{acceptance['total']} 项。",'',
        '|检查项|结果|','|---|---|']
    for key,ok in acceptance['checks'].items():text.append(f"|{key}|{'通过' if ok else '未通过'}|")
    text += ['', '## 推断和工程范围','',
        '配对bootstrap以初态为簇，先平均四类误差与三个固定策略，再重采样10000次。95%区间是描述性区间，不代表对所有训练随机种子的推断。逐训练种子结果、每次失败、计时及全部原始回合均保留在summary.json与study/episodes.json。',
        '', '固定路线、精确测量、全受控车辆及已知±0.5 m/s²误差界是重要限制。SUMO暂停等待控制器，不代表计算延迟闭环或实车验证。当前未复现并击败已发表的完整方法，不能据此宣称达到顶会投稿成熟度。',
        '', '[理论与相关研究](theory_v12_single.md) · [冻结协议](../artifacts/v12/single/protocol.json) · [完整汇总](../artifacts/v12/single/summary.json)',
        '', '```powershell','python tools/verify_single_v12.py --replay','python tools/study_single_v12.py --out runs/my_single_reproduction','```','']
    (ROOT/'docs/v12_single_results.md').write_text('\n'.join(text),encoding='utf-8',newline='\n')
    hashes={str(path.relative_to(D)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in D.rglob('*') if path.is_file() and path.name!='sha256.json'}
    (D/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n',newline='\n')
    print(json.dumps({'acceptance':acceptance,'performance':p,'gate':gate}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',default='runs/v12_single_study');a=p.parse_args();publish(a.study)
