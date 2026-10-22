"""Publish all V13 outcomes, including failed standalone and gate criteria."""
import argparse,hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_ego_v13 import summarize
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'artifacts/v13/ego'


def publish(study):
    if not (D/'study').exists():shutil.copytree(study,D/'study')
    p=json.loads((D/'protocol.json').read_text());rows=json.loads((D/'study/episodes.json').read_text())
    s=summarize(p,rows,json.loads((D/'calibration/calibration.json').read_text()));(D/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    base=s['methods']['rollout'];av=s['seed_average'];entries=[base,av['raw'],av['untrained'],av['augmented'],av['gate']]
    labels=['Planner','Neural policy','Untrained tail','Learned tail','Gated']
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,key,ylabel in zip(axes,['ego_time','ego_variation','ego_cost'],['Ego arrival (s)','Ego command variation (m/s²)','Ego cost (s)']):
        values=[r[key] for r in entries];bars=ax.bar(labels,values,color=['#64748b','#d8a24b','#9ca3af','#0f766e','#2563eb'])
        ax.set_ylim(0,max(values)*1.18);ax.set_ylabel(ylabel);ax.tick_params(axis='x',labelrotation=20);ax.spines[['top','right']].set_visible(False)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+max(values)*.025,f'{v:.3f}',ha='center',fontsize=9)
    fig.suptitle('Independent SUMO test: ego-specific outcomes',fontsize=13);fig.savefig(D/'ego_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');rr=[r for r in s['counterfactual_gate_ablations'] if r['group']=='all']
    xs=['Uncalibrated cost','Calibrated cost','Joint cost + time']
    for ax,key,ylabel in zip(axes,['joint_harm_rate','mean_retained_cost_gain'],['Harmful admission / all episodes (%)','Mean retained cost gain (s)']):
        vals=[r[key]*(100 if key=='joint_harm_rate' else 1) for r in rr];bars=ax.bar(xs,vals,color=['#9ca3af','#0f766e','#2563eb'])
        ax.set_ylabel(ylabel);ax.tick_params(axis='x',labelrotation=12);ax.spines[['top','right']].set_visible(False)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
        ax.margins(y=.2)
    fig.suptitle('Gate ablations from paired full-episode trajectories');fig.savefig(D/'gate_ablation.png',dpi=180);plt.close(fig)
    t=['# 自车收益与双指标性能门控：独立 SUMO 验证','',
        '**主体已改为自车。** 所有到达时间和指令变化量主指标都属于学习车辆，变化量仅累计到该车到达；另外两车的平均时间另行报告。',
        '',f"实验在提交3bac214冻结后运行：128个独立初态×四类误差×14种控制器，共{s['executions']}次实际SUMO执行。校准另有256个独立初态、1024次执行。三个全新训练种子各进行2097152步Double DQN模型训练，均保留最终检查点，没有模仿标签。",'',
        '**两种学习执行方式分别报告。** 纯网络直接提案；学习辅助规划器使用网络给出额外的未来反馈序列，在线规划器再选择动作。不能将后者的结果写成纯RL优于基线。',
        '', '|方法|自车到达 s|自车指令变化 m/s²|自车综合代价 s|其他两车平均到达 s|', '|---|---:|---:|---:|---:|']
    for label,r in zip(['强规划基线','纯Double DQN','未训练网络辅助规划','训练后网络辅助规划','双指标门控'],entries):
        t.append(f"|{label}|{r['ego_time']:.4f}|{r['ego_variation']:.4f}|{r['ego_cost']:.4f}|{r['other_time']:.4f}|")
    pa=s['performance']['augmented'];pg=s['performance']['gate'];g=s['gate_diagnostics']['all']
    t += ['',f"相对强规划基线，学习辅助规划的自车代价降低{pa['ego_cost_gain_percent']:.3f}%，自车到达时间降低{pa['ego_time_gain_percent']:.3f}%，指令变化降低{pa['ego_variation_gain_percent']:.3f}%。相对同构未训练网络对照，代价降低{s['learning_gain_vs_untrained_percent']:.3f}%。",'',
        f"门控代价降低{pg['ego_cost_gain_percent']:.3f}%，到达时间降低{pg['ego_time_gain_percent']:.3f}%。纯网络前置检验：{'通过' if s['individual_ego_verified'] else '未通过'}；MARL扩展{'允许进入下一验证阶段' if s['marl_expansion_allowed'] else '继续保留阻断条件，不把辅助规划器的成绩替代纯网络成绩'}。",'',
        '## 门控保留收益和过滤退化','',
        '门控要求成本改善上界≤−0.05秒、到达时间差上界≤0秒。两维残差最大值统一校准，性能门控在一次合流开始时执行一次，安全证书每0.5秒复核。','',
        '|初态分组|策略回合数|放行率|有害放行占全部|放行后退化率|保留净收益|经验联合覆盖|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group,label in [('all','全部'),('low_conflict','低冲突'),('high_conflict','高冲突')]:
        r=s['gate_diagnostics'][group];conditional='无放行' if r['conditional_harm_rate'] is None else f"{r['conditional_harm_rate']*100:.2f}%"
        retained='无正收益' if r['retained_gain_percent'] is None else f"{r['retained_gain_percent']:.2f}%"
        t.append(f"|{label}|{r['n']}|{r['acceptance_rate']*100:.2f}%|{r['joint_harm_rate']*100:.2f}%|{conditional}|{retained}|{r['empirical_joint_coverage']*100:.2f}%|")
    t += ['', '有害指成本或自车到达时间中任一项退化。低/高冲突按初态ETA差1.25秒预先划分，不能等同于任意简单/复杂道路。策略回合数包含三个模型和四类误差重复；独立初态数为128。','',
        '共形界针对可交换回合的边际风险，不是碰撞概率，也不是放行子集的条件错误率保证。经验联合覆盖率照实保留，不能为了接近90%而调整已冻结阈值。','',
        '未校准成本门控与仅成本校准门控用同场景已执行的完整轨迹重组，不计作额外SUMO运行。双指标门控本身另作实际执行，并检查其轨迹与选中控制器一致。','',
        '|门控消融|放行率|有害放行占全部|保留代价收益 s|','|---|---:|---:|---:|']
    for r in rr:t.append(f"|{r['variant']}|{r['acceptance_rate']*100:.2f}%|{r['joint_harm_rate']*100:.2f}%|{r['mean_retained_cost_gain']:.4f}|")
    cost_only=next(r for r in rr if r['variant']=='calibrated_cost_only');joint=next(r for r in rr if r['variant']=='joint_cost_time')
    t+=['',f"与仅成本校准相比，联合时间约束将本批有害放行从{cost_only['joint_harmful_admissions']}次降到{joint['joint_harmful_admissions']}次；额外减少幅度有限，同时少保留了一部分收益。双指标的价值首先是约束含义清楚，当前数据不足以证明它显著优于仅成本校准。"]
    t+=['','## 冻结验收','',f"通过{s['acceptance']['passed']}/{s['acceptance']['total']}项。未通过的项目保留，不事后改阈值。",'','|检查|结果|','|---|---|']
    for key,value in s['acceptance']['checks'].items():t.append(f"|{key}|{'通过' if value else '未通过'}|")
    t+=['','## 统计与工程边界','',
        '配对bootstrap以初态为簇，先平均四类误差和三个固定模型，再重采样10000次。95%区间是近似描述性推断，不代表训练随机种子的总体保证。每个模型的指标、原始失败、计时和完整核验材料均在summary.json及study/中。',
        '', '只验证固定路线、精确状态、三辆全受控车辆与已知误差界。共同安全层能够制动规则车辆；这不是非合作开放交通。未统计通信延迟，未做实车与真实延迟闭环。',
        '', '[理论和创新边界](theory_ego_v13.md) · [冻结协议](../artifacts/v13/ego/protocol.json) · [完整统计](../artifacts/v13/ego/summary.json)',
        '', '```powershell','python tools/verify_ego_v13.py --replay','python tools/study_ego_v13.py --out runs/my_ego_reproduction','```','']
    (ROOT/'docs/v13_ego_results.md').write_text('\n'.join(t),encoding='utf-8',newline='\n')
    hashes={str(path.relative_to(D)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in D.rglob('*') if path.is_file() and path.name!='sha256.json'}
    (D/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({'acceptance':s['acceptance'],'performance':s['performance'],'gate':s['gate_diagnostics'],
        'individual_ego_verified':s['individual_ego_verified']}),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--study',default='runs/v13_ego_study');v=a.parse_args();publish(v.study)
