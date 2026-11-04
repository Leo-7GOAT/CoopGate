"""Manuscript composition from fully completed evidence; no provisional results."""
from types import SimpleNamespace
from powered_mappo_manuscript_text_20261103 import compose as preserved_compose


def compose(p,h,eq,table,num,sub,frac,summation,cert,comparison,data,old_data,figure):
    seeds=('4703','4704','4705');summary=comparison['confirmation_methods']
    verdict='满足' if cert['joint_deployment_certified'] else '未同时满足'
    join=lambda values:'、'.join(str(v) for v in values)
    p('基于候选搜索与置信门控的多车协同学习控制','Title')
    p('MAPPO 分散策略与模型规划回退的独立闭环验证','Subtitle')
    p('李东旭','Subtitle')
    h('摘要')
    p('本文研究车辆协同汇入中学习决策与传统规划的切换。三个分散 actor 通过集中式 critic 进行联合 MAPPO 训练，'
      '部署时由局部动作偏好生成有限联合候选，并以物理可恢复性、逐车相对性能、连续相位与观测历史预测进行筛选。'
      '我们将附加拒绝条件纳入候选搜索，并在保留原有候选的基础上引入最多四个局部次优修复指令，避免单个候选被否决后直接放弃全部替代方案。'
      '有限预测收益分数与采用后的统计风险分别定义，后者以独立初态的任一车辆成本或到达时间退化为对象。'
      '固定三模型在 6144 个分层校准初态与 1024 个 IID 混合确认初态上完成原定 59392 个 SUMO 方法回合及完整独立回放。'
      '确认集平均配对成本改善为 '+join(num(summary[s]['mean_paired_percentage_gain'])+'%' for s in seeds)
      +'，整组模型'+verdict+'全部预定风险、采用率、收益和执行条件。'
      '相同候选集合的名义规划与独立随机弃权用于区分门控筛选和单纯减少采用的作用。'
      '结论限定于固定路线三车任务、已知执行器类别及声明的仿真分布。')
    p('关键词  MAPPO；候选搜索；置信门控；相对性能；可恢复控制')
    h('1 引言')
    p('学习型决策可能改善多车汇入效率，但团队平均回报不能回答当前采用的动作是否使某辆车变差。'
      '传统规划能够提供明确的动力学约束和回退行为，因此本文关注在同一闭环中选择实际执行指令。'
      '门控需要同时控制采用后的逐车退化、保持足够采用率，并产生相对强基线的实际收益。')
    p('本文针对一种具体失效方式改进候选搜索：原选择器可能先挑选一个名义上最优的指令，再因额外历史或相位条件否决它，'
      '从而丢弃其余可能通过的候选。我们在排序后的搜索中逐个检验这些条件，并利用各 actor 的局部次优偏好增加有限替代指令。'
      '方案保持同一物理基线、演员参数及收益门槛；其作用需要由完整闭环结果验证，不能由候选集合扩大直接推出。')
    p('本文的研究内容是候选级的相对性能筛选、观测历史条件下的物理分析，以及采用后风险与用途的独立联合验证。'
      'MAPPO、学习加备份、预测安全过滤和统计校准均有既有研究，本文不主张首次提出这些范式。'
      '开发数据用于选择固定方案，正式样本不重新选择模型或阈值。')

    def methods():
        h('5 候选搜索与置信门控')
        p('每次决策先计算同一传统基线 b，并只推进一次基线观测器和命令记忆。三个 actor 使用各自的局部输入，'
          '对公共动作掩码内的加速度等级排序。局部原始 logits 表示偏好，不作为可行性或安全概率。'
          '原有联合贪心候选及其混合修复全部保留；另构造单个活动角色采用局部第二选择、其余采用基线的三个请求，'
          '以及全部角色采用第二选择的一个请求。每个请求经共同物理修复后去重，已完成车辆仍执行基线指令。')
        eq(sub('C','+'),' = ',sub('C','greedy'),' ∪ {P(',sub('u','i','(2)'),',',sub('b','−i'),') : i∈',sub('I','active'),
           '} ∪ {P(',sub('u','all','(2)'),')}', '    |',sub('C','+'),'∖',sub('C','greedy'),'| ≤ 4')
        p('P 是现有可恢复性检查使用的有限指令修复算子，式中的上标 2 表示局部排序第二项，不表示平方。'
          '当合法动作只有一个时重复该动作；相同 logits 按动作索引稳定排序。修复后与基线或已有候选相同的指令不重复加入。'
          '该过程只提供有限联合候选，不求解全部 13³ 个联合动作的全局最优。')
        p('候选先按同一名义预测排序，再在冻结参数与创新路径上比较执行候选一步后转回基线的成本与始终执行基线的成本。'
          '任一车辆的成本或到达时间不得增加，连续越界相位不得延后，最坏相对团队收益至少为 0.25%。')
        eq('s(x,u) = ',sub('min','ω∈Ω'),frac([sub('V','B,ω'),'(x)−',sub('Q','B,ω'),'(x,u)'],[sub('V','B,ω'),'(x)']),
           '    s(x,u) ≥ 0.0025')
        p('Ω 为固定有限预测路径。收益分数 s 不是后验置信概率，也不覆盖全部真实未来。'
          '对具有可用观测窗口的单位增益零阻力类别，额外拟合滞后和持续偏差，并检查一对历史预测；增益阻力类别不构造该见证。'
          '拟合记忆须处于命令历史传播区间内，候选仍须通过相同逐车、相位与收益条件。')
        p('拒绝一个候选后继续检查后续候选，只有未找到许可候选时才执行已计算的共同基线。'
          '任一路径已经否决或给出低于门槛的收益上界时允许提前拒绝；接受必须完成所有规定检查。'
          '实际已发命令用于下一时刻记忆推进，未执行的学习请求不进入物理状态更新。')

    def remaining():
        h('6 5 候选包含关系与闭环限制',2)
        p('在相同扩展状态、基线与物理修复规则下，附加构造只向原列表添加去重候选，因此原候选集合包含于新集合。'
          '若每个候选的预测与许可谓词不受其他候选影响，且所有比较完整完成，则许可子集也保持包含关系。'
          '对同一名义目标在许可候选加基线中取最小值，新集合的最小预测成本不会更高。这是有限集合优化的条件性质。')
        eq(sub('C','0'),' ⊆ ',sub('C','+'), '    ',sub('F','0'),' ⊆ ',sub('F','+'),
           '    ',sub('min','u∈F₊∪{b}'),sub('Q̂','B'),'(x,u) ≤ ',sub('min','u∈F₀∪{b}'),sub('Q̂','B'),'(x,u)')
        p('上述结论不比较不同闭环后续状态，不保证真实累计成本或采用后风险单调改善，也不覆盖未完成预测和资源不足的情况。'
          '实际收益仍取决于模型误差、后续切换与候选选择引起的状态变化，必须通过独立闭环评估。')
        h('7 独立验证的统计对象')
        p('每个独立初态的所有方法共享真实参数和外部创新。A 表示整回合至少一次实际采用学习指令；'
          'H 表示任一车辆成本或到达时间比配对基线增加超过 10⁻⁹，或候选与基线发生记录的执行失败。')
        eq('R = Pr(H|A) = ',frac('Pr(H∩A)','Pr(A)'), '    Δ = 100 ',frac([sub('J','B'),'−',sub('J','G')],sub('J','B')))
        p('校准使用 6144 个初态，四个执行器类别各 1536 个，类别内独立抽取场景与扰动。'
          '混合条件风险按各类别联合有害概率上界与采用概率下界加权后作比，内部对八个组成界分配该目标的误差。'
          '强滞后保留独立的精确条件二项界。平衡分层采样的全部采用记录不能直接充当完整混合分布的 IID 二项样本。')
        eq('R ≤ min(1,',frac(summation('h',None,[sub('w','h'),sub('U','h')]),summation('h',None,[sub('w','h'),sub('L','h')])),')')
        p('确认使用 1024 个独立完整混合初态，不固定各类数量，不替换失败。采用率采用精确单侧二项下界，'
          '配对百分比收益采用单侧 t 下界；后者在正态 IID 下精确，在一般有限方差条件下为渐近推断。'
          '三个固定模型均须满足混合及强滞后风险上界不超过 5%、采用率下界至少 20%、收益下界严格超过 0.5%，'
          '且主基线和目标零观测执行失败。联合决策使用交并检验最大 p 值。')
        p('本研究登记为 attempt 6，水平为 0.05/42。早期 0.025、0.05/6、0.05/12、0.05/20、0.05/30 的分配保持消耗，'
          '不因失败或执行中断收回。组成界不自动形成任意用途的同时置信区域，也不据此重建全部历史探索的统一 FWER。')
        h('8 实验与结果')
        table('表 1 原定独立研究范围',['阶段','独立初态','方法数','完整方法回合'],
            [('分层校准','6144','7','43008'),('IID 确认','1024','16','16384'),('总计','7168','分阶段固定','59392')],[4.5,3.8,3.2,4.9])
        p('五种场景、四类执行器和四类创新模式均在收集前固定。校准执行基线、三目标和三个同候选集合名义对照；'
          '确认增加三个名义策略随机弃权对照与六个共享物理检查的 MPS/MAMPS 适配器。'
          '名义与弃权对照获得相同局部次优候选提供器，从而控制候选集合差异。MPS/MAMPS 直接筛选原始联合提案，'
          '不是候选集合匹配的门控消融，也不是对应原论文全部实验条件的复现。')
        p('独立弃权概率仅由完整校准的目标采用数除以名义采用数确定，确认使用独立硬币流。无效概率不裁剪，并仍完成原定确认。'
          '源码与原登记在采集前公开。一次采集中断后保留全部 642 条通过身份与压缩校验的已提交记录，另执行原表中的 58750 个缺失方法回合，'
          '所有原定记录接受完整控制器和物理回放。原中断进程的退出码及丢失在途尝试数未知，总物理尝试数不能写成已知。')
        p('三个模型整组'+verdict+'预定条件，联合 p='+num(cert['joint_family_p_value'],6)
          +'，登记水平='+num(cert['allocated_error'],6)+'；主基线与目标的观测执行失败计数为 '+str(cert['observed_execution_failures'])+'。'
          '零观测失败不等于总体零失败风险。')
        risk=cert['primary_risk']['members']
        rows=[(s,num(100*risk[s+':mixture']['pointwise_risk_upper'])+'%',num(100*risk[s+':strong_lag']['pointwise_risk_upper'])+'%',
            num(100*cert['adoption']['members'][s]['adoption_lower'])+'%',num(cert['paired_gain']['members'][s]['gain_lower_percent'])+'%') for s in seeds]
        table('表 2 三个固定模型的正式单侧界',['模型','混合风险上界','强滞后风险上界','采用率下界','收益下界'],rows,[1.5,3.9,4.0,3.5,3.5])
        figure('formal_acceptance','图 1 原定风险 采用率及收益条件的联合验证')
        order=['baseline']+[prefix+s for s in seeds for prefix in ('','nominal_','thinned_','mps_','mamps_')]
        labels={'baseline':'传统基线'}
        for s in seeds:labels.update({s:'目标 '+s,'nominal_'+s:'同池名义 '+s,'thinned_'+s:'名义弃权 '+s,'mps_'+s:'MPS '+s,'mamps_'+s:'MAMPS '+s})
        rows=[(labels[m],str(summary[m]['adopted'])+'/1024',str(summary[m]['harmful_adopted']),num(summary[m]['mean_paired_percentage_gain'])+'%',str(summary[m]['execution_failures'])) for m in order]
        table('表 3 全部确认方法的完整闭环结果',['方法','采用回合','有害采用回合','平均配对收益','执行失败'],rows,[4.0,3.4,3.0,3.6,2.4])
        figure('formal_control_comparisons','图 2 同候选集合及采用率对照的描述性结果')
        rows=[]
        for s in seeds:
            q=comparison['adoption_differences'][s]
            rows.append((s,num(comparison['calibration_matching']['probabilities'][s],4),num(100*q['observed_difference']),
                         '['+num(100*q['lower'])+', '+num(100*q['upper'])+']'))
        table('表 4 目标与弃权对照的采用率差',['模型','校准硬币概率','实际差 百分点','区间 百分点'],rows,[1.6,4.0,5.0,5.8])
        p('全部匹配概率有效状态为 '+str(comparison['calibration_matching']['all_matching_valid'])+'。'
          '区间来自配对不一致事件的四个精确二项单侧界，并按原规定在三组比较内分配误差。覆盖零不证明等价，'
          '校准估计不保证确认集或总体采用率精确相同。对照的收益和损害点值是描述性比较，不另行产生部署证书。')
        rows=[]
        for s in seeds:
            q=data['development_paired_table'][s]
            rows.append((s,str(q['old_adopt'])+' / '+str(q['new_adopt']),num(q['old_mean_gain_percent'])+'%',
                         num(q['new_mean_gain_percent'])+'%',num(q['gain_change_pp']),str(q['old_harm'])+' / '+str(q['new_harm'])))
        table('表 5 独立开发初态上的整套候选改动',['模型','旧 新采用数','旧平均收益','新平均收益','差 百分点','旧 新损害数'],rows,[1.5,3.2,3.0,3.0,3.0,2.7])
        p('表 5 使用完成的 320 个新开发初态和 2240 次实际 SUMO 方法回合，旧方法为后置拒绝的观测历史门控。'
          '比较同时包含候选级持续搜索和局部次优扩展，未单独隔离两者的因果贡献。更早的复用数据晋级筛选仍未通过；'
          '前一次完整独立研究亦未通过，不能以当前开发结果替换旧结论。开发初态未进入本轮正式样本。')
        fallback=data['exact_physical_fallback']
        p('完整记录额外核对 '+str(fallback['successful_zero_adoption_exact_fallbacks'])+' 个成功零采用回合与配对基线的指令、时间及物理轨迹完全相同，'
          '并检查 '+str(fallback['paired_observed_prefixes_checked'])+' 个方法回合的可观察首次采用前缀。'
          '失败记录保留，未将失败的零采用回合计为成功回退。该核对不证明未测试状态或未来扰动上的安全。')
        rows=[]
        for s in seeds:
            q=data['deployed_role_participation']['confirmation'][s]
            rows.append((s,*[str(v) for v in q['actual_command_change_steps']],str(q['multi_role_command_change_steps'])))
        table('表 6 确认集目标的实际角色控制改变',['模型','前车步数','汇入车步数','后车步数','多角色同时改变'],rows,[1.8,3.0,3.0,3.0,5.6])
        p('各角色改变均按活动车辆实际指令与同一决策状态下基线的差异计算。训练审计、局部策略网络和部署记录分别支持训练与执行事实；'
          '这些结果不单独证明集中式 critic 相对 IPPO 的因果优势，也不把整个过滤后的闭环收益归因于裸 MAPPO 策略。')
        h('9 计算代价与复现边界')
        rows=[(s,str(summary[s]['callbacks']),num(summary[s]['callback_p95_ms']),num(summary[s]['callback_max_ms']),str(summary[s]['callback_over_500ms'])) for s in seeds]
        table('表 7 完整实际在线回调耗时',['模型','回调数','P95 毫秒','最大毫秒','超过 500 毫秒'],rows,[2.4,3.4,3.2,3.6,3.8])
        figure('formal_callback_cost','图 3 完整在线回调耗时与 500 毫秒控制周期')
        p('桌面硬件为 Ryzen 5 9600X、约 64 GB 内存与 RTX 5060 Ti 16 GB。计时包含推理、候选构造、物理检查、全部实际预测与回退工作，'
          '未用超时截断隐藏慢调用，不统计通信延迟。保留记录使用原实际计时，新记录使用恢复运行的实际计时。'
          '混合批次与共享桌面负载不能支持因果加速或硬实时结论，超过控制周期的调用限制直接部署能力。')
        p('源码、固定权重、原登记与恢复记录在 https://github.com/Leo-7GOAT/CoopGate 公开。'
          '逐文件校验、已提交记录身份、实际原生退出凭据及完整回放用于复核；原中断退出与在途计数保持未知。'
          '未来修改门控或模型须使用新的独立验证，不能依据正式结果挑选种子或回收已消耗误差。')
        h('10 讨论与结论')
        p('本文将 MAPPO 联合提案与传统基线放在同一闭环中评估，以有限候选级筛选实现置信门控，并通过独立采用后风险、采用率和收益条件控制结果解释。'
          '本轮固定三模型整组'+verdict+'预定要求。候选包含与条件辨识分析解释了算法结构，也限定了能由预测分数推出的结论。')
        p('研究范围仍为固定路线三车、已知执行器类别和精确同步观测。有限预测路径不是连续参数与未来创新的完整覆盖，'
          '历史拟合也未解决一般隐状态辨识；未验证真实车辆、视觉感知、自由横向规划及硬实时执行。'
          '未完整适配的现有先进方法和集中式 critic 的独立贡献不作为已证实优越性。')

    active=False
    dummy=SimpleNamespace(paragraph_format=SimpleNamespace(),runs=[])
    def filtered_p(text,style=None):
        if not active:return dummy
        # Use the verified official abstract landing page rather than the old
        # PDF address that timed out during this revision's reference check.
        if text.startswith('[7] '):
            text=text.replace('https://papers.nips.cc/paper_files/paper/2024/file/b589d92785e39486e978fa273d0dc343-Paper-Conference.pdf',
                              'https://proceedings.neurips.cc/paper_files/paper/2024/hash/b589d92785e39486e978fa273d0dc343-Abstract-Conference.html')
        return p(text,style)
    def filtered_h(text,level=1):
        nonlocal active
        if level==1:
            active=text.startswith(('2 ','3 ','4 ','6 ')) or text=='参考文献'
            if text.startswith('5 '):methods()
            if text.startswith('7 '):remaining()
        if active:return h('4 MAPPO 联合训练' if text=='4 真实 MAPPO 联合训练' else text,level)
    def filtered_eq(*args):
        if active:return eq(*args)
    # Reuse already audited physical/training/conditional-theory prose and native
    # equations. Old numerical results, methods section and figures are omitted.
    preserved_compose(filtered_p,filtered_h,filtered_eq,lambda *a:None,num,sub,frac,summation,
        old_data['certificate'],old_data['comparison'],old_data,lambda *a:None)
