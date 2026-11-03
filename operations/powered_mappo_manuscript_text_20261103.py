"""Complete-study manuscript text; every result is supplied by a terminal bundle."""
from docx.shared import Pt


def compose(p,h,eq,table,num,sub,frac,summation,cert,comparison,data,figure):
    seeds=('4703','4704','4705')
    risk=cert['primary_risk']['members'];summary=comparison['confirmation_methods']
    verdict='通过' if cert['joint_deployment_certified'] else '未通过'
    join=lambda values:'、'.join(str(v) for v in values)
    p('基于观测历史与相对性能门控的多车学习控制协同方法','Title')
    p('CoopGate 真实 MAPPO 可恢复基线与独立闭环验证','Subtitle')
    p('李东旭','Subtitle')
    h('摘要')
    p('学习策略的平均回报提高，并不能说明当前采用的动作优于传统规划控制。本文研究固定路线三车汇入中的基线相对性能门控：'
      '三个分散局部 actor 由集中式 critic 进行联合 MAPPO 训练；部署时，物理可恢复性证书、逐车预测约束、'
      '连续越界相位检查和基于实际观测历史的滞后偏差预测共同决定采用学习指令或执行同一传统基线。'
      '我们推导闭环性能的基线优势分解，给出历史观测条件可辨识性及其反例，并区分有限预测评分与采用后统计风险。'
      '三个固定模型各完成 2,097,152 个联合 on-policy 步；独立研究使用 4,096 个分层校准初态及 1,024 个 IID 混合确认初态，'
      '完成 45,056 次 SUMO 运行和完整控制器物理复核。确认集平均配对团队成本改善分别为 '
      +join(num(summary[s]['mean_paired_percentage_gain'])+'%' for s in seeds)+'；整组模型'+verdict+'全部预定条件。'
      '另以原确认初态完成 6,144 次预登记门控机制消融，并在独立开发数据上完成 960 次随机合法提案对照。'
      '结论限定于声明的执行器类别、同步观测和仿真分布，不构成逐状态安全概率或真车实时保证。')
    p('关键词  MAPPO；性能门控；执行器记忆；基线相对风险；安全回退')
    h('1 引言')
    p('车辆在汇入中需要协调加速与让行。学习策略可以提出联合动作，传统控制则提供明确的动力学约束及可恢复行为。'
      '真正的部署问题是：何时有足够证据采用学习动作，证据不足时能否准确执行同一基线？'
      '策略 softmax 概率是动作偏好，并不等于物理可行性概率或相对基线收益概率。')
    p('本文将完整部署闭环作为评价对象。任一车辆的实际成本或到达时间退化均记为损害，避免团队平均收益掩盖个体损失；'
      '同时要求非平凡采用率与严格正的收益下界，避免全回退获得表面安全。主要工作包括：'
      '构造执行器记忆相关的物理与相对性能门控；解释观测历史增加的信息及其适用条件；'
      '在真实联合 MAPPO 上使用独立完整闭环、实际采用率对照、机制消融与全回调计时验证。'
      '标准 PPO、系统辨识代数和统计检验均沿用已有思想，本文不将术语组合或仿真数量本身作为创新。')
    h('2 相关工作与定位')
    p('MAPPO 使用合作任务中的 PPO 与集中式价值估计[1]。本文沿用其训练范式，研究训练后如何选择实际执行方案。'
      '集中式 critic 仅用于训练；共享物理过滤器和门控仍集中执行，因此不声称整个系统完全分散。'
      'MPS 与 MAMPS 已研究学习策略和备份策略的在线切换[2,3]，预测安全过滤器也已有完整框架[4]。'
      '本文不是首次提出学习加备份或多智能体安全过滤。')
    p('SPIBB 通过基线约束研究安全策略改进[5]；CSPI-MT 已研究阈值策略的多候选改进检验与阈值选择[6]；'
      'DMPS 让恢复规划同时考虑短期进展与长期回报[7]。因此统计校准和考虑性能的恢复均已有先例。'
      '本文的具体对象是固定联合策略在实际采用条件下的任一车辆退化，并将执行器记忆、相位约束和收益见证组合后独立验证。')
    p('UPSi 使用模型集合和可达集表达预测不确定性[8]；组合式及分散式多智能体盾牌研究了可扩展的局部验证[9,10]。'
      '本文的历史拟合是有限假设见证，不是后验分布，也未覆盖全部连续参数和未来扰动。'
      '实验中的 MPS/MAMPS 是共享当前物理模型的适配器，不是原论文全部条件的复现；'
      '本文未完整适配 CSPI-MT、DMPS、UPSi，因而不主张相对这些方法的数值优越性。')
    h('3 物理模型、任务与传统基线')
    p('任务包含主路前车、汇入车和主路后车，车辆沿给定路线运动。物理积分步长 0.1 s，控制周期 0.5 s，回合时域 25 s。'
      '离散加速度提案包含 13 个等级，范围 −5 至 2.5 m/s²，步长 0.625 m/s²。公共动作掩码限制单周期指令变化。'
      '执行器包含增益 g、阻力 d、记忆保留系数 ρ 和外部创新 ξ；位置采用更新后速度进行半隐式积分。')
    eq(sub('m','k+1'),' = ρ',sub('m','k'),' + (1−ρ)',sub('u','k'),
       '    ',sub('a','k'),' = g',sub('m','k+1'),' − d',sub('v','k','2'),' + ',sub('ξ','k'))
    p('类别为名义、增益阻力、轻度滞后与强滞后。在线门控只获知声明的类别区间，'
      '不读取真实随机参数、隐藏执行器记忆或未来创新。强滞后时间常数位于 0.3–0.5 s，创新幅值界为 0.1 m/s²。'
      '执行器指令跟踪误差与外部创新不同，不能将二者的界混用。')
    p('传统基线采用确定性联合候选规划及执行器可恢复控制。根据已发命令传播记忆区间，'
      '停车证书检查固定路线几何与输入约束，直至进入可保持的停止集合。有效性以真实执行器属于声明类别、'
      '包络有效和同步观测时序成立为条件。无法出具证书表示验证未成功，不说明碰撞必然发生。'
      '该实现不是通用横纵向 MPC，不能据此声称解决城市驾驶的全部规控问题。')
    eq(sub('J','i'),' = ',sub('T','i'),' + 0.05',summation('t=0','N−1',[sub('I','t,i'),' |',sub('u','t,i'),'−',sub('u','t−1,i'),'|']),
       '    ',sub('J','team'),' = ',frac('1','3'),summation('i=1','3',sub('J','i')))
    p('Tᵢ 为截断到达时间，I 表示该车尚未到达；第二项衡量所发指令的变化，不能直接等同于实测车身 jerk。'
      '按秒表示总成本时，权重单位为 s³/m。未完成回合保留 25 s 截断成本及已执行前缀。')
    h('4 真实 MAPPO 联合训练')
    p('三个独立参数化的局部 actor 均采用两层宽度 256 的 ReLU 网络。每车读取 18 个局部特征及由其确定性计算的 8 个运动学特征，'
      '集中式 critic 使用 37 维公共全局特征、两层宽度 256 的 Tanh 网络估计团队价值。critic 不参与部署提案。'
      '三个 actor 都由联合回合的团队奖励更新，而不是仅训练汇入车并把另外两辆固定车辆称为多智能体。')
    eq(sub('π','θ'),' (z | o,M) = ',sub('π','θ₁'),' (',sub('z','1'),' | ',sub('o','1'),',',sub('M','1'),') ',
       sub('π','θ₂'),' (',sub('z','2'),' | ',sub('o','2'),',',sub('M','2'),') ',sub('π','θ₃'),' (',sub('z','3'),' | ',sub('o','3'),',',sub('M','3'),')')
    p('rollout 保存物理过滤前的采样提案和旧策略 log probability。过滤器可以将多个提案映射到同一执行指令，'
      '因此不能用执行指令替换 PPO 的采样动作。固定过滤器并入环境转移核后，似然比更新无需对过滤器求导；'
      '这不意味着 clipped surrogate 保证每次更新均改进性能。')
    eq(sub('δ','t'),' = ',sub('r','t'),' + γ(1−',sub('d','t'),')V(',sub('s','t+1'),')−V(',sub('s','t'),')',
       '    ',sub('Â','t'),' = ',sub('δ','t'),' + γλ(1−',sub('b','t'),')',sub('Â','t+1'))
    p('d 表示真正终端，b 阻止优势跨回合复位传播。25 s 为任务终端；仅因 rollout 长度结束时使用预复位后继观测 bootstrap。'
      '每个 actor 使用本车采样动作的似然比及团队优势更新，非活动车辆由掩码排除。')
    eq(sub('r','t,i'),' = exp(log ',sub('π','θᵢ'),'(',sub('z','t,i'),'|',sub('o','t,i'),') − log ',sub('π','old,i'),'(',sub('z','t,i'),'|',sub('o','t,i'),'))')
    p('每批 64 个环境、每环境 64 步，更新 5 个 epoch，minibatch 1024，clip 0.2，entropy 0.01，γ=λ=1。'
      '前车与后车 actor 学习率为 10⁻⁴，汇入 actor 为 10⁻⁵，critic 为 3×10⁻⁴。三个模型的声明配置相同。'
      '每个模型实际完成 2,097,152 个新联合步，并独立核验 on-policy 提案、GAE 及所有角色非零参数更新。'
      '初始化由已验证单车 PPO 及更早的规则初始化继承，另有后车滑行先验，不声称从零训练。')
    p('训练采用匹配任务的批量解析环境；后续验证实际启动 SUMO 并执行闭环。联合训练步数、仿真回合数和独立验证初态数分别报告，不相互替代。')
    h('5 观测历史与相对性能门控')
    p('每次控制先生成相同传统基线，再根据三个局部 actor 的提案构造有限联合候选。学习候选须通过可恢复性检查；'
      '在冻结预测路径上，候选与基线的逐车到达时间及成本不得退化，归一化团队收益最低须达到 0.25%。'
      '连续相位检查拒绝会延后越界事件的候选，减少离散采样下被掩盖的延误。')
    eq('s(x,u) = ',sub('min','ω∈Ω'),frac([sub('V','B,ω'),' (x)−',sub('Q','B,ω'),' (x,u)'],[sub('V','B,ω'),' (x)']),
       '    s(x,u) ≥ 0.0025')
    p('Ω 是冻结的有限参数与创新路径集合。V 与 Q 表示采用相同基线延续的剩余预测成本，Q 包含当前候选一步。'
      '分数 s 是有限预测中的最小相对收益，不是后验置信概率，也不是全部真实未来的性能下界。'
      '任一路径否决即可提前拒绝；只有所有必需路径均完成，才能接受。该提前终止不改变许可集合。')
    p('对单位增益零阻力的类别，利用已观测控制边界位置、速度及实际命令历史，联合拟合滞后与持续偏差。'
      '拟合采用 65 点网格及有界局部求精，随后增加候选与基线的一对历史预测。'
      '预测仍须满足逐车不退化、相位条件及同一收益门槛。增益阻力类别和缺少有效观测窗口时不构造该见证。'
      '拟合不是全局最优证明，缺少激励时也不自动保证唯一识别。')
    p('门控拒绝时执行同一上游基线，记忆根据实际命令推进一次；不能用未执行的学习动作更新状态。'
      '零采用的成功回合及首次采用前的轨迹前缀必须与实际基线一致，实验逐条检查这一性质。')
    h('6 数学与物理分析')
    h('6 1 闭环性能分解及模型误差',2)
    p('将时刻、命令历史、记忆信息与终止状态纳入扩充状态。令 Vᴮ 为真实基线的有限时域成本到达函数，'
      'Aᴮ 为当前动作加基线延续相对 Vᴮ 的优势成本。条件期望望远镜求和得到')
    eq('E[',sub('J','G'),'−',sub('J','B'),'] = E[',summation('t','N−1',sub('A','B,t')),
       '] = E[',summation('t','N−1',[sub('Â','B,t'),' + ',sub('ε','t')]),']')
    p('执行基线的步骤具有零真实基线优势，学习采用步骤的预测收益与模型误差共同决定最终改善。'
      '若每个采用步骤的真实优势成本都非正，则期望总成本不增；当前有限预测检查并未验证这一真实条件。'
      '执行器隐藏记忆、未来创新与后续切换都可能造成误差项，解释了预测通过但实际逐车退化的反例。'
      '该分解沿用标准性能差分思想，不将其本身作为新强化学习定理。')
    h('6 2 滞后与持续偏差的条件可辨识性',2)
    p('考虑单位增益、零阻力、不发生速度裁剪的五个物理子步，窗口内命令 u 与创新 b 均为常数，初始记忆 m₀ 已知。'
      '由 mₛ=ρˢm₀+(1−ρˢ)u 及半隐式积分构造消去 b 的观测组合')
    eq('Z = ',sub('p','5'),'−',sub('p','0'),'−0.2',sub('v','0'),'−0.3',sub('v','5'),
       ' = 0.01(',sub('m','0'),'−u)φ(ρ)')
    eq('φ(r)=2r+r²−r⁴−2r⁵', '    b=2(',sub('v','5'),'−',sub('v','0'),')−0.2',summation('s=1','5',sub('m','s')))
    p('当 m₀≠u 且 φ 在声明区间严格单调时，可依次唯一确定 ρ 与 b。轻度滞后区间的 φ′ 为正，'
      '强滞后区间的 φ′ 为负，分别有保守绝对下界 1.98 与 0.47。'
      '该结论有已知记忆、激励和常偏差条件，不是当前多窗口数值优化的全局求解保证。')
    h('6 3 可辨识性的边界与创新反例',2)
    p('若子步创新可以变化，上式额外出现 0.01∑(3−s)bₛ。其偏离常数部分的幅值界为 εᵦ 时，残差最多为 0.06εᵦ。'
      '取 m₀=0、u=1、ρᴬ=0.75、ρᴮ=0.76，并令 bᴬₛ=0、bᴮₛ=(ρᴮ)ˢ−(ρᴬ)ˢ，'
      '两执行器在所有五个子步的位置和速度完全相同，虽然滞后不同。所需创新最大约 0.017216 m/s²，低于声明的 0.1 界。'
      '精确有理数递推验证了该反例。因此即便无观测噪声，单窗口内也不能无条件区分任意有界创新与滞后变化。'
      '这不反驳常偏差条件下的唯一性，而是说明历史见证不能替代未来不确定性覆盖。')
    h('6 4 采用率 风险与随机弃权',2)
    p('更严格的拒绝规则可能使整回合采用事件缩小，但条件有害率未必单调下降，因为分母也改变。'
      '相同初态下只有实际采用前状态相同、拒绝后执行同一基线并正确推进记忆，才能讨论采用事件包含关系。'
      '此外，若部署分布与测试分布的总变差至多 ε，原联合有害率界为 U、采用率下界为 L>ε，则部署条件风险至多 min(1,(U+ε)/(L−ε))。'
      '本文没有估计真实道路的 ε，因而不据此作分布外认证。')
    p('确认阶段的随机弃权对照以独立回合硬币 C 决定运行名义门控 N 或同一基线。条件于校准后固定的概率 q，且 q>0，有')
    eq('Pr(',sub('A','C'),') = q Pr(',sub('A','N'),')',
       '    ',sub('R','C'),' = ',sub('R','N'),'    E[',sub('Δ','C'),'] = q E[',sub('Δ','N'),']')
    p('因此独立随机弃权仅缩放采用率与平均收益，不改变名义门控的采用后条件风险。'
      '这是检验选择场景是否有价值的对照依据；校准比例估计仍不保证确认集实际采用率精确相同。')
    h('7 独立验证的统计对象')
    p('每个独立初态对全部方法使用相同真实参数与外部创新路径。A 为整回合实际采用学习，'
      'H 为任一车辆成本或到达时间比配对基线增加超过 10⁻⁹，或候选与基线发生执行失败。')
    eq('R = Pr(H|A) = ',frac('Pr(H∩A)','Pr(A)'),
       '    Δ = 100 ',frac([sub('J','B'),'−',sub('J','G')],[sub('J','B')]))
    p('4,096 个校准初态按四个执行器类别各取 1,024 个，类别内独立抽取场景和扰动。'
      '分层混合风险使用各类别联合有害概率上界 Uₕ 与采用概率下界 Lₕ 加权后作比，'
      '不能把平衡采样的全部采用回合直接当成混合分布 IID 二项样本。强滞后条件风险单独保留精确二项界。')
    eq('R ≤ min(1, ',frac(summation('h',None,[sub('w','h'),sub('U','h')]),summation('h',None,[sub('w','h'),sub('L','h')])) ,')')
    p('1,024 个确认初态独立来自完整混合分布，不强制类别平衡、不替换失败。采用率使用精确二项单侧下界，'
      '平均配对百分比收益使用单侧 t 下界：正态 IID 假设下精确，一般有限方差条件下为渐近推断。'
      '三个固定模型都须满足混合与强滞后风险上界≤5%、采用率下界≥20%、收益下界严格>0.5%，且主基线与目标零观测执行失败。')
    p('全部条件组成交并检验，联合 p 值取组成检验 p 值的最大值。研究登记为 attempt 5，水平 0.05/30，'
      '旧 0.025、0.05/6、0.05/12、0.05/20 的消耗均保留，不挑种子或转移旧证书。'
      '图表中的组成界不自动构成任意用途的同时置信区域；也不声称重建全部历史探索的统一 FWER。')
    h('8 实验与工程实现')
    table('表 1 本次独立研究及配对消融范围',['阶段','独立初态','方法数','实际 SUMO 运行'],
        [('分层校准','4096','7','28672'),('IID 确认','1024','16','16384'),
         ('主研究合计','5120','分阶段固定','45056'),('原门控配对','复用确认集','3','3072'),
         ('仅相位配对','复用确认集','3','3072')],[4.5,3.6,3.0,5.3])
    p('源代码 76d3a772336e 与登记 99afdd57422c 在采集前公开，冻结 163 个源文件，另有 165 文件逐字节快照。'
      '主研究的基线加三目标共 20,480 次，其余公平对照不增加独立初态数。校准另含三个名义门控，'
      '确认增加三个随机弃权、三个 MPS 和三个 MAMPS 适配器。每个方法实际运行并计时自己的完整回调。')
    p('执行器随机参数及创新路径用于 SUMO 闭环；独立复核根据完整记录重建物理递推，并重新执行参考控制器逐条比较。'
      '控制器异常、碰撞、未完成和约束违反保留成本与已执行前缀，不缩减样本。后端或证据完整性错误则停止并保留实验。'
      '仿真接口一致性不等于真车或独立高保真动力学验证。')
    p('采用率对照的概率仅由完整校准的目标采用数除以名义采用数确定，确认硬币使用独立流。'
      '无效比例不裁剪成合法值，仍完成全部确认并记录无效匹配。两组机制消融在主研究采集前固定全部确认初态与 actor。'
      '另外的随机合法提案对照复用完整 320 初态开发集，用三组独立局部随机流替代 actor，保留相同门控。'
      '它新增 960 次 SUMO，不增加独立初态或统计错误额度，不属于主证书。')
    h('9 完成结果')
    competence=data['single_competence']['metrics']
    p('作为联合训练父模型的单车 PPO 4601 先完成 128 初态、四种扰动的闭环能力检查，相对强单车规划的匝道成本、'
      '到达时间和指令变化改善分别为 '+join(num(competence[k]['aggregate_mean_gain_percent'])+'%' for k in ('cost','time','variation'))
      +'。该结果是父模型的开发证据；另一个单车种子 4603 未通过预定区间条件，其失败保留，不能推广为所有训练种子均优越。')
    p('主研究整组模型'+verdict+'预定条件；联合 p='+num(cert['joint_family_p_value'],6)
      +'，登记水平='+num(cert['allocated_error'],6)+'；主基线与目标的观测执行失败计数为 '+str(cert['observed_execution_failures'])+'。'
      '零观测失败不意味着总体失败概率为零。若任一条件不满足，本研究即保留为未通过。')
    rows=[]
    for s in seeds:
        rows.append((s,num(100*risk[s+':mixture']['pointwise_risk_upper'])+'%',
            num(100*risk[s+':strong_lag']['pointwise_risk_upper'])+'%',
            num(100*cert['adoption']['members'][s]['adoption_lower'])+'%',
            num(cert['paired_gain']['members'][s]['gain_lower_percent'])+'%'))
    table('表 2 固定三模型的正式单侧界',['模型','混合风险上界','强滞后风险上界','采用率下界','收益下界'],rows,[1.5,3.9,4.0,3.5,3.5])
    figure('formal_acceptance','图 1 正式风险、采用率和收益界；虚线为未改变的预定门槛')
    labels={'baseline':'传统基线'}
    for s in seeds:
        labels.update({s:'目标 '+s,'nominal_'+s:'名义 '+s,'thinned_'+s:'弃权 '+s,'mps_'+s:'MPS '+s,'mamps_'+s:'MAMPS '+s})
    order=['baseline']+[prefix+s for s in seeds for prefix in ('','nominal_','thinned_','mps_','mamps_')]
    rows=[]
    for method in order:
        r=summary[method];rows.append((labels[method],str(r['adopted'])+'/1024',str(r['harmful_adopted']),
                                      num(r['mean_paired_percentage_gain'])+'%',str(r['execution_failures'])))
    table('表 3 全部确认方法的实际闭环结果',['方法','采用回合','有害采用回合','平均配对收益','执行失败'],rows,[4.0,3.4,3.0,3.6,2.4])
    p('有害采用按任一车辆的实际退化定义。零采用失败仍进入执行失败和收益统计。'
      '以上为整个部署控制器的结果，不能归因于集中式 critic 单独优越。原始联合策略相对强团队规划的跨种子优势尚未建立，'
      '已有 IPPO 对照也未支持所有种子一致优越；不以真实 MAPPO 的训练事实替代优越性的实验依据。')
    rows=[]
    for s in seeds:
        q=comparison['adoption_differences'][s]
        rows.append((s,num(comparison['calibration_matching']['probabilities'][s],4),
            num(100*q['observed_difference'])+' 个百分点','['+num(100*q['lower'])+', '+num(100*q['upper'])+']'))
    table('表 4 确认集目标与弃权对照的实际采用差',['模型','校准硬币概率','实际采用率差','区间（百分点）'],rows,[1.6,4.0,5.0,5.8])
    p('匹配概率全部有效状态为 '+str(comparison['calibration_matching']['all_matching_valid'])+'。'
      '区间由配对不一致事件的四个精确二项单侧界作差，并对三组比较分配规定误差。'
      '覆盖零不证明等价，不把估计匹配描述成实际采用率精确相同。')
    figure('formal_control_comparisons','图 2 全确认集的目标门控及公平对照；比例为有害采用数除以实际采用数')
    h('9 1 预定场景的完整确认结果',2)
    scenes=data['diagnosis']['by_stage']['confirmation']['by_scene'];rows=[]
    if sum(group['baseline']['initials'] for group in scenes.values())!=1024:
        raise ValueError('Every confirmation initial must remain in the scene table')
    for scene,group in sorted(scenes.items()):
        n=group['baseline']['initials']
        if any(group[s]['initials']!=n for s in seeds):
            raise ValueError('Scene summaries must retain the same paired methods')
        rows.append((scene,str(n),*[num(group[s]['mean_paired_percentage_gain'])+'%' for s in seeds],
            ' / '.join(str(group[s]['adopted']) for s in seeds),
            ' / '.join(str(group[s]['harmful_adopted']) for s in seeds)))
    table('表 5 预定场景中的确认结果',['场景','初态数','4703 收益','4704 收益','4705 收益','采用数 三模型','有害采用 三模型'],
          rows,[3.0,1.2,2.3,2.3,2.3,2.8,2.5])
    p('场景名称及采样规则在收集前固定，各行保留所有对应确认初态，不按效果重新分组。'
      '三模型计数顺序为 4703、4704、4705，收益仍为相对同一初态实际基线的百分比成本改善。'
      '采用和性能退化计数用于核对场景覆盖与收益差异；整体证书不自动构成每个场景的条件风险证书。')
    h('9 2 门控机制的全确认集消融',2)
    rows=[]
    for variant,label in (('compact','原门控'),('phase','仅相位')):
        for s in seeds:
            r=data['mechanism_ablations'][variant]['results'][s];a=r['methods']['comparator'];b=r['methods']['history'];d=r['paired_discordance']
            rows.append((label+' '+s,str(a['adopted'])+' / '+str(b['adopted']),
                str(a['harmful_adopted'])+' / '+str(b['harmful_adopted']),
                join(d[k] for k in ('harm_removed','harm_introduced','harm_retained')),
                num(r['mean_paired_gain_difference_history_minus_comparator'])))
    table('表 6 消融与历史门控的完整配对结果',['比较','采用数 旧/历史','有害采用 旧/历史','移除/新增/保留损害','收益差百分点'],rows,[2.8,3.4,3.4,3.8,3.0])
    p('全部 6,144 次额外实际运行均使用相同固定 actor、物理模型、收益门槛和确认初态。'
      '“原门控”比较整个新增相位与历史机制，“仅相位”隔离额外历史预测的作用。'
      '采用数、配对收益损失和新增损害都保留；这些是完整描述性机制比较，不另行产生部署证书。')
    h('9 3 随机合法提案与真实角色参与',2)
    rows=[]
    learned=data['random_proposals']['original_mappo_results']
    random=data['random_proposals']['results']
    for label,results,keys in (('MAPPO',learned,tuple('fresh:history:'+s for s in seeds)),
                              ('随机',random,('random_0','random_1','random_2'))):
        for key in keys:
            r=results[key];s=key.split(':')[-1]
            rows.append((label+' '+str(s),str(r['adopted'])+'/320',str(r['harmful_adopted']),
                         num(r['mean_paired_percentage_gain'])+'%',str(r['execution_failures'])))
    table('表 7 相同门控下的已学习与随机合法提案',['提案','采用回合','有害采用','平均配对收益','失败'],rows,[4.4,3.3,3.0,3.3,2.4])
    p('随机提案不加载 actor checkpoint，三个车的随机流独立，且不读取真实隐藏参数与未来创新。'
      '比较用于检验收益是否主要来自门控搜索合法动作；不因原始策略经过训练就预设它优于随机提案。'
      '该开发对照与主确认集不同，不能据其结果改变主研究的候选、门槛或证书。')
    p('在此开发集上，三组已训练提案的最小平均收益为 '
      +num(min(learned['fresh:history:'+s]['mean_paired_percentage_gain'] for s in seeds))
      +'%，三组随机提案的最大平均收益为 '+num(max(r['mean_paired_percentage_gain'] for r in random.values()))
      +'%。两组采用率不同，该比较不构成同采用率优势证明，也未把 PPO 更新与继承初始化的贡献单独分离。')
    rows=[]
    for s in seeds:
        r=data['deployed_roles']['results'][s]
        rows.append((s,*[str(v) for v in r['executed_differs_from_baseline']],str(r['multi_role_execution_steps'])))
    table('表 8 开发集上实际偏离基线的角色控制步',['模型','前车','汇入车','后车','至少两角色同时改变'],rows,[1.8,3.0,3.0,3.0,5.6])
    p('角色计数来自完整 320 初态开发集，检查活动车辆实际执行命令与同一状态下基线命令的差异。'
      '所有角色均有多个提案和非零实际控制改变，结合训练审计支持真实全角色 MAPPO；'
      '它仍不单独证明合作的因果增益或集中式 critic 相对 IPPO 的优势。')
    h('9 4 保留的失败研究与开发数据',2)
    old=data['prior_attempt4']['certificate']
    p('前一独立研究完成 37,888 次 SUMO，但未通过整组风险要求。其混合风险上界为 '
      +join(num(100*old['primary_risk']['members'][s+':mixture']['pointwise_risk_upper'])+'%' for s in seeds)
      +'，失败与全部原始记录均保留。新门控的开发依次包括 644 次历史验证、276 次复用初态相位消融和 1,280 次新的 320 初态功效试验。'
      '已知反例集合用于开发，不能将其损害减少当作独立泛化收益。')
    recovery=data['prior_paired_recovery']
    p('前一研究的 3,072 个配对门控消融结果亦全部保留：新增耦合预测路径没有减少确认集损害，平均收益反而分别下降 '
      +join(num(-recovery['results'][s]['mean_paired_gain_difference_coupled_minus_original']) for s in seeds)
      +' 个百分点。该比较保留先前 1,375 条持久记录并实际补完 1,697 次缺失运行；'
      '原中断进程退出状态及丢失的在途尝试总数未知，不将它们改写为成功完成。')
    p('样本量由读取功效试验效果前冻结的 12 设计规则确定。规则在经验分层重采样及规定损害压力假设下，'
      '选择 4,096 校准与 1,024 确认；其模拟通过概率下界不是对真实研究成功率的保证。'
      '扩大样本量是推断设计变化，不是算法改进，也没有降低原风险、采用率或收益门槛。')
    h('10 计算代价、复现与落地边界')
    rows=[]
    for s in seeds:
        r=summary[s];rows.append((s,str(r['callbacks']),num(r['callback_p95_ms']),num(r['callback_max_ms']),str(r['callback_over_500ms'])))
    table('表 9 目标门控的完整实际在线回调',['模型','回调数','P95 毫秒','最大毫秒','超过 500 毫秒'],rows,[2.4,3.4,3.2,3.6,3.8])
    figure('formal_callback_cost','图 3 各方法完整在线回调耗时；共享桌面负载条件下测量，不构成硬实时证明')
    p('实验桌面为 Ryzen 5 9600X、约 64 GB 系统内存及 RTX 5060 Ti 16 GB。'
      '计时包含实际推理、物理验证、所有预测路径及历史拟合，未计通信延迟，也未用超时截断掩盖慢调用。'
      '仿真按同步控制周期推进；P95 和最大值必须与 500 ms 周期共同检查。'
      '共享负载、不同采集批次及系统调度影响耗时，不以非配对批次推断因果加速。')
    p('代码、登记与固定权重的公开入口为 https://github.com/Leo-7GOAT/CoopGate；本轮归档含实际退出凭据及逐文件 SHA256，公开交付状态以仓库清单与正式 Release 资产为准，上传中的草稿不计为公开完成。'
      '完整诊断重新计算所有方法统计，并检查零采用轨迹、首次采用前缀及随机弃权分支与各自实际基线或名义运行一致。'
      '正式样本不参与重新选阈值，未来修改必须单独验证。')
    h('11 讨论与结论')
    p('本文提供可复核的联合学习与传统控制协同实现，并将“能否采用”拆分为物理恢复、有限性能见证与采用后统计验证。'
      '整组固定模型的最终判断为'+verdict+'。其工程意义取决于实际收益、逐车损害、采用率与计算成本是否共同满足需要，'
      '不能以零碰撞、真实训练或仿真数量替代其中任何一项。')
    p('当前边界包括固定三车与路线、已知执行器类别、精确同步观测、有限预测假设及共享集中式门控；'
      '未覆盖视觉感知、自由横向规划、通信不确定性、真实传感器噪声和车辆实机。'
      '历史辨识反例说明未来扰动覆盖仍是理论缺口；原始策略跨种子优势和集中 critic 的因果价值也需进一步证据。'
      '论文不以这些未验证部分作为已实现贡献。')
    h('参考文献')
    refs=[
        '[1] Yu C, et al. The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. NeurIPS, 2022. https://arxiv.org/abs/2103.01955',
        '[2] Bastani O. Safe Reinforcement Learning with Nonlinear Dynamics via Model Predictive Shielding. https://arxiv.org/abs/1905.10691',
        '[3] Zhang W, Bastani O, Kumar V. MAMPS: Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding. https://arxiv.org/abs/1910.12639',
        '[4] Wabersich K P, Zeilinger M N. A predictive safety filter for learning-based control of constrained nonlinear dynamical systems. Automatica, 2021. https://arxiv.org/abs/1812.05506',
        '[5] Laroche R, Trichelair P, Tachet des Combes R. Safe Policy Improvement with Baseline Bootstrapping. ICML, 2019. https://proceedings.mlr.press/v97/laroche19a.html',
        '[6] Cho B M, et al. CSPI-MT: Calibrated Safe Policy Improvement with Multiple Testing for Threshold Policies. 2024. https://arxiv.org/abs/2408.12004',
        '[7] Banerjee A, Rahmani K, Biswas J, Dillig I. Dynamic Model Predictive Shielding for Provably Safe Reinforcement Learning. NeurIPS, 2024. https://papers.nips.cc/paper_files/paper/2024/file/b589d92785e39486e978fa273d0dc343-Paper-Conference.pdf',
        '[8] Frauenknecht B, et al. Uncertainty-Aware Predictive Safety Filters for Probabilistic Neural Network Dynamics. Reinforcement Learning Journal / RLC, 2026. https://arxiv.org/abs/2604.26836',
        '[9] Brorholt A H, Larsen K G, Schilling C. Compositional Shielding and Reinforcement Learning for Multi-Agent Systems. AAMAS, 2025. https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p399.pdf',
        '[10] El Yamani Y, Krasowski H, Vanneaux E. Decentralized Safe Multi-Agent Reinforcement Learning via Predictive Shielding. Preprint, 2026. https://arxiv.org/abs/2609.07618']
    for ref in refs:
        paragraph=p(ref);paragraph.paragraph_format.first_line_indent=Pt(0)
        for run in paragraph.runs:run.font.size=Pt(9)
