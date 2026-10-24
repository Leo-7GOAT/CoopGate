# V39 公平对照预注册协议

本文件只定义下一轮实验的比较合同，不读取任何结果、不生成实验 artifact，也不启动
SUMO。实现位于 `src/coopgate/fair_comparison_v39.py`，协议测试位于
`tests/test_fair_comparison_v39.py`。

## 1. 对 V35 现状的审计

V35 已经做对的部分是：五类规则使用同一个 V34 恢复基线、同一个执行器记忆区间、
同一个停车证书、同一组固定演员、同一初态/真实动力学/外生扰动，并且对方法顺序做
固定轮转和反转。V35 也准确披露了有限路径 margin 不是概率，MPS/MAMPS 只是共享
判据适配器。

V35 尚不能作为“等计算公平对照”的原因如下。

| V35 方法 | 原有实现 | 审计结论 |
|---|---|---|
| `baseline` | 共同 V34 恢复基线 | 可作为项目内部完整复现和负对照 |
| `mps_shared` | 学习联合动作不通过物理检查时整体退回 | MPS adapter；没有复现原优化器、原 benchmark 或原定理条件 |
| `mamps_shared` | 按事故关联 flag 单调移除学习智能体 | MAMPS adapter；同样不是原论文完整复现 |
| `nominal_performance` | 候选投影后比较一条名义完整续接轨迹 | 名义 rollout comparator；不是扰动风险保证 |
| `bank_performance` | 对候选逐个检查最多 33 条固定路径 | finite-bank comparator；margin 不是校准概率 |

原 V35 方法的昂贵工作量并不相同：基线不调用学习提案和性能预测；MPS/MAMPS 的
物理检查次数随分支变化；名义性能规则和 bank 规则的预测路径数、候选数也不同。
它们虽然共享实验条件，却没有共享完全相同的计算预算。因此 V39 不直接把 V35 的
运行时间差解释为算法优劣，也不把 `bank_performance` 改名成“calibrated gate”。

项目旧有 calibrated gate 的核心不等式是

\[
\widehat\Delta(x)+q_g\le -\epsilon,
\]

其中 `predicted_delta + calibration_quantile <= -minimum_gain`。V39 完整复现这个**决策
不等式**，但把它接到共同的 V35 恢复执行链后，闭环系统仍明确标为 adapter。旧门控
是 episode-level，V39 公平执行器按共同决策时刻调用，所以不能宣称旧闭环系统的完整
复现或直接继承旧证书。

## 2. 固定比较族和血缘标签

预注册方法固定为以下七种，不能看结果后删减：

1. `always_learned`：始终执行同一个学习提案，负对照；允许暴露物理不可行。
2. `always_baseline`：始终执行同一个恢复基线，安全负对照。
3. `uncalibrated_confidence_threshold`：固定普通置信度阈值；没有校准声明。
4. `existing_calibrated_gate`：完整复现项目既有校准不等式，但 V35 闭环是 adapter。
5. `advantage_lcb_adapter`：固定路径 bank 上的 advantage LCB；不是原始 CSPI 的观测
   因果估计器或其定理复现。
6. `mps_adapter`：共同物理判据下的 all-or-none MPS 适配器。
7. `mamps_adapter`：共同物理判据下的单调逐智能体 MAMPS 适配器。

代码为每个方法分别保存 `decision_rule_status` 和 `closed_loop_status`。`full_reproduction`
只表示项目内已明确写出的决策规则逐式复现；凡替换原论文优化器、动力学、恢复集合、
信息结构或调用频率的实现均标为 `adapter`。负对照标为 `protocol_control`，不伪装成
文献方法。

## 3. 共同证据包与严格等预算

每个决策状态只构造一次不可变 `DecisionEvidence`，所有七个规则消费完全相同的字节：

- 同一 observed kinematics、上一条实际命令、决策序号、执行器类别和历史区间；
- 同一个固定训练策略给出的联合学习提案；
- 同一个 V35 恢复基线联合命令；
- 同一置信度、预测成本差、预先冻结的校准分位数；
- 同一个 33 路径 advantage 样本向量；
- 三智能体全部八个学习/基线 mask 的共同物理检查；
- 同一个 500 ms 截止和同一个工作账本。

`ProtocolSpec` 在构造时必须提供恢复基线实现、校准文件、置信度定义、固定预测 bank 和
物理 oracle 的 SHA-256；每个 case 还必须绑定实际 learned policy 的 SHA-256。没有这
六类真实 hash 就不能创建 evaluator，防止只冻结阈值却在执行前后替换模型或证据生成器。

正常完成的证据包必须恰好记录 1 次观测、1 次学习提案、1 次基线提案、33 条预测
路径和 8 个 mask 检查。即使 always-baseline 不需要预测，它也执行并计入同样的 shadow
work；因此比较的是选择规则，而不是“某方法少算了东西”。MPS/MAMPS 从预计算的八个
mask 表中选择，不得额外调用 oracle。共同证据计算一旦超时，所有方法统一回退基线并
把 timeout 保留为失败事件，不能给任何方法单独延长预算。

“同一观测”指同一决策状态下的反事实比较共享完全相同的观测字节和信息字段。不同
控制器实际执行后会进入不同后续状态，不能谎称后续数值观测仍然相同；它们只共享同一
观测**模式和可用信息集合**。

## 4. 完全配对的实验单位

每个 `CaseIdentity` 同时绑定：策略 ID、场景、初态、动力学类别、外生扰动及三者字节
哈希。每个 case 必须恰好运行七种方法。方法次序只由 case 索引固定轮转并交替反向，
不由结果决定。

所有方法使用相同的：

- 初始状态、场景、真实动力学参数和外生创新 tape；
- 控制周期、episode 截止、单次决策 timeout；
- 失败、碰撞、中止及 contract violation 的保存规则；
- capped per-vehicle arrival time + `0.05 × command variation` 代价；
- adoption 定义：至少一个 active agent 的实际命令与共同基线相差超过 `1e-9`；
- harm 定义：控制器失败，或任一车辆成本/到达时间比配对基线差超过 `1e-9`。

若 gated 方法整个 episode 零 adoption，它的所有成本、到达时间和失败身份必须与配对
baseline 完全相同，否则协议直接拒绝结果。always-learned 是故意保留的不安全负对照；
其物理不可行提案不被静默改写成安全门控结果。

多个场景/动力学 cell 可以共享一个 `independent_cluster`，但统计汇总先在 cluster 内
聚合，再计算样本量；重复 cell、三模型或四种动力学不会被伪装成新增独立初态。

## 5. 预注册主指标与 usefulness 约束

主指标固定为：**在发生学习 adoption 的独立 cluster 中，harm 概率的同时单侧精确
上界**。总比较误差预算固定为 `0.05`，预先分给风险、adoption、gain 三个 family 的
比例分别为 `0.50/0.25/0.25`，再在六个非 baseline 候选内作 Bonferroni 分配；不能
把同一个 `0.05` 重复用于三个筛选步骤。零 adoption 的风险上界记为 1，而不是 0 或
空白胜利。

为防止 always-baseline 或近似 always-baseline 通过“从不采用所以从不出错”获胜，方法
进入主排名前必须同时满足：

1. adoption probability 的同时单侧 Clopper–Pearson 下界不低于 `0.20`；
2. 对每个独立 cluster 先计算
   `100 × (cluster baseline mean cost - cluster method mean cost) / cluster baseline mean cost`，
   再对这些配对百分比作同时单侧 t 下界；该下界不低于 `0.5%`。

gain 下界使用固定 Bonferroni 误差分配下的单侧 paired t 区间；它在独立 cluster 的
百分比收益服从正态时具有对应的有限样本解释，非正态时只作预注册诊断，不包装成分布
无关定理。代码同时报告 `ratio_of_means_gain_percent` 作为描述量，但它不参与资格判断：
先平均绝对 gain 再除以平均 baseline，会把随机且异质的 baseline 分母当作固定量，通常
不等于平均配对百分比。harm/adoption 界仍为精确二项界。满足 usefulness 约束的方法先
最小化主风险上界，再最大化 gain 下界，再最大化 adoption 下界，最后按冻结方法顺序
打破完全相等。若没有方法满足约束，报告 `selected_method = null`，不能退回
always-baseline 并宣布“最安全”。

always-baseline 始终显示在结果表中，但 `eligible=false`。所有中止、timeout、碰撞、原始
封顶成本以及 baseline 失败均保留；不得只在双方完成的子集上建立主结论。

## 6. 当前边界

V39 现阶段完成的是可执行的公平比较合同、选择器和独立汇总器，不是实验结果。它没有：

- 读取 V29/V32/V35/V36/V38/V39 正在生成或已经封存的效果结果；
- 为任何 adapter 创建新的理论保证；
- 复现原始 MPS/MAMPS/CSPI 的完整系统；
- 生成新独立初态、校准样本或 SUMO episode；
- 改动任何冻结控制器、阈值或 artifact。

后续只有在新计划提交并冻结、case/cluster 哈希和 calibration bytes 全部预注册后，才能
另行启动数据收集。
