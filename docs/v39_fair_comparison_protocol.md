# V39 公平对照预注册协议

本文件定义正式实验前的比较合同，不报告效果结果。选择器位于
`src/coopgate/fair_comparison_v39.py`，真实证据生成器位于
`src/coopgate/fair_runtime_v41.py`，持久化执行与审计位于
`tools/run_fair_runtime_v41.py`，正式执行入口为 `tools/execute_fair_comparison_v41.py`。

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

预注册方法固定为以下八种，不能看结果后删减：

1. `proposed_bank_performance_gate`：实际 V35
   `RecoveryJointGate(..., "bank_performance")` 及恢复闭环，是待检验的 proposed method；
   finite-bank margin 仍不冒充校准概率。
2. `always_learned`：始终执行同一个学习提案，负对照；允许暴露物理不可行。
3. `always_baseline`：始终执行同一个恢复基线，安全负对照。
4. `uncalibrated_confidence_threshold`：固定普通置信度阈值；没有校准声明。
5. `existing_calibrated_gate`：完整复现项目既有校准不等式，但 V35 闭环是 adapter。
6. `advantage_lcb_adapter`：固定路径 bank 上的 advantage LCB；不是原始 CSPI 的观测
   因果估计器或其定理复现。
7. `mps_adapter`：共同物理判据下的 all-or-none MPS 适配器。
8. `mamps_adapter`：共同物理判据下的单调逐智能体 MAMPS 适配器。

代码为每个方法分别保存 `decision_rule_status` 和 `closed_loop_status`。`full_reproduction`
只表示项目内已明确写出的决策规则逐式复现；凡替换原论文优化器、动力学、恢复集合、
信息结构或调用频率的实现均标为 `adapter`。负对照标为 `protocol_control`，不伪装成
文献方法。

## 3. 共同证据包与相同预算规则

每个决策状态只构造一次不可变 `DecisionEvidence`，所有八个规则消费完全相同的字节：

- 同一 observed kinematics、上一条实际命令、决策序号、执行器类别和历史区间；
- 同一个固定训练策略给出的联合学习提案；
- 同一个 V35 恢复基线联合命令；
- 同一置信度、预测成本差、预先冻结的校准分位数；
- 同一个 33 路径 advantage 样本向量；
- 三智能体全部八个学习/基线 mask 的共同物理检查；
- proposed bank gate 的实际命令、候选数、完整预测/replay identity 及 forecast-call 计数；
- 同一个 500 ms 截止和同一个工作账本。

`ProtocolSpec` 在构造时必须提供恢复基线实现、proposed bank-gate 声明的实现依赖集合、
校准文件、置信度定义、固定预测 bank 和物理 oracle 的 SHA-256；每个 case 还必须绑定
实际 learned policy 的 SHA-256。没有这七类真实 hash 就不能创建 evaluator，防止只冻结
阈值却在执行前后替换模型或证据生成器。bank hash 覆盖 V35 实际执行的缩放后 `±0.1`
ordered float64 tape bytes，而不是未缩放的 legacy `±0.5` 源 bank。

正常完成的证据包必须恰好记录 1 次观测、1 次学习提案、1 次基线提案、33 对原始学习/
基线预测（66 次 kernel 调用）和 8 个 mask 检查，另外计入 proposed gate 的候选搜索。
普通置信分数固定为 `sigmoid(nominal paired team-cost gain / 0.05 seconds)`，名义预测
不完整时置零；此分数不是校准概率。尺度与公式在正式结果产生前冻结。

同一 case 内，只有观测字节及完整已执行命令历史完全一致，才能共享已测量的不可变
证据包，包括原始计时、超时和失败。每种 selector 归因同样的证据成本，实际构造次数
与缓存复用次数另计；always-baseline 也消费该包。该设计比较共享证据下的完整 selector，
不测量各方法独立部署时的延迟。不同后续状态需要分别计算，计算次数不强行相等。
MPS/MAMPS 从预计算的 mask 表选择，不额外调用 oracle。

500 ms 是在计算阶段之间检查的协作式截止，单次 kernel 可以越过它；超时结果被丢弃，
所有消费该包的方法统一退回已计算的基线，并保留 timeout 失败标记。它不是硬实时上限。
重放使用记录的单调时钟检查点，能验证截止决策，不能独立证明原始操作系统耗时。

等 shadow computation 和同一 raw 学习提案不代表可选 command 集相同：proposed V35
bank gate 会投影/修复并搜索最多 28 个候选 command，而普通阈值、MPS、MAMPS 只在 raw
学习/基线 mask 中选择。因此这个协议比较的是八个**完整部署 selector**，不能表述成
“只隔离门控判据”的纯 ablation。若论文需要该因果结论，必须另加 projected-candidate-
matched ablation；当前不能靠改措辞获得。

“同一观测”指同一决策状态下的反事实比较共享完全相同的观测字节和信息字段。不同
控制器实际执行后会进入不同后续状态，不能谎称后续数值观测仍然相同；它们只共享同一
观测**模式和可用信息集合**。

## 4. 完全配对的实验单位

每个 `CaseIdentity` 同时绑定：策略 ID、场景、初态、动力学类别、外生扰动及三者字节
哈希。每个 case 必须恰好运行八种方法。方法次序只由 case 索引固定轮转并交替反向，
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

计划中的主样本不是固定配平 cell。它从冻结的 scenario × plant × disturbance × model
均匀混合分布中**有放回 IID 抽取** 640 个 cluster；每个 cluster 只含一个新初态和一个
类别抽样，然后与八种方法完全交叉。无拒绝采样、无事后补 cell、无按结果选择。因此
主 exact-binomial 目标是该预注册随机设计下的混合边际概率。若另报固定 cell 表，只能
作为次要描述，不能把异质的固定分层 cell 当成 IID 重复后套用 pooled binomial 界。

## 5. 预注册主指标与 usefulness 约束

主指标固定为：**在发生学习 adoption 的独立 cluster 中，harm 概率的同时单侧精确
上界**。总比较误差预算固定为 `0.05`，预先分给风险、adoption、gain 三个 family 的
比例分别为 `0.50/0.25/0.25`，再在七个非 baseline 候选内作 Bonferroni 分配；不能
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

## 6. Freeze-before-run 计划与执行骨架

`tools/freeze_fair_comparison_v39.py` 是单独的 freeze/check builder。它在任何运行前把以下
**实际字节**逐文件写入依赖 hash 表并在 `check` 时重算：

- V32 model plan 和 1802/1812/1822 三个 `policy.pt`；
- V13 calibration JSON 与 protocol JSON（只绑定 bytes，builder 不读取效果数值）；
- `confidence_v12.py`、`shield.py`、V34 baseline、V35 forecast/bank gate、V27 oracle；
- V20 固定 33-path bank 的名称与 float64 tape bytes；
- V39 核心、builder、测试和本协议。

V13 calibration 的 701/711/721 到 1802/1812/1822 映射仅供 comparator adapter 使用，
不把旧模型证书转移给新模型。真实 proposed method 固定为 V35
`RecoveryJointGate(model, profile, "bank_performance")`；它的实际 command、完整预测标志、
候选数、forecast-call 数和独立 replay identity 均进入共同 evidence packet。原要求的七个
参考方法全部保留后，再加入待检验方法，所以公平族必须是八种；只比较七个 reference
而漏掉 proposed gate 没有论文判别意义。

主计划固定为 640 个 IID cluster、每 block 4 个 cluster、160 blocks、八方法完全交叉，
即 5120 个 SUMO episode。每 episode 最多 50 个控制决策，共最多 256000 个 packet；
500 ms 检查阈值之和为 35.56 controller-hours，这不是实际运行时间的上界。调用上限为
8,448,000 对固定 path forecast（16,896,000 次 raw kernel 调用）、244,992,000 次
proposed-gate 内部 forecast call、2,048,000 次物理 mask check。共享包减少实际计算，
独立重放也有额外开销。实际调用数与 timeout 必须原样保留，不能按效果补跑。

builder 只有 `freeze` 和 `check`，没有 simulator launch 命令；`route_block` 只是无文件写入
的单决策路由骨架，用于证明一个冻结 case 的同一 evidence hash 能送入全部八个 selector。
V41 运行器已经实现从实际 V35 replay 构造 proposed 字段。正式使用顺序为：提交源码，
freeze/check 科学计划并提交，再 bind/提交执行依赖，最后 watch。独立开发种子
139881001 用于编译预热及 SUMO 集成测试，不进入正式效果样本。

每个回合先将结果与 trace 事务提交到 SQLite，再进行独立动力学递推和确定性决策重放。
物理审计不调用仿真的执行器函数；证据重放会再次调用同一控制器实现，因此不宣称是
独立实现的门控算法。重放失败保留未封存前缀，不覆盖、不自动重启、不替换样本。
仅当原生进程退出成功、5120 回合和 160 块完整、文件 hash 链一致，才允许最终汇总。

## 7. 当前边界

现阶段已有比较合同、真实运行器、持久化和汇总审计；开发集成检查不是正式效果结果。
当前工作没有：

- 读取 V29/V32/V35/V36/V38/V39 正在生成或已经封存的效果结果；
- 为任何 adapter 创建新的理论保证；
- 复现原始 MPS/MAMPS/CSPI 的完整系统；
- 将开发 SUMO 回合计入正式 640 个 cluster，或按开发效果调参；
- 改动任何冻结控制器、阈值或 artifact。

后续只有在新计划提交并冻结、case/cluster 哈希和 calibration bytes 全部预注册后，才能
另行启动数据收集。
