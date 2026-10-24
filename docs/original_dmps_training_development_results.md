# 原作者 DMPS 训练流程：真实仿真小型接入已完成

在八个原生状态恢复案例完整通过后，按提交 7460f963 的固定协议执行原 MCTSTrainer。
工具会话 35152 正常退出。32 个真实训练控制决策、4 次梯度更新和另一个固定初态的
12 步评估前缀全部完成；共 220 个真实物理子步的独立执行器与运动学重算一致。
这两个前缀没有观察到路由几何违反或 SUMO 碰撞，不据此推断总体安全率。

## 核实了什么

原作者 mcts.py、mcts_trainer.py、models.py、utils.py 按固定 Git blob 校验后直接
加载，未修改函数体。训练使用原回放缓冲区、双 critic 更新、延迟 actor 更新和
Polyak 目标网络更新。记录包装器只观察原 take_step 输入输出。

回放共 37 行，其中 32 行对应真实 SUMO 控制决策，另外 5 行是原训练器为被拒绝
提案加入的吸收惩罚样本。独立从逐步记录重建全部五个回放数组，与保存值逐项完全
一致；没有将这 5 行合成样本当作物理执行或独立场景。

| 参数组 | 相对初始权重的 L2 变化 |
|---|---:|
| actor | 0.10453748 |
| critic1 | 0.17327804 |
| critic2 | 0.17155419 |
| actor target | 0.00079737 |
| critic1 target | 0.00100135 |
| critic2 target | 0.00100180 |

全部权重有限。以上变化是初始/结束检查点独立计算的数值差，不是性能提升指标。
保存了两个检查点、全部原回放数组、32 条真实训练调用、12 条评估调用和完整实际
轨迹。原 evaluate 的 env-only reset 问题通过显式 test_planner.reset 的评估入口
处理，这一接入差异在协议中披露。

## 尚未完成的工作

32 步和 4 次更新只足以检验学习流程，不能代表收敛训练、原论文完整复现或竞争力。
训练只有一个固定初态，评估也只有另一个前缀，没有进行收益排名或统计显著性检验。
规划仍使用真实模拟器状态、执行器记忆和未来扰动带，因此具有特权信息；独立进程
载入很慢，也不适合直接部署。下一步需要在统一可用信息的规划模型上训练，并按匹配
训练/规划预算完成独立测试，才可作为公平对照。现有冻结八方法实验未被替换。

原作者模块固定于 [DMPS 提交 f37df2d](https://github.com/arkobnrjee/dmps/tree/f37df2d3900cfdc330ebb39e9b89d05dd9b85a7e)。

- [运行前协议](original_dmps_training_development_protocol.md)
- [完成与文件哈希](../artifacts/literature_reproduction/original_dmps_training_development/complete.json)
- [独立回放、物理轨迹及权重复核](../artifacts/literature_reproduction/original_dmps_training_development/verification.json)
- [固定原版模块加载器](../tools/load_original_dmps.py)
