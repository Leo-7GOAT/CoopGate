# 可执行区间 actor：128 次真实交互接入核验

按提交 ddfdaca1f 固定的实现、协议和审计，工具会话 20284 实际退出 0。原生
进程 35572 完成 128 次 SUMO 决策、640 个物理子步；实际观测、原奖励、吸收
标记与五个回放数组全部核验。146 行回放中 128 行来自实际执行，18 行是
原作者对被拒提案的合成惩罚，二者分开记录。

在线/目标 actor 输出钩子分别调用 118/44 次，critic 各更新 44 次，actor 更新
22 次；critic1/critic2/两目标 critic 钩子分别调用 116/44/44/44 次。没有只改
部署输出而遗漏 target actor。全部绑定源码和原作者四文件身份保持。

两个 50 步完整训练回合均未完成任务，最后为 28 步预算前缀。这次探针只验证
接入，不能用它宣称学习能力、收益或安全得到改善。没有保存供正式配方继承的
权重，也没有执行新方法的正式效能测试。

另有 12 项检查通过，涵盖活动坐标梯度、端点动作支持、完成车辆巡航、float32
边界、参数/优化器身份、500 个封存原生状态的解码、新审计钩子要求，以及用
父最终检查点重放原 actor/MCTS。最后两条完整原生轨迹的选择重放不新增 SUMO
样本，改动已记录执行动作会被拒绝。

下一阶段按已固定协议重新初始化，训练预算与父实验相同，独立新鲜三方测试
全部保留。此探针不替代最终任务完成检验、总体风险验证或公平方法族比较。

- [固定协议](dmps_feasible_actor_protocol.md)
- [探针实际退出证明](../artifacts/literature_reproduction/dmps_feasible_actor_native_probe/observed_execution_exit.json)
- [全部物理与回放核验](../artifacts/literature_reproduction/dmps_feasible_actor_native_probe/audit.json)

完成回执 SHA256：`55bbde54474a8ad086526c42832f8aed53de8b1a14e8a5675dd2687482ab8b2b`。
