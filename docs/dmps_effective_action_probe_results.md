# 执行语义输入训练接入：128 次真实交互通过核验

按提交 fdd1ba3a3 固定的实现和探针协议，工具会话 66751 实际退出 0。
原生进程 54428 完成 128 次 SUMO 决策、640 个物理子步；全部物理重建、
实际观测/奖励/吸收标记和五个经验数组核验通过。137 行回放中只有 128 行是
实际执行，另外 9 行为原算法的拒绝提案惩罚。

原更新器的 critic 各更新 44 次，actor 更新 22 次。输入钩子调用分别为
critic1=90、critic2=44、critic1_target=44、critic2_target=44，包含目标自举
和 actor 更新，未只修改部署时输入。原作者函数体与全部绑定源码身份未变。

两个完整训练回合均在 50 步截止且未完成，最后一条为 28 步预算前缀。
这次探针只证明训练接入可执行，不能说明策略有效或任务已改善；没有学习
效能测试、额外风险证书或 MARL 结论。探针不保存供长训练继承的检查点。

下一阶段按相同预定配方重新初始化，使用旧名义配方相同的随机流、奖励及
32,768 次真实决策预算；只改变四个 critic 的动作输入表示。新测试保留所有
失败，与旧策略及传统参考逐条件对照，结束并全量审计前不报告修复效果。

- [固定实验协议](dmps_effective_action_protocol.md)
- [完整物理及回放审计](../artifacts/literature_reproduction/dmps_effective_action_native_probe/audit.json)
- [实际退出回执](../artifacts/literature_reproduction/dmps_effective_action_native_probe/observed_execution_exit.json)

完成回执 SHA256：`ab40c0ad0b9504364dd1db8e4ebb2ca879bd1837c36329a3d1c815c023259390`。
