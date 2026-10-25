# 课程初态已在原生 SUMO 核验，尚不是学习效果

按提交 8f67fa9f 的协议完成四阶段、每阶段两个固定种子的全部八个条件。执行工具
会话 50046 和独立复核会话 80675 均正常退出。实际 312 个控制决策、1,560 个
物理子步的执行器、名义预测与原生 SUMO 状态重建一致，没有碰撞或路由几何违反。

所有新初态均在实验记账时间零点之前通过原生车辆定位和速度设置实现。逐车位置、
速度及重建距离偏置与预定值在 1e-8 容差内吻合；没有在执行中注入状态。记录了
原生仿真时间原点、目标初态、实际初态及课程阶段。

两个 near_goal 条件均在持续最大允许制动及共同到达后巡航规则下，完成了真实
三车任务。其余六个条件按本探针的制动控制停止未完成，完整保留。由此确认可以
在实际 SUMO 中提供三车共同完成的训练数据，而不是在奖励或日志中合成成功。

这次没有训练网络，没有使用原 MCTS 策略作性能比较，也没有新增风险证书。
较长训练配方的 0/10 结果保持不变；下一步必须另行冻结课程训练预算和独立测试，
再判断任务完成率及相对传统基线的收益是否真正改善。

- [运行前协议](dmps_curriculum_probe_protocol.md)
- [完整执行](../artifacts/literature_reproduction/dmps_curriculum_native_probe/complete.json)
- [独立原生位置与物理子步复核](../artifacts/literature_reproduction/dmps_curriculum_native_probe/verification.json)
- [初态实现](../src/coopgate/dmps_curriculum_initials.py)

完成回执 SHA256：`f0cb25059ed571d03bbe8e385c357e22c6104012cec639956e78af4ed144b78d`。
