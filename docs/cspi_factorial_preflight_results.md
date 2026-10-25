# 原作者 CSPI 补充分析：执行前检查通过，未采集正式效果

补充计划在父因子实验尚无 execution/blocks/analysis 目录时独占生成，SHA256：
`01f02985ab61ba5af2e71f513cfd429acc879f8446963c8b12d48edd7badddb1`。
冻结的是完整 640 个父 cluster 的分配、四组切分、来源、五个对照及错误处理。
它不修改父计划或正在运行的风险/OOD/训练任务。

在独立 baseline Python 环境中，新检查与原支持适配器共 **18 项通过**。其中
使用固定的合成分支日志实际调用了原作者 cutoff_selection、safety_test_single
和 safety_test；估计均值/协方差由已有独立 DR 实现交叉核对。这只是接入检查，
不是交通效果或新的风险证书。原文件字符串的四项 escape deprecation 警告
保留，没有为消除警告而修改原作者函数。

检查覆盖固定完整切分、不从结果派生随机分配、拟合只消费 fit 标签、预测点不
改变拟合参数、采用前失败保留、失败参考不可评估、团队改善不能掩盖个车退化、
缺失/非有限配对不删除，以及随机日志不能识别配对个体损害的反例。

另在冻结的 SUMO 运行环境实际尝试导出入口：它在缺少原生成功终止回执时按
预期拒绝，未创建 data 目录，也没有读取部分轨迹。该拒绝检查不启动父仿真，
不构成采集失败或需要绕过的限制。两个环境分别检查补充计划身份一致；原有
联合风险、公平对照、运行中 OOD、完整因子四份方案的源码/运行身份仍全部通过。

只在独立 baseline 环境增加了 pytest 8.3.5 及其测试依赖，未更改冻结仿真环境。
正式统计执行时还会记录实际 Python 与科学计算包版本。

下一步必须先完成原 3,200 次因子执行、原生退出与全量审计并提交，随后导出
随机观察日志，再运行原 CSPI/CSPI-MT。现在不能报告比较结果或通过率。

- [预先固定的完整协议](cspi_factorial_prospective_protocol.md)
- [冻结的分配与切分](../artifacts/literature_reproduction/cspi_factorial_prospective/plan.json)
- [缺少终止回执时的实际拒绝检查](../artifacts/literature_reproduction/cspi_factorial_preconditions_check.json)
