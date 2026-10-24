# 原作者基线接入：已验证的接口问题与复现边界

2026-10-24。本检查服务于真实文献对照，不替换已冻结的八方法共享证据实验。
从 GitHub 连接读取作者仓库，固定提交与文件 Git blob；执行前验证原始字节。
第三方源码保存在仓库外的本地 `papper/reference_sources`，公开仓库只保存检查器与结果。
未启动原论文全量训练、绘图或任何新的正式 SUMO 实验。

## DMPS：规划器观测必须随评估环境重置

原代码提交为 `f37df2d3900cfdc330ebb39e9b89d05dd9b85a7e`。
[main.py](https://github.com/arkobnrjee/dmps/blob/f37df2d3900cfdc330ebb39e9b89d05dd9b85a7e/main.py)
通过 `MCTSTrainer` 执行 DMPS；
[mcts_trainer.py](https://github.com/arkobnrjee/dmps/blob/f37df2d3900cfdc330ebb39e9b89d05dd9b85a7e/mcts_trainer.py)
第 246 行仅调用 `test_env.reset()`，规划器另有自己的 `obs` 缓存。

用未修改的 [mcts.py](https://github.com/arkobnrjee/dmps/blob/f37df2d3900cfdc330ebb39e9b89d05dd9b85a7e/mcts.py)
和确定性标量环境执行四项检查：首次仅重置环境，首个提案需要恢复规划时，出现
`TypeError: object of type 'NoneType' has no len()`；规划器自行 reset 后相同调用成功；
已有运行后的环境单独 reset 可留下旧观测；预测检查能够恢复完整状态及步数。
这定位了复现所需的重置同步契约，不等于原论文全部实验失败，也没有修改上游源码。

后续接入应使用 `test_planner.reset()` 同步两者，并明确披露这一兼容性修正。
需要继续验证真实环境状态克隆、随机数、执行器记忆、动作缩放及训练/测试分离。
当前三车控制器不能仅添加一个有限候选搜索就称为 DMPS：其
[原论文](https://papers.neurips.cc/paper_files/paper/2024/file/b589d92785e39486e978fa273d0dc343-Paper-Conference.pdf)
包含恢复规划、价值网络及训练交互，完整方法比较必须计入训练与规划预算。

证据：[四项接口检查](../artifacts/literature_reproduction/dmps_reset_contract.json)。
入口：[原源码检查器](../tools/audit_dmps_source_contract.py)，参数 `--source` 指向固定
Git blob `68a83a6eaf518cc4ed7d2f07ccf7cd7f7ccc686c` 的 mcts.py，`--output` 必须为新文件。

## CSPI-MT：分数的点质量与阈值相等不能忽略

原代码提交为 `b5af605a38a7866c52745c153e02cb4bf049ea01`，文件为
[Figures_2_and_3_for_KDD.py](https://github.com/facebookresearch/CSPI-MT/blob/b5af605a38a7866c52745c153e02cb4bf049ea01/Figures_2_and_3_for_KDD.py)。
只提取原 `PDE` 函数 AST 执行，没有执行其顶层大规模实验，也没有更改函数。

固定 256 个样本，真实潜在结果为 Y(0)=0、Y(1)=1，已知倾向概率 0.5，结果模型精确。
这样双稳健处理效应分数恒为 1，政策差可直接用两个阈值指示函数的差独立计算。
无阈值重合时，原函数的点估计和协方差均与独立计算一致。两个反例为：

| 分数 | 基线阈值 | 候选阈值 | 实际政策价值差 | 上游估计 |
|---:|---:|---:|---:|---:|
| 全为 0 | 0 | 0 | 0 | 1 |
| 全为 0.5 | 0 | 0.5 | 0 | -1 |

原因是原函数以双端包含的区间表示政策分歧；在 `1[S>=c]` 约定下，分歧应直接由
两个指示函数相减得到。连续分数下端点概率可为零；不能据此推断原连续分数实验错误。
但离散化、裁剪或饱和分数可能有点质量，接入本项目之前必须处理并披露边界约定。

[原论文第 2、4、5 节](https://arxiv.org/html/2408.12004v1)使用独立单位的处理分配、
重叠性、结果模型、阈值选择与独立测试。将一条闭环轨迹内的逐步状态当作 IID 单位，
或将固定扰动路径的均值下界命名为 CSPI，都不能构成复现。可考虑以完整回合作为单位、
在回合开始前分配完整控制器的实验，但这与在线逐步门控是不同的部署决策，必须分别评价。

证据：[阈值边界检查](../artifacts/literature_reproduction/cspi_threshold_contract_with_endpoint.json)。
入口：[原估计器检查器](../tools/audit_cspi_source_contract.py)，固定 Git blob 为
`84323a8cf90351cf69fbbc59300523ddca54f625`。初次仅等阈值检查的记录也保留，未覆盖。

## 尚未完成的对照要求

本次没有新增学习收益、安全率或风险证书。原始方法复现仍需先通过这些接口检查，
再运行其原训练/估计流程，记录必要修改，并在匹配信息、训练预算与测试分布下比较。
当前 MPS/MAMPS 和 advantage-LCB 继续标为适配器；共享证据延迟不能代替独立部署延迟。
冻结实验及其统计验收不因这些检查而改变。
