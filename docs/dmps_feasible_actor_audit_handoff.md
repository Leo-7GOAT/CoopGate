# 可执行区间 actor 的一次性原生退出与审计衔接

训练及科学审计在启动前固定。新增操作入口复用已验证的 Windows 原生句柄
等待函数及成功退出回执函数，code object 不变；仅在私有 globals 副本提供
本实验的输出目录、已冻结审计器和原生命令身份检查，不修改原模块全局变量。

新命令必须真实包含 train_dmps_feasible_actor.py，且 PID、父进程、创建时间
与本实验启动记录匹配；没有伪装成旧命令来通过身份检查。原等待函数继续在
进程存活时保留实际句柄，以实际退出码和完整数量回执为准；非零退出保留
原前缀，不因观察超时重启。成功才调用本实验已冻结的全量物理/策略审计。

独立 binding 在采集目录外保存新入口、原等待函数、本实验审计器与实际
protocol/native_start 的哈希。原 observer 记录的 observer_source_sha256
仍指原等待函数源码；本 binding 明确补全新的分派/审计依赖，不能把二者混淆。
它不改变训练、测试、随机流、阈值或统计，也不是周期任务。

两项检查验证原函数 code object/原模块不变，以及新命令、PID/父进程和创建
时间均须匹配。实际附着与最后审计是否完成由原生观察文件和最终核验报告判断，
不能由这些检查预先宣布策略有效。

- [新操作入口](../tools/watch_dmps_feasible_actor.py)
- [已验证的原生等待实现](../tools/watch_dmps_effective_action.py)
- [新实验冻结审计](../tools/verify_dmps_feasible_actor.py)
