# 项目名称
基于类脑强化学习的车道保持控制研究

# 研究目标
在自动驾驶仿真环境中（highway_env），实现稳定的车道保持控制策略，
并探索类脑神经网络（RWTA / SNN）在强化学习中的鲁棒性优势。

# 方法概述
本文在策略梯度框架（PPO）下，构建两类策略模型进行对比：

1. Baseline：
   - 使用传统人工神经网络（MLP）
   - 标准强化学习方法

2. Ours：
   - 使用类脑模型（RWTA / SNN）
   - 引入动态熵调节、课程学习（curriculum learning）
   - 强化鲁棒性

# 核心研究问题
1. 类脑模型是否可以提升自动驾驶策略的稳定性？
2. 在复杂交通环境（dense traffic）下是否更鲁棒？
3. 在噪声 / 执行器故障下性能如何？

# 实验任务
- highway / merge / roundabout 场景训练
- standard / dense traffic 测试
- 鲁棒性评估（noise / failure）

# 评价指标
- mean_steps（平均存活时间）
- max_steps（最大步数）
- collision_rate（碰撞率）
- success_rate（成功率）

# 当前实验进展（摘要）

在 highway_standard 场景下：

- baseline 最大 step：150
- ours 可达到：200 step

关键改进：
- 提升 gamma（0.996 → 0.998）
- 提升 GAE lambda（0.98 → 0.99）

效果：
- 成功率提升
- 最大存活时间达到上限

详细实验见 RESULT.md