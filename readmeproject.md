# LANE: 自动驾驶场景强化学习项目说明

本 README 面向两类用途：

- 代码使用者：快速理解如何训练、测试、导出视频
- 毕业论文作者：系统说明本项目的强化学习建模、训练输入输出、实验指标、方法改进点，以及答辩时老师可能追问的技术细节

注意：仓库中很多历史参数名仍保留了 `gymip` 这一旧名称，但当前 LANE 主任务实际训练的是道路驾驶场景，不再是原始的倒立摆任务。

## 1. 项目概述

本项目将强化学习用于自动驾驶决策控制，目标是在不同道路场景中学习合理的离散驾驶动作策略。当前主要场景包括：

- `highway`：高速公路多车道跟驰/变道
- `merge`：匝道汇入主路
- `roundabout`：环岛进入与驶出

项目主要包含两套训练流程：

- `run_RL_base.py`：基线版训练器
- `run_RL_ours.py`：改进版训练器

其中改进版在基线 PPO 的基础上加入了：

- rollout 式采样与小批量更新
- GAE 优势估计
- curriculum learning 课程式训练
- 场景自适应稳定化参数配置
- warm start 热启动
- 更丰富的评测指标与鲁棒性测试

## 2. 项目核心文件

推荐先看以下文件：

- `env_lane.py`：自动驾驶环境封装，定义状态、动作、奖励、终止逻辑
- `run_RL_base.py`：基线训练主循环
- `run_RL_ours.py`：改进训练主循环
- `model_rwta.py`：RWTA / RWTAspike 策略网络
- `model_critic.py`：评论家网络
- `memory_lib.py`：基线版经验缓存
- `test_video.py`：录制测试视频
- `test_robustness.py`：鲁棒性评测
- `compare_experiments.py`：对比基线与改进版实验结果
- `checkpoint_utils.py`：checkpoint、日志、视频目录管理

## 3. 本项目为什么不需要“喂数据集”

这是强化学习项目，不是监督学习项目，因此它不依赖一个静态标注数据集。

监督学习通常是：

- 输入：固定数据样本 `x`
- 标签：人工标注目标 `y`
- 学习目标：拟合 `x -> y`

而本项目的强化学习是：

- 智能体在仿真环境中观察当前状态 `s_t`
- 根据策略网络输出动作 `a_t`
- 环境执行动作并返回奖励 `r_t`、下一状态 `s_{t+1}`、是否结束 `done`
- 训练时在线收集轨迹 `(s_t, a_t, r_t, s_{t+1}, done)`
- 再利用 PPO/GAE 对策略和价值网络进行更新

因此，本项目的“训练数据”不是预先存在的图片或表格，而是训练过程中由环境交互动态生成的轨迹数据。

这也是答辩中一个很常见的问题：

- 问：你没有数据集，模型怎么训练？
- 答：本项目采用强化学习，训练样本来自智能体与仿真道路环境的在线交互。每一次 episode 都是在环境中重新采样得到新的状态转移序列，而不是从静态数据集中读取样本。

## 4. 强化学习建模

### 4.1 马尔可夫决策过程

本项目可抽象为一个马尔可夫决策过程（MDP）：

- 状态 `s_t`：当前道路交通观测
- 动作 `a_t`：当前时刻选择的离散驾驶行为
- 奖励 `r_t`：环境反馈和奖励整形后的即时回报
- 状态转移 `P(s_{t+1}|s_t,a_t)`：由交通仿真器决定
- 策略 `pi(a_t|s_t)`：策略网络给出的动作分布

智能体的目标是最大化长期累计回报。

### 4.2 状态定义

本项目使用 `highway_env` 的 `KinematicObservation` 作为环境观测。

环境配置核对结果如下：

- observation type: `Kinematics`
- `vehicles_count = 5`
- 每辆车的特征数为 5
- 单步原始观测形状为 `(5, 5)`

具体特征为：

- `presence`
- `x`
- `y`
- `vx`
- `vy`

也就是说，单步观测本质上是 5 辆车的运动学特征矩阵。项目中会将其：

1. 展平为 25 维向量
2. 按 `(state + 1.0) / 2.0` 做归一化
3. 裁剪到 `[0, 1]`
4. 转成形状为 `(1, 25)` 的 `torch.FloatTensor`

所以，网络真正接收的输入是：

- 输入张量形状：`(1, 25)`
- 输入含义：当前交通局部观测的归一化向量

### 4.3 动作定义

本项目使用 `DiscreteMetaAction`，动作空间大小为 5。

实际动作映射为：

- `0 -> LANE_LEFT`
- `1 -> IDLE`
- `2 -> LANE_RIGHT`
- `3 -> FASTER`
- `4 -> SLOWER`

训练时：

- 策略网络输出动作分布
- 从分布中采样动作，用于探索

验证/测试/视频时：

- 默认使用 `argmax` 选最大概率动作
- 即采用更接近贪心策略的确定性执行方式

### 4.4 奖励定义

本项目不是直接使用环境原始奖励，而是在 `env_lane.py` 中做了奖励整形。

奖励由以下项组合而成：

- 原始环境奖励
- 高速行驶奖励
- 右侧车道奖励或路权相关奖励
- 存活奖励
- 稳定驾驶奖励
- 任务完成奖励
- 变道惩罚
- 连续变道惩罚
- 摇摆式变道惩罚
- 低速惩罚
- 停车惩罚
- 越界/离路惩罚
- 碰撞惩罚

并且不同场景下奖励重点不同：

- `highway`：强调稳定通行、高速、减少无意义变道
- `merge`：强调安全并线、及时汇入主路、避免长期犹豫或停车
- `roundabout`：强调安全让行、避免碰撞、正确驶出环岛

这是论文中非常重要的一点：本项目性能很大程度上依赖奖励设计，属于“任务建模能力”的一部分，而不仅仅是“网络结构能力”。

### 4.5 终止条件

一个 episode 结束的典型原因包括：

- 发生碰撞
- 环境自然终止或截断
- 达到最大步数
- 在 `merge` 或 `roundabout` 中完成场景目标

例如：

- `highway` 最大步数约为 150
- `merge` 最大步数约为 45
- `roundabout` 最大步数约为 90

## 5. 策略网络与价值网络

### 5.1 策略网络

训练脚本支持多种策略模型：

- `mlp3soft`
- `mlp3relu`
- `rwtaprob`
- `rwtaspk`
- `snnbptt`

当前自动驾驶实验中最常用的是：

- `rwtaspk`

其特点可以概括为：

- 输入 25 维状态
- 输出 5 维动作概率或动作分数
- 具有 RWTA/脉冲式响应结构

### 5.2 价值网络

`model_critic.py` 中的评论家网络输入状态，输出一个 5 维向量，而不是单一标量。

这意味着评论家网络学习的是“每个动作对应的价值估计”，而不是传统 PPO 中唯一的标量 `V(s)`。训练时会：

- 先得到评论家输出的 5 维价值向量
- 再与策略当前动作概率做加权，形成状态平均价值估计
- 对已执行动作的位置单独构造目标值

这是一处比较有辨识度的实现细节，答辩时很值得主动说明，因为它和很多教材里的“标量价值函数”写法不同。

## 6. 训练是如何进行的

### 6.1 基线版训练流程

基线脚本：`run_RL_base.py`

整体流程如下：

1. 初始化环境、策略网络、评论家网络、经验缓存
2. `env.init_train()` 重置环境
3. 读取当前观测 `observation`
4. 策略网络输出动作分布
5. 采样动作并执行 `env.make_action(...)`
6. 得到 `next_state, reward, done`
7. 将 `(s, a, logprob, r, s', done)` 写入 `MemoryBuffer`
8. 每累计一定步数后执行 PPO 更新
9. 每隔固定 episode 做验证
10. 若验证结果更优，则保存 best checkpoint

基线版有两个重要特点：

- 经验缓存是 `memory_lib.py` 中的 `MemoryBuffer`
- 更新频率是“每若干 environment steps 触发一次”

### 6.2 改进版训练流程

改进脚本：`run_RL_ours.py`

改进版的训练逻辑更完整，整体可以概括为：

1. 初始化环境、策略网络、评论家网络
2. 根据场景自动修正更稳定的训练超参数
3. 如果指定 `warm_start`，则加载已有 best checkpoint
4. 根据 curriculum 设置本轮噪声强度与熵系数
5. 进入训练 episode
6. 在线采样轨迹并写入 `RolloutBuffer`
7. 当 rollout 长度达到 `rollout_steps` 时：
   - 计算优势函数与回报目标
   - 按 mini-batch 进行 PPO 多轮更新
8. 周期性做验证评估
9. 若验证指标更优，保存 best 模型
10. 根据验证结果更新 curriculum

改进版的核心提升点包括：

- `RolloutBuffer` 替代简单缓存
- `GAE` 替代更粗糙的优势估计
- `mini-batch + multiple PPO epochs`
- 自适应 curriculum
- 不同场景下的稳定化参数配置

### 6.3 GAE 与 PPO 的专业解释

改进版中使用了广义优势估计 GAE。其思想是：

- 利用当前步与后续若干步的时序差分误差
- 在偏差和方差之间做折中
- 获得更平滑、更稳定的优势估计

可以在论文中写成：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * delta_{t+1} + ...
```

PPO 的核心思想是：

- 新旧策略之比不能变化过大
- 使用 clip 约束策略更新幅度
- 提高训练稳定性

可以概括为：

```text
r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
L_clip = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
```

如果老师问“为什么选择 PPO”，可以回答：

- PPO 比 REINFORCE 方差更低
- 不像 TRPO 那样实现复杂
- 在离散动作控制任务中训练稳定、工程可复现性较好

## 7. 改进版相对于基线版的主要差异

| 方面 | 基线版 | 改进版 |
| --- | --- | --- |
| 训练脚本 | `run_RL_base.py` | `run_RL_ours.py` |
| 经验组织 | `MemoryBuffer` | `RolloutBuffer` |
| 优势估计 | 基础优势/TD 风格 | GAE |
| 更新方式 | 定步触发更新 | rollout 达阈值再批量更新 |
| 小批量训练 | 不突出 | 支持 `mini_batch_size` |
| PPO epoch | 支持 | 支持且配合更合理的 rollout 结构 |
| curriculum | 无 | 有 |
| warm start | 无 | 有 |
| 稳定化 profile | 无 | 有 |
| 日志指标 | 较少 | 更丰富 |

## 8. 训练输入与输出到底是什么

这是论文和答辩中最容易被问的部分。

### 8.1 训练脚本层面的输入

从命令行角度，训练脚本接收的输入包括：

- 场景选择：`--road_scenario`
- 交通密度：`--traffic_level`
- 模型类型：`--model`
- 学习率：`--lr`
- 折扣因子：`--gamma`
- 熵系数：`--entropy`
- PPO 轮数：`--PPO_epochs`
- 裁剪系数：`--eps_clip`
- rollout 长度：`--rollout_steps`
- mini-batch 大小：`--mini_batch_size`
- GAE 参数：`--gae_lambda`
- curriculum 参数
- seed / 线程 / GPU 等系统参数

### 8.2 算法层面的输入

真正送进策略网络的输入是：

- 一个 `(1, 25)` 的状态张量

真正送进评论家网络的输入也是：

- 同一个状态张量 `(1, 25)`

### 8.3 算法层面的输出

策略网络输出：

- 5 维动作分布或动作分数

评论家网络输出：

- 5 维动作价值估计

环境输出：

- 下一状态 `s_{t+1}`
- 即时奖励 `r_t`
- 是否结束 `done`
- 额外统计信息 `info`

### 8.4 训练完成后的输出

训练会产生以下结果：

- 模型 checkpoint
- 文本日志
- 对比报告
- 鲁棒性评测结果
- 测试视频

## 9. 训练产物保存在什么地方

训练产物目录采用统一组织方式：

```text
training_runs/
  baseline/
    highway_standard/
      models/
      logs/
    highway_dense/
      models/
      logs/
    merge/
      models/
      logs/
    roundabout/
      models/
      logs/
  ours/
    ...
```

其中：

- `models/`：权重文件
- `logs/`：训练日志

RWTA/RWTAspike 的 checkpoint 通常会拆成多个文件，例如：

- `*_w_1.pt`：权重
- `*_b_1.pt`：偏置
- `*_m_1.pt`：连接 mask
- `*_w_2.pt` / `*_b_2.pt` / `*_m_2.pt`：冗余备份

评论家网络通常保存为：

- `critic_current_1.pt`
- `critic_best_1.pt`

## 10. 日志中常见字段说明

改进版日志中常见标签包括：

- `init`：新训练开始
- `resume`：从 checkpoint 恢复
- `arguments`：启动参数
- `profile`：自动稳定化参数修正
- `warm_start`：热启动加载来源
- `train`：训练 episode 结果
- `update`：一次 rollout 更新结果
- `val`：验证结果
- `val_save`：当前 best checkpoint 对应的验证结果
- `curriculum`：课程学习状态变化

其中 `train` 一般记录：

- episode 编号
- episode return
- step 数
- 碰撞情况
- 变道次数
- 噪声强度
- 当前 entropy
- 当前车辆数

## 11. 如何启动训练

### 11.1 单个命令训练

基线版示例：

```bash
cd /root/autodl-tmp/SVPG2023/LANE
python3 run_RL_base.py --model rwtaspk --road_scenario highway --traffic_level standard --cuda 0 --ignore_checkpoint
```

改进版示例：

```bash
cd /root/autodl-tmp/SVPG2023/LANE
python3 run_RL_ours.py --model rwtaspk --road_scenario merge --cuda 0 --ignore_checkpoint
```

### 11.2 脚本批量训练

串行版本：

```bash
bash train_lane_scenario_suite.sh
```

并行版本：

```bash
bash train_lane_scenario_suite_parallel.sh
```

其中并行脚本会将日志写入：

```text
train_parallel_logs/
```

## 12. 视频导出是如何工作的

如果你说的“`video.py`”指的是当前仓库中的视频脚本，那么对应文件实际上是：

- `test_video.py`

### 12.1 视频脚本做了什么

`test_video.py` 的流程如下：

1. 读取命令行参数
2. 根据 `--road_scenario` 和 `--traffic_level` 创建测试环境
3. 自动寻找最新的 best checkpoint，或者读取你显式指定的 `--checkpoint_prefix`
4. 如未使用 `--no_video`，则在 `env.init_test(record_video=True)` 时启用 `RecordVideo`
5. 加载策略模型
6. 在测试环境中逐步执行动作
7. 支持按 `failure_rate` 注入随机动作故障
8. 回合结束后将 `.mp4` 视频保存到视频目录

### 12.2 视频保存目录

视频目录结构如下：

```text
video_logs_lane/
  base/
    highway_standard/
    highway_dense/
    merge/
    roundabout/
  ours/
    highway_standard/
    highway_dense/
    merge/
    roundabout/
```

一次录制会在对应场景目录下创建时间戳子目录，例如：

```text
video_logs_lane/ours/merge/20260409_212929_285913/
```

视频文件名通常包含：

- 算法类别
- 场景名称
- 失败率标签
- 可选 seed
- 时间戳

例如：

```text
ours_merge_test_failure020_20260409_212929_285913-episode-0.mp4
```

### 12.3 自动选模型的规则

若不显式提供 `--checkpoint_prefix`，`test_video.py` 会：

1. 根据 `--checkpoint_kind` 选择 `baseline` 或 `ours`
2. 根据 `--road_scenario` 和 `--traffic_level` 限定搜索范围
3. 在对应模型目录下寻找最新修改时间的 `*_best_w_1.pt`
4. 自动加载该 checkpoint family

因此，执行视频脚本前，最好先确认对应场景已经训练并保存过 best 模型。

### 12.4 常用视频命令

录制改进版 merge 场景视频：

```bash
cd /root/autodl-tmp/SVPG2023/LANE
python3 test_video.py --road_scenario merge --checkpoint_kind ours
```

录制 baseline 高速高密度视频：

```bash
python3 test_video.py --road_scenario highway --traffic_level dense --checkpoint_kind baseline
```

录制带 20% 执行动作故障的视频：

```bash
python3 test_video.py --road_scenario roundabout --checkpoint_kind ours --failure_rate 0.2
```

只测试不录视频：

```bash
python3 test_video.py --road_scenario merge --checkpoint_kind ours --no_video
```

显式指定某个模型前缀：

```bash
python3 test_video.py --checkpoint_prefix /root/autodl-tmp/SVPG2023/LANE/training_runs/ours/merge/models/ppo_..._best
```

## 13. 视频中的动作故障实验是什么意思

`test_video.py` 支持 `--failure_rate` 参数，表示每一步以一定概率强制替换为随机动作。

其意义在于：

- 模拟执行器故障
- 模拟方向盘失控
- 检验策略在非理想控制条件下的鲁棒性

这类实验很适合写在毕业论文的“鲁棒性测试”一节中，因为它不只是测平均回报，还测策略对执行误差的承受能力。

## 14. 评测指标有哪些

项目中的常见评测指标包括：

- `mean_return`：平均累计回报
- `mean_length`：平均存活步数
- `collision_rate`：碰撞率
- `mean_lane_change`：平均变道次数
- `success_rate`：任务成功率
- `mean_speed`：平均速度

这些指标分别反映：

- `return`：总体优化目标
- `length`：是否能稳定生存更久
- `collision_rate`：安全性
- `lane_change`：决策是否平滑
- `success_rate`：是否完成任务
- `mean_speed`：效率

论文里建议不要只报一个 `return`，最好至少同时报告：

- `success_rate`
- `collision_rate`
- `mean_speed`
- `mean_length`

这样老师会更容易接受你的结论，因为自动驾驶任务本质上是一个多目标任务。

## 15. 鲁棒性测试怎么做

`test_robustness.py` 已内置三类鲁棒性评测：

- 动作故障：`failure_rate`
- 输入高斯噪声：`input_noise_levels`
- 权重高斯噪声：`weight_noise_levels`

示例：

```bash
cd /root/autodl-tmp/SVPG2023/LANE
python3 test_robustness.py --device cpu --episodes 10
```

这可以直接作为论文中“抗扰动能力分析”的实验基础。

## 16. 如何做基线与改进版对比

可以使用：

- `compare_experiments.py`

它会自动：

- 读取 baseline 与 ours 的 checkpoint
- 推断对应日志
- 解析训练与验证摘要
- 在多个场景、故障率和输入噪声下做对比

这非常适合生成论文中的对比表格与附录数据。

## 17. 论文里建议强调的创新点

如果你的论文基于当前仓库，建议从以下几个层面组织“方法改进点”：

### 17.1 训练层面

- 从简单经验缓存更新提升为 rollout + mini-batch PPO 更新
- 引入 GAE 估计优势，降低方差
- 引入 curriculum learning，逐步增加训练难度
- 场景自适应超参数稳定化
- 支持 warm start，提高复杂场景收敛效率

### 17.2 环境层面

- 针对 `merge` 和 `roundabout` 重构奖励整形
- 定制交通流初始化蓝图
- 根据场景设置不同的目标步数和完成条件
- 用安全性、通行效率、任务完成度联合定义训练目标

### 17.3 评测层面

- 不只看奖励，还看成功率、碰撞率、平均速度
- 引入动作故障、输入噪声、权重噪声三类鲁棒性测试
- 使用视频可视化展示策略行为可解释性

## 18. 答辩老师常问问题与回答思路

### Q1. 你的训练输入和输出到底是什么？

答：

- 输入是一个 `(1, 25)` 的交通状态向量
- 输出是 5 维离散动作分布，对应左变道、保持、右变道、加速、减速

### Q2. 你没有数据集，为什么模型还能学习？

答：

- 本项目是强化学习，不依赖静态标注数据集
- 样本由智能体与环境在线交互产生
- 训练数据是每一步交互形成的轨迹 `(s, a, r, s', done)`

### Q3. 为什么选择 PPO，而不是 DQN 或 REINFORCE？

答：

- DQN 更偏向基于 Q 值的离散控制，但这里需要更稳定的策略优化
- REINFORCE 方差较大
- PPO 在工程实现与稳定性之间取得了很好的平衡

### Q4. 你的奖励函数为什么这么复杂？

答：

- 自动驾驶是多目标任务，仅靠原始奖励难以同时兼顾安全、效率与平滑性
- 奖励整形将这些工程目标显式编码到训练中
- 不同场景下风险重点不同，因此需要场景化设计

### Q5. 奖励整形会不会导致“投机取巧”？

答：

- 是的，奖励整形如果设计不当会产生 reward hacking
- 因此本项目同时引入多指标验证和视频行为检查
- 不是只看 return，而是联合看成功率、碰撞率、速度与行为合理性

### Q6. 你的评论家为什么输出 5 维，而不是 1 维？

答：

- 这里实现的是“动作条件价值向量”估计
- 每个动作对应一个价值分量
- 再结合当前策略概率形成状态平均价值
- 这是本工程实现的一个特点，与经典标量 `V(s)` 略有不同

### Q7. 为什么观测只有 25 维，会不会太少？

答：

- 当前采用的是局部运动学观测，强调可控性与训练稳定性
- 25 维足够表达近邻车辆相对位置和速度
- 这是一种“低维状态强化学习”方案，适合做方法验证
- 若要更接近真实自动驾驶，可扩展为图结构、多车轨迹或图像观测

### Q8. 你的模型怎么保证泛化能力？

答：

- 通过不同场景、不同交通密度训练
- 通过 curriculum、噪声扰动和故障注入增强鲁棒性
- 通过在验证集场景和鲁棒性测试中报告性能来评估泛化

### Q9. 为什么视频测试要用 `argmax` 而不是采样？

答：

- 训练时采样动作是为了探索
- 测试和演示时使用 `argmax` 更稳定、更可复现
- 更适合展示模型学到的“最优或近似最优”行为

### Q10. 你怎么证明改进版比基线版更好？

答：

- 在相同场景下对比：
  - 成功率是否更高
  - 碰撞率是否更低
  - 平均速度是否更优
  - 收敛速度是否更快
- 再结合鲁棒性实验和可视化视频给出证据

### Q11. 课程学习在这里起什么作用？

答：

- 先在较干净或较简单的条件下学习基本策略
- 再逐步引入噪声、提高车辆密度或增加难度
- 这样能降低训练早期的不稳定性

### Q12. 这个项目的局限性是什么？

答：

- 环境仍是仿真环境，不是现实车辆
- 观测较低维，感知问题被简化
- 动作为离散元动作，不是连续控制
- 奖励设计带有人为先验
- 真实世界迁移仍需进一步验证

## 19. 论文写作建议

推荐按照以下结构写方法章节：

1. 问题定义：自动驾驶离散决策任务建模为 MDP
2. 状态空间：25 维运动学观测
3. 动作空间：5 个离散元动作
4. 奖励函数：安全、效率、平滑、完成度的联合建模
5. 策略网络：RWTAspike
6. 价值网络：动作条件价值估计
7. 训练算法：PPO + GAE
8. 方法改进：curriculum、warm start、稳定化配置
9. 评测体系：回报、安全、成功率、鲁棒性、视频可视化

推荐按照以下结构写实验章节：

1. 实验环境与硬件配置
2. 场景说明：highway / merge / roundabout
3. 训练超参数设置
4. 基线与改进版对比
5. 收敛曲线分析
6. 鲁棒性实验
7. 可视化视频分析
8. 消融实验

## 20. 一个可以直接写进论文的方法概括

可以参考下面这段描述进行改写：

> 本文将自动驾驶决策问题建模为离散动作空间下的马尔可夫决策过程。智能体基于局部交通运动学观测生成离散控制动作，并通过与道路仿真环境的在线交互持续采样轨迹数据。训练过程中采用 PPO 作为主优化框架，并在改进方案中引入 GAE、课程学习、热启动与场景自适应稳定化机制，以提高复杂交通场景下策略学习的稳定性、收敛速度与鲁棒性。实验从平均回报、成功率、碰撞率、平均速度以及故障扰动下的性能变化等多个维度对模型进行评估。

## 21. 快速结论

如果你只想抓住最核心的几句话，可以记住：

- 本项目不是“喂数据集训练分类器”，而是“在仿真环境中在线交互学习策略”
- 输入是 25 维状态，输出是 5 维离散驾驶动作
- 训练样本来自环境交互轨迹，不是静态标注数据
- 改进版核心是 PPO + GAE + curriculum + warm start
- `test_video.py` 负责加载 best 模型并输出 `.mp4` 视频
- 论文与答辩要同时讲清楚：任务建模、奖励设计、训练算法、评测指标、鲁棒性分析
