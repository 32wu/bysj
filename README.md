# SVPG2023 / LANE 项目总说明

## 配套文档

- [README_thesis.md](README_thesis.md)：答辩 / 论文展示版，适合给老师、评审或答辩展示时快速讲清项目。
- [README_dev.md](README_dev.md)：开发维护版，适合后续继续改代码、复现实验、排查问题时查阅。
- [README_bysj.md](README_bysj.md)：个人实验速查笔记，偏日常命令和操作备忘。
- [ENV_SETUP.md](ENV_SETUP.md)：环境汇总与一键安装说明，适合新云服务器直接复现。

## 1. 项目定位

这个仓库最初来源于论文 **Spiking Variational Policy Gradient for Brain Inspired Reinforcement Learning** 的实验代码。原始仓库覆盖了 MNIST、GYMIP、DOOM、AI2THOR、ROBOTARM 等任务。

当前这份工作目录已经演化成你的毕设主线版本，核心关注点是：

- 保留原始 SVPG / RWTA 思路与部分兼容代码
- 将原来 `gymip` 这条训练入口改造成 **LANE 自动驾驶小车任务**
- 在 `highway_env` 上训练和评估四类道路场景
- 对比 **基础算法（base / baseline）** 与 **改进算法（ours）**
- 自动整理训练权重、日志、视频、对比报告

一句话理解：

> 这是一个“基于原始 SVPG 仓库改造而来的、小车车道驾驶毕设实验平台”。

## 2. 当前主线任务

当前最重要、最常用的部分都在 `LANE/` 目录下，主线任务包括：

- 训练基础算法与改进算法
- 在四种驾驶场景上做实验
- 自动保存最优权重、训练日志、测试视频
- 对比基线与改进算法的鲁棒性表现

四种主场景为：

- `highway_standard`：直道标准车流
- `highway_dense`：直道高密度车流
- `merge`：匝道汇入
- `roundabout`：环岛场景

## 3. 快速开始

### 3.1 环境准备

推荐直接使用仓库根目录的一键安装脚本：

```bash
bash setup_bysj_env.sh
conda activate bysj
```

如果你只想手动创建环境，也可以直接使用仓库根目录的环境导出文件：

```bash
conda env create -f bysj_env.yml
conda activate bysj
```

当前环境文件里比较关键的依赖包括：

- `python=3.10`
- `torch=2.10.0`
- `gym=0.26.2`
- `gymnasium=1.2.3`
- `highway-env=1.10.2`
- `mujoco=2.2.0`
- `spikingjelly=0.0.0.0.8`
- `snntorch=0.9.4`

### 3.2 进入主工作目录

```bash
cd LANE
```

### 3.3 最常用命令

训练基础算法示例：

```bash
python3 run_RL_base.py --model rwtaspk --road_scenario highway --traffic_level standard --ignore_checkpoint --skip_post_tests
```

训练改进算法示例：

```bash
python3 run_RL_ours.py --model rwtaspk --road_scenario merge --ignore_checkpoint
```

录制测试视频示例：

```bash
python3 test_video.py --road_scenario merge --checkpoint_kind ours
python3 test_video.py --road_scenario highway --traffic_level standard --checkpoint_kind base
```

自动生成论文对比报告示例：

```bash
python3 compare_experiments.py --device cpu
```

鲁棒性测试示例：

```bash
python3 test_robustness.py --device cpu --max_models 2
```

## 4. 主目录文件夹说明

仓库根目录当前最重要的内容如下：

```text
SVPG2023/
  LICENSE
  README.md
  README_bysj.md
  bysj_env.yml
  LANE/
```

各项作用如下：

| 路径 | 作用 |
|---|---|
| `LICENSE` | 开源许可证。 |
| `README.md` | 当前这份总说明文档。 |
| `README_bysj.md` | 你的个人实验速查笔记，记录常用命令、训练方式、备份方式。 |
| `bysj_env.yml` | 当前毕设环境导出文件，推荐用它复现你的运行环境。 |
| `LANE/` | 毕设主代码目录，训练、测试、评估、模型、环境都在里面。 |

## 5. `LANE/` 目录结构说明

`LANE/` 是整个项目的工作核心。当前常见目录如下：

```text
LANE/
  checkpoint_utils.py
  compare_experiments.py
  env_gymip.py
  env_lane.py
  memory_lib.py
  model_adversarial.py
  model_convert.py
  model_critic.py
  model_mlp.py
  model_rwta.py
  model_snnbptt.py
  plot_robustness.py
  plot_training_curve.py
  prepare_gymip.py
  run_RL_base.py
  run_RL_ours.py
  test_render.py
  test_robustness.py
  test_video.py
  train_lane_scenario_suite.sh
  train_lane_scenario_suite_parallel.sh

  ann2snn/
  comparison_reports/
  env/
  log_model/
  log_text/
  train_parallel_logs/
  training_runs/
  video_logs/
  video_logs_lane/
  __pycache__/
```

### 5.1 子文件夹说明

| 文件夹 | 作用 |
|---|---|
| `ann2snn/` | ANN 转 SNN 流程的临时输出目录。当前 LANE 主线几乎不常用，但兼容保留。 |
| `comparison_reports/` | `compare_experiments.py` 生成的对比报告目录，通常包含 CSV 和 Markdown。 |
| `env/` | vendored 的 `gym 0.26.2` 源码副本，主要给旧版倒立摆环境和 XML 改造使用。 |
| `log_model/` | 老版本遗留的权重输出目录。现在主要作为历史兼容路径。 |
| `log_text/` | 老版本遗留的文本日志目录。现在主要作为历史兼容路径。 |
| `train_parallel_logs/` | 并行训练脚本的终端输出日志。 |
| `training_runs/` | 当前正式使用的训练产物目录，按 `baseline/ours + 场景 + models/logs` 分类。 |
| `video_logs/` | 旧版倒立摆录视频目录，`test_render.py` 会写到这里。 |
| `video_logs_lane/` | 当前正式使用的小车视频目录，按 `base/ours + 场景 + 时间戳` 分类。 |
| `__pycache__/` | Python 运行生成的缓存目录，可删，会自动重建。 |

### 5.2 训练与视频输出目录规则

#### `training_runs/`

当前训练结果按下面结构保存：

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
```

#### `video_logs_lane/`

当前测试视频按下面结构保存：

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

每次录制还会在场景目录下再建一层时间戳目录，避免覆盖旧视频。

## 6. 核心 `.py` 文件逐个说明

仓库根目录没有业务 Python 文件，所有主代码都在 `LANE/*.py`。下面按“你自己维护的核心脚本”逐个说明。

### 6.1 训练与评估入口脚本

| 文件 | 作用 |
|---|---|
| `LANE/run_RL_base.py` | 基础算法训练入口。沿用原始 SVPG 风格训练循环，支持 PPO / REINFORCE，当前已接入 `env_lane.py` 并按场景保存到 `training_runs/baseline/`。 |
| `LANE/run_RL_ours.py` | 改进算法训练入口。加入 rollout buffer、GAE、梯度裁剪、自适应课程学习、warm start、场景稳定化策略，是当前论文/毕设最核心的训练脚本。 |
| `LANE/test_video.py` | 加载指定或自动发现的最优权重，在指定场景中测试并录像。支持区分 `base` / `ours`，视频自动分类保存。 |
| `LANE/test_robustness.py` | 对若干最优模型做鲁棒性测试，包括执行器故障、输入噪声、权重噪声。当前自动发现逻辑只会默认选取 `best` 权重。 |
| `LANE/compare_experiments.py` | 自动比较 baseline 与 ours，在多个场景和干扰条件下评估表现，并输出 CSV / Markdown 报告。适合直接生成论文表格素材。 |
| `LANE/test_render.py` | 旧版快速渲染脚本。它录的是倒立摆随机动作视频，主要用于验证 `RecordVideo` 和无头渲染是否正常。 |

### 6.2 环境与运行支撑模块

| 文件 | 作用 |
|---|---|
| `LANE/env_lane.py` | 当前主环境封装。把 `highway_env` 封装成统一训练接口，负责四种场景配置、观测处理、奖励塑形、成功判定、交通补车、录像封装。 |
| `LANE/env_gymip.py` | 旧版倒立摆环境封装。原始项目的 `gymip` 兼容层，主要配合 `prepare_gymip.py` 和 `env/` 使用。 |
| `LANE/checkpoint_utils.py` | 统一管理权重、日志、视频路径的工具模块。负责 `training_runs` 目录组织、最新 checkpoint 自动发现、日志路径推断、最终只保留最优权重的清理逻辑。 |
| `LANE/memory_lib.py` | 传统训练循环用的经验缓存。`run_RL_base.py` 使用它存储状态、动作、奖励等轨迹数据。 |
| `LANE/prepare_gymip.py` | 为旧版倒立摆环境批量生成不同杆长/杆厚 XML 文件。当前 LANE 主线不依赖它，但老任务兼容仍保留。 |

### 6.3 模型定义文件

| 文件 | 作用 |
|---|---|
| `LANE/model_rwta.py` | RWTA 概率版与脉冲版模型定义，是当前主实验最重要的 actor 模型文件。包含前向传播、PPO / REINFORCE 学习、保存/加载、加噪、连接删除等逻辑。 |
| `LANE/model_critic.py` | 价值网络定义。训练时用于估计状态价值或动作价值，帮助 PPO / GAE 更新。最终小车推理主要用 actor，但训练与恢复训练会用到 critic。 |
| `LANE/model_mlp.py` | 常规 MLP baseline。可作为非脉冲对照网络。 |
| `LANE/model_snnbptt.py` | 基于 `snntorch` 的 BPTT 脉冲网络基线。 |
| `LANE/model_convert.py` | ANN2SNN 转换逻辑，依赖 `spikingjelly`。把 ANN 模型转换成 SNN 用于旧版对照实验。 |
| `LANE/model_adversarial.py` | 对抗扰动辅助模型，主要用于 `run_RL_base.py` 末尾的旧版鲁棒性/对抗测试流程。 |

### 6.4 绘图与结果处理脚本

| 文件 | 作用 |
|---|---|
| `LANE/plot_training_curve.py` | 根据训练日志画收敛曲线。需要手动把 `log_path` 改成你的真实日志路径。 |
| `LANE/plot_robustness.py` | 根据 `compare_experiments.py` 生成的 CSV 画鲁棒性对比图。需要手动改 `csv_path`。 |

## 7. 第三方 / 兼容代码说明

### 7.1 `LANE/env/` 不是你当前主线自己写的新算法文件

`LANE/env/` 里是一整套 vendored 的 `gym 0.26.2` 源码镜像，主要为了：

- 保留原始 `GYMIP` 任务兼容性
- 支持 `prepare_gymip.py` 生成并使用自定义倒立摆 XML
- 避免完全依赖系统外部 `gym` 安装时的版本差异

### 7.2 `LANE/env/gym/**/*.py` 子树如何理解

这个目录下有很多 Python 文件，它们大致分成以下几类：

| 子路径 | 作用 |
|---|---|
| `LANE/env/gym/__init__.py`、`core.py`、`error.py`、`logger.py`、`version.py` | Gym 包本体的基础入口与公共工具。 |
| `LANE/env/gym/envs/mujoco/*.py` | MuJoCo 环境实现，其中 `inverted_pendulum.py` / `inverted_pendulum_v4.py` 与旧版倒立摆任务关系最紧密。 |
| `LANE/env/gym/envs/classic_control/*.py` | 经典控制环境，如 cartpole、pendulum、mountain_car。当前 LANE 主线基本不用。 |
| `LANE/env/gym/envs/box2d/*.py` | Box2D 环境，如 lunar_lander、car_racing。当前 LANE 主线基本不用。 |
| `LANE/env/gym/envs/toy_text/*.py` | toy text 环境，如 taxi、frozen_lake。当前 LANE 主线不用。 |
| `LANE/env/gym/spaces/*.py` | Gym 空间定义，如 Box、Discrete、Tuple 等。 |
| `LANE/env/gym/utils/*.py` | Gym 工具函数，如随机种子、视频保存、环境检查。 |
| `LANE/env/gym/vector/*.py` | 向量化环境支持。当前 LANE 主线未直接使用。 |
| `LANE/env/gym/wrappers/*.py` | 各类环境包装器，例如视频录制、归一化、观测变换。 |

结论可以简单记为：

> `LANE/env/` 是“旧环境和上游依赖镜像”，不是你当前论文主线最核心、最常改动的业务代码区域。

## 8. 整个项目的运行逻辑

当前小车主线的运行流程如下。

### 8.1 训练阶段总流程

1. 启动训练脚本：`run_RL_base.py` 或 `run_RL_ours.py`
2. 解析命令行参数，生成实验名 `EXP_NAME`
3. 调用 `checkpoint_utils.py`，自动激活当前实验的输出目录
4. 创建环境 `env_lane.GymLane`
5. 根据 `--model` 创建 actor；同时创建 `model_critic.Critic`
6. 进入训练循环：环境交互、收集轨迹、更新策略
7. 定期做验证集评估
8. 若验证表现更好，则保存为 `*_best*`
9. 训练结束时清理冗余 `current` / 备份文件，只保留最优权重
10. 后续测试、录像、对比脚本默认优先加载 `best` 权重

### 8.2 小车真正跑起来时的链路

当你执行：

```bash
python3 test_video.py --road_scenario merge --checkpoint_kind ours
```

内部流程是：

1. `test_video.py` 自动发现指定类别和场景下最新的 `best` checkpoint
2. 构造 `env_lane.GymLane` 测试环境
3. `model_rwta.py` 加载 actor 权重
4. 每一步从环境取观测
5. actor 输出动作分布
6. 取最大概率动作控制车辆
7. 环境返回下一个观测和奖励
8. 如果开启录像，则自动存入 `video_logs_lane/`

### 8.3 为什么参数里还是写 `task=gymip`

这是原始仓库改造留下的兼容设计。

现在很多训练脚本里虽然仍然使用 `--task gymip` 这个参数名，但在 LANE 主线中它实际上已经指向：

- `env_lane.py`
- `highway_env` 小车环境
- 四种道路场景任务

所以可以把它理解为：

> `gymip` 这个字符串只是保留了旧入口名字，当前真正跑的是小车车道环境。

## 9. 训练方式说明

### 9.1 基础算法 `run_RL_base.py`

基础算法脚本的特点：

- 更接近原始 SVPG / RWTA 训练骨架
- 使用 `MemoryBuffer` 存储经验
- 支持 `ppo` 和 `reinforce`
- 定期保存 `current` checkpoint
- 按验证平均回报选择 `best`
- 默认训练结束后还会跑一整套旧版后处理测试
- 如果你只想训练并快速结束，建议加 `--skip_post_tests`

它适合：

- 做基线复现
- 与改进算法做公平对比
- 保留原始仓库风格的训练方式

### 9.2 改进算法 `run_RL_ours.py`

改进算法脚本是当前主线的核心创新版本，主要增强点包括：

- 使用 `RolloutBuffer` 收集固定长度 rollout
- 使用 `GAE` 计算优势函数
- 支持 mini-batch PPO 更新
- 支持 `grad_clip` 梯度裁剪
- 支持 `reward_scale`
- 支持自适应课程学习 `curriculum_mode=adaptive`
- 支持噪声上限逐步提升与熵系数逐步衰减
- 支持 `warm_start`，从 baseline 或已有 ours 权重继续起步
- 支持 `lane_profile=auto`，自动把超参数收敛到更稳的小车配置
- 验证时不是只看 return，而是综合比较：成功率、碰撞率、速度、回报

它适合：

- 作为论文中的“改进算法”
- 跑小车稳定控制与鲁棒性实验
- 做场景迁移、课程学习、热启动实验

### 9.3 base 与 ours 的差异总结

| 维度 | `run_RL_base.py` | `run_RL_ours.py` |
|---|---|---|
| 目标 | 基线复现 | 毕设主改进算法 |
| 轨迹缓存 | `MemoryBuffer` | `RolloutBuffer` |
| 优势估计 | 较传统流程 | `GAE` |
| 超参稳定化 | 基本没有 | `lane_profile=auto` |
| 课程学习 | 无 | 支持固定/自适应课程学习 |
| warm start | 无 | 支持 |
| 验证标准 | 以回报为主 | 成功率、碰撞率、速度、回报综合 |
| 后处理 | 默认会做旧版测试 | 更专注 LANE 主线训练 |

## 10. 场景与训练任务说明

当前主线支持的道路场景来自 `env_lane.py` 中的 `SCENARIO_PRESETS`：

| 场景 | 参数写法 | 含义 |
|---|---|---|
| `highway_standard` | `--road_scenario highway --traffic_level standard` | 直道标准车流 |
| `highway_dense` | `--road_scenario highway --traffic_level dense` | 直道高密度车流 |
| `merge` | `--road_scenario merge` | 匝道汇入 |
| `roundabout` | `--road_scenario roundabout` | 环岛 |

补充说明：

- `highway` 场景通过 `traffic_level` 区分 `light / standard / dense`
- `merge` 和 `roundabout` 也支持 `traffic_level`，但当前实验通常默认 `standard`
- `env_lane.py` 里还包含场景专用交通蓝图、成功条件与奖励塑形规则

## 11. 权重与日志的组织方式

### 11.1 训练权重

当前正式权重默认只保留最优结果，位于：

```text
LANE/training_runs/<baseline|ours>/<scenario>/models/
```

对于 RWTA / RWTA-spike 实验，一个最优模型通常保留 4 个文件：

- `*_best_w_1.pt`：actor 权重矩阵
- `*_best_b_1.pt`：actor 偏置
- `*_best_m_1.pt`：actor 连接掩码
- `*critic_best_1.pt`：critic 权重

### 11.2 训练日志

训练日志位于：

```text
LANE/training_runs/<baseline|ours>/<scenario>/logs/
```

日志中通常包含：

- 初始化参数
- 当前实验使用的模型目录 / 日志目录
- 训练过程中的 `train` 记录
- 验证记录 `val`
- 最优模型保存记录 `val_save`
- 最终 checkpoint 清理记录 `checkpoint_cleanup`

### 11.3 为什么现在只保留 4 个文件

原始仓库每份 checkpoint 会保存 `_1` 和 `_2` 两套副本，还会留 `current`。为了减少混乱，当前项目已经改成：

- 训练中可以临时写 `current`
- 训练结束时自动清理
- 只保留 `best` 的主文件

这样做更适合你的论文实验整理和统一归档。

## 12. 视频录制方式

### 12.1 视频保存结构

当前正式视频目录：

```text
LANE/video_logs_lane/
  base/
  ours/
```

每个算法目录下再按场景和时间戳分类：

```text
video_logs_lane/ours/merge/20260331_170529_070225/
```

### 12.2 文件命名方式

视频文件名会自动包含：

- 算法类别 `base` / `ours`
- 场景类别
- 测试模式
- 额外标签 `video_tag`
- 时间戳

例如：

```text
ours_merge_test_failure020-seed1_20260331_170529_070225-episode-0.mp4
```

## 13. 常用训练命令整理

下面给出比较实用的命令模板。默认建议先 `cd LANE`。

### 13.1 基础算法四场景

```bash
python3 run_RL_base.py --model rwtaspk --road_scenario highway --traffic_level standard --ignore_checkpoint --skip_post_tests
python3 run_RL_base.py --model rwtaspk --road_scenario highway --traffic_level dense --ignore_checkpoint --skip_post_tests
python3 run_RL_base.py --model rwtaspk --road_scenario merge --ignore_checkpoint --skip_post_tests
python3 run_RL_base.py --model rwtaspk --road_scenario roundabout --ignore_checkpoint --skip_post_tests
```

### 13.2 改进算法四场景

```bash
python3 run_RL_ours.py --model rwtaspk --road_scenario highway --traffic_level standard --ignore_checkpoint
python3 run_RL_ours.py --model rwtaspk --road_scenario highway --traffic_level dense --ignore_checkpoint
python3 run_RL_ours.py --model rwtaspk --road_scenario merge --ignore_checkpoint
python3 run_RL_ours.py --model rwtaspk --road_scenario roundabout --ignore_checkpoint
```

### 13.3 批量脚本

顺序训练套件：

```bash
bash train_lane_scenario_suite.sh
```

并行训练套件：

```bash
bash train_lane_scenario_suite_parallel.sh
```

说明：

- `train_lane_scenario_suite.sh`：串行执行，简单稳妥
- `train_lane_scenario_suite_parallel.sh`：并行跑多个任务，终端输出保存到 `train_parallel_logs/`

## 14. 常用测试与分析命令

### 14.1 录制视频

```bash
python3 test_video.py --road_scenario highway --traffic_level standard --checkpoint_kind base
python3 test_video.py --road_scenario highway --traffic_level dense --checkpoint_kind ours
python3 test_video.py --road_scenario merge --checkpoint_kind ours
python3 test_video.py --road_scenario roundabout --checkpoint_kind ours
```

如果你要自定义标签：

```bash
python3 test_video.py --road_scenario merge --checkpoint_kind ours --video_tag demo1
```

### 14.2 鲁棒性测试

```bash
python3 test_robustness.py --device cpu --max_models 2
```

默认会自动寻找最新的 `best` 模型做测试。

### 14.3 自动对比论文结果

```bash
python3 compare_experiments.py --device cpu
```

输出通常会进入：

```text
LANE/comparison_reports/
```

### 14.4 绘图脚本

在手动改好路径后可使用：

```bash
python3 plot_training_curve.py
python3 plot_robustness.py
```

## 15. 当前项目里“常用”和“历史兼容”的边界

### 15.1 当前最常用、最建议关注的文件

如果你后续继续做毕设、整理实验、录视频、写论文，最值得重点关注的是：

- `LANE/run_RL_ours.py`
- `LANE/run_RL_base.py`
- `LANE/env_lane.py`
- `LANE/model_rwta.py`
- `LANE/checkpoint_utils.py`
- `LANE/test_video.py`
- `LANE/test_robustness.py`
- `LANE/compare_experiments.py`
- `LANE/training_runs/`
- `LANE/video_logs_lane/`

### 15.2 历史兼容、保留但不是主线重点的文件

这些文件依然有价值，但主要用于兼容旧仓库思路，不是你现在小车论文主线的首要关注对象：

- `LANE/env_gymip.py`
- `LANE/prepare_gymip.py`
- `LANE/model_convert.py`
- `LANE/model_adversarial.py`
- `LANE/test_render.py`
- `LANE/video_logs/`
- `LANE/log_model/`
- `LANE/log_text/`
- `LANE/env/`

## 16. 你当前项目最推荐的使用顺序

如果按“做实验 -> 出结果 -> 录视频 -> 写论文”来走，推荐顺序是：

1. 用 `run_RL_base.py` 和 `run_RL_ours.py` 训练各场景模型
2. 到 `training_runs/` 查看是否已生成对应场景的最优权重和日志
3. 用 `test_video.py` 录制关键场景视频
4. 用 `test_robustness.py` 和 `compare_experiments.py` 输出对比结果
5. 用 `plot_training_curve.py`、`plot_robustness.py` 进一步做图
6. 把论文正文中的表格、曲线、视频案例与 `comparison_reports/`、`video_logs_lane/` 对齐整理

## 17. 备注

- 现在默认的测试与录像流程都会优先使用 **最优权重 `best`**。
- 训练结束后也会自动清理冗余 checkpoint，只保留最终最有价值的权重文件。
- `README_bysj.md` 更适合当作你个人日常命令速查；本 README 更适合作为项目正式说明。
- 如果未来你继续扩展这份仓库，建议继续以 `LANE/` 为主工作区，而不是把新逻辑再散回旧版兼容目录里。

---

如果把整个仓库分成一句话总结：

> `SVPG2023` 是原始 SVPG 代码仓的毕设演化版，而 `LANE/` 是你当前“小车自动驾驶四场景训练、评估、录像、对比”的完整实验平台。
