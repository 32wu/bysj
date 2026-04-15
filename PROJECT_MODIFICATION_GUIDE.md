# SVPG2023 / LANE 项目修改指南

更新时间：2026-04-14  
适用范围：当前工作区中的 `SVPG2023/` 仓库，尤其是 `LANE/` 主线代码。

## 1. 先看结论

这个仓库虽然保留了原始 SVPG 项目的很多历史痕迹，但当前真正需要维护的主线几乎都在 `LANE/` 目录，核心目标是：

- 在 `highway_env` 上训练自动驾驶离散动作策略。
- 对比 `baseline/base` 与 `ours` 两套训练流程。
- 自动管理权重、日志、视频和对比报告。

以后改项目时，建议优先遵守这三条：

- 改环境和奖励，先看 `LANE/env_lane.py`。
- 改训练流程，优先看 `LANE/run_RL_ours.py`，基线对照看 `LANE/run_RL_base.py`。
- 改权重、日志、目录命名，不要在脚本里到处拼路径，统一改 `LANE/checkpoint_utils.py`。

## 2. 仓库结构总览

```text
SVPG2023/
  LICENSE
  README*.md
  bysj_env.yml
  PROJECT_MODIFICATION_GUIDE.md
  LANE/
    run_RL_base.py
    run_RL_ours.py
    env_lane.py
    checkpoint_utils.py
    model_*.py
    test_*.py
    compare_experiments.py
    training_runs/
    video_logs_lane/
    comparison_reports/
    env/
```

理解这个仓库最重要的一句话：

> 根目录负责文档和环境说明，`LANE/` 负责训练、测试、评估和实验产物。

## 3. 主数据流

### 3.1 训练主线

```text
run_RL_base.py / run_RL_ours.py
  -> env_lane.py
  -> model_rwta.py / model_mlp.py / model_snnbptt.py
  -> model_critic.py
  -> checkpoint_utils.py
  -> training_runs/
```

### 3.2 视频测试主线

```text
test_video.py
  -> checkpoint_utils.py 自动找 checkpoint
  -> env_lane.py
  -> model_rwta.py
  -> video_logs_lane/
```

### 3.3 对比分析主线

```text
test_robustness.py -> 控制台鲁棒性统计
compare_experiments.py -> comparison_reports/CSV + Markdown
plot_training_curve.py / plot_robustness.py -> 二次画图
```

## 4. 根目录文件说明

| 路径 | 作用 | 修改建议 |
|---|---|---|
| `LICENSE` | 开源许可证。 | 一般不改。 |
| `README.md` | 仓库总说明，偏总览。 | 项目结构变化后同步更新。 |
| `README_dev.md` | 开发维护说明，偏代码维护。 | 改主线流程时建议同步更新。 |
| `README_thesis.md` | 答辩 / 论文展示版说明。 | 结果或叙述变化时更新。 |
| `README_bysj.md` | 个人常用命令速查表。 | 更像备忘，不是主规范。 |
| `readmeproject.md` | 更完整的项目说明，包含答辩常见解释。 | 可作为论文叙述补充。 |
| `bysj_env.yml` | Conda 环境导出文件，记录依赖。 | 环境依赖变化后再更新。 |
| `PROJECT_MODIFICATION_GUIDE.md` | 本文档，作为后续修改入口。 | 新增模块或目录变化时更新。 |

## 5. `LANE/` 顶层源码逐文件说明

| 文件 | 作用 | 后续修改时优先关注什么 |
|---|---|---|
| `checkpoint_utils.py` | 统一管理 `training_runs/`、历史 `log_model/` / `log_text/`、checkpoint 自动发现、清理和路径推断。 | 改目录结构、命名规则、自动找最新模型时先改这里。 |
| `compare_experiments.py` | 自动对比 baseline 与 ours，生成 CSV 和 Markdown 报告。 | 改默认对比场景、指标、报告格式时改这里。 |
| `conversation_history_20260409_165153.md` | 一次历史对话记录。 | 参考资料，不是代码。 |
| `conversation_history_20260410.md` | 一次历史对话记录。 | 参考资料，不是代码。 |
| `env_gymip.py` | 原始倒立摆 `gymip` 任务封装，属于历史兼容入口。 | 只有维护旧实验时才改。 |
| `env_lane.py` | 当前主线环境封装：场景注册、交通配置、奖励塑形、终止逻辑、视频目录、测试接口都在这里。 | 改场景、奖励、成功判定、车流密度、视频命名时都从这里入手。 |
| `memory_lib.py` | 基线训练脚本使用的旧式经验缓存。 | 只会影响 `run_RL_base.py`。 |
| `model_adversarial.py` | 辅助对抗/扰动网络定义，当前 LANE 主线里基本不是核心。 | 除非专门做对抗实验，否则通常不动。 |
| `model_convert.py` | ANN 到 SNN 的转换工具，依赖 `spikingjelly`。 | 维护 `ann2snn` 路线时再看。 |
| `model_critic.py` | 价值网络定义，训练时与 actor 配套使用。 | 改 critic 结构、保存加载、梯度裁剪时改这里。 |
| `model_mlp.py` | 普通 MLP actor，实现 PPO / REINFORCE 所需接口。 | 新增普通前馈策略时参考它。 |
| `model_rwta.py` | 当前主线最重要的 actor 文件，定义 `RWTAprob` 和 `RWTAspike`，含保存、加载、噪声注入、PPO 学习接口。 | 改 RWTA 网络、熵控制、噪声、checkpoint 组成时重点看这里。 |
| `model_snnbptt.py` | 用 `snntorch` 实现的 BPTT 脉冲网络 actor。 | 只在 SNN 路线实验时改。 |
| `monitor_training_eta.py` | 读取结构化训练日志，估算当前训练速度和 ETA。 | 日志格式变了时要同步适配。 |
| `plot_robustness.py` | 从对比 CSV 画鲁棒性图的脚本模板。 | 论文画图时可复用，但当前是半手工脚本。 |
| `plot_training_curve.py` | 从日志画训练曲线的脚本模板。 | 论文画图时可复用，但需要替换输入路径。 |
| `prepare_gymip.py` | 生成倒立摆 XML 扰动资产。 | 只服务历史 `gymip` 任务。 |
| `run_RL_base.py` | 基线训练入口，保留更接近旧代码的训练骨架。 | 做对照实验或兼容旧逻辑时改它。 |
| `run_RL_ours.py` | 当前主线训练入口，包含 rollout buffer、GAE、mini-batch PPO、课程学习、warm start、并行采样等增强。 | 新方法、新稳定化策略、新训练实验优先改它。 |
| `test_render.py` | 旧版倒立摆随机动作录像脚本。 | 仅用于 legacy 演示。 |
| `test_robustness.py` | 对 checkpoint 做动作故障、输入噪声、权重噪声评估。 | 改鲁棒性实验设置、默认噪声列表时改这里。 |
| `test_video.py` | 自动寻找 checkpoint 并录制 LANE 视频。 | 改视频测试入口、默认加载逻辑、标签命名时改这里。 |
| `train_lane_scenario_suite.sh` | 顺序跑四个场景的批处理脚本。 | 改默认批量训练组合时改这里。 |
| `train_lane_scenario_suite_parallel.sh` | 并行跑四个场景的批处理脚本，并把终端输出写入 `train_parallel_logs/`。 | 改并行训练编排、线程数、日志目录时改这里。 |
| `train_remaining_routes.log` | 一份剩余路线训练相关日志。 | 属于运行产物，不建议手工维护。 |

## 6. `LANE/` 目录说明

| 路径 | 作用 | 是否建议手改 |
|---|---|---|
| `ann2snn/` | ANN 转 SNN 过程的临时目录。 | 否，除非维护转换流程。 |
| `comparison_reports/` | 对比脚本生成的 CSV 和 Markdown 报告。 | 否，通常重新生成。 |
| `env/` | vendored `gym 0.26.2` 源码树，主要给旧版 MuJoCo 倒立摆任务使用。 | 谨慎。只有改 legacy gym 环境时才动。 |
| `log_model/` | 老版权重目录。 | 否，兼容保留。 |
| `log_text/` | 老版日志目录。 | 否，兼容保留。 |
| `train_parallel_logs/` | 并行训练脚本的终端输出日志。 | 否，运行产物。 |
| `training_runs/` | 当前正式训练产物目录，按 `baseline/ours + 场景 + models/logs` 分类。 | 否，除非做清理或备份。 |
| `video_logs/` | legacy 倒立摆视频目录。 | 否。 |
| `video_logs_lane/` | 当前正式小车视频目录。 | 否，通常由脚本自动写入。 |
| `__pycache__/` | Python 缓存。 | 否，可删可重建。 |

## 7. `env/` 目录怎么理解

`LANE/env/` 不是当前 LANE 主线的核心逻辑，而是一份 vendored 的 Gym 源码副本，主要为了兼容旧版 MuJoCo 倒立摆实验。

重点文件只有少数几个：

| 路径 | 作用 |
|---|---|
| `LANE/env/gym/envs/mujoco/inverted_pendulum_v4.py` | 支持通过 `self_xml` 指定 XML 文件的自定义倒立摆环境。 |
| `LANE/env/gym/envs/mujoco/inverted_double_pendulum_v4.py` | 支持通过 `self_xml` 指定 XML 文件的自定义双倒立摆环境。 |
| `LANE/env/gym/envs/mujoco/assets/*.xml` | 倒立摆及其扰动版本 XML 资产。 |
| `LANE/env/gym/envs/registration.py` | 注册环境时会涉及 `InvertedPendulum-v4` / `InvertedDoublePendulum-v4`。 |
| 其余 `env/gym/*` 文件 | 大多是 Gym 上游副本，通常不需要为 LANE 主线修改。 |

## 8. 生成物目录命名规则

### 8.1 训练结果

当前正式训练结果位于：

```text
LANE/training_runs/
  baseline/
  ours/
```

每个算法目录下按场景继续分：

- `highway_standard`
- `highway_dense`
- `merge`
- `roundabout`

每个场景再分：

- `models/`：checkpoint
- `logs/`：训练日志

### 8.2 视频结果

当前正式视频位于：

```text
LANE/video_logs_lane/
  base/
  ours/
    <scenario>/
      <timestamp>/
```

视频目录和命名由 `env_lane.py` 自动生成，不建议手写。

## 9. 高频改动该去哪里

| 想改什么 | 首选文件 |
|---|---|
| 新增道路场景 | `LANE/env_lane.py` |
| 改车流密度 / `vehicles_count` / `traffic_level` | `LANE/env_lane.py` |
| 改奖励函数 | `LANE/env_lane.py` 的 `_shape_reward()` |
| 改场景成功判定 | `LANE/env_lane.py` 的 `_scenario_completed_now()` |
| 改视频目录或命名 | `LANE/env_lane.py` 的 `_build_video_recording_target()` 和 `LANE/test_video.py` |
| 改基线训练流程 | `LANE/run_RL_base.py` |
| 改主线训练策略 | `LANE/run_RL_ours.py` |
| 改 rollout / GAE / PPO 更新 | `LANE/run_RL_ours.py` |
| 改 warm start / 自动加载 | `LANE/run_RL_ours.py` + `LANE/checkpoint_utils.py` |
| 改 checkpoint 命名、查找、清理 | `LANE/checkpoint_utils.py` |
| 新增 actor 模型 | `LANE/model_*.py` + 两个训练入口 + 测试脚本 |
| 改 critic 结构 | `LANE/model_critic.py` |
| 改鲁棒性评估指标 | `LANE/test_robustness.py` |
| 改基线/改进算法自动对比逻辑 | `LANE/compare_experiments.py` |
| 改日志 ETA 统计 | `LANE/monitor_training_eta.py` |

## 10. 推荐阅读顺序

如果以后要接着改项目，建议按这个顺序重新熟悉：

1. `README.md`
2. `README_dev.md`
3. `LANE/env_lane.py`
4. `LANE/run_RL_ours.py`
5. `LANE/checkpoint_utils.py`
6. `LANE/model_rwta.py`
7. `LANE/test_video.py`
8. `LANE/compare_experiments.py`

## 11. 修改时的边界规则

- `training_runs/`、`comparison_reports/`、`video_logs_lane/`、`train_parallel_logs/` 基本都属于产物目录，优先通过脚本重生成，不要手工维护。
- `log_model/` 和 `log_text/` 是历史兼容目录，新代码优先写 `training_runs/`。
- `env/gym/` 大部分是上游拷贝，除非你明确在维护 legacy `gymip`，否则不要把 LANE 主线改动放进去。
- 如果新增场景、模型或输出规则，记得同时检查 `test_video.py`、`test_robustness.py`、`compare_experiments.py` 和批处理脚本是否需要同步更新。

## 12. 一句话维护策略

以后每次改项目时，可以直接按下面的思路定位：

- 环境问题看 `env_lane.py`
- 训练问题看 `run_RL_ours.py`
- 路径和 checkpoint 问题看 `checkpoint_utils.py`
- 模型问题看 `model_rwta.py` / `model_critic.py`
- 测试展示问题看 `test_video.py` / `compare_experiments.py`

如果想继续扩展项目，优先让新逻辑沿着这条主线长：  
`env_lane.py -> run_RL_ours.py -> checkpoint_utils.py -> test/compare 脚本`
