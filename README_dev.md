# SVPG2023 / LANE 开发维护说明

## 1. 文档定位

这份文档面向后续继续维护、扩展和复现实验的人。相比主 `README.md`，这里更关注：

- 哪些脚本是主线入口
- 每个目录和模块是怎么配合的
- 想改场景、改奖励、改训练逻辑时该去哪里
- 项目里有哪些“历史兼容代码”和“当前主线代码”

## 2. 当前主线在哪里

当前真正需要重点维护的是 `LANE/` 目录。

最核心的主线文件是：

- `LANE/run_RL_base.py`
- `LANE/run_RL_ours.py`
- `LANE/env_lane.py`
- `LANE/model_rwta.py`
- `LANE/model_critic.py`
- `LANE/checkpoint_utils.py`
- `LANE/test_video.py`
- `LANE/test_robustness.py`
- `LANE/compare_experiments.py`

## 3. 当前代码的主数据流

### 3.1 训练流

训练时的主链路是：

```text
run_RL_base.py / run_RL_ours.py
  -> env_lane.py
  -> model_rwta.py / model_mlp.py / model_snnbptt.py
  -> model_critic.py
  -> checkpoint_utils.py
  -> training_runs/
```

含义如下：

- `run_RL_base.py` / `run_RL_ours.py`：训练调度入口
- `env_lane.py`：小车环境与奖励逻辑
- `model_*.py`：actor 网络实现
- `model_critic.py`：训练时的 critic
- `checkpoint_utils.py`：输出目录、checkpoint 自动发现、清理逻辑
- `training_runs/`：最终权重和日志落盘位置

### 3.2 测试流

测试和录像的主链路是：

```text
test_video.py
  -> checkpoint_utils.py 自动找 best
  -> env_lane.py
  -> model_rwta.py 加载 actor
  -> video_logs_lane/
```

### 3.3 对比与鲁棒性流

```text
test_robustness.py -> 读取 best 权重 -> 输出控制台统计
compare_experiments.py -> 读取 baseline/ours best -> 输出 comparison_reports/
plot_*.py -> 基于日志或 CSV 画图
```

## 4. 各目录维护建议

### 4.1 当前正式输出目录

| 路径 | 是否主线 | 说明 |
|---|---|---|
| `LANE/training_runs/` | 是 | 当前正式训练权重与日志目录。 |
| `LANE/video_logs_lane/` | 是 | 当前正式小车视频目录。 |
| `LANE/comparison_reports/` | 是 | 论文对比报告目录。 |
| `LANE/train_parallel_logs/` | 是 | 并行训练终端输出。 |

### 4.2 历史兼容目录

| 路径 | 是否主线 | 说明 |
|---|---|---|
| `LANE/log_model/` | 否 | 老版权重目录，当前保留作兼容。 |
| `LANE/log_text/` | 否 | 老版日志目录，当前保留作兼容。 |
| `LANE/video_logs/` | 否 | 旧版倒立摆录视频目录。 |
| `LANE/ann2snn/` | 否 | ANN2SNN 临时文件目录。 |
| `LANE/env/` | 否 | vendored gym 源码树，主要服务旧版倒立摆流程。 |

## 5. 各核心脚本怎么维护

### 5.1 `run_RL_base.py`

适合保留为基线，不建议在这里塞太多新功能。它更像：

- 原始训练逻辑兼容入口
- 对照组脚本
- 基础实验复现脚本

如果你需要新增论文主线改进，优先加到 `run_RL_ours.py`。

### 5.2 `run_RL_ours.py`

这是当前最值得维护的训练入口。所有与毕设核心方法相关的改动，原则上优先放这里，比如：

- 训练稳定化
- 新的课程学习策略
- 新的 warm start 逻辑
- 新的最优模型评估指标
- 新的噪声注入与鲁棒性训练方式

### 5.3 `env_lane.py`

如果你要改环境相关逻辑，基本都在这里。常见修改点包括：

- 新增场景：改 `SCENARIO_PRESETS`
- 改车流密度：改 `TRAFFIC_VEHICLE_COUNT` 或场景交通蓝图
- 改奖励：改 `_shape_reward()`
- 改成功判定：改 `_scenario_completed_now()`
- 改视频目录：改 `_build_video_recording_target()`

### 5.4 `checkpoint_utils.py`

这个文件统一管理：

- `training_runs` 的目录结构
- `baseline` / `ours` 分类
- checkpoint 自动发现
- 日志路径推断
- 训练结束后的冗余权重清理

如果你后续还要继续整理实验产物，优先改这里，不要在各个脚本里重复拼路径。

## 6. 当前权重规则

当前每个最优实验通常只保留 4 个文件：

- actor：`*_best_w_1.pt`
- actor：`*_best_b_1.pt`
- actor：`*_best_m_1.pt`
- critic：`*critic_best_1.pt`

原因：

- `w / b / m` 是 RWTA actor 推理真正需要的三部分
- `critic_best_1.pt` 用于训练恢复或分析
- `_2` 和 `current` 已在训练结束时自动清理，减少目录混乱

## 7. 当前默认加载规则

### 7.1 测试视频

`test_video.py` 默认：

- 自动寻找最新的 `best` checkpoint
- 根据 `--checkpoint_kind` 区分 `base` 或 `ours`
- 默认不会优先读 `current`

### 7.2 鲁棒性评估

`test_robustness.py` 当前自动发现逻辑也已经收紧为：

- 只自动枚举 `*_best_w_1.pt`

### 7.3 实验对比

`compare_experiments.py` 会：

- 自动发现 `baseline` 与 `ours` 的最新 `best`
- 读取对应日志和权重
- 生成统一格式报告

## 8. 你如果要扩展项目，建议按下面改

### 8.1 新增一个道路场景

建议顺序：

1. 在 `env_lane.py` 的 `SCENARIO_PRESETS` 中注册新场景
2. 补充该场景的交通配置、成功条件、奖励塑形
3. 确认 `checkpoint_utils.scenario_dirname()` 输出符合目录命名预期
4. 根据需要在 `video_logs_lane` 骨架中加入新场景目录
5. 修改训练脚本或批量脚本，把新场景加入训练入口
6. 修改 `compare_experiments.py` 的场景套件

### 8.2 更换或新增模型

建议顺序：

1. 在 `model_*.py` 中加入新模型定义
2. 在 `run_RL_base.py` / `run_RL_ours.py` 中补模型分支
3. 确认保存 / 加载规则与 `checkpoint_utils` 匹配
4. 更新测试脚本中的模型构造逻辑

### 8.3 调整训练策略

如果是主线改进，优先看这些位置：

- `run_RL_ours.py` 中的 `compute_gae()`
- `run_RL_ours.py` 中的 `update_policy()`
- `run_RL_ours.py` 中的课程学习相关函数
- `run_RL_ours.py` 中的最优模型选择函数 `is_better_lane_checkpoint()`

## 9. 常用维护命令

进入主线目录：

```bash
cd LANE
```

跑基础算法：

```bash
python3 run_RL_base.py --model rwtaspk --road_scenario highway --traffic_level standard --ignore_checkpoint --skip_post_tests
```

跑改进算法：

```bash
python3 run_RL_ours.py --model rwtaspk --road_scenario merge --ignore_checkpoint
```

录视频：

```bash
python3 test_video.py --road_scenario merge --checkpoint_kind ours
```

做对比：

```bash
python3 compare_experiments.py --device cpu
```

并行训练：

```bash
bash train_lane_scenario_suite_parallel.sh
```

## 10. 当前项目的几个“坑点”

### 10.1 参数里仍然叫 `task=gymip`

这是历史兼容，不代表现在跑的是倒立摆。当前主线 `gymip` 实际已经被 LANE 小车环境接管。

### 10.2 不要优先往 `log_model/` 和 `log_text/` 里写新结果

当前正式输出目录已经是：

- `training_runs/`
- `video_logs_lane/`

旧目录尽量只保留兼容用途。

### 10.3 `env/` 不建议随便删

虽然小车主线主要走 `gymnasium + highway_env`，但旧版 `prepare_gymip.py` 和兼容任务仍然会用到这套 vendored gym 代码。

### 10.4 `run_RL_base.py` 默认会在训练后跑很长的旧版后处理测试

如果你只是想训练出模型，记得加：

```bash
--skip_post_tests
```

### 10.5 `apply best` 是当前默认策略

现在测试、录像、对比都默认偏向最优模型。如果你手动传 `--checkpoint_prefix` 指向别的 checkpoint，那才会覆盖默认行为。

## 11. 文件阅读优先级建议

如果你是后续维护者，建议按下面顺序读代码：

1. `README.md`
2. `README_dev.md`
3. `LANE/run_RL_ours.py`
4. `LANE/env_lane.py`
5. `LANE/checkpoint_utils.py`
6. `LANE/test_video.py`
7. `LANE/compare_experiments.py`
8. `LANE/run_RL_base.py`
9. 其他兼容模块

## 12. 结论

当前仓库虽然源于原始 SVPG 论文代码，但如果从维护角度看，实际上已经形成了一个比较明确的主线：

- `LANE` 是核心工作区
- `run_RL_ours.py` 是核心训练入口
- `env_lane.py` 是核心环境逻辑
- `checkpoint_utils.py` 是核心产物管理模块
- `training_runs/` 与 `video_logs_lane/` 是当前正式结果目录

如果以后继续扩展，最稳妥的原则是：

> 新功能优先接到 `LANE` 主线，不要把新逻辑再散回旧兼容目录里。
