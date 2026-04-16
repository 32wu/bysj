# 系统总体结构

本项目由 4 个核心模块组成：

--------------------------------------
1. 环境（Environment）
--------------------------------------
文件：env_lane.py

- 基于 highway_env 构建自动驾驶环境
- 支持：
  - highway / merge / roundabout
  - traffic_level（light / standard / dense）
- 包含 reward shaping 和交通建模

--------------------------------------
2. 强化学习框架（RL Framework）
--------------------------------------
文件：run_RL_base.py / run_RL_ours.py

- PPO / REINFORCE
- GAE（优势估计）
- entropy 正则
- rollout buffer
- 并行采样（ours）

--------------------------------------
3. 模型（Models）
--------------------------------------

Baseline：
- MLP（model_mlp.py）

Ours：
- RWTA（model_rwta.py）【主创新】
- SNN（model_snnbptt.py）
- ANN2SNN（model_convert.py）

--------------------------------------
4. 数据与训练（Memory）
--------------------------------------
文件：memory_lib.py

- 存储 trajectory
- 支持 PPO 更新

--------------------------------------
5. 测试与评估（Evaluation）
--------------------------------------
- test_robustness.py（鲁棒性测试）
- plot_robustness.py（鲁棒性曲线）
- plot_training_curve.py（收敛曲线）

--------------------------------------
数据流（核心流程）

state → model → action → env → reward → buffer → update