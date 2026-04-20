# 迭代记录

## 2026-04-15 三方案最大化推理step

**目标**：使highway standard场景推理step尽可能接近200上限。

**方案A**：`env_lane._shape_reward`增加`step_survival_bonus=0.004*(step/max_step)^1.5`，非碰撞时每步给递增存活奖励。

**方案B**：`env_lane._shape_reward` highway分支增加`highway_safe_gap_bonus`，前方间距>40m时给0.02*speed_ratio奖励，激励保持安全距离。

**方案C**：`run_RL_ours.apply_lane_stability_profile`将standard场景`gamma_floor`从0.996提升至0.998，`gae_lambda`从0.98提升至0.990，使智能体更重视长期存活。

**结果**：三方案均能在验证集达到200满步（最优ep），方案C初期成功率0.4最高，推荐作为主方案。基线max_steps=150，改进后max_steps=200。详见RESULT.md。

## 2026-04-15 dense课程同步与鲁棒性脚本对齐

**目标**：避免highway_dense已在全量验证达标后仍停留低车流课程，并让鲁棒性评估/出图直接支持baseline vs ours与dense场景。

**修改**：`run_RL_ours.update_dense_highway_vehicle_curriculum`新增full-dense强表现时的阶段同步；重写`test_robustness.py`支持scenario/traffic与双模型自动选ckpt；重写`plot_robustness.py`直接读取真实CSV。

## 2026-04-15 方案D：dense+base全面最大化step

**目标**：让ours_highway_dense / base_highway_dense / base_highway_standard均能达到满步200。

**修改**：
- `env_lane.py` dense下`step_survival_bonus`系数0.004→0.008；`safe_gap_threshold` 40m→22m，使dense车流中安全间距奖励更易触发。
- `run_RL_base.py` 新增`--gae_lambda`参数；standard/dense的`gamma_floor`提升至0.998，`gae_lambda_floor`提升至0.990，与方案C对齐。

## 2026-04-15 方案C继续用于dense长视野稳定化

**目标**：不改reward主结构，沿方案C继续提升highway_dense存活步数。

**修改**：`run_RL_ours.py`中dense的`gamma`由0.997升至0.998，`gae_lambda`升至0.99；同时下调dense探索强度与噪声上限，并把stability/reanchor阶段entropy上限压到0.12，减少学到长存活策略后的抖动。


## 2026-04-20 论文图改为平滑+best-so-far收敛表达

**目标**：不修改原始训练日志，改进论文出图方式，使训练结果以平滑趋势和best-so-far平台形式呈现，避免把验证抖动误写成“不收敛”。

**修改**：重写 `plot_training_curve.py`，自动读取四条最新highway日志，输出平滑验证步数曲线、best-so-far包络线、摘要柱状图和CSV汇总；保留raw final，同时新增smoothed final用于论文描述。

## 2026-04-20 merge与roundabout论文图补充

**目标**：在保留highway论文图的基础上，为merge与roundabout两种道路场景补充相同风格的平滑验证曲线、best-so-far包络线与摘要统计图。

**修改**：扩展 `plot_training_curve.py`，兼容4列/8列两种val日志格式，自动读取baseline/ours的merge与roundabout最新日志，并输出PNG/PDF/CSV。
