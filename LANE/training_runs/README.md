# LANE Training Outputs

训练输出目录按“算法 / 场景 / 产物类型”分层：

- `baseline/`: 基础算法训练产物
- `ours/`: 改进算法训练产物
- `highway_standard`、`highway_dense`、`merge`、`roundabout`: 四种场景单独存放
- `models/`: 权重与 checkpoint
- `logs/`: 文本训练日志

训练脚本 `run_RL_base.py` 与 `run_RL_ours.py` 已自动写入对应目录。
