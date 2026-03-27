#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

python3 run_RL_base.py --model rwtaspk --road_scenario highway --traffic_level standard --cuda "$CUDA_DEVICE" --skip_post_tests --ignore_checkpoint
python3 run_RL_ours.py --model rwtaspk --road_scenario highway --traffic_level dense --cuda "$CUDA_DEVICE"
python3 run_RL_ours.py --model rwtaspk --road_scenario merge --cuda "$CUDA_DEVICE"
python3 run_RL_ours.py --model rwtaspk --road_scenario roundabout --cuda "$CUDA_DEVICE"
