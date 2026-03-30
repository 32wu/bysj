#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"
LOG_DIR="${LOG_DIR:-train_parallel_logs}"
mkdir -p "$LOG_DIR"

PIDS=()
NAMES=()
LAST_PID=""

launch_job() {
    local job_name="$1"
    shift
    echo "[launch] $job_name" >&2
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"         python3 "$@" --cuda 0 --thread "$CPU_THREADS_PER_JOB" --ignore_checkpoint         > "$LOG_DIR/${job_name}.log" 2>&1 &
    LAST_PID=$!
}

launch_job highway_standard run_RL_base.py --model rwtaspk --road_scenario highway --traffic_level standard --skip_post_tests
PIDS+=("$LAST_PID")
NAMES+=("highway_standard")
launch_job highway_dense run_RL_ours.py --model rwtaspk --road_scenario highway --traffic_level dense
PIDS+=("$LAST_PID")
NAMES+=("highway_dense")
launch_job merge run_RL_ours.py --model rwtaspk --road_scenario merge
PIDS+=("$LAST_PID")
NAMES+=("merge")
launch_job roundabout run_RL_ours.py --model rwtaspk --road_scenario roundabout
PIDS+=("$LAST_PID")
NAMES+=("roundabout")

cleanup() {
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM

for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    name="${NAMES[$idx]}"
    if wait "$pid"; then
        echo "[done] $name"
    else
        echo "[fail] $name" >&2
        exit 1
    fi
done

echo "[all_done] parallel scenario suite finished"
