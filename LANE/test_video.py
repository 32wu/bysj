# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch

import checkpoint_utils
import env_lane
import model_rwta


def parse_args():
    parser = argparse.ArgumentParser(description="Switchable road-scenario video test runner")
    parser.add_argument(
        "--road_scenario",
        type=str,
        default="roundabout",
        choices=sorted(env_lane.SCENARIO_PRESETS.keys()),
        help="Road scenario to test.",
    )
    parser.add_argument(
        "--traffic_level",
        type=str,
        default="standard",
        choices=sorted(env_lane.TRAFFIC_VEHICLE_COUNT.keys()),
        help="Traffic level for the selected scenario.",
    )
    parser.add_argument(
        "--failure_rate",
        type=float,
        default=0.2,
        help="Probability of injecting a random steering/action failure at each step.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default=None,
        help="Optional explicit checkpoint prefix. If omitted, auto-load the latest best checkpoint for the selected scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional environment reset seed for reproducible tests.",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Disable video recording.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.failure_rate <= 1.0:
        parser.error("--failure_rate must be between 0 and 1.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available.")
    return args


def build_model(device):
    return model_rwta.RWTAspike(
        input_size=25,
        output_size=5,
        hid_num=8,
        hid_size=8,
        spk_response_window="uni",
        spk_full_time=42,
        spk_resp_time=40,
        remove_connection_pattern="none",
        optimizer_name="rmsprop",
        optimizer_learning_rate=0.001,
        entropy_ratio=5.0,
        device=device,
    )


def resolve_checkpoint_prefix(args):
    if args.checkpoint_prefix:
        return checkpoint_utils.normalize_prefix(args.checkpoint_prefix)
    return checkpoint_utils.find_latest_checkpoint_prefix(
        kind="ours",
        road_scenario=args.road_scenario,
        traffic_level=args.traffic_level,
        best_only=True,
    )


def main():
    args = parse_args()

    print("🎬 自动驾驶导演组：【毕设压轴】执行器故障/方向盘失控测试启动...")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    record_video = not args.no_video

    env = env_lane.GymLane(
        dev=device,
        road_scenario=args.road_scenario,
        traffic_level=args.traffic_level,
    )
    env.init_test(record_video=record_video, seed=args.seed)
    if record_video:
        print(f"📹 本次录像将保存到: {env.video_folder}")
    else:
        print("📹 当前关闭录像，仅做环境测试。")

    model = build_model(device)

    checkpoint_prefix = resolve_checkpoint_prefix(args)
    checkpoint_path = checkpoint_utils.resolve_checkpoint_file(checkpoint_prefix + "_w_1")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"没有找到固定权重: {checkpoint_path}")
    model.load_model(checkpoint_prefix)
    print(f"✅ 大脑组装成功！当前使用模型: {checkpoint_prefix}")
    print(
        f"🛣️ 当前场景: {args.road_scenario} | 交通密度: {args.traffic_level} | 故障率: {args.failure_rate * 100:.1f}%"
    )

    done = False
    step = 0

    try:
        while not done:
            original_state = env.get_test_observation()

            with torch.no_grad():
                raw_out = model(original_state)
                if isinstance(raw_out, (tuple, list)):
                    action_data = raw_out[0]
                else:
                    action_data = raw_out

                if not torch.is_tensor(action_data):
                    action_data = torch.tensor(action_data, dtype=torch.float32, device=device)

                if action_data.dim() > 1:
                    action_scores = torch.sum(action_data, dim=0)
                else:
                    action_scores = action_data

            if np.random.rand() < args.failure_rate:
                random_action = np.random.randint(0, env.action_num)
                action_scores = torch.zeros_like(action_scores)
                if action_scores.dim() > 0:
                    action_scores[random_action] = 100.0
                else:
                    action_scores = torch.tensor(
                        [100.0 if i == random_action else 0.0 for i in range(env.action_num)],
                        dtype=torch.float32,
                        device=device,
                    )

            try:
                _next_state, _reward, done_flag, _info, step_record = env.make_action(action_scores)
                step = step_record[0]
                if env.done_signal == 1 or done_flag:
                    done = True
            except Exception as exc:
                print(f"❌ 环境交互报错: {exc}")
                break
    finally:
        env.close()

    print(f"✅ 杀青！在 {args.failure_rate * 100:.1f}% 的方向盘失控率下，小车存活了 {step} 步。")
    if record_video:
        print(f"📁 录像已安全保存至: {env.video_folder}")


if __name__ == "__main__":
    main()
