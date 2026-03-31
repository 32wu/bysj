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
        "--checkpoint_kind",
        type=str,
        default="ours",
        choices=["ours", "base", "baseline"],
        help="Which checkpoint family to auto-load when --checkpoint_prefix is not provided.",
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
    parser.add_argument(
        "--video_tag",
        type=str,
        default="",
        help="Optional extra label to append to recorded video filenames.",
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
    checkpoint_kind = "baseline" if args.checkpoint_kind in ["base", "baseline"] else "ours"
    return checkpoint_utils.find_latest_checkpoint_prefix(
        kind=checkpoint_kind,
        road_scenario=args.road_scenario,
        traffic_level=args.traffic_level,
        best_only=True,
    )


def resolve_video_run_kind(args, checkpoint_prefix):
    inferred_kind = checkpoint_utils.infer_run_kind_from_prefix(checkpoint_prefix)
    if inferred_kind is not None:
        return inferred_kind
    return "baseline" if args.checkpoint_kind in ["base", "baseline"] else "ours"


def build_video_tag(args):
    tag_parts = [f"failure{int(round(args.failure_rate * 100)):03d}"]
    if args.seed is not None:
        tag_parts.append(f"seed{args.seed}")
    if args.video_tag:
        tag_parts.append(args.video_tag)
    return '_'.join(tag_parts)


def main():
    args = parse_args()

    print("🎬 自动驾驶导演组：【毕设压轴】执行器故障/方向盘失控测试启动...")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    record_video = not args.no_video

    checkpoint_prefix = resolve_checkpoint_prefix(args)
    video_run_kind = resolve_video_run_kind(args, checkpoint_prefix)

    env = env_lane.GymLane(
        dev=device,
        road_scenario=args.road_scenario,
        traffic_level=args.traffic_level,
    )
    env.init_test(
        record_video=record_video,
        seed=args.seed,
        video_tag=build_video_tag(args) if record_video else None,
        video_run_kind=video_run_kind,
    )
    if record_video:
        print(f"📹 本次录像目录: {env.video_folder}")
        print(f"🗂️ 录像分类: {env.video_run_kind}/{checkpoint_utils.scenario_dirname(args.road_scenario, args.traffic_level)}")
        print(f"📝 文件名前缀: {env.video_name_prefix}")
    else:
        print("📹 当前关闭录像，仅做环境测试。")

    model = build_model(device)

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
        if env.video_path_hint is not None:
            print(f"🎞️ 录像文件模式: {env.video_path_hint}")


if __name__ == "__main__":
    main()
