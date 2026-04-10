#!/usr/bin/env python3
"""Monitor LANE training logs and estimate remaining time to finish."""

import argparse
import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

EVENT_RE = re.compile(r'^(init|resume|finish),\s*(.+)$')
TRAIN_RE = re.compile(r'^train,\s*(\d+),')
TRAIN_NUM_RE = re.compile(r'train_num=(\d+)')
TIMESTAMP_FORMATS = ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S')


def parse_timestamp(raw_text):
    raw_text = raw_text.strip()
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw_text, fmt)
        except ValueError:
            continue
    return None


def parse_log(log_path):
    lines = Path(log_path).read_text(errors='replace').splitlines()
    status = {
        'first_start_time': None,
        'active_start_time': None,
        'active_start_episode': 0,
        'finish_time': None,
        'train_num': None,
        'last_train_index': 0,
        'last_train_line': None,
        'line_count': len(lines),
    }
    for line in lines:
        if not line:
            continue
        if line.startswith('arguments,'):
            match = TRAIN_NUM_RE.search(line)
            if match is not None:
                status['train_num'] = int(match.group(1))
            continue
        event_match = EVENT_RE.match(line)
        if event_match is not None:
            event_name = event_match.group(1)
            timestamp = parse_timestamp(event_match.group(2))
            if event_name == 'finish':
                status['finish_time'] = timestamp
            else:
                if status['first_start_time'] is None:
                    status['first_start_time'] = timestamp
                status['active_start_time'] = timestamp
                status['active_start_episode'] = status['last_train_index']
            continue
        train_match = TRAIN_RE.match(line)
        if train_match is not None:
            train_index = int(train_match.group(1))
            if train_index >= status['last_train_index']:
                status['last_train_index'] = train_index
                status['last_train_line'] = line
    if status['active_start_time'] is None:
        status['active_start_time'] = status['first_start_time']
    return status


def safe_seconds(delta):
    if delta is None:
        return None
    return max(0.0, delta.total_seconds())


def format_seconds(seconds):
    if seconds is None:
        return 'n/a'
    return str(timedelta(seconds=int(round(max(0.0, seconds)))))


def format_timestamp(timestamp):
    if timestamp is None:
        return 'n/a'
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')


def format_rate(rate):
    if rate is None or rate <= 0:
        return 'n/a'
    return f'{rate * 60.0:.2f} epi/min'


def compute_summary(status, sample_history, override_train_num=None):
    now_wall = datetime.now()
    train_num = override_train_num if override_train_num is not None else status['train_num']
    target_index = None
    if train_num is not None and train_num > 0:
        target_index = max(0, train_num - 1)

    last_train_index = status['last_train_index']
    progress_percent = None
    remaining_episodes = None
    if target_index not in (None, 0):
        progress_percent = 100.0 * min(1.0, last_train_index / target_index)
        remaining_episodes = max(0, target_index - last_train_index)

    elapsed_total_seconds = None
    if status['first_start_time'] is not None:
        end_time = status['finish_time'] if status['finish_time'] is not None else now_wall
        elapsed_total_seconds = safe_seconds(end_time - status['first_start_time'])

    elapsed_active_seconds = None
    completed_active = None
    if status['active_start_time'] is not None:
        end_time = status['finish_time'] if status['finish_time'] is not None else now_wall
        elapsed_active_seconds = safe_seconds(end_time - status['active_start_time'])
        completed_active = max(0, last_train_index - status['active_start_episode'])

    recent_rate = None
    if len(sample_history) >= 2:
        start_time, start_index = sample_history[0]
        end_time, end_index = sample_history[-1]
        delta_time = end_time - start_time
        delta_episode = end_index - start_index
        if delta_time > 0 and delta_episode > 0:
            recent_rate = delta_episode / delta_time

    active_rate = None
    if elapsed_active_seconds and elapsed_active_seconds > 0 and completed_active and completed_active > 0:
        active_rate = completed_active / elapsed_active_seconds

    total_rate = None
    if elapsed_total_seconds and elapsed_total_seconds > 0 and last_train_index > 0:
        total_rate = last_train_index / elapsed_total_seconds

    eta_rate = recent_rate or active_rate or total_rate
    remaining_seconds = None
    eta_time = None
    if status['finish_time'] is not None and remaining_episodes == 0:
        remaining_seconds = 0.0
        eta_time = status['finish_time']
    elif remaining_episodes is not None and eta_rate is not None and eta_rate > 0:
        remaining_seconds = remaining_episodes / eta_rate
        eta_time = now_wall + timedelta(seconds=remaining_seconds)

    return {
        'now_wall': now_wall,
        'train_num': train_num,
        'target_index': target_index,
        'last_train_index': last_train_index,
        'progress_percent': progress_percent,
        'remaining_episodes': remaining_episodes,
        'elapsed_total_seconds': elapsed_total_seconds,
        'elapsed_active_seconds': elapsed_active_seconds,
        'recent_rate': recent_rate,
        'active_rate': active_rate,
        'total_rate': total_rate,
        'eta_rate': eta_rate,
        'remaining_seconds': remaining_seconds,
        'eta_time': eta_time,
    }


def print_summary(log_path, status, summary):
    finished = status['finish_time'] is not None
    status_text = 'finished' if finished else 'running'
    print(f'log: {log_path}')
    print(f'status: {status_text}')
    print(f"checked_at: {format_timestamp(summary['now_wall'])}")
    print(f"last_train: {summary['last_train_index']}")
    if summary['target_index'] is not None:
        print(f"target_train: {summary['target_index']}")
    if summary['progress_percent'] is not None:
        print(f"progress: {summary['progress_percent']:.2f}%")
    if status['last_train_line'] is not None:
        print(f"last_train_line: {status['last_train_line']}")
    print(f"elapsed_total: {format_seconds(summary['elapsed_total_seconds'])}")
    print(f"elapsed_active: {format_seconds(summary['elapsed_active_seconds'])}")
    print(f"recent_speed: {format_rate(summary['recent_rate'])}")
    print(f"active_speed: {format_rate(summary['active_rate'])}")
    print(f"overall_speed: {format_rate(summary['total_rate'])}")
    print(f"remaining: {format_seconds(summary['remaining_seconds'])}")
    print(f"eta_finish: {format_timestamp(summary['eta_time'])}")
    if finished:
        print(f"finish_time: {format_timestamp(status['finish_time'])}")


def build_argument_parser():
    parser = argparse.ArgumentParser(description='Estimate ETA for a LANE training log.')
    parser.add_argument('--log', required=True, help='Path to the structured training log file.')
    parser.add_argument('--train-num', type=int, default=None, help='Override total train_num if needed.')
    parser.add_argument('--poll-seconds', type=float, default=30.0, help='Refresh interval in seconds when watching.')
    parser.add_argument('--history-points', type=int, default=6, help='How many polling samples to use for recent speed.')
    parser.add_argument('--once', action='store_true', help='Print a single ETA snapshot and exit.')
    parser.add_argument('--clear', action='store_true', help='Clear the terminal before each refresh.')
    return parser


def main():
    args = build_argument_parser().parse_args()
    log_path = os.path.abspath(args.log)
    if not os.path.exists(log_path):
        print(f'log file not found: {log_path}', file=sys.stderr)
        return 1

    sample_history = deque(maxlen=max(2, args.history_points))
    while True:
        status = parse_log(log_path)
        sample_history.append((time.monotonic(), status['last_train_index']))
        summary = compute_summary(status, sample_history, override_train_num=args.train_num)

        if args.clear:
            print('[2J[H', end='')
        print_summary(log_path, status, summary)

        if args.once or status['finish_time'] is not None:
            return 0
        print('', flush=True)
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == '__main__':
    raise SystemExit(main())
