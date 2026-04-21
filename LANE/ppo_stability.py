STABLE_HIGHWAY_STANDARD_PPO = {
    'lr': 1.0e-4,
    'critic_lr': 1.5e-4,
    'gamma': 0.998,
    'gae_lambda': 0.95,
    'entropy': 0.015,
    'entropy_min': 0.001,
    'entropy_decay': 0.999,
    'entropy_warmup_scale': 1.10,
    'entropy_warmup_episodes': 50,
    'PPO_epochs': 4,
    'eps_clip': 0.20,
    'adv_norm': 1,
    'reward_scale': 0.80,
    'grad_clip': 0.50,
}


def scheduled_entropy(
    initial_entropy,
    entropy_min,
    entropy_decay,
    episode_index,
    adaptive_entropy=None,
    entropy_warmup_scale=1.0,
    entropy_warmup_episodes=0,
):
    episode_index = max(0, int(episode_index))
    initial_entropy = float(initial_entropy)
    entropy_min = float(entropy_min)
    entropy_decay = float(entropy_decay)
    entropy_warmup_scale = max(1.0, float(entropy_warmup_scale))
    entropy_warmup_episodes = max(0, int(entropy_warmup_episodes))

    warmup_multiplier = 1.0
    if entropy_warmup_episodes > 0 and episode_index < entropy_warmup_episodes:
        warmup_progress = float(episode_index) / max(1, entropy_warmup_episodes)
        warmup_multiplier = 1.0 + (entropy_warmup_scale - 1.0) * (1.0 - warmup_progress)

    decay_value = max(entropy_min, initial_entropy * warmup_multiplier * (entropy_decay ** episode_index))
    if adaptive_entropy is None:
        return decay_value
    return max(entropy_min, min(float(adaptive_entropy), decay_value))


def should_restore_best_checkpoint(current_metrics, best_length, best_collision):
    best_length = float(best_length)
    best_collision = float(best_collision)
    if best_length < 80.0:
        return False
    current_length = float(current_metrics.get('mean_length', 0.0))
    current_collision = float(current_metrics.get('collision_rate', 1.0))
    return (
        current_length < max(24.0, 0.45 * best_length) and
        current_collision >= min(1.0, best_collision + 0.20)
    )
