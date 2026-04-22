import glob
import os
import shutil


LANE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(LANE_DIR, 'log_model')
LOG_ROOT = os.path.join(LANE_DIR, 'log_text')
ARTIFACT_ROOT = os.path.join(LANE_DIR, 'training_runs')
MODEL_SUBDIR = 'models'
LOG_SUBDIR = 'logs'
RUN_KIND_ALIASES = {
    'base': 'baseline',
    'baseline': 'baseline',
    'ours': 'ours',
    'improved': 'ours',
    'improve': 'ours',
    'optimized': 'ours',
}
KNOWN_RUN_KINDS = ('baseline', 'ours')
ACTIVE_MODEL_DIR_ENV = 'LANE_ACTIVE_MODEL_DIR'
ACTIVE_LOG_DIR_ENV = 'LANE_ACTIVE_LOG_DIR'
ACTIVE_RUN_KIND_ENV = 'LANE_ACTIVE_RUN_KIND'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def normalize_run_kind(run_kind=None):
    if run_kind in [None, '', 'any', 'all']:
        return None
    normalized = RUN_KIND_ALIASES.get(str(run_kind).strip().lower())
    if normalized is None:
        raise ValueError(f'Unsupported run kind: {run_kind}')
    return normalized


def get_current_run_kind():
    return normalize_run_kind(os.environ.get(ACTIVE_RUN_KIND_ENV))


def get_legacy_model_root(create=False):
    if create:
        ensure_dir(MODEL_ROOT)
    return MODEL_ROOT


def get_legacy_log_root(create=False):
    if create:
        ensure_dir(LOG_ROOT)
    return LOG_ROOT


def get_artifact_root(create=False):
    if create:
        ensure_dir(ARTIFACT_ROOT)
    return ARTIFACT_ROOT


def get_run_root(run_kind=None, create=False):
    normalized_kind = normalize_run_kind(run_kind)
    if normalized_kind is None:
        normalized_kind = get_current_run_kind()
    if normalized_kind is None:
        path = get_artifact_root(create=create)
    else:
        path = os.path.join(get_artifact_root(create=create), normalized_kind)
    if create:
        ensure_dir(path)
    return path


def get_model_root(create=False, run_kind=None):
    return get_run_root(run_kind=run_kind, create=create)


def get_log_root(create=False, run_kind=None):
    return get_run_root(run_kind=run_kind, create=create)


def scenario_dirname(road_scenario, traffic_level='standard'):
    traffic_level = traffic_level or 'standard'
    if road_scenario == 'highway':
        return f'{road_scenario}_{traffic_level}'
    if traffic_level != 'standard':
        return f'{road_scenario}_{traffic_level}'
    return road_scenario


def get_legacy_scenario_model_dir(road_scenario, traffic_level='standard', create=False):
    path = os.path.join(get_legacy_model_root(create=create), scenario_dirname(road_scenario, traffic_level))
    if create:
        ensure_dir(path)
    return path


def get_scenario_root(road_scenario, traffic_level='standard', run_kind=None, create=False):
    path = os.path.join(get_run_root(run_kind=run_kind, create=create), scenario_dirname(road_scenario, traffic_level))
    if create:
        ensure_dir(path)
    return path


def get_scenario_model_dir(road_scenario, traffic_level='standard', run_kind=None, create=False):
    path = os.path.join(
        get_scenario_root(road_scenario, traffic_level, run_kind=run_kind, create=create),
        MODEL_SUBDIR,
    )
    if create:
        ensure_dir(path)
    return path


def get_scenario_log_dir(road_scenario, traffic_level='standard', run_kind=None, create=False):
    path = os.path.join(
        get_scenario_root(road_scenario, traffic_level, run_kind=run_kind, create=create),
        LOG_SUBDIR,
    )
    if create:
        ensure_dir(path)
    return path


def set_active_run_kind(run_kind=None):
    normalized_kind = normalize_run_kind(run_kind)
    if normalized_kind is None:
        os.environ.pop(ACTIVE_RUN_KIND_ENV, None)
    else:
        os.environ[ACTIVE_RUN_KIND_ENV] = normalized_kind
    return normalized_kind


def set_active_model_dir(path, create=False):
    if create:
        ensure_dir(path)
    os.environ[ACTIVE_MODEL_DIR_ENV] = path
    return path


def set_active_log_dir(path, create=False):
    if create:
        ensure_dir(path)
    os.environ[ACTIVE_LOG_DIR_ENV] = path
    return path


def activate_scenario_output_dirs(run_kind, road_scenario, traffic_level='standard', create=False):
    normalized_kind = set_active_run_kind(run_kind)
    model_dir = get_scenario_model_dir(
        road_scenario,
        traffic_level,
        run_kind=normalized_kind,
        create=create,
    )
    log_dir = get_scenario_log_dir(
        road_scenario,
        traffic_level,
        run_kind=normalized_kind,
        create=create,
    )
    set_active_model_dir(model_dir, create=create)
    set_active_log_dir(log_dir, create=create)
    return model_dir, log_dir


def activate_scenario_model_dir(road_scenario, traffic_level='standard', create=False, run_kind=None):
    model_dir, _ = activate_scenario_output_dirs(
        run_kind=run_kind,
        road_scenario=road_scenario,
        traffic_level=traffic_level,
        create=create,
    )
    return model_dir


def get_active_model_dir(create=False):
    active_dir = os.environ.get(ACTIVE_MODEL_DIR_ENV)
    if active_dir is None:
        active_dir = get_legacy_model_root(create=False)
    if create:
        ensure_dir(active_dir)
    return active_dir


def get_active_log_dir(create=False):
    active_dir = os.environ.get(ACTIVE_LOG_DIR_ENV)
    if active_dir is None:
        active_dir = get_legacy_log_root(create=False)
    if create:
        ensure_dir(active_dir)
    return active_dir


def _find_existing_checkpoint_file(file_name):
    matches = []
    search_patterns = [
        os.path.join(get_artifact_root(create=False), '**', MODEL_SUBDIR, file_name),
        os.path.join(get_legacy_model_root(create=False), '**', file_name),
    ]
    for pattern in search_patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    matches = sorted({os.path.normpath(path) for path in matches})
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_checkpoint_file(name):
    file_name = name if name.endswith('.pt') else name + '.pt'
    if os.path.isabs(file_name):
        return os.path.normpath(file_name)
    if os.path.dirname(file_name):
        return os.path.normpath(file_name)
    if ACTIVE_MODEL_DIR_ENV not in os.environ:
        existing_file = _find_existing_checkpoint_file(file_name)
        if existing_file is not None:
            return existing_file
    return os.path.join(get_active_model_dir(create=True), file_name)


def normalize_prefix(prefix_value):
    if prefix_value is None:
        return None
    prefix_value = os.path.normpath(prefix_value)
    if prefix_value.endswith('_w_1.pt'):
        return prefix_value[:-len('_w_1.pt')]
    if prefix_value.endswith('.pt'):
        return prefix_value[:-len('.pt')]
    return prefix_value


def resolve_checkpoint_prefix_path(prefix):
    normalized_prefix = normalize_prefix(prefix)
    if normalized_prefix is None:
        return None
    if os.path.isabs(normalized_prefix):
        return normalized_prefix
    if os.path.dirname(normalized_prefix):
        return os.path.normpath(normalized_prefix)
    probe_suffixes = ['_w_1.pt', '_w_2.pt', '_1.pt', '.pt']
    for suffix in probe_suffixes:
        existing_file = _find_existing_checkpoint_file(normalized_prefix + suffix)
        if existing_file is not None and existing_file.endswith(suffix):
            return existing_file[:-len(suffix)]
    return os.path.join(get_active_model_dir(create=True), normalized_prefix)


def iter_checkpoint_family_files(prefix):
    prefix_path = resolve_checkpoint_prefix_path(prefix)
    if prefix_path is None:
        return []
    return sorted(
        {os.path.normpath(path) for path in glob.glob(prefix_path + '*.pt')}
    )


def checkpoint_family_exists(prefix):
    return len(iter_checkpoint_family_files(prefix)) > 0


def copy_checkpoint_family(source_prefix, target_prefix):
    source_prefix_path = resolve_checkpoint_prefix_path(source_prefix)
    target_prefix_path = resolve_checkpoint_prefix_path(target_prefix)
    copied_files = []
    for source_path in iter_checkpoint_family_files(source_prefix):
        suffix = source_path[len(source_prefix_path):]
        target_path = target_prefix_path + suffix
        ensure_dir(os.path.dirname(target_path))
        shutil.copy2(source_path, target_path)
        copied_files.append(target_path)
    return copied_files


def prune_checkpoint_family(prefix, keep_endings=None):
    keep_endings = tuple(keep_endings or ())
    kept_files = []
    removed_files = []
    for path in iter_checkpoint_family_files(prefix):
        if keep_endings and path.endswith(keep_endings):
            kept_files.append(path)
            continue
        os.remove(path)
        removed_files.append(path)
    return kept_files, removed_files


def cleanup_final_best_checkpoints(
    actor_best_prefix,
    actor_current_prefix=None,
    critic_best_prefix=None,
    critic_current_prefix=None,
    keep_current=False,
):
    keep_endings = ('_1.pt', '_w_1.pt', '_b_1.pt', '_m_1.pt')
    summary = {
        'promoted': [],
        'kept': [],
        'removed': [],
    }

    if actor_best_prefix and (not checkpoint_family_exists(actor_best_prefix)) and actor_current_prefix and checkpoint_family_exists(actor_current_prefix):
        summary['promoted'].extend(copy_checkpoint_family(actor_current_prefix, actor_best_prefix))
    if critic_best_prefix and (not checkpoint_family_exists(critic_best_prefix)) and critic_current_prefix and checkpoint_family_exists(critic_current_prefix):
        summary['promoted'].extend(copy_checkpoint_family(critic_current_prefix, critic_best_prefix))

    actor_best_ready = actor_best_prefix is not None and checkpoint_family_exists(actor_best_prefix)
    critic_best_ready = critic_best_prefix is None or checkpoint_family_exists(critic_best_prefix)

    if actor_best_ready:
        kept_files, removed_files = prune_checkpoint_family(actor_best_prefix, keep_endings=keep_endings)
        summary['kept'].extend(kept_files)
        summary['removed'].extend(removed_files)
    if critic_best_prefix and checkpoint_family_exists(critic_best_prefix):
        kept_files, removed_files = prune_checkpoint_family(critic_best_prefix, keep_endings=keep_endings)
        summary['kept'].extend(kept_files)
        summary['removed'].extend(removed_files)

    if actor_best_ready and actor_current_prefix:
        if keep_current:
            kept_files, removed_files = prune_checkpoint_family(actor_current_prefix, keep_endings=keep_endings)
            summary['kept'].extend(kept_files)
            summary['removed'].extend(removed_files)
        else:
            _, removed_files = prune_checkpoint_family(actor_current_prefix, keep_endings=())
            summary['removed'].extend(removed_files)
    if critic_best_ready and critic_current_prefix:
        if keep_current:
            kept_files, removed_files = prune_checkpoint_family(critic_current_prefix, keep_endings=keep_endings)
            summary['kept'].extend(kept_files)
            summary['removed'].extend(removed_files)
        else:
            _, removed_files = prune_checkpoint_family(critic_current_prefix, keep_endings=())
            summary['removed'].extend(removed_files)

    return summary
def summarize_checkpoint_cleanup(summary):
    return 'promoted=%d, kept=%d, removed=%d' % (
        len(summary.get('promoted', [])),
        len(summary.get('kept', [])),
        len(summary.get('removed', [])),
    )


def checkpoint_prefix_from_weight_file(path):
    return os.path.normpath(path[:-len('_w_1.pt')])


def infer_run_kind_from_prefix(prefix):
    normalized_prefix = normalize_prefix(prefix)
    artifact_root = os.path.normpath(get_artifact_root(create=False))
    try:
        rel_path = os.path.relpath(normalized_prefix, artifact_root)
    except ValueError:
        rel_path = None
    if rel_path is not None and not rel_path.startswith('..'):
        first_part = rel_path.split(os.sep, 1)[0]
        if first_part in KNOWN_RUN_KINDS:
            return first_part
    basename = os.path.basename(normalized_prefix)
    if ('_rep' in basename) and ('adaptive' not in basename) and (not basename.startswith('ours_model')):
        return 'baseline'
    if ('adaptive' in basename) or basename.startswith('ours_model'):
        return 'ours'
    return None


def is_baseline_checkpoint(prefix):
    inferred_kind = infer_run_kind_from_prefix(prefix)
    if inferred_kind is not None:
        return inferred_kind == 'baseline'
    basename = os.path.basename(prefix)
    return ('_rep' in basename) and ('adaptive' not in basename) and (not basename.startswith('ours_model'))


def is_optimized_checkpoint(prefix):
    inferred_kind = infer_run_kind_from_prefix(prefix)
    if inferred_kind is not None:
        return inferred_kind == 'ours'
    basename = os.path.basename(prefix)
    return ('adaptive' in basename) or basename.startswith('ours_model')


def iter_weight_files(pattern='*_w_1.pt', road_scenario=None, traffic_level='standard', run_kind=None, include_legacy=True):
    normalized_kind = normalize_run_kind(run_kind)
    search_patterns = []
    if road_scenario is None:
        if normalized_kind is None:
            search_patterns.append(os.path.join(get_artifact_root(create=False), '**', MODEL_SUBDIR, pattern))
        else:
            search_patterns.append(
                os.path.join(get_run_root(run_kind=normalized_kind, create=False), '**', MODEL_SUBDIR, pattern)
            )
        if include_legacy:
            search_patterns.append(os.path.join(get_legacy_model_root(create=False), '**', pattern))
    else:
        scenario_name = scenario_dirname(road_scenario, traffic_level)
        if normalized_kind is None:
            for kind_name in KNOWN_RUN_KINDS:
                search_patterns.append(
                    os.path.join(get_run_root(run_kind=kind_name, create=False), scenario_name, MODEL_SUBDIR, pattern)
                )
        else:
            search_patterns.append(
                os.path.join(get_run_root(run_kind=normalized_kind, create=False), scenario_name, MODEL_SUBDIR, pattern)
            )
        if include_legacy:
            search_patterns.append(os.path.join(get_legacy_scenario_model_dir(road_scenario, traffic_level), pattern))
    weight_files = []
    seen = set()
    for search_pattern in search_patterns:
        for path in glob.glob(search_pattern, recursive=True):
            normalized_path = os.path.normpath(path)
            if normalized_path in seen:
                continue
            seen.add(normalized_path)
            weight_files.append(normalized_path)
    return weight_files


def _candidate_log_names(prefix_basename):
    candidate_names = [f'log_{prefix_basename}.txt']
    if prefix_basename.endswith('_best'):
        candidate_names.append(f'log_{prefix_basename[:-5]}.txt')
    if prefix_basename.endswith('_current'):
        candidate_names.append(f'log_{prefix_basename[:-8]}.txt')
    ordered_names = []
    seen = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        ordered_names.append(name)
    return ordered_names


def _search_log_file_by_name(file_name):
    matches = []
    search_patterns = [
        os.path.join(get_artifact_root(create=False), '**', LOG_SUBDIR, file_name),
        os.path.join(get_legacy_log_root(create=False), file_name),
    ]
    for pattern in search_patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    if not matches:
        return None
    matches = sorted(
        {os.path.normpath(path) for path in matches},
        key=lambda path: os.path.getmtime(path),
        reverse=True,
    )
    return matches[0]


def infer_log_path(prefix):
    normalized_prefix = normalize_prefix(prefix)
    prefix_basename = os.path.basename(normalized_prefix)
    candidate_names = _candidate_log_names(prefix_basename)
    candidate_dirs = []
    prefix_dir = os.path.dirname(normalized_prefix)
    if prefix_dir and os.path.basename(prefix_dir) == MODEL_SUBDIR:
        candidate_dirs.append(os.path.join(os.path.dirname(prefix_dir), LOG_SUBDIR))
    active_log_dir = os.environ.get(ACTIVE_LOG_DIR_ENV)
    if active_log_dir:
        candidate_dirs.append(active_log_dir)
    candidate_dirs.append(get_legacy_log_root(create=False))
    seen_dirs = set()
    for candidate_dir in candidate_dirs:
        normalized_dir = os.path.normpath(candidate_dir)
        if normalized_dir in seen_dirs:
            continue
        seen_dirs.add(normalized_dir)
        for file_name in candidate_names:
            candidate_path = os.path.join(normalized_dir, file_name)
            if os.path.exists(candidate_path):
                return candidate_path
    for file_name in candidate_names:
        candidate_path = _search_log_file_by_name(file_name)
        if candidate_path is not None:
            return candidate_path
    return None


def find_latest_checkpoint_prefix(kind=None, road_scenario=None, traffic_level='standard', best_only=True):
    normalized_kind = normalize_run_kind(kind)
    pattern = '*_best_w_1.pt' if best_only else '*_w_1.pt'
    candidates = []
    for path in iter_weight_files(
        pattern=pattern,
        road_scenario=road_scenario,
        traffic_level=traffic_level,
        run_kind=normalized_kind,
        include_legacy=True,
    ):
        prefix = checkpoint_prefix_from_weight_file(path)
        if normalized_kind == 'baseline' and not is_baseline_checkpoint(prefix):
            continue
        if normalized_kind == 'ours' and not is_optimized_checkpoint(prefix):
            continue
        candidates.append((os.path.getmtime(path), prefix))
    if not candidates:
        scope = scenario_dirname(road_scenario, traffic_level) if road_scenario is not None else 'all scenarios'
        kind_name = normalized_kind or 'checkpoint'
        raise FileNotFoundError(f'No {kind_name} checkpoint was found under {scope}.')
    candidates.sort(reverse=True)
    return candidates[0][1]
