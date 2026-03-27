import glob
import os


LANE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(LANE_DIR, 'log_model')
LOG_ROOT = os.path.join(LANE_DIR, 'log_text')
ACTIVE_MODEL_DIR_ENV = 'LANE_ACTIVE_MODEL_DIR'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_model_root(create=False):
    if create:
        ensure_dir(MODEL_ROOT)
    return MODEL_ROOT


def get_log_root(create=False):
    if create:
        ensure_dir(LOG_ROOT)
    return LOG_ROOT


def scenario_dirname(road_scenario, traffic_level='standard'):
    traffic_level = traffic_level or 'standard'
    if road_scenario == 'highway':
        return f'{road_scenario}_{traffic_level}'
    if traffic_level != 'standard':
        return f'{road_scenario}_{traffic_level}'
    return road_scenario


def get_scenario_model_dir(road_scenario, traffic_level='standard', create=False):
    path = os.path.join(get_model_root(create=create), scenario_dirname(road_scenario, traffic_level))
    if create:
        ensure_dir(path)
    return path


def set_active_model_dir(path, create=False):
    if create:
        ensure_dir(path)
    os.environ[ACTIVE_MODEL_DIR_ENV] = path
    return path


def activate_scenario_model_dir(road_scenario, traffic_level='standard', create=False):
    return set_active_model_dir(
        get_scenario_model_dir(road_scenario, traffic_level, create=create),
        create=create,
    )


def get_active_model_dir(create=False):
    active_dir = os.environ.get(ACTIVE_MODEL_DIR_ENV, MODEL_ROOT)
    if create:
        ensure_dir(active_dir)
    return active_dir


def resolve_checkpoint_file(name):
    file_name = name if name.endswith('.pt') else name + '.pt'
    if os.path.isabs(file_name):
        return os.path.normpath(file_name)
    if os.path.dirname(file_name):
        return os.path.normpath(file_name)
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


def checkpoint_prefix_from_weight_file(path):
    return os.path.normpath(path[:-len('_w_1.pt')])


def iter_weight_files(pattern='*_w_1.pt', road_scenario=None, traffic_level='standard'):
    if road_scenario is None:
        search_root = get_model_root(create=False)
        search_pattern = os.path.join(search_root, '**', pattern)
        return glob.glob(search_pattern, recursive=True)
    scenario_dir = get_scenario_model_dir(road_scenario, traffic_level, create=False)
    search_pattern = os.path.join(scenario_dir, pattern)
    return glob.glob(search_pattern)


def is_baseline_checkpoint(prefix):
    basename = os.path.basename(prefix)
    return ('_rep' in basename) and ('adaptive' not in basename) and (not basename.startswith('ours_model'))


def is_optimized_checkpoint(prefix):
    basename = os.path.basename(prefix)
    return ('adaptive' in basename) or basename.startswith('ours_model')


def find_latest_checkpoint_prefix(kind=None, road_scenario=None, traffic_level='standard', best_only=True):
    pattern = '*_best_w_1.pt' if best_only else '*_w_1.pt'
    candidates = []
    for path in iter_weight_files(pattern=pattern, road_scenario=road_scenario, traffic_level=traffic_level):
        prefix = checkpoint_prefix_from_weight_file(path)
        if kind == 'baseline' and not is_baseline_checkpoint(prefix):
            continue
        if kind == 'ours' and not is_optimized_checkpoint(prefix):
            continue
        candidates.append((os.path.getmtime(path), prefix))
    if not candidates:
        scope = scenario_dirname(road_scenario, traffic_level) if road_scenario is not None else 'all scenarios'
        kind_name = kind or 'checkpoint'
        raise FileNotFoundError(f'No {kind_name} checkpoint was found under {scope}.')
    candidates.sort(reverse=True)
    return candidates[0][1]
