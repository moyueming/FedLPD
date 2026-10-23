from copy import deepcopy
from pathlib import Path
import json
from .config import v14_config, validate_config, environment_config, smoke_config

def study_config():
    c = v14_config()
    c['purpose'] = 'task_study'
    c['study'] = {'version': 'v17', 'scenario': 'original', 'freeze_path': None}
    c['env']['task_sampling_audit'] = True
    return validate_config(c)

def build_study_matrix(base=None):
    base = deepcopy(base or study_config())
    jobs = []
    for scenario in ('original', 'complementary'):
        for env in ('kuaisim', 'recsim'):
            for method in ('local', 'fedavg', 'full', 'central_shared'):
                for seed in (2026, 2027, 2028):
                    c = deepcopy(base)
                    c['study']['scenario'] = scenario
                    if scenario == 'complementary':
                        c['env']['training_candidate_bias'] = {'strength': 0.8, 'groups': 5, 'version': 1}
                    else:
                        c['env'].pop('training_candidate_bias', None)
                    c = environment_config(c, env, concrete=True)
                    c.update(method=method, seed=seed)
                    c['training']['requests_per_client'] = 80000 if env == 'kuaisim' else 40000
                    jobs.append(validate_config(smoke_config(c) if c.get('smoke') else c))
    return jobs

def _execution_identity(c):
    from .config import stable_hash
    c = deepcopy(c)
    c['study']['freeze_path'] = None
    return stable_hash(c)

def freeze_study(jobs, path, data_dir=None, response_path=None):
    from .training import _assets
    from .protocol import source_identity
    from .config import run_id, stable_hash
    from .artifacts import atomic_json
    path = Path(path).resolve()
    value = dict(schema_version=1, version='v17', source_hash=source_identity(), interpretation='Full-budget task-design diagnostic, no calibrated J*; all planned scenarios are retained.', jobs={run_id(c): _execution_identity(_assets(c, data_dir, response_path)) for c in jobs})
    if len(value['jobs']) != len(jobs):
        raise ValueError('Duplicate study run IDs')
    value['hash'] = stable_hash(value)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Study freeze differs; use a new output root')
    else:
        atomic_json(path, value)
    return path

def check_study_freeze(c):
    from .protocol import source_identity
    from .config import run_id, stable_hash
    filename = c['study'].get('freeze_path')
    if not filename:
        raise ValueError('Task study requires a frozen matrix')
    value = json.loads(Path(filename).read_text(encoding='utf-8'))
    if value.get('hash') != stable_hash({k: v for k, v in value.items() if k != 'hash'}):
        raise ValueError('Study freeze checksum changed')
    if value.get('version') != 'v17' or value['source_hash'] != source_identity():
        raise ValueError('Study source changed')
    if value['jobs'].get(run_id(c)) != _execution_identity(c):
        raise ValueError('Run differs from frozen study config/assets')
    return value
