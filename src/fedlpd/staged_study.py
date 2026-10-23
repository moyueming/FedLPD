from pathlib import Path
import json
from .artifacts import atomic_json, fingerprint
from .config import run_id, validation_steps, stable_hash
from .task_study import build_study_matrix

def build_staged_matrix(base=None):
    return [c for c in build_study_matrix(base) if c['study']['scenario'] == 'complementary' and c['method'] in ('local', 'full', 'central_shared')]

def pause_points(jobs):
    return {run_id(c): min((s for s in validation_steps(c) if c['training']['requests_per_client'] // 2 <= s < c['training']['requests_per_client'])) for c in jobs}

def freeze_policy(jobs, path):
    from .protocol import source_identity
    from .task_study import _execution_identity
    value = dict(version='v17.1', source_hash=source_identity(), pause_points=pause_points(jobs), configs={run_id(c): _execution_identity(c) for c in jobs}, decision_split='validation', continue_policy='explicit finish command; all 18 matched runs', final_budgets={'kuaisim': 80000, 'recsim': 40000})
    value['hash'] = stable_hash(value)
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError('Staged policy changed; use a fresh output root')
    atomic_json(path, value)
    return value

def _pause_metadata(directory, step):
    meta = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    c = meta['config']
    if meta['status'] != 'paused' or meta['requests_per_client'] != step or (not 0 < step < meta['budget_per_client']) or (step not in validation_steps(c)):
        raise ValueError('Run is not paused at the scheduled validation point: ' + str(directory))
    if stable_hash(c) != meta['configuration_hash']:
        raise ValueError('Pause configuration identity differs')
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    validation = [r for r in rows if r['split'] == 'validation']
    expected = {s for s in validation_steps(c) if s <= step}
    if len(validation) != len(expected) or {r['requests_per_client'] for r in validation} != expected:
        raise ValueError('Incomplete interim validation trajectory')
    if any((r['eval_sessions'] != c['evaluation']['validation_sessions'] for r in validation)):
        raise ValueError('Validation trajectory session count differs')
    selected = [r for r in rows if r['split'] == 'validation' and r['requests_per_client'] == step]
    if len(selected) != 1 or selected[0]['eval_sessions'] != c['evaluation']['validation_sessions']:
        raise ValueError('Missing fixed-sample interim validation')
    if any((r['requests_per_client'] > step for r in rows)):
        raise ValueError('Metrics extend beyond pause point')
    if not meta['smoke'] and selected[0]['eval_sessions'] != 500:
        raise ValueError('Formal interim validation requires 500 sessions')
    return meta

def seal_pause(directory, step):
    import torch
    directory = Path(directory)
    meta = _pause_metadata(directory, step)
    if meta['config'].get('trial', {}).get('version') in ('v20', 'v21', 'v22', 'v23'):
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'fedlpd.checkpoint_validation', str(directory / 'resume.pt'), str(step), meta['configuration_hash'], str(meta['config']['env']['num_clients'])], check=True)
    else:
        state = torch.load(directory / 'resume.pt', map_location='cpu', weights_only=False)
        if state['requests_per_client'] != step or state['configuration_hash'] != meta['configuration_hash']:
            raise ValueError('Resume checkpoint differs from interim metadata')
        if len(state['agents']) != meta['config']['env']['num_clients']:
            raise ValueError('Incomplete resume agents')
        del state
    required = {'run.json', 'config.json', 'metrics.jsonl', 'evaluation_sessions.jsonl', 'resume.pt', 'validation-initializations.json', 'test-initializations.json', 'message_ledger.jsonl', 'source/source.zip', 'source/dependencies.json'}
    if meta['config'].get('study'):
        required.add('task-sampling.json')
    from .config import scheduled_round
    if meta['method'] not in ('local', 'local_fixed', 'random', 'central_shared') and scheduled_round(meta['config'], step) > 0:
        required.add('rounds.jsonl')
    if meta['config'].get('trial'):
        required.add(f'checkpoints/request-{step}.pt')
    files = {p.relative_to(directory).as_posix() for p in directory.glob('*.json*') if p.name not in ('pause-retention.json', 'retention.json')}
    files.update(required)
    receipt = dict(version=meta['config'].get('trial', {}).get('version', 'v17.1'), requests_per_client=step, configuration_hash=meta['configuration_hash'], files={name: fingerprint(directory / name) for name in sorted(files)})
    atomic_json(directory / 'pause-retention.json', receipt)
    return receipt

def verify_pause(directory, step):
    directory = Path(directory).resolve()
    receipt = json.loads((directory / 'pause-retention.json').read_text(encoding='utf-8'))
    if receipt.get('version') not in ('v17.1', 'v19', 'v20', 'v21', 'v22', 'v23') or receipt.get('requests_per_client') != step:
        raise ValueError('Pause receipt point differs')
    required = {'resume.pt', 'run.json', 'config.json', 'metrics.jsonl', 'evaluation_sessions.jsonl', 'validation-initializations.json', 'test-initializations.json', 'message_ledger.jsonl', 'source/source.zip', 'source/dependencies.json'}
    if receipt['version'] == 'v17.1':
        required.add('task-sampling.json')
    if not required <= receipt.get('files', {}).keys():
        raise ValueError('Incomplete pause receipt')
    for name, digest in receipt['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or fingerprint(path) != digest:
            raise ValueError('Pause evidence checksum failed: ' + str(path))
    meta = _pause_metadata(directory, step)
    if receipt['version'] != meta['config'].get('trial', {}).get('version', 'v17.1'):
        raise ValueError('Pause version differs')
    if receipt['configuration_hash'] != meta['configuration_hash']:
        raise ValueError('Pause receipt configuration differs')
    return receipt

def build_stage_report(runs, output, phase):
    from .staged_study_reporting import build_report
    return build_report(runs, output, phase)
