from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import zipfile
from .artifacts import atomic_json, fingerprint
from .config import load_config, v14_config, v16_config, v18_config, build_matrix
from .protocol import source_identity

def utc():
    return datetime.now(timezone.utc).isoformat()

class Journal:

    def __init__(self, path):
        self.path = Path(path)
        self.started = time.monotonic()
        now = utc()
        self.value = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {'schema_version': 1, 'first_started_utc': now, 'attempts': [], 'interpretation': 'Monotonic attempt time, not summed GPU task time. Offline gaps excluded. Abruptly killed attempts retain a lower bound from their last heartbeat. Final tar export excluded.'}
        for old in self.value['attempts']:
            if old['status'] == 'running':
                old['status'] = 'interrupted_lower_bound'
                for stage in old['stages']:
                    if stage['status'] == 'running':
                        stage['status'] = 'interrupted_lower_bound'
        self.attempt = dict(started_utc=now, status='running', elapsed_seconds=0.0, stages=[])
        self.value['attempts'].append(self.attempt)
        self.stage_started = None
        self.flush()

    def flush(self):
        elapsed = time.monotonic() - self.started
        self.attempt.update(elapsed_seconds=elapsed, last_observed_utc=utc())
        if self.stage_started is not None:
            self.attempt['stages'][-1]['elapsed_seconds'] = time.monotonic() - self.stage_started
        self.value['total_observed_attempt_seconds'] = sum((a['elapsed_seconds'] for a in self.value['attempts']))
        atomic_json(self.path, self.value)

    def end_stage(self, status='completed'):
        if self.stage_started is not None:
            self.flush()
            self.attempt['stages'][-1].update(status=status, finished_utc=utc())
            self.stage_started = None
        self.flush()

    def start_stage(self, name):
        self.end_stage()
        self.stage_started = time.monotonic()
        self.attempt['stages'].append(dict(name=name, started_utc=utc(), status='running', elapsed_seconds=0.0))
        self.flush()
        print(f'[{utc()}] {name}', flush=True)

    def finish(self, status, error=None):
        self.end_stage(status)
        self.attempt.update(status=status, finished_utc=utc())
        if error:
            self.attempt['error'] = error
        self.value['last_finished_utc'] = self.attempt['finished_utc']
        self.flush()

    def command(self, name, args):
        self.start_stage(name)
        with subprocess.Popen([str(x) for x in args]) as process:
            try:
                while True:
                    try:
                        code = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        self.flush()
                if code:
                    raise subprocess.CalledProcessError(code, args)
            except BaseException:
                if process.poll() is None:
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=30)
                raise
        self.end_stage()

def _verify_analysis(path):
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('analysis_manifest.json'))
        if set(z.namelist()) != set(manifest['files']) | {'analysis_manifest.json'}:
            raise ValueError('Analysis archive members differ')
        for name, digest in manifest['files'].items():
            h = sha256()
            with z.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != digest:
                raise ValueError('Analysis archive checksum failed: ' + name)
    return manifest

def archive_analysis(runs_root, output, expected_runs=48):
    from .scheduler import verify_completed_retention
    root = Path(runs_root).resolve()
    output = Path(output).resolve()
    run_paths = sorted(root.rglob('run.json'))
    if len(run_paths) != expected_runs:
        raise ValueError('Analysis archive requires the complete run matrix')
    identities = {}
    for path in run_paths:
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta.get('status') != 'completed':
            raise ValueError('Cannot archive incomplete runs')
        verify_completed_retention(path.parent, meta['configuration_hash'])
        identities[path.parent.relative_to(root).as_posix()] = meta['configuration_hash']
    if output.exists():
        manifest = _verify_analysis(output)
        if manifest['runs_root'] != str(root) or manifest['runs'] != identities or manifest['source_hash'] != source_identity():
            raise ValueError('Analysis archive belongs to different inputs')
        for name in manifest['files']:
            if Path(name).name in ('run.json', 'config.json') and fingerprint(root / name) != manifest['files'][name]:
                raise ValueError('Archived analysis identity changed')
    else:
        files = {}
        for path in run_paths:
            receipt = json.loads((path.parent / 'retention.json').read_text(encoding='utf-8'))
            selected = set(receipt['files']) | {'retention.json'}
            selected.update((p.name for p in path.parent.glob('*.jsonl')))
            for name in sorted(selected):
                asset = (path.parent / name).resolve()
                if not asset.is_relative_to(path.parent.resolve()) or asset.is_symlink():
                    raise ValueError('Unsafe analysis asset')
                if name == 'resume.pt':
                    continue
                if not asset.is_file():
                    raise ValueError('Evidence was removed before analysis archiving: ' + str(asset))
                files[asset.relative_to(root).as_posix()] = fingerprint(asset)
        manifest = dict(schema_version=1, source_hash=source_identity(), runs_root=str(root), runs=identities, files=files, interpretation='Raw sessions, round evidence and compact final/comparison models; replay buffers excluded.')
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_suffix('.zip.pending')
        with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for name in files:
                z.write(root / name, name)
            z.writestr('analysis_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        _verify_analysis(pending)
        os.replace(pending, output)
    atomic_json(output.with_suffix('.receipt.json'), dict(sha256=fingerprint(output), files=len(manifest['files']), runs=identities, source_hash=source_identity(), bytes=output.stat().st_size))
    return output

def execute_workflow(root, runs, config, data, response, devices, affinities, run, archive, *, release='v14', scope='all'):
    if scope not in ('all', 'baselines') or (scope == 'baselines' and release != 'v18'):
        raise ValueError('Invalid formal matrix scope')
    root = Path(root)
    final = root / 'final'
    final.mkdir(parents=True, exist_ok=True)
    freeze = root / 'protocol/protocol_freeze.json'
    targets = root / 'calibration/cost_targets.json'
    cli = [sys.executable, '-u', '-m', 'fedlpd']
    cfg = ['--config', config]
    assets = ['--data-dir', data, '--response', response]
    bound = ['taskset', '-c', affinities.split(';')[0]]
    audits = [root / 'audit-kuaisim', root / 'audit-recsim']
    if not freeze.exists():
        run('data_audit', cli + ['validate-data', '--data-dir', data])
        for env, audit in zip(('kuaisim', 'recsim'), audits):
            if not (audit / 'environment_validation.json').exists():
                run('audit_' + env, bound + cli + ['validate-env'] + cfg + ['--environment', env, '--sessions', '100', '--output', audit, '--device', devices[0]] + assets)
        check = "from fedlpd.config import load_config; from fedlpd.protocol import _preflight_evidence; import sys; from pathlib import Path; c,r,d,p=sys.argv[1:]; _preflight_evidence(load_config(c),None,[Path(r)/'audit-kuaisim',Path(r)/'audit-recsim'],d,p)"
        run('validate_audits', [sys.executable, '-c', check, config, root, data, response])
        run('calibration', bound + cli + ['calibrate'] + cfg + ['--output', root / 'calibration', '--device', devices[0]] + assets)
        run('freeze', cli + ['freeze'] + cfg + ['--targets', targets, '--environment-reports', *audits, '--output', root / 'protocol'] + assets)
    matrix = root / 'matrix-formal.json'
    run('matrix', cli + ['matrix'] + cfg + ['--output', matrix, '--freeze', freeze, '--targets', targets])
    if scope == 'baselines':
        run('select_baselines', [sys.executable, '-m', 'fedlpd.baseline_release', 'select', '--matrix', matrix])
    run('formal', cli + ['run-matrix', '--matrix', matrix, '--output', runs, '--devices', ','.join(devices), '--jobs-per-device', '1', '--cpu-affinities', affinities] + assets)
    archive(runs, final / 'analysis-evidence.zip')
    if scope == 'baselines':
        run('baseline_report', [sys.executable, '-m', 'fedlpd.baseline_release', 'report', '--runs', runs, '--output', final / 'baseline-results'])
        return
    if release == 'v16':
        run('donor_report', [sys.executable, '-u', '-m', 'fedlpd.donor_reporting', '--archive', final / 'analysis-evidence.zip', '--output', final / 'donor-filter'])
    run('diagnostics', bound + cli + ['diagnose', '--runs-root', runs, '--output', final / 'diagnostics', '--device', devices[0]])
    run('finalize', cli + ['finalize-results', '--runs-root', runs, '--output', final / 'paper-results.zip', '--diagnostics-root', final / 'diagnostics'])
    run('report', cli + ['report', '--runs-root', runs, '--output', final / 'tables-and-figures'])

def validate_release_config(c, release):
    from .task_study import study_config
    from .config import v19_config, v20_config, v21_config, v22_config, v23_config
    builders = {'v14': v14_config, 'v16': v16_config, 'v18': v18_config}
    builders['task-study-v17'] = study_config
    builders['optimization-v19'] = v19_config
    builders['optimization-v20'] = v20_config
    builders['optimization-v21'] = v21_config
    builders['optimization-v22'] = v22_config
    builders['optimization-v23'] = v23_config
    builders['optimization-v23-2'] = v23_config
    if release not in builders:
        raise ValueError('Unknown formal release')
    expected = builders[release]()
    expected['training']['cpu_threads'] = 4
    expected['response']['num_workers'] = 0
    if release == 'optimization-v23-2':
        expected['cost']['require_calibrated_target'] = True
        expected['evaluation']['validation_every_round'] = True
    expected['cost'].update(controlled_load=True, jobs_per_device=1)
    if c != expected:
        raise ValueError('The deployment must use the exact formal-' + release + ' configuration')

def validate_runtime(config, preflight, devices, affinities, release='v14'):
    from .deployment import worker_bindings, cpu_allocation
    from .resources import _gpu_snapshot
    p = json.loads(Path(preflight).read_text(encoding='utf-8'))
    c = load_config(config)
    validate_release_config(c, release)
    if p['status'] != 'passed' or p['source_hash'] != source_identity():
        raise ValueError('Preflight source changed')
    if len(devices) != 4 or devices != p['devices'] or worker_bindings(devices) != p['worker_bindings']:
        raise ValueError('GPU allocation changed')
    groups = p['cpu']['groups']
    if cpu_allocation(4, 4)['groups'] != groups or affinities != ';'.join((','.join(map(str, g)) for g in groups)):
        raise ValueError('CPU allocation changed')
    for device in devices:
        gpu = _gpu_snapshot(device)
        if not gpu['process_query_success'] or gpu['other_compute_pids']:
            raise ValueError('GPU is not idle: ' + str(gpu))
    return c

def isolated_runtime_check(config, preflight, devices, affinities, release='v14'):
    code = 'from fedlpd.full_workflow import validate_runtime; import sys,json; c,p,d,a,r=sys.argv[1:]; validate_runtime(c,p,json.loads(d),a,r)'
    subprocess.run([sys.executable, '-c', code, str(config), str(preflight), json.dumps(devices), affinities, release], check=True)

def main():
    env = os.environ
    release = env.get('FEDLPD_RELEASE', 'v14')
    scope = env.get('FEDLPD_MATRIX_SCOPE', 'all')
    root = Path(env.get('FEDLPD_COMPLETE_ROOT', 'results/complete-' + release)).resolve()
    runs = Path(env.get('FEDLPD_RUNS', 'runs/formal-' + release)).resolve()
    config = Path(env['FEDLPD_CONFIG']).resolve()
    data = Path(env['FEDLPD_DATA_DIR']).resolve()
    response = Path(env['FEDLPD_RESPONSE']).resolve()
    devices = env['FEDLPD_DEVICES'].split(',')
    affinities = env['FEDLPD_CPU_AFFINITIES']
    isolated_runtime_check(config, env['FEDLPD_PREFLIGHT'], devices, affinities, release)
    c = load_config(config)
    if not data.is_dir() or not response.is_file():
        raise ValueError('Existing prepared assets are required')
    identity = dict(source_hash=source_identity(), config=c, runs=str(runs), data=str(data), response=str(response), devices=devices, affinities=affinities, mode='full_without_short_pilots')
    if release == 'v18':
        identity['matrix_scope'] = scope
    manifest = root / 'workflow_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != identity:
        raise ValueError('Workflow identity changed; preserve this result directory and use a new one')
    atomic_json(manifest, identity)
    journal = Journal(root / 'final/workflow-timing.json')

    def archive(*args):
        expected = 12 if scope == 'baselines' else len(build_matrix(c, include_random=True))
        journal.start_stage('analysis_archive')
        archive_analysis(*args, expected_runs=expected)
        journal.end_stage()
    try:
        execute_workflow(root, runs, config, data, response, devices, affinities, journal.command, archive, release=release, scope=scope)
    except BaseException as exc:
        journal.finish('failed', f'{type(exc).__name__}: {exc}')
        raise
    journal.finish('completed')
    atomic_json(root / 'final/workflow_manifest.json', identity)
    print('Complete: ' + str(root / 'final'), flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
