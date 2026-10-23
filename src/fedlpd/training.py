from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json, os, platform, random, signal, time, uuid, zipfile
import numpy as np
import torch
from .config import validate_config, stable_hash, task_hash, protocol_hash, evaluation_hash, method_config_hash, backbone_hash, scheduled_round, validation_steps, run_id
from .artifacts import atomic_json, fingerprint, LogBook, restore_log_positions
from .evaluation import evaluate, initialization_manifest
from .costs import TrainingTimer, MessageLedger, payload_descriptor, target_outcome

def _atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with tmp.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def _assets(config, data_dir, response_path):
    from .protocol import source_identity
    from .config import environment_config
    c = environment_config(config, config['environment'], concrete=True)
    assets = {'implementation_sha256': source_identity()}
    if c['environment'] == 'recsim':
        data_dir = None
        response_path = None
        if not c.get('smoke', False):
            from .envs.recsim_backend import load_backend
            assets['recsim_backend'] = load_backend()[1]
    if data_dir:
        p = Path(data_dir)
        if not c.get('smoke', False) and (p / 'metadata.json').exists():
            from .data.preparation import validate_prepared_config
            validate_prepared_config(json.loads((p / 'metadata.json').read_text(encoding='utf-8-sig')), c)
        for name in ('manifest.json', 'metadata.json'):
            if (p / name).exists():
                assets['data_' + name] = fingerprint(p / name)
    if response_path:
        assets['response'] = fingerprint(response_path)
    c['assets'] = assets
    return c

def _seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)

def _checkpoint_agents(agents):
    return [a.save_state(include_replay=False) for a in agents]

def load_agents_from_checkpoint(run_dir, requests_per_client=None, device='cpu'):
    from .baselines import make_agent
    directory = Path(run_dir)
    if requests_per_client is None:
        paths = list((directory / 'checkpoints').glob('request-*.pt'))
        if not paths:
            raise FileNotFoundError('No model checkpoints')
        path = max(paths, key=lambda x: int(x.stem.split('-')[-1]))
    else:
        path = directory / 'checkpoints' / f'request-{requests_per_client}.pt'
    state = torch.load(path, map_location=device, weights_only=False)
    agents = []
    for payload in state['agents']:
        a = make_agent(payload['history_shape'], payload['context_dim'], payload['item_features'], state['config'], device)
        a.load_state(payload, load_replay=False)
        agents.append(a)
    return (agents, state['config'])

def _hardware(device, config):
    info = dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(), torch=str(torch.__version__), device_type=str(device).split(':')[0], cpu_threads=config['training']['cpu_threads'], client_order=list(range(config['env']['num_clients'])), precision='float32', gpu_name=None)
    if str(device).startswith('cuda'):
        info['gpu_name'] = torch.cuda.get_device_name(device)
    return info

def _retention_check(out, c, request, validation_records):
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    test = {r['requests_per_client']: r for r in metrics if r.get('split') == 'test'}
    if request not in test or test[request]['eval_sessions'] != c['evaluation']['final_sessions']:
        raise ValueError('Missing final test evaluation; retaining recovery material')
    if c['method'] != 'random':
        expected = {0, request} | set(range(c['evaluation']['every'], request + 1, c['evaluation']['every']))
        comparison = c['evaluation']['comparison_requests']
        if comparison <= request:
            expected.add(comparison)
        if not expected <= test.keys():
            raise ValueError('Missing required test curve checkpoints')
        if (c['evaluation'].get('matched_comparison_all_methods', False) or (c['environment'] == 'kuaisim' and c['method'] in ('full', 'local'))) and comparison <= request:
            if test[comparison]['eval_sessions'] != c['evaluation']['final_sessions']:
                raise ValueError('Missing matched comparison evaluation')
        if c['method'] in ('local', 'fedavg', 'full', 'donor_filter_off', 'central_shared', 'fedrag', 'gpfedrec'):
            if {r['requests_per_client'] for r in validation_records} != set(validation_steps(c)):
                raise ValueError('Missing validation trajectory; retaining recovery material')
    needed = ['run.json', 'config.json', 'packages.json', 'metrics.jsonl', 'evaluation_sessions.jsonl', 'message_ledger.jsonl', 'resources.jsonl', 'source/source.zip', 'source/dependencies.json', 'test-initializations.json', 'validation-initializations.json']
    if c.get('diagnostics', {}).get('enabled'):
        needed.append('learning.jsonl')
    if c.get('study'):
        needed.append('task-sampling.json')
    for name in needed:
        if not (out / name).is_file():
            raise ValueError('Missing retained artifact ' + name)
    metadata = json.loads((out / 'run.json').read_text(encoding='utf-8'))
    for attempt in metadata.get('prior_attempts', []):
        archive = out / attempt['archive']
        if not archive.is_file() or fingerprint(archive) != attempt['archive_sha256']:
            raise ValueError('Prior attempt archive changed')
        needed.append(attempt['archive'])
    ledger = [json.loads(x) for x in (out / 'message_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    if len({e['event_id'] for e in ledger}) != len(ledger):
        raise ValueError('Duplicate communication ledger events')
    if sum((e['payload_nbytes'] for e in ledger)) != metadata['communication_bytes']:
        raise ValueError('Incomplete communication ledger')
    for e in ledger:
        if e['payload_nbytes'] != e['metadata_nbytes'] + sum((t['nbytes'] for t in e['tensors'])):
            raise ValueError('Invalid communication ledger bytes')
    if c['method'] not in ('local', 'local_fixed', 'random', 'central_shared'):
        if sum((e['message_type'] == 'projector_init' for e in ledger)) != c['env']['num_clients']:
            raise ValueError('Missing initialization communication')
        needed.append('rounds.jsonl')
        if not (out / 'rounds.jsonl').exists():
            if request > c['training']['warmup']:
                raise ValueError('Missing federation rounds')
            (out / 'rounds.jsonl').touch()
        rounds = [json.loads(x) for x in (out / 'rounds.jsonl').read_text(encoding='utf-8').splitlines()]
        expected = list(range(c['training']['warmup'] + c['training']['communication_every'], request + 1, c['training']['communication_every']))
        if [r['requests_per_client'] for r in rounds] != expected:
            raise ValueError('Incomplete federation round schedule')
        if sum((r['communication_bytes'] for r in rounds)) != sum((e['payload_nbytes'] for e in ledger if e['phase'] == 'federation')):
            raise ValueError('Round and communication ledger totals disagree')
    sessions = {}
    for line in (out / 'evaluation_sessions.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        key = (row['split'], row['requests_per_client'], row['eval_kind'], row['client_id'], row['session'])
        if key in sessions:
            raise ValueError('Duplicate evaluation session')
        if row['length'] != len(row['rewards']) or not np.isfinite(row['rewards']).all():
            raise ValueError('Invalid session rewards')
        sessions[key] = row
    for metric in metrics:
        for client in metric['clients']:
            values = []
            for e in range(metric['eval_sessions']):
                key = (metric['split'], metric['requests_per_client'], metric['eval_kind'], client['client_id'], e)
                if key not in sessions:
                    raise ValueError('Missing evaluation session')
                values.append(sessions[key]['rewards'])
            from .metrics import client_metrics
            actual = client_metrics(values, c['agent']['gamma'])
            if any((not np.isclose(actual[k], client[k], atol=1e-10) for k in ('J', 'A', 'L'))):
                raise ValueError('Evaluation session totals differ from summary')
    if c['environment'] == 'recsim' and c['method'] in ('full', 'alignment_off') and (c['evaluation']['comparison_requests'] <= request):
        name = f"checkpoints/request-{c['evaluation']['comparison_requests']}.pt"
        if not (out / name).exists():
            raise ValueError('Missing mechanism checkpoint')
        needed.append(name)
    if c['storage'].get('analysis_checkpoints', False) and c['method'] != 'random':
        for step in {request, c['evaluation']['comparison_requests']}:
            if step > request:
                continue
            name = f'checkpoints/request-{step}.pt'
            if not (out / name).is_file():
                raise ValueError('Missing analysis checkpoint')
            if name not in needed:
                needed.append(name)
    return {name: fingerprint(out / name) for name in needed}

def run_experiment(config, output_dir, data_dir=None, response_path=None, device='cpu', resume=None, stop_after=None, diagnostic_federation=None):
    from .baselines import make_agent
    from .envs import make_env
    from .federation import federated_round
    from .protocol import check_frozen_protocol, snapshot_source
    from .resources import resource_snapshot, cost_resource_reasons
    invocation_started = time.perf_counter()
    validate_config(config)
    if diagnostic_federation is not None and (config.get('phase') != 'coordinate_diagnostic' or config.get('purpose') != 'pilot' or (not config.get('diagnostics', {}).get('enabled')) or config.get('cost', {}).get('controlled_load')):
        raise ValueError('Diagnostic federation callback requires uncontrolled pilot coordinate_diagnostic')
    coordinate_round = diagnostic_federation or federated_round
    if not config['smoke'] and config.get('purpose', 'formal') in ('formal', 'cost_remeasure') and (not config.get('cost', {}).get('freeze_path')):
        raise ValueError('Formal runs require a precommitted protocol freeze')
    if config['environment'] == 'recsim':
        data_dir = None
        response_path = None
    c = validate_config(_assets(config, data_dir, response_path))
    out = Path(output_dir)
    frozen = None
    if not c['smoke'] and c.get('purpose', 'formal') in ('formal', 'cost_remeasure'):
        frozen = check_frozen_protocol(c)
    if c.get('study') and (not c['smoke']):
        from .task_study import check_study_freeze
        check_study_freeze(c)
    if c.get('trial') and (not c['smoke']):
        from .optimization_trial import check_trial_freeze
        check_trial_freeze(c)
    if str(device).startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable')
    if str(device).startswith('cuda'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.cuda.set_device(torch.device(device))
        torch.cuda.reset_peak_memory_stats(device)
    if not c['smoke'] and c['environment'] == 'kuaisim' and (not data_dir or not response_path):
        raise ValueError('Formal KuaiSim requires prepared real data and a frozen response checkpoint')
    old = {}
    if (out / 'run.json').exists():
        old = json.loads((out / 'run.json').read_text(encoding='utf-8'))
        if resume is None:
            raise FileExistsError(f'Run already exists: {out}; use --resume')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(c['training']['cpu_threads']))
    torch.use_deterministic_algorithms(bool(c['training'].get('deterministic', True)))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    _seed(c['seed'])
    timer = TrainingTimer(device)
    compact = c.get('storage', {}).get('mode', 'compact') == 'compact'
    n = c['env']['num_clients']
    budget = c['training']['requests_per_client']
    diagnostic = c.get('diagnostics', {}).get('enabled', False)
    rngs = [np.random.default_rng(np.random.SeedSequence([c['seed'], k, 1947])) for k in range(n)]
    replay_rngs = [np.random.default_rng(np.random.SeedSequence([c['seed'], k, 1948])) for k in range(n)]
    fed_rng = np.random.default_rng(np.random.SeedSequence([c['seed'], 9817]))
    envs = []
    agents = []
    observations = []
    for k in range(n):
        envseed = int(np.random.SeedSequence([c['seed'], k, 7829]).generate_state(1)[0])
        env = make_env(c['environment'], k, envseed, c, data_dir=data_dir, split='train', response_path=response_path, device=device)
        with timer.region('training_elapsed_seconds'), timer.region('initialization_seconds'):
            obs = env.reset()
            envs.append(env)
            observations.append(obs)
            if k and (not np.array_equal(envs[0].catalog_features, env.catalog_features)):
                raise ValueError('Clients must share catalog semantics')
            if c['method'] != 'random':
                torch.manual_seed(c['seed'] + 1009 * k)
                a = make_agent(obs.history.shape, obs.context.size, env.catalog_features, c, device)
                if agents:
                    a.load_projector(agents[0].projector_state())
                agents.append(a)
    central = c['method'] == 'central_shared'
    if central:
        from .centralized import bind_shared_heads, central_update
        bind_shared_heads(agents)
    request = 0
    episodes = [0] * n
    session_rewards = [[] for _ in range(n)]
    federation_state = {}
    aggregation_count = 0
    distillation_count = 0
    communication_bytes = 0
    validation_records = []
    identity = stable_hash(c)
    attempt_id = old.get('attempt_id', uuid.uuid4().hex)
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state['configuration_hash'] != identity:
            raise ValueError('Resume configuration/protocol differs from checkpoint')
    prior_attempts = list(old.get('prior_attempts', []))
    if resume and old.get('status') in ('failed', 'running'):
        archive = out / 'attempts' / (attempt_id + '-' + uuid.uuid4().hex + '.zip')
        archive.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as saved:
            for path in sorted(out.glob('*.json*')):
                saved.write(path, path.name)
        prior_digest = fingerprint(archive)
        prior_attempts.append(dict(attempt_id=attempt_id, status=old['status'], requests_per_client=old.get('requests_per_client'), error=old.get('error'), communication_bytes=old.get('communication_bytes'), timing=old.get('timing'), archive=archive.relative_to(out).as_posix(), archive_sha256=prior_digest))
        attempt_id = uuid.uuid4().hex
    restart_overhead = None if resume and old.get('status') in ('failed', 'running') else 0.0
    if resume:
        request = state['requests_per_client']
        episodes = state['episodes']
        session_rewards = state['session_rewards']
        observations = state['observations']
        for a, s in zip(agents, state['agents']):
            a.load_state(s)
        for e, s in zip(envs, state['environments']):
            e.load_state_dict(s)
        for r, s in zip(rngs, state['client_rngs']):
            r.bit_generator.state = s
        for r, s in zip(replay_rngs, state['replay_rngs']):
            r.bit_generator.state = s
        fed_rng.bit_generator.state = state['federation_rng']
        federation_state = state['federation_state']
        timer.values.update(state['timing'])
        validation_records = state['validation_records']
        aggregation_count = state['completed_aggregation_count']
        distillation_count = state['distillation_update_count']
        communication_bytes = state['communication_bytes']
        torch.set_rng_state(state['torch_rng'].cpu())
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if str(device).startswith('cuda') and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        restore_log_positions(out, state['log_positions'])
    source = snapshot_source(out / 'source')
    atomic_json(out / 'config.json', c)
    atomic_json(out / 'packages.json', json.loads((out / 'source/dependencies.json').read_text(encoding='utf-8')))
    manifests = {split: initialization_manifest(c, split) for split in ('test', 'validation')}
    for split, manifest in manifests.items():
        atomic_json(out / (split + '-initializations.json'), manifest)
    initial_resources = resource_snapshot(device, c['training']['cpu_threads'])
    target = frozen['target'] if frozen else None
    if c['cost'].get('require_calibrated_target'):
        from .trial_costs import load_trial_target
        target = load_trial_target(c)
    hardware = _hardware(device, c)
    cost_reasons = []
    hardware['cpu_model'] = initial_resources.get('cpu_model')
    gpu = initial_resources.get('gpu') or {}
    hardware['gpu_driver'] = gpu.get('driver_version')
    hardware['cuda_runtime'] = initial_resources.get('cuda_runtime')
    hardware['cpu_affinity_count'] = len(initial_resources['cpu_affinity']) if initial_resources.get('cpu_affinity') else None
    cost_reasons.extend(cost_resource_reasons(initial_resources, c['cost']['controlled_load'], c['cost']['jobs_per_device']))
    if c['smoke']:
        cost_reasons.append('smoke')
    if c.get('purpose') not in ('formal', 'cost_remeasure'):
        cost_reasons.append('nonformal purpose')
    if not c['cost'].get('controlled_load'):
        cost_reasons.append('controlled load not declared')
    if c['cost'].get('jobs_per_device', 1) != 1:
        cost_reasons.append('shared device allocation')
    if restart_overhead is None:
        cost_reasons.append('unrecoverable timing boundary after abnormal restart')
    cost_reasons = sorted(set(cost_reasons + old.get('persistent_cost_ineligible_reasons', [])))
    metadata = dict(environment=c['environment'], method=c['method'], seed=c['seed'], smoke=c['smoke'], design_version='2.0', run_id=run_id(c), attempt_id=attempt_id, prior_attempts=prior_attempts, purpose=c.get('purpose', 'formal'), phase=c.get('phase'), status='running', performance_run_id=c.get('performance_run_id'), performance_attempt_id=c.get('performance_attempt_id'), publication_eligible=False, performance_eligible=False, cost_eligible=False, cost_ineligible_reasons=cost_reasons, persistent_cost_ineligible_reasons=cost_reasons, restart_overhead_seconds=restart_overhead, environment_provenance=[getattr(e, 'provenance', {}) for e in envs], task_hash=task_hash(c), protocol_hash=protocol_hash(c), training_protocol_hash=protocol_hash(c), evaluation_hash=evaluation_hash(c), backbone_hash=backbone_hash(c), method_config_hash=method_config_hash(c), configuration_hash=identity, config=c, budget_per_client=budget, device=str(device), hardware=hardware, cost_target=target, cost_target_hash=target['hash'] if target else None, cost_protocol_hash=stable_hash(dict(hardware=hardware, cost={k: v for k, v in c['cost'].items() if not k.endswith('_path')}, targets_sha256=frozen['freeze']['targets_sha256'] if frozen else None)), protocol_freeze_hash=frozen['freeze_sha256'] if frozen else None, python=platform.python_version(), torch=str(torch.__version__), platform=platform.platform(), data_dir=str(Path(data_dir).resolve()) if data_dir else None, response_path=str(Path(response_path).resolve()) if response_path else None, **source)
    counts = []
    for agent in agents:
        backbone = sum((p.numel() for name in ('adapter', 'projector', 'actor', 'critic') for p in getattr(agent, name).parameters()))
        auxiliary = sum((p.numel() for name in ('item_embedding', 'reward_model', 'dynamics_model') if hasattr(agent, name) for p in getattr(agent, name).parameters()))
        counts.append(dict(backbone=backbone, method_specific=auxiliary, total_trainable=backbone + auxiliary))
    metadata['parameter_counts_by_client'] = counts
    atomic_json(out / 'run.json', metadata)
    stop = [False]
    old_handlers = {}

    def handle(*_):
        stop[0] = True
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, handle)
    except ValueError:
        pass

    def record_resources(logs, stage, snapshot=None):
        observed = snapshot if snapshot is not None else resource_snapshot(device, c['training']['cpu_threads'])
        logs.write('resources.jsonl', dict(stage=stage, requests_per_client=request, attempt_id=attempt_id, **observed))
        cost_reasons[:] = sorted(set(cost_reasons + cost_resource_reasons(observed, c['cost']['controlled_load'], c['cost']['jobs_per_device'])))

    def save_resume(logs, ledger):
        record_resources(logs, 'checkpoint')
        timer.synchronize()
        logs.durable()
        with timer.region('checkpoint_seconds'):
            payload = dict(config=c, configuration_hash=identity, requests_per_client=request, agents=[a.save_state() for a in agents], environments=[e.state_dict() for e in envs], observations=observations, episodes=episodes, session_rewards=session_rewards, client_rngs=[r.bit_generator.state for r in rngs], replay_rngs=[r.bit_generator.state for r in replay_rngs], federation_rng=fed_rng.bit_generator.state, federation_state=federation_state, timing=deepcopy(timer.values), torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None, log_positions=logs.positions(), communication_bytes=ledger.total, validation_records=validation_records, completed_aggregation_count=aggregation_count, distillation_update_count=distillation_count)
            _atomic_torch(out / 'resume.pt', payload)

    def save_model():
        analysis = c['storage'].get('analysis_checkpoints', False) and c['method'] != 'random' and (request in (budget, c['evaluation']['comparison_requests']))
        analysis = analysis or (bool(c.get('trial')) and request == budget // 2)
        if compact and (not analysis) and (not (c['environment'] == 'recsim' and c['method'] in ('full', 'alignment_off') and (request == c['evaluation']['comparison_requests']))):
            return
        with timer.region('checkpoint_seconds'):
            _atomic_torch(out / 'checkpoints' / f'request-{request}.pt', dict(config=c, agents=_checkpoint_agents(agents), requests_per_client=request, task_hash=task_hash(c), federation_state=deepcopy(federation_state)))

    def assessment(logs, split, kind):
        with timer.region('validation_seconds' if split == 'validation' else 'test_seconds'):
            record = evaluate(agents, c, request, kind, data_dir, response_path, device, logs, split=split, manifest=manifests[split], completed_aggregation_count=aggregation_count, target=target['J_star'] if target else None)
        timer.values['evaluation_seconds'] = timer.values['validation_seconds'] + timer.values['test_seconds']
        if split == 'validation':
            validation_records.append(record)
        if diagnostic and split == 'validation':
            logs.write('learning.jsonl', dict(requests_per_client=request, stage=kind, clients=[dict(client_id=k, **a.learning_summary()) for k, a in enumerate(agents)]))
    try:
        with LogBook(out) as logs:
            record_resources(logs, 'start', initial_resources)
            (out / 'message_ledger.jsonl').touch(exist_ok=True)
            ledger = MessageLedger(run_id(c), logs, communication_bytes)
            if resume is None:
                if c['method'] not in ('local', 'local_fixed', 'random', 'central_shared'):
                    with timer.region('training_elapsed_seconds'), timer.region('initialization_seconds'):
                        messages = [payload_descriptor('download', 'server', f'client:{k}', 'projector_init', a.projector_state()) for k, a in enumerate(agents)]
                        ledger.record(messages, 0, 0, 'initialization')
                if c['method'] != 'random':
                    if c.get('trial') or c['method'] in ('full', 'local', 'fedavg', 'donor_filter_off', 'central_shared', 'fedrag', 'gpfedrec') or diagnostic:
                        assessment(logs, 'validation', 'validation')
                    assessment(logs, 'test', 'curve')
            if c['method'] == 'random':
                if c.get('trial'):
                    assessment(logs, 'validation', 'validation')
                request = budget
                assessment(logs, 'test', 'final')
                save_model()
                timer.values['training_elapsed_seconds'] = 0.0
                timer.values['initialization_seconds'] = 0.0
            else:
                while request < budget and (stop_after is None or request < stop_after):
                    with timer.region('training_elapsed_seconds'):
                        for k, (env, agent) in enumerate(zip(envs, agents)):
                            obs = observations[k]
                            with timer.region('action_seconds'):
                                slate, action = agent.act(obs, rngs[k], explore=True)
                            if c.get('study'):
                                env.record_exposure(slate)
                            with timer.region('environment_seconds'):
                                result = env.step(slate)
                            with timer.region('rl_seconds'):
                                agent.observe(obs, action, slate, result.reward, result.observation, result.done, episodes[k])
                                if (request + 1) % c['training']['update_every'] == 0:
                                    for _ in range(c['training']['updates_per_request']):
                                        if central:
                                            central_update(agents, k, replay_rngs[k])
                                        else:
                                            agent.update(replay_rngs[k])
                            session_rewards[k].append(result.reward)
                            observations[k] = result.observation
                            if result.done:
                                if not compact:
                                    logs.write('training_sessions.jsonl', dict(requests_per_client=request + 1, client_id=k, episode=episodes[k], return_sum=float(sum(session_rewards[k])), length=len(session_rewards[k]), terminal_reason=result.info.get('terminal_reason', result.info.get('termination_reason', 'done'))))
                                episodes[k] += 1
                                session_rewards[k] = []
                                with timer.region('environment_seconds'):
                                    observations[k] = env.reset()
                        request += 1
                    federating = c['method'] not in ('local', 'local_fixed', 'central_shared') and request > c['training']['warmup'] and ((request - c['training']['warmup']) % c['training']['communication_every'] == 0)
                    if federating:
                        if diagnostic:
                            assessment(logs, 'validation', 'pre_federation')
                        with timer.region('training_elapsed_seconds'), timer.region('federation_seconds'):
                            try:
                                detail = coordinate_round(agents, c['method'], c, fed_rng, scheduled_round(c, request), state=federation_state)
                            except Exception as exc:
                                partial = getattr(exc, 'federation_log', None)
                                if partial:
                                    ledger.record(partial.pop('messages', []), request, scheduled_round(c, request))
                                    logs.write('rounds.jsonl', dict(partial, requests_per_client=request, scheduled_round_index=scheduled_round(c, request)))
                                raise
                            messages = detail.pop('messages', [])
                            ledger.record(messages, request, scheduled_round(c, request))
                            detail['communication_bytes'] = sum((m['payload_nbytes'] for m in messages))
                            aggregation_count += detail.get('completed_aggregation_count', 0)
                            distillation_count += detail.get('distillation_update_count', 0)
                        logs.write('rounds.jsonl', dict(detail, requests_per_client=request, scheduled_round_index=scheduled_round(c, request), cumulative_aggregation_count=aggregation_count, cumulative_distillation_count=distillation_count))
                    if federating and diagnostic:
                        assessment(logs, 'validation', 'post_federation')
                    final = request == budget
                    comparison = request == c['evaluation']['comparison_requests']
                    if (c.get('trial') or c['method'] in ('local', 'fedavg', 'full', 'donor_filter_off', 'central_shared', 'fedrag', 'gpfedrec') or diagnostic) and request in validation_steps(c) and (not (federating and diagnostic)):
                        assessment(logs, 'validation', 'validation')
                    if final or comparison or request % c['evaluation']['every'] == 0:
                        enough = final or (comparison and (c['evaluation'].get('matched_comparison_all_methods', False) or (c['environment'] == 'kuaisim' and c['method'] in ('local', 'full'))))
                        assessment(logs, 'test', 'final' if final else 'comparison' if enough else 'curve')
                    if final or comparison or request % c['training']['checkpoint_every'] == 0 or (c.get('trial') and request == budget // 2):
                        save_model()
                    with timer.region('logging_seconds'):
                        if final or request % c['evaluation']['every'] == 0:
                            logs.write('progress.jsonl', dict(requests_per_client=request, budget_per_client=budget, timing=timer.values.copy()))
                            logs.flush()
                    if final or request % c['training']['resume_every'] == 0 or stop[0] or (stop_after is not None and request >= stop_after):
                        save_resume(logs, ledger)
                    if stop[0] or (stop_after is not None and request >= stop_after):
                        break
            if c['method'] == 'random' and (not compact):
                save_resume(logs, ledger)
            record_resources(logs, 'end')
            complete = request == budget
            if c.get('study'):
                atomic_json(out / 'task-sampling.json', dict(scenario=c['study']['scenario'], clients=[e.sampling_audit() for e in envs]))
                metadata.update(centralized_reference=central, centralized_global_updates=agents[0].updates if central else None, rl_updates_by_client=[a.updates for a in agents] if not central else None, study_scenario=c['study']['scenario'])
            metadata.update(status='validating' if complete else 'paused', requests_per_client=request, timing=timer.values, communication_bytes=ledger.total, scheduled_round_index=scheduled_round(c, request), completed_aggregation_count=aggregation_count, distillation_update_count=distillation_count, wall_seconds_this_invocation=time.perf_counter() - invocation_started, gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0, rounds_to_target=target_outcome(validation_records, target['J_star'], c['method']) if target else None)
            metadata['timing']['invocation_wall_seconds'] = metadata['wall_seconds_this_invocation']
            logs.durable()
            atomic_json(out / 'run.json', metadata)
            if complete:
                _retention_check(out, c, request, validation_records)
                eligible = not c['smoke'] and c.get('purpose') == 'formal'
                metadata.update(status='completed', publication_eligible=eligible, performance_eligible=eligible, cost_eligible=not cost_reasons and c['method'] in ('local', 'fedavg', 'full', 'fedrag', 'gpfedrec'))
                atomic_json(out / 'run.json', metadata)
                receipt = _retention_check(out, c, request, validation_records)
                atomic_json(out / 'retention.json', dict(schema_version=2, files=receipt, configuration_hash=identity))
                if compact:
                    (out / 'resume.pt').unlink(missing_ok=True)
    except BaseException as exc:
        metadata.update(status='failed', requests_per_client=request, error=f'{type(exc).__name__}: {exc}', timing=timer.values, communication_bytes=ledger.total if 'ledger' in locals() else communication_bytes)
        atomic_json(out / 'run.json', metadata)
        raise
    finally:
        for sig, old_handler in old_handlers.items():
            signal.signal(sig, old_handler)
    return out
