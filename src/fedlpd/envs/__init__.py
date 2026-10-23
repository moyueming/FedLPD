from .validation import validate_environment

def make_env(name, client_id, seed, config, data_dir=None, split='train', response_path=None, device='cpu', evaluation=False):
    c = dict(config.get('env', config))
    c['smoke'] = bool(config.get('smoke', c.get('smoke', False)))
    c['_split'] = split
    c['_sampling_environment'] = name.lower()
    if name.lower() == 'recsim':
        from .recsim import ControlledRecSim
        return ControlledRecSim(client_id, seed, c, evaluation=evaluation)
    if name.lower() in ['kuaisim', 'kuairand']:
        if c['smoke'] and data_dir is None:
            from .recsim import ControlledRecSim
            c['_sampling_environment'] = 'recsim'
            env = ControlledRecSim(client_id, seed, c, evaluation=evaluation)
            env.provenance.update({'backend': 'kuaisim-smoke-fixture', 'publication_eligible': False})
            return env
        from .kuaisim import KuaiSessionEnvironment
        return KuaiSessionEnvironment(client_id, seed, c, data_dir, response_path, split, device, evaluation)
    raise ValueError(f'Unknown environment: {name}')
