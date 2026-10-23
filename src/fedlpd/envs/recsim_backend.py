from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

@lru_cache(maxsize=1)
def load_backend():
    import recsim
    path = Path(recsim.__file__).parent / 'simulator' / 'environment.py'
    spec = importlib.util.spec_from_file_location('_fedlpd_official_recsim_environment', path)
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load official RecSim dispatcher')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (module.SingleUserEnvironment, {'recsim_package_version': importlib.metadata.version('recsim'), 'dispatcher_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
