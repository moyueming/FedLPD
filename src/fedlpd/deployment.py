from pathlib import Path
import os
import json
import subprocess
import sys
import torch
from .resources import _device_uuid

def isolated_worker_bindings(devices):
    if all((torch.device(d).type == 'cpu' for d in devices)):
        return worker_bindings(devices)
    result = subprocess.run([sys.executable, '-m', 'fedlpd.deployment', json.dumps(devices)], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(result.stdout)

def worker_bindings(devices):
    bindings = []
    seen = set()
    for name in devices:
        device = torch.device(name)
        if device.type == 'cpu':
            bindings.append(dict(parent_device=name, child_device='cpu', uuid=None))
            continue
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA worker requested but CUDA is unavailable')
        index = device.index or 0
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError('Worker GPU is not visible: ' + name)
        uuid = _device_uuid(getattr(torch.cuda.get_device_properties(device), 'uuid', None))
        if uuid is None:
            raise RuntimeError('Actual GPU UUID unavailable; cannot safely isolate ' + name)
        if uuid in seen:
            raise ValueError('Two workers resolve to the same physical GPU')
        seen.add(uuid)
        bindings.append(dict(parent_device=name, child_device='cuda:0', uuid=uuid))
    return bindings

def allocate_cpu_groups(allowed, topology, workers=4, threads=4, quota=None):
    if workers < 1 or threads < 1:
        raise ValueError('Positive workers and threads required')
    required = workers * threads
    if quota is not None and quota < required:
        raise ValueError(f'CPU quota {quota:g} is below required {required}')
    unique = {}
    for cpu in sorted(set(allowed)):
        if cpu not in topology:
            raise ValueError('Missing CPU topology for ' + str(cpu))
        unique.setdefault(tuple(topology[cpu]), cpu)
    cores = [unique[key] for key in sorted(unique)]
    if len(cores) < required:
        raise ValueError(f'Need {required} allocated physical cores; found {len(cores)}')
    return [cores[i * threads:(i + 1) * threads] for i in range(workers)]

def cpu_allocation(workers=4, threads=4):
    if not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('Server allocation requires Linux CPU affinity')
    allowed = sorted(os.sched_getaffinity(0))
    topology = {}
    for cpu in allowed:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology[cpu] = tuple((int((base / name).read_text()) for name in ('physical_package_id', 'core_id')))
    quotas = []
    evidence = {}
    roots = [Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup/cpu'), Path('/sys/fs/cgroup/cpu,cpuacct')]
    candidates = set(roots)
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if controllers and 'cpu' not in controllers.split(','):
            continue
        for root in roots:
            path = (root / relative.lstrip('/')).resolve()
            while path.is_relative_to(root):
                candidates.add(path)
                if path == root:
                    break
                path = path.parent
    for base in sorted(candidates):
        path = base / 'cpu.max'
        if path.is_file():
            value = path.read_text().strip()
            evidence[str(path)] = value
            quota, period = value.split()
            if quota != 'max':
                quotas.append(int(quota) / int(period))
        q = base / 'cpu.cfs_quota_us'
        p = base / 'cpu.cfs_period_us'
        if q.is_file() and p.is_file():
            quota = int(q.read_text())
            period = int(p.read_text())
            evidence[str(q)] = quota
            evidence[str(p)] = period
            if quota > 0:
                quotas.append(quota / period)
    quota = min(quotas) if quotas else None
    return dict(allowed=allowed, quota_cores=quota, quota_evidence=evidence, groups=allocate_cpu_groups(allowed, topology, workers, threads, quota))
if __name__ == '__main__':
    print(json.dumps(worker_bindings(json.loads(sys.argv[1]))))
