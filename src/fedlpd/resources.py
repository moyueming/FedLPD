from __future__ import annotations
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import uuid
import torch

def _affinity():
    if not hasattr(os, 'sched_getaffinity'):
        return None
    try:
        return sorted(os.sched_getaffinity(0))
    except OSError:
        return None

def _thread_ids():
    if platform.system() != 'Linux':
        return [0]
    try:
        tids = sorted((int(path.name) for path in Path('/proc/self/task').iterdir() if path.name.isdigit()))
    except OSError as exc:
        raise RuntimeError(f'Unable to enumerate Linux process threads: {exc}') from exc
    if not tids:
        raise RuntimeError('Unable to enumerate any Linux process threads')
    return tids

def set_cpu_affinity(cpus: list[int] | None):
    if cpus is None:
        return _affinity()
    if not isinstance(cpus, list) or not cpus or any((type(cpu) is not int or cpu < 0 for cpu in cpus)):
        raise ValueError('cpu_affinity must be a nonempty list of nonnegative integer CPU IDs')
    if not hasattr(os, 'sched_setaffinity') or not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('Explicit CPU affinity is unsupported on this operating system')
    requested = set(cpus)
    available = _affinity()
    if available is None:
        raise RuntimeError('Unable to inspect the available CPU affinity allocation')
    if not requested <= set(available):
        raise ValueError('Requested CPU affinity exceeds the available process allocation')
    for _ in range(3):
        for tid in _thread_ids():
            try:
                os.sched_setaffinity(tid, requested)
            except ProcessLookupError:
                continue
            except OSError as exc:
                raise RuntimeError(f'Unable to apply CPU affinity to thread {tid}: {exc}') from exc
        verified = True
        for tid in _thread_ids():
            try:
                verified = set(os.sched_getaffinity(tid)) == requested and verified
            except ProcessLookupError:
                continue
            except OSError as exc:
                raise RuntimeError(f'Unable to verify CPU affinity for thread {tid}: {exc}') from exc
        actual = _affinity()
        if verified and actual == sorted(requested):
            return actual
    raise RuntimeError('The operating system did not apply requested CPU affinity to every process thread')

def _cpu_model():
    if platform.system() == 'Linux':
        try:
            for line in Path('/proc/cpuinfo').read_text(encoding='utf-8').splitlines():
                if line.lower().startswith(('model name', 'hardware')) and ':' in line:
                    return line.split(':', 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or 'unknown'

def _memory():
    try:
        import psutil
        return (int(psutil.Process(os.getpid()).memory_info().rss), 'current_rss')
    except (ImportError, OSError, RuntimeError):
        pass
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (int(value * (1 if platform.system() == 'Darwin' else 1024)), 'peak_rss')
    except (ImportError, OSError, ValueError):
        return (None, 'unavailable')

def _smi(fields, query):
    command = ['nvidia-smi', f'--query-{query}={fields}', '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=3)
    return [[field.strip() for field in row] for row in csv.reader(result.stdout.splitlines()) if row]

def _device_uuid(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        if len(value) == 16:
            return 'GPU-' + str(uuid.UUID(bytes=value))
        value = value.decode('ascii')
    value = str(value).strip()
    if not value:
        return None
    if value.startswith(('GPU-', 'MIG-')):
        return value
    try:
        return 'GPU-' + str(uuid.UUID(value))
    except ValueError:
        return None

def _gpu_snapshot(device):
    gpu = {'name': None, 'uuid': None, 'physical_index': None, 'pci_bus_id': None, 'driver_version': None, 'total_memory_bytes': None, 'process_query_success': False, 'process_query_error': None, 'compute_processes': [], 'other_compute_pids': []}
    try:
        logical = torch.device(device)
        logical_index = logical.index if logical.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(logical)
        gpu['name'] = str(props.name)
        gpu['total_memory_bytes'] = int(props.total_memory)
        actual_uuid = _device_uuid(getattr(props, 'uuid', None))
        inventory = _smi('index,uuid,name,pci.bus_id,driver_version', 'gpu')
        if any((len(row) != 5 for row in inventory)):
            raise ValueError('Malformed GPU inventory from nvidia-smi')
        if actual_uuid is None:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
            if logical_index < len(visible):
                token = visible[logical_index].strip()
                if token.startswith(('GPU-', 'MIG-')):
                    matches = [row[1] for row in inventory if row[1].startswith(token)]
                    if len(matches) == 1:
                        actual_uuid = matches[0]
        if actual_uuid is None:
            raise ValueError('Actual CUDA device UUID is unavailable; refusing to guess a physical GPU')
        matches = [row for row in inventory if row[1].lower() == actual_uuid.lower()]
        if len(matches) != 1:
            raise ValueError('Actual CUDA UUID does not uniquely match the nvidia-smi GPU inventory')
        index, actual_uuid, name, bus, driver = matches[0]
        gpu.update(uuid=actual_uuid, physical_index=int(index), name=name, pci_bus_id=bus, driver_version=driver)
        processes = _smi('gpu_uuid,pid,process_name,used_gpu_memory', 'compute-apps')
        for row in processes:
            if len(row) != 4:
                raise ValueError('Malformed GPU process query from nvidia-smi')
            process_uuid, pid, name, memory = row
            if process_uuid.lower() != actual_uuid.lower():
                continue
            pid = int(pid)
            if pid < 1:
                raise ValueError('Invalid GPU compute process PID')
            try:
                memory_bytes = int(float(memory) * 1024 * 1024)
            except ValueError:
                memory_bytes = None
            gpu['compute_processes'].append({'pid': pid, 'gpu_uuid': process_uuid, 'process_name': name, 'used_memory_bytes': memory_bytes})
        gpu['other_compute_pids'] = sorted({row['pid'] for row in gpu['compute_processes'] if row['pid'] != os.getpid()})
        gpu['process_query_success'] = True
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, AssertionError, AttributeError) as exc:
        gpu['process_query_error'] = f'{type(exc).__name__}: {exc}'
    return gpu

def resource_snapshot(device, cpu_threads):
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    device_type = torch.device(device).type
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    rss, rss_kind = _memory()
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'platform': platform.platform(), 'operating_system': platform.system(), 'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda, 'cpu_model': _cpu_model(), 'cpu_logical_count': os.cpu_count(), 'cpu_affinity': _affinity(), 'cpu_threads': cpu_threads, 'load_average': load, 'rss_bytes': rss, 'rss_kind': rss_kind, 'device': str(device), 'device_type': device_type, 'gpu': _gpu_snapshot(device) if device_type == 'cuda' else None}

def cost_resource_reasons(snapshot, controlled_load, jobs_per_device):
    reasons = []
    if not controlled_load:
        reasons.append('undeclared_controlled_load')
    if snapshot.get('device_type') == 'cuda':
        if int(jobs_per_device) > 1:
            reasons.append('multiple_workers_per_gpu')
        gpu = snapshot.get('gpu') or {}
        if not gpu.get('process_query_success', False):
            reasons.append('gpu_process_query_failed')
        if gpu.get('other_compute_pids'):
            reasons.append('other_gpu_compute_processes')
    return reasons
