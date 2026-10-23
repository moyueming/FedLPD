from pathlib import Path
from hashlib import sha256
import json, os, uuid
import numpy as np

def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value)}')

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if os.name == 'posix':
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

def fingerprint(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

class LogBook:

    def __init__(self, directory):
        self.directory = Path(directory)
        self.handles = {}
        self.count = 0
        self.pending = {}

    def write(self, name, value):
        self.pending.setdefault(name, []).append(json.dumps(value, allow_nan=False, default=json_default) + '\n')
        self.count += 1

    def flush(self):
        for name, lines in self.pending.items():
            if not lines:
                continue
            if name not in self.handles:
                self.handles[name] = (self.directory / name).open('a', encoding='utf-8')
            self.handles[name].writelines(lines)
            lines.clear()
        for h in self.handles.values():
            h.flush()

    def positions(self):
        self.flush()
        return {p.name: p.stat().st_size for p in self.directory.glob('*.jsonl')}

    def durable(self):
        self.flush()
        for h in self.handles.values():
            os.fsync(h.fileno())

    def close(self):
        self.flush()
        for h in self.handles.values():
            h.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def truncate_logs(directory, request):
    for p in Path(directory).glob('*.jsonl'):
        good = []
        for line in p.read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                break
            if item.get('requests_per_client', 0) <= request:
                good.append(line)
        temp = p.with_suffix('.rewrite')
        temp.write_text(''.join((l + '\n' for l in good)), encoding='utf-8')
        os.replace(temp, p)

def restore_log_positions(directory, positions):
    for path in Path(directory).glob('*.jsonl'):
        size = int(positions.get(path.name, 0))
        if path.stat().st_size < size:
            raise ValueError('Committed log was truncated: ' + path.name)
        with path.open('r+b') as stream:
            stream.truncate(size)
