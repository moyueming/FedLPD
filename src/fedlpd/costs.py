from contextlib import contextmanager
import json
import time
import numpy as np

def payload_descriptor(direction, sender, receiver, message_type, arrays):
    tensors = []
    for name, value in arrays.items():
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        a = np.asarray(value)
        tensors.append(dict(name=name, shape=list(a.shape), dtype=a.dtype.str, nbytes=int(a.nbytes)))
    header = dict(schema_version=1, message_type=message_type, tensors=tensors)
    metadata_nbytes = len(json.dumps(header, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    return dict(direction=direction, sender=sender, receiver=receiver, message_type=message_type, schema_version=1, tensors=tensors, metadata_nbytes=metadata_nbytes, payload_nbytes=metadata_nbytes + sum((t['nbytes'] for t in tensors)))

class MessageLedger:

    def __init__(self, run_id, logs, total=0):
        self.run_id = run_id
        self.logs = logs
        self.total = int(total)

    def record(self, messages, request, round_index, phase='federation'):
        for index, message in enumerate(messages):
            expected = sum((t['nbytes'] for t in message['tensors'])) + message['metadata_nbytes']
            if expected != message['payload_nbytes']:
                raise ValueError('Invalid wire byte accounting')
            row = dict(message, run_id=self.run_id, event_id=f'{phase}:{request}:{index}', phase=phase, requests_per_client=request, request_step=request, round_index=round_index)
            self.logs.write('message_ledger.jsonl', row)
            self.total += expected

class TrainingTimer:
    KEYS = ('action_seconds', 'environment_seconds', 'rl_seconds', 'federation_seconds', 'initialization_seconds', 'validation_seconds', 'test_seconds', 'evaluation_seconds', 'checkpoint_seconds', 'logging_seconds', 'training_elapsed_seconds', 'synchronization_seconds')

    def __init__(self, device='cpu', values=None, clock=time.perf_counter):
        self.device = str(device)
        self.clock = clock
        self.values = {key: 0.0 for key in self.KEYS}
        self.values.update(values or {})
        self.active = set()

    def synchronize(self):
        if self.device.startswith('cuda'):
            import torch
            start = time.perf_counter()
            torch.cuda.synchronize(self.device)
            self.values['synchronization_seconds'] += time.perf_counter() - start

    @contextmanager
    def region(self, name):
        if name in self.active:
            raise ValueError('Overlapping timer category ' + name)
        self.synchronize()
        start = self.clock()
        self.active.add(name)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = self.clock() - start
            self.values[name] = self.values.get(name, 0.0) + elapsed
            self.active.remove(name)

def target_outcome(records, target, method):
    if method in ('local', 'random'):
        return dict(status='N/A', rounds_to_target=None, first_hit_requests=None)
    for row in sorted(records, key=lambda r: r['requests_per_client']):
        if row['macro']['J'] >= target:
            return dict(status='reached', rounds_to_target=row['scheduled_round_index'], first_hit_requests=row['requests_per_client'])
    return dict(status='NR', rounds_to_target=None, first_hit_requests=None)
