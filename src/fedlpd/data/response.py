from __future__ import annotations
import copy
import json
from collections import OrderedDict
from pathlib import Path
from threading import RLock
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .preparation import PreparedData, file_hash, LABELS, validate_data, verified_data, asset_file_signature
_FROZEN_MODEL_CACHE = OrderedDict()
_FROZEN_MODEL_LOCK = RLock()

class ResponseModel(nn.Module):

    def __init__(self, n_users, feature_dim, hidden=32):
        super().__init__()
        self.user = nn.Embedding(n_users, hidden)
        self.history = nn.GRU(feature_dim + 4, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2 + feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, users, items, histories):
        encoded, _ = self.history(histories)
        lengths = (histories[:, :, -1] > 0).sum(-1)
        state = encoded[torch.arange(len(users), device=users.device), (lengths - 1).clamp_min(0)]
        state = state * (lengths > 0).unsqueeze(-1)
        return self.head(torch.cat((self.user(users), state, items), -1))

class ResponseDataset(Dataset):

    def __init__(self, data, rows, history_length):
        self.data = data
        self.rows = rows
        self.history_length = history_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        i = int(self.rows[index])
        d = self.data
        return (int(d.users[i]), np.array(d.catalog_features[d.items[i]]), d.history_tensor(i, self.history_length), np.array(d.labels[i]), int(d.user_client[d.users[i]]))

def _scores(model, loader, device, compute_auc=False, reference_rates=None):
    sums = np.zeros((6, 3, 3), dtype=np.float64)
    counts = np.zeros(6, dtype=np.int64)
    model.eval()
    all_p = []
    all_y = []
    all_clients = []
    with torch.no_grad():
        for users, items, histories, labels, clients in loader:
            p = torch.sigmoid(model(users.to(device), items.to(device), histories.to(device))).cpu().numpy()
            y = labels.numpy()
            p = np.clip(p, 1e-07, 1 - 1e-07)
            if compute_auc:
                all_p.append(p)
                all_y.append(y)
                all_clients.append(clients.numpy())
            errs = np.stack((-(y * np.log(p) + (1 - y) * np.log(1 - p)), (p - y) ** 2, y), axis=-1)
            for k in range(6):
                m = np.ones(len(y), dtype=bool) if k == 5 else clients.numpy() == k
                sums[k] += errs[m].sum(0)
                counts[k] += int(m.sum())
    result = {'all' if k == 5 else str(k): {'n': int(counts[k]), 'labels': {name: {'log_loss': float(sums[k, j, 0] / max(counts[k], 1)), 'brier': float(sums[k, j, 1] / max(counts[k], 1)), 'positive_rate': float(sums[k, j, 2] / max(counts[k], 1))} for j, name in enumerate(LABELS)}} for k in range(6)}
    if compute_auc:
        from scipy.stats import rankdata
        probs = np.concatenate(all_p) if all_p else np.empty((0, 3))
        labels = np.concatenate(all_y) if all_y else np.empty((0, 3))
        clients = np.concatenate(all_clients) if all_clients else np.empty(0)
        for k in range(6):
            mask = np.ones(len(labels), dtype=bool) if k == 5 else clients == k
            for j, label in enumerate(LABELS):
                y = labels[mask, j]
                p = probs[mask, j]
                positive = y == 1
                np_ = int(positive.sum())
                nn = len(y) - np_
                auc = float((rankdata(p)[positive].sum() - np_ * (np_ + 1) / 2) / (np_ * nn)) if np_ and nn else None
                key = 'all' if k == 5 else str(k)
                score = result[key]['labels'][label]
                score['auc'] = auc
                score['auc_null_reason'] = None if auc is not None else 'single_class' if len(y) else 'no_samples'
                score['mean_prediction'] = float(p.mean()) if len(p) else None
                bins = []
                weighted_gap = 0.0
                indexes = np.minimum((p * 10).astype(int), 9)
                for bin_id in range(10):
                    bm = indexes == bin_id
                    count = int(bm.sum())
                    prediction = float(p[bm].mean()) if count else None
                    observed = float(y[bm].mean()) if count else None
                    bins.append({'lower': bin_id / 10, 'upper': (bin_id + 1) / 10, 'n': count, 'mean_prediction': prediction, 'positive_rate': observed})
                    if count:
                        weighted_gap += count * abs(prediction - observed)
                score['calibration'] = {'bins': bins, 'ece': weighted_gap / len(y) if len(y) else None, 'definition': '10 equal-width probability bins; weighted absolute prediction-frequency gap'}
                if not len(y):
                    for field in ['log_loss', 'brier', 'positive_rate']:
                        score[field] = None
                if reference_rates is not None:
                    fallback = reference_rates[key] is None
                    probability = reference_rates['all' if fallback else key][j]
                    q = float(np.clip(probability, 1e-07, 1 - 1e-07))
                    score['reference'] = {'source': 'full_train_in_catalog_positive_frequency', 'scope': 'pooled' if key == 'all' or fallback else 'same_client', 'probability': float(probability), 'fallback_reason': 'no_client_training_in_catalog_targets' if fallback else None, 'log_loss': float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()) if len(y) else None, 'brier': float(((y - probability) ** 2).mean()) if len(y) else None}
    return result

def fit_response(data_dir, output_path, config=None, device='cpu'):
    c = {'epochs': 20, 'batch_size': 1024, 'num_workers': 4, 'hidden': 32, 'history_length': 20, 'seed': 1234, 'max_train_rows': 2000000, 'sample_window': 128, 'learning_rate': 0.001, **dict(config or {})}
    for field in ['epochs', 'batch_size', 'hidden', 'history_length', 'max_train_rows', 'sample_window']:
        if int(c[field]) <= 0:
            raise ValueError(f'{field} must be positive')
    validate_data(data_dir)
    d = PreparedData(data_dir)
    seed = int(c.get('seed', 1234))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = {s: np.flatnonzero((d.splits == j) & (d.items >= 0)) for j, s in enumerate(['train', 'validation', 'test'])}
    if any((not len(x) for x in rows.values())):
        raise ValueError('Response fitting requires in-catalog targets in all splits')
    train_targets = rows['train'].copy()
    train_clients = np.asarray(d.user_client[d.users[train_targets]])
    reference_rates = {'all': np.asarray(d.labels[train_targets]).mean(0)}
    for client in range(5):
        targets = train_targets[train_clients == client]
        reference_rates[str(client)] = np.asarray(d.labels[targets]).mean(0) if len(targets) else None
    cap = int(c.get('max_train_rows', 2000000))
    if len(rows['train']) > cap:
        windows = []
        window = min(int(c['sample_window']), cap)
        order = np.argsort(d.users[rows['train']], kind='stable')
        grouped = rows['train'][order]
        cuts = np.flatnonzero(np.diff(d.users[grouped])) + 1
        for ur in np.split(grouped, cuts):
            windows.extend([ur[i:i + window] for i in range(0, len(ur), window)])
        rng.shuffle(windows)
        selected = []
        n = 0
        for w in windows:
            if n + len(w) > cap:
                continue
            selected.append(w)
            n += len(w)
        rows['train'] = np.concatenate(selected)
    history_length = int(c.get('history_length', 20))
    hidden = int(c.get('hidden', 32))
    batch = int(c.get('batch_size', 1024))
    loaders = {s: DataLoader(ResponseDataset(d, r, history_length), batch_size=batch, shuffle=s == 'train', num_workers=int(c.get('num_workers', 0)), generator=torch.Generator().manual_seed(seed)) for s, r in rows.items()}
    model = ResponseModel(len(d.user_client), d.catalog_features.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(c.get('learning_rate', 0.001)))
    best = float('inf')
    best_state = None
    best_epoch = None
    trace = []
    for epoch in range(int(c.get('epochs', 10))):
        model.train()
        losses = []
        for users, items, histories, labels, _ in loaders['train']:
            logits = model(users.to(device), items.to(device), histories.to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        scores = _scores(model, loaders['validation'], device)
        val = float(np.mean([x['log_loss'] for x in scores['all']['labels'].values()]))
        trace.append({'epoch': epoch, 'training_loss': float(np.mean(losses)), 'validation_log_loss': val})
        if val < best:
            best = val
            best_epoch = epoch
            best_state = copy.deepcopy(model.cpu().state_dict())
            model.to(device)
    if best_state is None:
        raise ValueError('epochs must be positive')
    model.load_state_dict(best_state)
    payload = {'model': best_state, 'n_users': len(d.user_client), 'feature_dim': d.catalog_features.shape[1], 'hidden': hidden, 'history_length': history_length, 'metadata_sha256': file_hash(Path(data_dir) / 'metadata.json'), 'config': c, 'labels': LABELS}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    selection_path = path.with_suffix('.train_rows.npy')
    np.save(selection_path, rows['train'], allow_pickle=False)
    report = {'trace': trace, 'selected_validation_loss': best, 'selected_epoch': best_epoch, 'checkpoint_sha256': file_hash(path), 'prepared_metadata_sha256': payload['metadata_sha256'], 'train_targets': len(rows['train']), 'sampling': {'seed': seed, 'window_size': int(c['sample_window']), 'effective_window_size': min(int(c['sample_window']), cap), 'available_train_targets': len(train_targets), 'selected_targets': len(rows['train']), 'selection_file': selection_path.name, 'selection_sha256': file_hash(selection_path), 'history_source': 'complete user logs strictly preceding target timestamp'}, 'validation': _scores(model, loaders['validation'], device, True, reference_rates), 'test': _scores(model, loaders['test'], device, True, reference_rates), 'limitation': 'Conditional independent Bernoulli labels; observational log-fitted simulation, not causal slate identification. Quality metrics are descriptive; no policy-performance-dependent acceptance threshold.'}
    path.with_suffix('.metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return path

class FrozenResponse:

    def __init__(self, data_dir, path, device='cpu'):
        self.data = verified_data(data_dir)
        target = torch.device(device)
        if target.type == 'cuda' and target.index is None:
            target = torch.device('cuda', torch.cuda.current_device())
        self.device = str(target)
        report_path = Path(path).with_suffix('.metrics.json')
        signature = asset_file_signature([path, report_path])
        key = (self.data._validation_signature, signature, self.device)
        with _FROZEN_MODEL_LOCK:
            if key not in _FROZEN_MODEL_CACHE:
                if not report_path.is_file():
                    raise ValueError('Response checkpoint integrity report is missing')
                report = json.loads(report_path.read_text(encoding='utf-8-sig'))
                digest = file_hash(path)
                if report.get('checkpoint_sha256') != digest:
                    raise ValueError('Response checkpoint failed integrity verification')
                p = torch.load(path, map_location='cpu', weights_only=False)
                if p['metadata_sha256'] != file_hash(Path(data_dir) / 'metadata.json'):
                    raise ValueError('Response checkpoint belongs to different prepared data')
                model = ResponseModel(p['n_users'], p['feature_dim'], p['hidden']).to(self.device)
                model.load_state_dict(p['model'])
                model.eval()
                model.requires_grad_(False)
                if asset_file_signature([path, report_path]) != signature:
                    raise ValueError('Response assets changed during integrity verification')
                _FROZEN_MODEL_CACHE[key] = (model, p['history_length'], digest)
            _FROZEN_MODEL_CACHE.move_to_end(key)
            self.model, self.history_length, self.checkpoint_sha256 = _FROZEN_MODEL_CACHE[key]
            while len(_FROZEN_MODEL_CACHE) > 4:
                _FROZEN_MODEL_CACHE.popitem(last=False)

    @torch.inference_mode()
    def probabilities(self, user_id, slate_ids, history):
        h = np.zeros((self.history_length, self.data.catalog_features.shape[1] + 4), dtype=np.float32)
        for j, (iid, labels) in enumerate(history[-self.history_length:]):
            if iid >= 0:
                h[j, :-4] = self.data.catalog_features[iid]
            h[j, -4:-1] = labels
            h[j, -1] = 1
        n = len(slate_ids)
        return torch.sigmoid(self.model(torch.full((n,), int(user_id), device=self.device, dtype=torch.long), torch.as_tensor(np.array(self.data.catalog_features[slate_ids]), device=self.device), torch.as_tensor(np.repeat(h[None], n, axis=0), device=self.device))).cpu().numpy()
