from __future__ import annotations
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from threading import RLock
import numpy as np
import pandas as pd
LABELS = ['is_click', 'is_long_view', 'is_like']
_ARRAY_NAMES = ('users', 'items', 'timestamps', 'splits', 'labels', 'previous', 'user_client', 'catalog_features')
_VERIFIED_DATA_CACHE = OrderedDict()
_VERIFIED_DATA_LOCK = RLock()
_ASSET_CACHE_LIMIT = 4

def asset_file_signature(paths):
    signature = []
    for value in paths:
        path = Path(value).resolve()
        try:
            st = path.stat()
            signature.append((str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
        except FileNotFoundError:
            signature.append((str(path), None))
    return (os.getpid(), tuple(signature))

def prepared_signature(directory):
    root = Path(directory)
    return asset_file_signature([root / 'metadata.json', root / 'manifest.json', *(root / (key + '.npy') for key in _ARRAY_NAMES)])

def verified_data(directory, *, force=False):
    with _VERIFIED_DATA_LOCK:
        signature = prepared_signature(directory)
        if not force and signature in _VERIFIED_DATA_CACHE:
            _VERIFIED_DATA_CACHE.move_to_end(signature)
            return _VERIFIED_DATA_CACHE[signature]
        data = PreparedData(directory)
        _validate_prepared(data)
        if prepared_signature(directory) != signature:
            raise ValueError('Prepared assets changed during integrity verification')
        data._validation_signature = signature
        if signature not in _VERIFIED_DATA_CACHE:
            _VERIFIED_DATA_CACHE[signature] = data
        _VERIFIED_DATA_CACHE.move_to_end(signature)
        while len(_VERIFIED_DATA_CACHE) > _ASSET_CACHE_LIMIT:
            _VERIFIED_DATA_CACHE.popitem(last=False)
        return _VERIFIED_DATA_CACHE[signature]

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def validate_prepared_config(metadata, config):
    recorded = metadata.get('config', {})
    wanted = dict(config.get('env', {}), **config.get('data', {}))
    keys = ('catalog_size', 'feature_dim', 'num_clients', 'partition_seed', 'catalog_seed', 'min_item_count', 'train_fraction', 'val_fraction', 'duplicate_policy', 'static_feature_columns')
    for key in keys:
        defaults = {'val_fraction': 0.1, 'duplicate_policy': 'reject', 'static_feature_columns': []}
        actual = recorded.get(key, defaults.get(key))
        if key == 'val_fraction':
            actual = recorded.get('validation_fraction', actual)
        if key in wanted and actual != wanted[key]:
            raise ValueError(f'Prepared data configuration mismatch: {key}={actual!r}, requested {wanted[key]!r}; rebuild assets in a new directory')

def prepare_data(log_paths, output_dir, config=None, item_features_path=None, *, catalog_audit_path=None):
    c = {'catalog_size': 2000, 'feature_dim': 16, 'num_clients': 5, 'min_item_count': 5, 'train_fraction': 0.7, 'partition_seed': 2718, 'catalog_seed': 1729, 'duplicate_policy': 'reject', **dict(config or {})}
    c.setdefault('validation_fraction', c.get('val_fraction', 0.1))
    policy = c['duplicate_policy']
    if policy not in ('reject', 'drop_exact'):
        raise ValueError('Invalid duplicate_policy; use reject or drop_exact')
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Prepared output directory is not empty; preserve it and use a new directory')
    paths = [Path(p) for p in ([log_paths] if isinstance(log_paths, (str, Path)) else log_paths)]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Duplicate input paths')
    frames = []
    source_files = []
    source_hashes = set()
    comparison_columns = None
    for p in paths:
        digest = file_hash(p)
        if digest in source_hashes:
            raise ValueError('Duplicate input file contents: ' + str(p))
        source_hashes.add(digest)
        frame = pd.read_csv(p)
        aliases = {c.get('user_column', 'user_id'): 'user_id', c.get('item_column', 'video_id'): 'video_id', c.get('timestamp_column', 'time_ms'): 'timestamp', 'long_view': 'is_long_view'}
        for alias, canonical in aliases.items():
            if alias != canonical and alias in frame and (canonical in frame) and (not np.array_equal(frame[alias].to_numpy(), frame[canonical].to_numpy())):
                raise ValueError(f'Conflicting columns {alias} and {canonical}')
        frame = frame.rename(columns={a: b for a, b in aliases.items() if a in frame and b not in frame})
        required = ['user_id', 'video_id', 'timestamp'] + LABELS
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing required identity/time/feedback columns: {sorted(missing)}')
        columns = sorted(frame.columns)
        if comparison_columns is None:
            comparison_columns = columns
        elif columns != comparison_columns:
            raise ValueError('Input column schemas differ after alias normalization; resolve explicitly before preparation')
        source_files.append({'name': p.name, 'sha256': digest, 'input_rows': len(frame), 'exact_duplicate_rows': int(frame.duplicated().sum())})
        frames.append(frame)
    if not frames:
        raise ValueError('No input logs')
    raw = pd.concat(frames, ignore_index=True)
    frames.clear()
    del frame
    input_rows = len(raw)
    duplicate_mask = raw.duplicated(keep='first')
    duplicates = int(duplicate_mask.sum())
    if duplicates and policy == 'reject':
        raise ValueError(f'Duplicate complete records detected ({duplicates}); set duplicate_policy=drop_exact to retain first full rows explicitly')
    df = raw.loc[~duplicate_mask, required].copy()
    del raw, duplicate_mask
    if df.empty:
        raise ValueError('Input logs are empty')
    if df.isna().any().any():
        raise ValueError('Required log columns contain missing values; resolve explicitly before preparation')
    for field in ['user_id', 'video_id', 'timestamp']:
        values = pd.to_numeric(df[field], errors='raise')
        if not np.isfinite(values).all() or not (values == np.floor(values)).all() or (values < 0).any():
            raise ValueError(f'{field} must contain finite nonnegative integer values')
        df[field] = values.astype(np.int64)
    for label in LABELS:
        if not df[label].isin([0, 1]).all():
            raise ValueError(f'feedback {label} must be binary')
    within = sum((s['exact_duplicate_rows'] for s in source_files))
    deduplication = {'policy': policy, 'keep': 'first_in_input_order', 'comparison_columns': comparison_columns, 'comparison': 'all parsed columns after alias normalization; pandas equality treats nulls as equal', 'within_file_exact_duplicates': within, 'cross_file_exact_duplicates': duplicates - within, 'removed_rows': duplicates, 'projected_duplicate_rows_retained': int(df.duplicated().sum())}
    rows_after_deduplication = len(df)
    df = df.sort_values(['timestamp', 'user_id'], kind='stable').reset_index(drop=True)
    times = df.timestamp.to_numpy(dtype=np.int64)
    f = float(c.get('train_fraction', 0.7))
    v = float(c.get('validation_fraction', c.get('val_fraction', 0.1)))
    if not (0 < f < 1 and 0 < v and (f + v < 1)):
        raise ValueError('Invalid temporal fractions')
    cut1 = int(times[min(int(len(times) * f), len(times) - 1)])
    cut2 = int(times[min(int(len(times) * (f + v)), len(times) - 1)])
    train = df.timestamp < cut1
    if not train.any() or cut1 >= cut2:
        raise ValueError('Insufficient distinct timestamps for three splits')
    raw_users = sorted(df.loc[train, 'user_id'].unique().tolist())
    user_map = {u: i for i, u in enumerate(raw_users)}
    excluded = int((~df.user_id.isin(raw_users)).sum())
    df = df[df.user_id.isin(raw_users)].reset_index(drop=True)
    times = df.timestamp.to_numpy(dtype=np.int64)
    train = times < cut1
    counts = df.loc[train, 'video_id'].value_counts()
    counts = counts[counts >= int(c.get('min_item_count', 5))]
    if catalog_audit_path is not None:
        from ..experiment_checks import catalog_audit
        from ..artifacts import atomic_json
        if Path(catalog_audit_path).exists():
            raise FileExistsError('Catalog audit output already exists')
        atomic_json(catalog_audit_path, catalog_audit(df, train, counts, c, source_files, duplicates))
        return Path(catalog_audit_path)
    ordered = sorted(counts.index.tolist(), key=lambda x: (-int(counts[x]), int(x)))[:int(c.get('catalog_size', 2000))]
    if not ordered:
        raise ValueError('Training-only catalog is empty; inspect data/min_item_count')
    if not c.get('smoke', False):
        minimum = int(c.get('candidate_size', 100)) + (int(c.get('horizon', 20)) - 1) * int(c.get('slate_size', 5))
        if len(ordered) < minimum:
            raise ValueError(f'Training-only catalog must contain at least {minimum} items for legal session candidates')
    item_map = {x: i for i, x in enumerate(ordered)}
    users = df.user_id.map(user_map).to_numpy(dtype=np.int32)
    items = df.video_id.map(item_map).fillna(-1).to_numpy(dtype=np.int32)
    splits = np.where(times < cut1, 0, np.where(times < cut2, 1, 2)).astype(np.int8)
    labels = df[LABELS].to_numpy(dtype=np.float32)
    previous = np.full(len(df), -1, dtype=np.int64)
    last = {}
    for i, u in enumerate(users):
        previous[i] = last.get(int(u), -1)
        last[int(u)] = i
    nclients = int(c.get('num_clients', 5))
    if len(raw_users) < nclients:
        raise ValueError('Fewer training users than clients')
    rng = np.random.default_rng(int(c.get('partition_seed', 1729)))
    perm = rng.permutation(len(raw_users))
    user_client = np.empty(len(raw_users), dtype=np.int16)
    user_client[perm] = np.arange(len(raw_users)) % nclients
    feature_dim = int(c.get('feature_dim', 16))
    feature_rng = np.random.default_rng(int(c.get('catalog_seed', 31415)))
    features = feature_rng.normal(size=(len(ordered), feature_dim)).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-08)
    feature_meta = {'kind': 'fixed_public_id_features', 'seed': int(c.get('catalog_seed', 31415))}
    if item_features_path is not None:
        fields = c.get('static_feature_columns')
        if not fields:
            raise ValueError('static_feature_columns whitelist required; aggregate popularity features are forbidden')
        table = pd.read_csv(item_features_path).set_index(c.get('item_column', 'video_id'))
        values = table.reindex(ordered)[fields].to_numpy(dtype=np.float32)
        if np.isinf(values).any():
            raise ValueError('Static feature values must be finite or missing')
        med = np.nanmedian(values, axis=0)
        if not np.isfinite(med).all():
            raise ValueError('Empty/nonfinite static feature columns')
        values = np.where(np.isnan(values), med, values)
        mean = values.mean(0)
        std = np.maximum(values.std(0), 1e-06)
        features = ((values - mean) / std).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite normalized static features')
        feature_meta = {'kind': 'whitelisted_static', 'columns': fields, 'median': med.tolist(), 'mean': mean.tolist(), 'std': std.tolist(), 'source_sha256': file_hash(item_features_path)}
    arrays = {'users': users, 'items': items, 'timestamps': times, 'splits': splits, 'labels': labels, 'previous': previous, 'user_client': user_client, 'catalog_features': features}
    out.mkdir(parents=True, exist_ok=True)
    for key, array in arrays.items():
        np.save(out / (key + '.npy'), array, allow_pickle=False)
    split_statistics = {}
    for sid, name in enumerate(['train', 'validation', 'test']):
        mask = splits == sid
        stats = {'rows': int(mask.sum()), 'users': int(len(np.unique(users[mask]))), 'in_catalog_targets': int((mask & (items >= 0)).sum()), 'out_of_catalog_targets': int((mask & (items < 0)).sum()), 'missing_required_values': 0, 'unknown_item_rows': int((mask & (items < 0)).sum()), 'labels': {label: {'positive_rate': float(labels[mask, j].mean()) if mask.any() else None, 'positive_count': int(labels[mask, j].sum())} for j, label in enumerate(LABELS)}, 'clients': {}}
        for client in range(nclients):
            cm = mask & (user_client[users] == client)
            known = items[cm & (items >= 0)]
            stats['clients'][str(client)] = {'rows': int(cm.sum()), 'users': int(len(np.unique(users[cm]))), 'in_catalog_targets': int(len(known)), 'positive_rates': dict(zip(LABELS, labels[cm].mean(0).astype(float).tolist())) if cm.any() else {label: None for label in LABELS}, 'public_feature_mean': features[known].mean(0).astype(float).tolist() if len(known) else None}
        split_statistics[name] = stats
    metadata = {'format_version': 3, 'source_files': source_files, 'catalog_raw_ids': [int(x) for x in ordered], 'user_raw_ids': [int(x) for x in raw_users], 'feedback_aliases': {'long_view': 'is_long_view'}, 'feedback_source': 'https://github.com/chongminggao/KuaiRand#interaction-logs', 'train_cutoff': cut1, 'test_cutoff': cut2, 'config': c, 'labels': LABELS, 'feature_metadata': feature_meta, 'excluded_future_only_user_rows': excluded, 'split_counts': {s: int((splits == j).sum()) for j, s in enumerate(['train', 'validation', 'test'])}, 'input_rows': input_rows, 'retained_rows': len(df), 'duplicate_rows': duplicates, 'rows_after_deduplication': rows_after_deduplication, 'deduplication': deduplication, 'actual_split_fractions': {s: float((splits == j).mean()) for j, s in enumerate(['train', 'validation', 'test'])}, 'split_statistics': split_statistics, 'catalog_coverage': {s: float((items[splits == j] >= 0).mean()) for j, s in enumerate(['train', 'validation', 'test'])}, 'publication_eligible': not bool(c.get('smoke', False))}
    from ..experiment_checks import quantiles
    frequency = counts.loc[ordered].to_numpy()
    metadata['catalog_selection'] = {'decision_split': 'train', 'order': 'count_desc_then_raw_id_asc', 'requested_size': int(c['catalog_size']), 'actual_size': len(ordered), 'minimum_training_occurrences': int(c['min_item_count']), 'eligible_items_before_cap': len(counts), 'train_rows': int((splits == 0).sum()), 'train_target_rows': int(((splits == 0) & (items >= 0)).sum()), 'item_train_count_quantiles': quantiles(frequency), 'mean_item_train_count': float(frequency.mean())}
    metadata['array_hashes'] = {k: file_hash(out / (k + '.npy')) for k in arrays}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    validate_data(out)
    return out

class PreparedData:

    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text(encoding='utf-8-sig'))
        for key in _ARRAY_NAMES:
            setattr(self, key, np.load(self.directory / (key + '.npy'), mmap_mode='r', allow_pickle=False))

    def __getstate__(self):
        return {'directory': str(self.directory.resolve())}

    def __setstate__(self, state):
        self.__init__(state['directory'])

    def history_indices(self, index, length=20):
        result = []
        prev = int(self.previous[index])
        while prev >= 0 and len(result) < length:
            if self.timestamps[prev] < self.timestamps[index]:
                result.append(prev)
            prev = int(self.previous[prev])
        return np.array(result[::-1], dtype=np.int64)

    def history(self, index, length=20):
        ids = self.history_indices(index, length)
        return [(int(self.items[i]), self.labels[i].copy()) for i in ids]

    def initial_rows(self, client_id, split):
        sid = {'train': 0, 'validation': 1, 'val': 1, 'test': 2}[split]
        return np.flatnonzero((self.splits == sid) & (self.user_client[self.users] == client_id))

    def history_tensor(self, index, length=20):
        h = np.zeros((length, self.catalog_features.shape[1] + 4), dtype=np.float32)
        for j, (item, labels) in enumerate(self.history(index, length)):
            if item >= 0:
                h[j, :-4] = self.catalog_features[item]
            h[j, -4:-1] = labels
            h[j, -1] = 1
        return h

def validate_data(data_dir):
    return verified_data(data_dir, force=True).metadata

def _validate_prepared(d):
    expected = set(_ARRAY_NAMES)
    if set(d.metadata.get('array_hashes', {})) != expected:
        raise ValueError('Incomplete prepared array integrity manifest')
    for key, digest in d.metadata['array_hashes'].items():
        if file_hash(d.directory / (key + '.npy')) != digest:
            raise ValueError(f'Corrupted prepared array: {key}')
    n = len(d.users)
    for key in ['items', 'timestamps', 'splits', 'previous']:
        if getattr(d, key).shape != (n,):
            raise ValueError(f'Invalid prepared array shape: {key}')
    if d.labels.shape != (n, 3):
        raise ValueError('Invalid feedback array shape')
    if not np.isin(d.labels, [0.0, 1.0]).all():
        raise ValueError('Nonbinary prepared feedback')
    if not np.isfinite(d.catalog_features).all():
        raise ValueError('Nonfinite prepared public features')
    if np.any(d.users < 0) or np.any(d.users >= len(d.user_client)):
        raise ValueError('Invalid prepared user index')
    if np.any(d.items < -1) or np.any(d.items >= len(d.catalog_features)):
        raise ValueError('Invalid prepared item index')
    if not np.isin(d.splits, [0, 1, 2]).all():
        raise ValueError('Invalid prepared split')
    expected_splits = np.where(d.timestamps < d.metadata['train_cutoff'], 0, np.where(d.timestamps < d.metadata['test_cutoff'], 1, 2))
    if not np.array_equal(d.splits, expected_splits):
        raise ValueError('Prepared split disagrees with temporal cutoffs')
    previous = d.previous
    if np.any(previous < -1) or np.any(previous >= np.arange(n)):
        raise ValueError('Invalid history link; must point to an earlier row')
    valid = previous >= 0
    idx = np.flatnonzero(valid)
    if not np.all(d.timestamps[previous[valid]] <= d.timestamps[idx]):
        raise ValueError('Future/same-time information in history')
    if not np.all(d.users[previous[valid]] == d.users[idx]):
        raise ValueError('Cross-user history')
    for split in ['train', 'validation', 'test']:
        for client in range(int(d.metadata['config'].get('num_clients', 5))):
            if not len(d.initial_rows(client, split)):
                raise ValueError(f'No {split} initializations for client {client}')
    return d.metadata
