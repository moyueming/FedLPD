from __future__ import annotations
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile
from .artifacts import atomic_json, fingerprint
DIAGNOSTIC_FILES = ('diagnostics.json', 'client_pair_discrepancies.csv', 'latents.npz', 'discrepancies.png', 'discrepancies.pdf', 'latent_pca.png', 'latent_pca.pdf', 'latent_pca_reward.png', 'latent_pca_reward.pdf', 'plot_arrays.npz')
DETAIL_FILES = ('evaluation_sessions.jsonl', 'message_ledger.jsonl', 'rounds.jsonl', 'resources.jsonl', 'training_sessions.jsonl', 'progress.jsonl', 'resume.pt', 'packages.json', 'retention.json', 'test-initializations.json', 'validation-initializations.json', 'diagnostic_retention.json', 'source/source.zip', 'source/dependencies.json', 'worker.log')

def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def _digest(data):
    return hashlib.sha256(data).hexdigest()

def verify_paper_archive(path):
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate paper archive members')
            for name in names:
                p = PurePosixPath(name)
                if p.is_absolute() or '..' in p.parts or '\\' in name:
                    raise ValueError('Unsafe paper archive member')
            manifest = json.loads(archive.read('paper_manifest.json'))
            files = manifest['files']
            if set(names) != set(files) | {'paper_manifest.json'}:
                raise ValueError('Incomplete paper archive inventory')
            for name, digest in files.items():
                if _digest(archive.read(name)) != digest:
                    raise ValueError('Paper archive checksum failed: ' + name)
            if manifest.get('schema_version') != 1:
                raise ValueError('Unknown paper archive schema')
            return manifest
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid paper archive: ' + str(path)) from exc

def verify_archived_run(directory, configuration_hash):
    directory = Path(directory).resolve()
    link = _json(directory / 'paper_retention.json')
    path = Path(link['archive_path'])
    if not path.is_file() or fingerprint(path) != link['archive_sha256']:
        raise ValueError('Paper archive missing or changed')
    manifest = verify_paper_archive(path)
    matches = [r for r in manifest['runs'] if r['path'] == str(directory)]
    if len(matches) != 1 or matches[0]['configuration_hash'] != configuration_hash:
        raise ValueError('Paper archive run identity mismatch')
    row = matches[0]
    for name in ('run.json', 'config.json'):
        if fingerprint(directory / name) != manifest['files'][row['prefix'] + '/' + name]:
            raise ValueError('Archived run metadata changed')
    return (path, manifest, row)

def restore_exports(archive_path, prefix, output):
    manifest = verify_paper_archive(archive_path)
    output = Path(output).resolve()
    members = [n for n in manifest['files'] if n.startswith(prefix + '/')]
    if not members:
        raise ValueError('Requested results are not present in the paper archive')
    with zipfile.ZipFile(archive_path) as archive:
        for name in members:
            target = (output / name[len(prefix) + 1:]).resolve()
            if not target.is_relative_to(output):
                raise ValueError('Output path escapes results directory')
            if target.exists() and fingerprint(target) != manifest['files'][name]:
                raise FileExistsError('Refusing to overwrite different results: ' + str(target))
        for name in members:
            target = output / name[len(prefix) + 1:]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(archive.read(name))
    return output

def restore_archived_report(root, output):
    directories = [p.parent for p in Path(root).rglob('run.json')]
    archived = [p for p in directories if (p / 'paper_retention.json').exists()]
    if not archived:
        return None
    if len(archived) != len(directories):
        raise ValueError('Cannot mix archived and live runs; use the saved paper report')
    paths = set()
    manifests = []
    for directory in archived:
        path, manifest, _ = verify_archived_run(directory, _json(directory / 'run.json')['configuration_hash'])
        paths.add(path)
        manifests.append(manifest)
    if len(paths) != 1 or {str(p.resolve()) for p in directories} != {r['path'] for r in manifests[0]['runs']}:
        raise ValueError('Archived report must match the complete original run collection')
    return restore_exports(paths.pop(), 'report', output)

def _cleanup(manifest, archive_path):
    roots = [Path(p).resolve() for p in manifest['cleanup_roots']]
    pending = []
    for name, digest in manifest['temporary_files'].items():
        original = Path(name)
        path = original.resolve()
        if original.is_symlink() or not any((path.is_relative_to(r) for r in roots)):
            raise ValueError('Cleanup path escapes approved temporary roots')
        if path.exists():
            if not path.is_file() or fingerprint(path) != digest:
                raise ValueError('Temporary file changed; cleanup stopped: ' + name)
            pending.append(path)
    digest = fingerprint(archive_path)
    for row in manifest['runs']:
        atomic_json(Path(row['path']) / 'paper_retention.json', dict(schema_version=1, archive_path=str(archive_path), archive_sha256=digest, configuration_hash=row['configuration_hash']))
    for path in pending:
        path.unlink()
    atomic_json(archive_path.with_suffix(archive_path.suffix + '.receipt.json'), dict(archive_sha256=digest, cleanup_complete=True, planned_files=len(manifest['temporary_files']), retained_runs=len(manifest['runs'])))
