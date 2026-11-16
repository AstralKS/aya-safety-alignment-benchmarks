"""Preserve and clear a refused/aborted compute submission marker only when Kaggle proves absence.

Kaggle creates kernels under the slug derived from the metadata title. A refused SaveKernel
(400/409) can leave the durable marker behind while no kernel exists; this tool records the
proof, preserves the original marker, and allows exactly one fresh push.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v2_compute_jobs as compute


def slugify(title):
    return re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', title.lower())).strip('-')


def kaggle(*arguments, timeout=120):
    return subprocess.run(['kaggle', *arguments], capture_output=True, text=True, timeout=timeout,
                          encoding='utf-8', errors='replace')


def clear(path: Path):
    job = compute.verify(path)
    marker = path / 'submission-started.json'
    if not marker.exists():
        return {'cleared': False, 'reason': 'No submission marker present'}
    raw = marker.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    metadata = json.loads((path / 'kernel-metadata.json').read_bytes())
    owner = job['kernel'].split('/', 1)[0]
    candidates = {job['kernel'], owner + '/' + slugify(metadata['title'])}
    checks = {}
    for candidate in sorted(candidates):
        result = kaggle('kernels', 'status', candidate)
        checks[candidate] = {'kernel': candidate, 'returncode': result.returncode,
                             'output': result.stdout + result.stderr}
    listing = kaggle('kernels', 'list', '--user', owner, '--search', slugify(metadata['title'])[:30])
    absent = all(check['returncode'] != 0 and 'Cannot access kernel' in check['output']
                 and 'Permission' in check['output'] for check in checks.values())
    listed_absent = listing.returncode == 0 and listing.stdout.strip().splitlines()[-1].strip() == 'Not found'
    if not (absent and listed_absent):
        raise SystemExit('Remote absence not proven; the marker stays in place')
    receipt = {
        'kind': 'compute-submission-refused-v1',
        'kernel': job['kernel'],
        'title': metadata['title'],
        'directory': path.as_posix(),
        'marker_sha256': sha,
        'marker': json.loads(raw),
        'submission_log': (path / 'submission.log').read_text(encoding='utf-8')
        if (path / 'submission.log').exists() else None,
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'evidence': {'slug_checks': checks,
                     'kernels_list_returncode': listing.returncode,
                     'kernels_list_output': listing.stdout + listing.stderr,
                     'remote_kernel_absent': True},
        'decision': 'Kaggle refused or aborted SaveKernel before creating any kernel under the id or '
                    'the title-derived slug. The marker is preserved under a new name and one fresh push is allowed.',
        'resubmit_allowed': True,
    }
    receipt_path = path / ('submission-refused-' + sha[:12] + '.json')
    if receipt_path.exists():
        if json.loads(receipt_path.read_bytes()) != receipt:
            raise SystemExit('Existing refusal receipt differs')
    else:
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    marker.rename(path / ('submission-started-refused-' + sha[:12] + '.json'))
    return {'cleared': True, 'receipt': receipt_path.as_posix(), 'checks': len(checks)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(clear(arguments.directory), indent=2))
