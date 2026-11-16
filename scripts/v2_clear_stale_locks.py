"""Inspect and clear locks left behind by a dead baseline process.

The ledger locks are empty directories created atomically and are never removed by a contender,
so a killed process leaves them behind. Clearing is only allowed when no live process runs any
baseline script; the evidence and the removal are recorded as a receipt.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_CANDIDATES = (
    ROOT / 'scoring/budget.json.lock',
    ROOT / 'scoring/v2-glm/results/baseline-v2-full/run.lock',
)
BASELINE_MARKERS = ('v2_glm_scoring', 'run_v2_pipeline', 'v2_baseline_pipeline', 'v2_selfaware',
                    'v2_factscore_retrieval', 'v2_compute_jobs', 'v2_generation_queue', 'aya-v2')
GUARDS = (ROOT / 'runs/v2/baseline-pipeline.guard', ROOT / 'runs/v2/generation-queue.guard')


def python_processes():
    script = ('Get-CimInstance Win32_Process -Filter "Name=\'python.exe\'" | '
              'Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress')
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True,
                            text=True, timeout=60, encoding='utf-8', errors='replace')
    if result.returncode or not result.stdout.strip():
        return []
    data = json.loads(result.stdout)
    return data if isinstance(data, list) else [data]


def lock_candidates():
    """Static ledger locks plus any per-request journal locks left by a dead scorer."""
    candidates = list(LOCK_CANDIDATES)
    journals = ROOT / 'scoring/v2-glm/request-journals'
    if journals.is_dir():
        candidates.extend(sorted(journals.glob('*.lock')))
    return candidates


def inspect():
    processes = python_processes()
    live = [{'pid': item['ProcessId'], 'command': item.get('CommandLine') or ''} for item in processes]
    busy = [item for item in live if any(marker in item['command'] for marker in BASELINE_MARKERS)]
    locks = [{'path': lock.relative_to(ROOT).as_posix(), 'exists': lock.is_dir(),
              'empty': lock.is_dir() and not any(lock.iterdir())}
             for lock in lock_candidates()]
    guards = [{'path': guard.relative_to(ROOT).as_posix(), 'exists': guard.exists()} for guard in GUARDS]
    return {'live_python': live, 'busy_baseline': busy, 'locks': locks, 'guards': guards}


def clear(confirm):
    report = inspect()
    if report['busy_baseline']:
        raise SystemExit('A baseline process is still running; refusing to clear locks')
    for guard in report['guards']:
        if guard['exists']:
            raise SystemExit('A supervisor guard exists; confirm supervisors are dead and remove it first')
    removable = [item for item in report['locks'] if item['exists']]
    if not removable:
        return {'cleared': [], 'report': report}
    for item in removable:
        if not item['empty']:
            raise SystemExit('Lock is not empty; manual investigation required: ' + item['path'])
    if not confirm:
        return {'would_clear': [item['path'] for item in removable], 'report': report}
    receipt = {'kind': 'stale-baseline-lock-clear-v1', 'cleared_at': datetime.now(timezone.utc).isoformat(),
               'locks': removable, 'no_baseline_process': True, 'evidence': report}
    path = ROOT / ('scoring/v2-glm/results/baseline-v2-full/recovery/stale-locks-' +
                   datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    for item in removable:
        (ROOT / item['path']).rmdir()
    return {'cleared': [item['path'] for item in removable], 'receipt': path.relative_to(ROOT).as_posix()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'clear'])
    parser.add_argument('--confirm', action='store_true')
    arguments = parser.parse_args()
    result = inspect() if arguments.action == 'inspect' else clear(arguments.confirm)
    print(json.dumps(result, indent=2))
