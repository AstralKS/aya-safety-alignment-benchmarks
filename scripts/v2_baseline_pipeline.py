"""Resume-safe six-benchmark scoring/remote-compute orchestration (not generation)."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import launch_v2
import v2_benchmark_report as report
import v2_compute_jobs as compute
import v2_glm_scoring as scoring
from v2_glm_budget import ATTEMPT_BOUND, Ledger, atomic_replace, json_bytes, money, write_once

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'baseline-v2-full'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def kernel_status(kernel, attempts=10, delay=30):
    last = None
    for attempt in range(attempts):
        try:
            return launch_v2.kernel_status(kernel)
        except RuntimeError as error:
            if str(error) not in ('Kaggle status unavailable', 'Unrecognized Kaggle status'):
                raise
            last = error
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise RuntimeError('Kaggle status unavailable after retries: ' + str(last))


def checkpoint(root, state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    atomic_replace(root / 'runs/v2/baseline-pipeline-progress.json', json_bytes(state))


def load_state(root):
    path = root / 'runs/v2/baseline-pipeline-progress.json'
    return scoring.client.read_json(path.read_bytes()) if path.exists() else {
        'kind': 'aya-v2-baseline-pipeline-v1', 'run_id': RUN_ID, 'complete': False, 'fixed': {}, 'factscore': {}, 'compute': {}}


def require_state(state):
    scoring.require(state.get('kind') == 'aya-v2-baseline-pipeline-v1' and state.get('run_id') == RUN_ID,
                    'Unexpected pipeline state')


def source_fingerprint(available, tasks):
    identities = [(short, row['id'], row['_input_row_sha256'], sha(answer.encode()))
                  for row, answer, short in available if row['dataset'] in tasks]
    return sha(json.dumps(sorted(identities), ensure_ascii=False).encode())


def result_problems(packet):
    return [outcome for outcome in packet.get('results', []) if outcome.get('ok') is not True]


def require_usable_result(root, path):
    packet = scoring.client.read_json(Path(path).read_bytes())
    scoring.require(packet.get('halted', False) is False, 'Scoring halted; preserve the snapshot and investigate before continuing')
    problems = result_problems(packet)
    scoring.require(not problems, 'Scoring has unresolved judgments: ' + str(Counter(o.get('state') for o in problems)))
    return packet


def require_complete_extraction(root, path, binding):
    packet = scoring.client.read_json(Path(path).read_bytes())
    scoring.require(packet.get('kind') == 'glm-factscore-extractions-v1' and packet.get('v2_inputs') == binding,
                    'Atomic extraction input binding changed')
    responses = packet.get('responses')
    scoring.require(packet.get('halted', False) is False and isinstance(responses, list) and len(responses) == 360
                    and all(response.get('completed') is True for response in responses), 'Atomic extraction is incomplete')
    return packet


def build_report(root, state, write=False):
    return report.build(root, root / state['fixed']['snapshot'],
        root / state['compute']['selfaware']['receipt']['result_path'], root / state['factscore']['verification'],
        root / state['factscore']['ready'], root / state['compute']['factscore']['receipt']['result_path'], write=write)


def require_published_report(root, state):
    compiled = build_report(root, state, write=False)
    machine = scoring.client.read_json((root / 'scoring/v2-glm/benchmark-results.json').read_bytes())
    normalized = scoring.client.read_json(json_bytes(compiled))
    comparable = lambda value: {key: item for key, item in value.items() if key != 'verified_at'}
    scoring.require(comparable(machine) == comparable(normalized), 'Published benchmark checkpoint differs from verified results')
    markdown = report.render(compiled['metrics'], compiled['accounting'], compiled['artifacts'],
                             compiled['new_generation_finish_reasons']).encode('utf-8')
    scoring.require((root / 'benchmark.md').read_bytes() == markdown, 'Published benchmark report differs from verified results')


def active_generation(root):
    path = root / 'runs/v2/generation-progress.json'
    if not path.exists():
        return 2  # Never assume capacity when scheduler state is absent.
    progress = json.loads(path.read_bytes())
    states = [job['state'] for model in progress['models'].values() for job in model['jobs']]
    scoring.require(not any(status in ('error', 'cancel') for status in states), 'Generation failed; preserved output requires recovery')
    return sum(status in ('running', 'queued') for status in states)


def collect_compute(root, path, kind):
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = (root / path).resolve()
    job = compute.verify(path, root)
    marker, output = path / 'collected.json', path / 'output'
    filename = 'grades.jsonl' if kind == 'selfaware' else 'retrieval.jsonl'
    if not marker.exists() and output.exists():
        # A previous attempt finished downloading before writing its receipt; re-validate the
        # preserved output instead of downloading over it.
        scoring.require(len(list(output.rglob(filename))) == 1, 'Preserved compute output lacks a unique ' + filename)
    elif not marker.exists():
        scoring.require(not output.exists(), 'Interrupted compute download preserved; never download over it')
        result = subprocess.run(['kaggle', 'kernels', 'output', launch_v2.remote_kernel(job['kernel']), '-p', str(output)],
                                capture_output=True, text=True, timeout=180, encoding='utf-8', errors='replace')
        (path / 'download.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        result.check_returncode()
    matches = list(output.rglob(filename))
    scoring.require(len(matches) == 1, 'Compute output lacks a unique ' + filename)
    raw = matches[0].read_bytes()
    payload = json.loads((path / 'inputs.json').read_bytes())
    if kind == 'selfaware':
        rows = [json.loads(line) for line in raw.splitlines()]
        available, _ = scoring.load_available(root, ('tiny', '8b'))
        report.selfaware_table(rows, available)
    else:
        scoring.read_remote_retrieval(matches[0], payload)
    receipt = {'kind': kind, 'kernel': job['kernel'], 'result_path': matches[0].relative_to(root).as_posix(),
               'result_sha256': sha(raw), 'input_sha256': job['input_sha256']}
    if kind == 'factscore':
        manifest = matches[0].with_name('manifest.json')
        receipt['manifest_sha256'] = sha(manifest.read_bytes())
    if marker.exists():
        scoring.require(scoring.client.read_json(marker.read_bytes()) == receipt, 'Collected remote compute output changed')
    else:
        write_once(marker, json_bytes(receipt))
    return receipt


def advance_compute(root, state, kind, payload):
    jobs = state['compute']
    info = jobs.setdefault(kind, {})
    identifier = kind + '-' + sha(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode())[:12]
    path = root / 'runs/v2/compute' / identifier
    if not info.get('directory'):
        if path.exists():
            compute.verify(path, root)
            scoring.require(json.loads((path / 'inputs.json').read_bytes()) == payload, 'Prepared compute payload differs')
        else:
            path = compute.prepare(root, kind, payload)
        info.update(directory=path.relative_to(root).as_posix(), state='prepared')
        checkpoint(root, state)
    else:
        scoring.require((root / info['directory']).resolve() == path.resolve(), 'Compute inputs changed after preparation')
        compute.verify(path, root)
    if info.get('receipt'):
        info['receipt'] = collect_compute(root, path, kind)
        return info['receipt']
    if (path / 'collected.json').exists():
        info.update(state='complete', receipt=collect_compute(root, path, kind))
        checkpoint(root, state)
        return info['receipt']
    if not (path / 'submission-started.json').exists():
        other_active = sum(record.get('state') in ('running', 'queued', 'submitted')
                           for other, record in jobs.items() if other != kind)
        if active_generation(root) + other_active >= 2:
            return None
        compute.submit(path)  # Exclusive durable marker prevents replay even after process death.
        info['state'] = 'submitted'
        checkpoint(root, state)
    job = compute.verify(path, root)
    status = kernel_status(job['kernel'])
    info['state'] = status
    checkpoint(root, state)
    scoring.require(status not in ('error', 'cancel'), 'Remote ' + kind + ' failed; do not resubmit automatically')
    if status == 'complete':
        info['receipt'] = collect_compute(root, path, kind)
        checkpoint(root, state)
        return info['receipt']
    return None


def tick(root, state):
    active_generation(root)  # Fail before scoring or waiting for remote-compute readiness.
    require_state(state)
    available, binding = scoring.load_available(root, ('tiny', '8b'))
    facts = state['factscore']
    # Checkpointed pointers preserve failures; they never prove a stage succeeded.
    if facts.get('extractions'):
        scoring.require(facts.get('extraction_halted', False) is False, 'Atomic extraction is incomplete')
        require_complete_extraction(root, root / facts['extractions'], binding)
    if facts.get('verification'):
        require_usable_result(root, root / facts['verification'])
    if state['fixed'].get('snapshot'):
        require_usable_result(root, root / state['fixed']['snapshot'])
    state['counts'] = {short: dict(Counter(row['dataset'] for row, _, model in available if model == short))
                       for short in ('tiny', '8b')}
    fingerprint = source_fingerprint(available, scoring.FIXED_TASKS)
    if state['fixed'].get('source_fingerprint') != fingerprint:
        result = scoring.score_fixed(root, RUN_ID, dispatch=True)
        state['fixed'].update(snapshot=str(Path(result['snapshot']).relative_to(root)), source_fingerprint=fingerprint)
        checkpoint(root, state)
        require_usable_result(root, result['snapshot'])
    else:
        require_usable_result(root, root / state['fixed']['snapshot'])
    self_complete = all(state['counts'][short].get('selfaware') == 3369 for short in ('tiny', '8b'))
    if self_complete:
        advance_compute(root, state, 'selfaware', compute.selfaware_inputs(root))
    bio_complete = all(state['counts'][short].get('factscore_bios') == 180 for short in ('tiny', '8b'))
    if bio_complete:
        if not facts.get('extractions'):
            result = scoring.factscore_stage(root, RUN_ID + '-facts', 'extract', dispatch=True)
            facts.update(extractions=str(Path(result['snapshot']).relative_to(root)), extraction_halted=result['halted'])
            checkpoint(root, state)
            scoring.require(facts['extraction_halted'] is False, 'Atomic extraction is incomplete')
            require_complete_extraction(root, root / facts['extractions'], binding)
        if not facts.get('ready'):
            result = scoring.factscore_stage(root, RUN_ID + '-facts', 'retrieval-ready', extraction_path=root / facts['extractions'])
            facts['ready'] = str(Path(result['snapshot']).relative_to(root))
            checkpoint(root, state)
        ready = scoring.client.read_json((root / facts['ready']).read_bytes())
        receipt = advance_compute(root, state, 'factscore', ready)
        if receipt and not facts.get('verification'):
            result = scoring.factscore_stage(root, RUN_ID + '-facts', 'verify', dispatch=True,
                extraction_path=root / facts['extractions'], retrieval_path=root / receipt['result_path'])
            facts['verification'] = str(Path(result['snapshot']).relative_to(root))
            checkpoint(root, state)
            require_usable_result(root, result['snapshot'])
    generation_complete = all(sum(state['counts'][short].values()) == 9527 for short in ('tiny', '8b'))
    if generation_complete and facts.get('verification') and all(state['compute'].get(kind, {}).get('receipt') for kind in ('selfaware', 'factscore')):
        compiled = build_report(root, state, write=True)
        state.update(complete=True, report='benchmark.md', accounting=compiled['accounting'])
    state['v2_inputs'] = binding
    checkpoint(root, state)
    return state


def run(root=ROOT, interval=60):
    root = Path(root).resolve()
    guard = root / 'runs/v2/baseline-pipeline.guard'
    guard_raw = json_bytes({'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat()})
    write_once(guard, guard_raw)
    try:
        state = load_state(root)
        require_state(state)
        if state['complete']:
            require_published_report(root, state)
            return state
        while True:
            try:
                state = tick(root, state)
                print(json.dumps({'counts': state['counts'], 'compute': state['compute'], 'complete': state['complete']}, default=str), flush=True)
            except RuntimeError as error:
                if str(error).split(':')[0] == 'Kaggle status unavailable after retries':
                    print(json.dumps({'transient_status_error': str(error),
                                      'updated_at': datetime.now(timezone.utc).isoformat()}), flush=True)
                    time.sleep(interval * 5)
                    continue
                state['error'] = {'type': type(error).__name__, 'message': str(error), 'at': datetime.now(timezone.utc).isoformat()}
                state['accounting'] = Ledger(root / 'scoring/budget.json').totals()
                checkpoint(root, state)
                raise
            except Exception as error:
                state['error'] = {'type': type(error).__name__, 'message': str(error), 'at': datetime.now(timezone.utc).isoformat()}
                state['accounting'] = Ledger(root / 'scoring/budget.json').totals()
                checkpoint(root, state)
                raise
            if state['complete']:
                return state
            # Scheduling remote services, never polling the OpenCode background shell.
            time.sleep(interval)
    finally:
        if guard.exists() and guard.read_bytes() == guard_raw:
            guard.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    scoring.require(args.interval >= 30, 'Remote scheduling interval must be at least 30 seconds')
    print(json_bytes(run(args.root, args.interval)).decode())
