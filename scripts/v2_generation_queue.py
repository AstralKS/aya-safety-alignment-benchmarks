"""Isolated v2 queue: one active Kaggle generation job per Aya model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import launch_v2 as launch
import v2_kaggle_runner as runner

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_output_records(records, job, complete):
    examples = {row['id']: row for row in job['examples']}
    config = job['config']
    ids = set()
    for record in records:
        identifier = record.get('id')
        if identifier not in examples or identifier in ids:
            raise ValueError('Unknown or duplicate generated answer ID')
        ids.add(identifier)
        example = examples[identifier]
        expected = {**{key: example[key] for key in ('id', 'dataset', 'component', 'max_new_tokens')},
                    **{key: config[key] for key in ('model_id', 'revision', 'input_sha256',
                        'chat_template_sha256', 'selected_template_sha256')},
                    'example_sha256': runner.payload_hash(example)}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError('Generated answer provenance mismatch')
        if record.get('status') != 'ok' or not isinstance(record.get('response'), str) or not record['response'].strip():
            raise ValueError('Generation contains failed or empty answers')
        if record.get('finish_reason') not in ('eos', 'length'):
            raise ValueError('Invalid generation finish reason')
        prompt_sha = record.get('prompt_sha256')
        if not isinstance(prompt_sha, str) or len(prompt_sha) != 64 or any(c not in '0123456789abcdef' for c in prompt_sha):
            raise ValueError('Missing rendered prompt hash')
        if type(record.get('input_tokens')) is not int or record['input_tokens'] < 1:
            raise ValueError('Invalid input token count')
        if type(record.get('output_tokens')) is not int or not 1 <= record['output_tokens'] <= example['max_new_tokens']:
            raise ValueError('Invalid output token count')
    if complete and ids != set(examples):
        raise ValueError('Completed shard has missing answers')
    return records


def aggregate(root, model, groups):
    records = [record for group in groups for record in group]
    if len({record['id'] for record in records}) != len(records):
        raise ValueError('Duplicate IDs across generated shards')
    records.sort(key=lambda record: record['id'])
    content = ''.join(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n' for record in records).encode()
    path = root / 'runs/v2' / model / 'responses.jsonl'
    if not path.exists() or path.read_bytes() != content:
        atomic_bytes(path, content)
    return len(records)


def choose_next(states):
    failed = [path for path, state in states if state in ('error', 'cancel')]
    if failed:
        raise RuntimeError('Generation failed; preserved output needs explicit recovery: ' + str(failed[0]))
    if any(state in ('running', 'queued') for _, state in states):
        return None
    return next((path for path, state in states if state == 'unsubmitted'), None)


def validate_runtime_manifest(manifest, job):
    config = job['config']
    if (manifest.get('config') != config or manifest.get('resolved_revision') != config['revision']
            or manifest.get('model_id') != config['model_id'] or manifest.get('weight_dtype') != 'float16'
            or manifest.get('quantization') is not None or manifest.get('stage') != 'completed'
            or manifest.get('successful') != len(job['examples']) or manifest.get('errors') != 0
            or type(manifest.get('sample_count')) is not int or manifest['sample_count'] != len(job['examples'])
            or manifest.get('transformers') != '4.57.6' or manifest.get('accelerate') != '1.12.0'
            or manifest.get('chat_template_sha256') != config['chat_template_sha256']
            or manifest.get('selected_template_sha256') != config['selected_template_sha256']):
        raise ValueError('Kaggle runtime manifest violates frozen generation contract')
    devices = set()
    for gpu in manifest.get('gpu_info', []):
        if type(gpu.get('index')) is not int or gpu['index'] < 0:
            raise ValueError('Invalid GPU metadata')
        devices.update((str(gpu['index']), 'cuda:' + str(gpu['index'])))
    mapping = manifest.get('device_map')
    if not devices or not isinstance(mapping, dict) or not mapping or any(value not in devices for value in mapping.values()):
        raise ValueError('Kaggle model was not wholly GPU-placed')


def collect(path, job):
    output = path / 'output'
    marker = path / 'collected.json'
    if not marker.exists():
        if output.exists():
            raise RuntimeError('Interrupted download preserved; inspect before a second download: ' + str(output))
        result = subprocess.run(['kaggle', 'kernels', 'output', launch.remote_kernel(job['kernel']), '-p', str(output)],
                                capture_output=True, text=True, timeout=180, encoding='utf-8', errors='replace')
        (path / 'download.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        result.check_returncode()
    responses = list(output.rglob('responses.jsonl'))
    manifests = list(output.rglob('manifest.json'))
    if len(responses) != 1 or len(manifests) != 1:
        raise ValueError('Kaggle output lacks unique responses/runtime manifest')
    manifest = json.loads(manifests[0].read_bytes())
    validate_runtime_manifest(manifest, job)
    raw = responses[0].read_bytes()
    records = validate_output_records([json.loads(line) for line in raw.splitlines()], job, complete=True)
    receipt = {'kernel': job['kernel'], 'responses_sha256': runner.digest(raw),
               'runtime_manifest_sha256': runner.digest(manifests[0].read_bytes()), 'answers': len(records)}
    if marker.exists():
        if json.loads(marker.read_bytes()) != receipt:
            raise ValueError('Collected generation output changed')
    else:
        with marker.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2)
    return records


def ordered_jobs(root, model):
    paths = list((root / 'runs/v2' / model / 'jobs').iterdir())
    priority = {'selfaware': 0, 'hallulens_precisewikiqa': 1,
                'hallulens_nonexistentrefusal': 2, 'factscore_bios': 3}
    return sorted(paths, key=lambda path: (priority[json.loads((path / 'job.json').read_bytes())['examples'][0]['dataset']], path.name))


def status_with_retry(kernel, attempts=10, delay=30):
    last = None
    for attempt in range(attempts):
        try:
            return launch.kernel_status(kernel)
        except RuntimeError as error:
            if str(error) not in ('Kaggle status unavailable', 'Unrecognized Kaggle status'):
                raise
            last = error
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise RuntimeError('Kaggle status unavailable after retries: ' + str(last))


def tick(root=ROOT):
    summary = {'updated_at': datetime.now(timezone.utc).isoformat(), 'models': {}, 'complete': True}
    for model in ('tiny', '8b'):
        states, groups = [], []
        paths = ordered_jobs(root, model)
        for path in paths:
            job = launch.verify_job(path)
            if (path / 'collected.json').exists():
                state = 'complete'
                groups.append(collect(path, job))
            elif (path / 'submission-started.json').exists():
                state = status_with_retry(job['kernel'])
                if state == 'complete':
                    groups.append(collect(path, job))
                elif state in ('error', 'cancel'):
                    failed_output = path / 'failed-output'
                    if not failed_output.exists():
                        result = subprocess.run(['kaggle', 'kernels', 'output', launch.remote_kernel(job['kernel']), '-p', str(failed_output)],
                                                capture_output=True, text=True, timeout=180, encoding='utf-8', errors='replace')
                        (path / 'failure-download.log').write_text(result.stdout + result.stderr, encoding='utf-8')
            else:
                state = 'unsubmitted'
            states.append((path, state))
        # Publish every validated completed shard even if another job or submission fails.
        count = aggregate(root, model, groups)
        next_path = choose_next(states)
        if next_path is not None:
            launch.submit(next_path)
            states = [(path, 'queued' if path == next_path else state) for path, state in states]
        complete = count == 7549 and all(state == 'complete' for _, state in states)
        summary['complete'] = summary['complete'] and complete
        summary['models'][model] = {'new_answers_saved': count, 'target': 7549, 'complete': complete,
                                   'jobs': [{'name': path.name, 'state': state} for path, state in states]}
    atomic_json(root / 'runs/v2/generation-progress.json', summary)
    return summary


def supervise(root=ROOT, interval=60):
    guard = root / 'runs/v2/generation-queue.guard'
    with guard.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        while True:
            try:
                summary = tick(root)
            except RuntimeError as error:
                # Transient Kaggle status outage: exit is not focused on unknowable remote state.
                if str(error).split(':')[0] != 'Kaggle status unavailable after retries':
                    raise
                print(json.dumps({'transient_status_error': str(error),
                                  'updated_at': datetime.now(timezone.utc).isoformat()}), flush=True)
                time.sleep(interval * 5)
                continue
            print(json.dumps(summary), flush=True)
            if summary['complete']:
                return summary
            # Remote job scheduling, not polling the OpenCode background-shell output.
            time.sleep(interval)
    finally:
        guard.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['tick', 'run'])
    args = parser.parse_args()
    print(json.dumps(tick() if args.command == 'tick' else supervise()))
