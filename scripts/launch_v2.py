"""Build and explicitly submit isolated, pinned v2 Kaggle jobs."""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone

import v2_kaggle_runner as runner

ROOT = Path(__file__).resolve().parents[1]
TEMP = Path(r'C:\Users\AstralK\AppData\Local\Temp\opencode')


def encoded(value):
    data = json.dumps(value, ensure_ascii=False).encode('utf-8')
    return base64.b64encode(gzip.compress(data, mtime=0)).decode('ascii')


def make_jobs(model, model_contract, rows, shard_size, manifest_sha, generation_sha):
    runner.validate_examples(rows)
    if model not in ('tiny', '8b') or type(shard_size) is not int or not 1 <= shard_size <= 1000:
        raise ValueError('Invalid model/shard size')
    jobs = []
    for dataset in sorted({row['dataset'] for row in rows}):
        selected = [row for row in rows if row['dataset'] == dataset]
        for index, start in enumerate(range(0, len(selected), shard_size)):
            examples = selected[start:start + shard_size]
            input_sha = runner.payload_hash(examples)
            name = f'{model}-{dataset.replace("hallulens_", "")}-{index:03d}-{input_sha[:8]}'
            config = {'model_id': model_contract['model_id'], 'revision': model_contract['revision'],
                      'chat_template_sha256': model_contract['chat_template']['sha256'],
                      'selected_template_sha256': model_contract['chat_template']['selected_template_sha256'],
                      'eos_token_id': model_contract['eos_token_id'],
                      'pad_token_id': model_contract['pad_token_id'],
                      'run_name': name, 'input_sha256': input_sha,
                      'generation_sha256': generation_sha, 'v2_manifest_sha256': manifest_sha,
                      'checkpoint_repo': 'ASTRALK/aya-baselines-20261111'}
            runner.validate_config(config, examples)
            jobs.append({'config': config, 'examples': examples})
    return jobs


def notebook_for(job, source):
    import nbformat
    bootstrap = ('import os\nfrom kaggle_secrets import UserSecretsClient\n'
                 'os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")\n')
    code = source.split("if __name__ == '__main__':")[0]
    code += ('\nimport base64, gzip\n' +
             f'run({job["config"]!r}, json.loads(gzip.decompress(base64.b64decode({encoded(job["examples"])!r}))))\n')
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Aya v2: ' + job['config']['run_name']),
        nbformat.v4.new_code_cell(bootstrap), nbformat.v4.new_code_cell(code)],
        metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'}})
    nbformat.validate(notebook)
    return notebook


def write_job(destination, job, source):
    import nbformat
    path = destination / job['config']['run_name']
    path.mkdir(parents=True, exist_ok=False)
    notebook = notebook_for(job, source)
    nbformat.write(notebook, path / 'run.ipynb')
    (path / 'runner.py').write_text(source, encoding='utf-8')
    metadata = {'id': 'astralks/aya-v2-' + job['config']['run_name'],
                'title': 'Aya v2 ' + job['config']['run_name'], 'code_file': 'run.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
                'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    (path / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    receipt = {**job, 'kernel': metadata['id'], 'runner_sha256': runner.digest(source.encode()),
               'notebook_sha256': runner.digest((path / 'run.ipynb').read_bytes()),
               'metadata_sha256': runner.digest((path / 'kernel-metadata.json').read_bytes())}
    (path / 'job.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def verify_job(path):
    import nbformat
    job = json.loads((path / 'job.json').read_bytes())
    runner.validate_config(job['config'], job['examples'])
    source = (path / 'runner.py').read_text(encoding='utf-8')
    if runner.digest(source.encode()) != job['runner_sha256']:
        raise ValueError('Packaged runner changed')
    if runner.digest((path / 'run.ipynb').read_bytes()) != job['notebook_sha256']:
        raise ValueError('Packaged notebook changed')
    if runner.digest((path / 'kernel-metadata.json').read_bytes()) != job['metadata_sha256']:
        raise ValueError('Kaggle metadata changed')
    notebook = nbformat.read(path / 'run.ipynb', as_version=4)
    expected = notebook_for(job, source)
    if [cell.source for cell in notebook.cells] != [cell.source for cell in expected.cells]:
        raise ValueError('Notebook code/payload differs from job')
    metadata = json.loads((path / 'kernel-metadata.json').read_bytes())
    if metadata['id'] != job['kernel'] or metadata['is_private'] is not True:
        raise ValueError('Kaggle identity/privacy mismatch')
    return job


def prepare(root, model, shard_size):
    from prepare_v2_data import verify
    verify(root)
    manifest_path = root / 'data/v2/manifest.json'
    generation_path = root / 'data/v2/generation.jsonl'
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['new_generation_rows'] != 7549 or manifest['counts']['factscore_bios'] != 180:
        raise ValueError('The approved 180-biography inputs are not materialized')
    rows = [json.loads(line) for line in generation_path.read_text(encoding='utf-8').splitlines()]
    model_contract = manifest['generation_contract']['models'][model]
    jobs = make_jobs(model, model_contract, rows, shard_size,
                     runner.digest(manifest_path.read_bytes()), runner.digest(generation_path.read_bytes()))
    source = (root / 'scripts/v2_kaggle_runner.py').read_text(encoding='utf-8')
    destination = root / 'runs/v2' / model / 'jobs'
    paths = [write_job(destination, job, source) for job in jobs]
    print(json.dumps({'model': model, 'jobs': len(paths), 'new_answers': len(rows),
                      'job_paths': [str(path.relative_to(root)) for path in paths]}, indent=2))


def _status_of(slug):
    result = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True,
                            text=True, timeout=60, encoding='utf-8', errors='replace')
    if result.returncode:
        return None
    text = (result.stdout or '').lower()
    for state in ('complete', 'error', 'cancel', 'running', 'queued'):
        if 'kernelworkerstatus.' + state in text:
            return state
    raise RuntimeError('Unrecognized Kaggle status')


_REMOTE_SLUGS = {}


def kernel_status(kernel):
    """Kaggle slugifies underscores to hyphens, so resolve the real remote slug once."""
    cached = _REMOTE_SLUGS.get(kernel)
    if cached is not None:
        state = _status_of(cached)
        if state is None:
            raise RuntimeError('Kaggle status unavailable')
        return state
    for candidate in (kernel, kernel.replace('_', '-')):
        state = _status_of(candidate)
        if state is not None:
            _REMOTE_SLUGS[kernel] = candidate
            return state
    raise RuntimeError('Kaggle status unavailable')


def remote_kernel(kernel):
    kernel_status(kernel)
    return _REMOTE_SLUGS[kernel]


def submit(path):
    import nbformat
    from huggingface_hub import HfApi, get_token
    job = verify_job(path)
    token = os.environ.get('HF_TOKEN') or get_token()
    if not token:
        raise RuntimeError('HF credential is missing')
    HfApi(token=token).create_repo(job['config']['checkpoint_repo'], repo_type='dataset',
                                  private=True, exist_ok=True)
    # Exclusive durable marker blocks duplicate or ambiguous submission after a restart.
    with (path / 'submission-started.json').open('x', encoding='utf-8') as stream:
        json.dump({'kernel': job['kernel'], 'started_at': datetime.now(timezone.utc).isoformat()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    with tempfile.TemporaryDirectory(prefix='aya-v2-submit-', dir=TEMP) as directory:
        transport = Path(directory)
        notebook = nbformat.read(path / 'run.ipynb', as_version=4)
        notebook.cells[1].source = 'import os\nos.environ["HF_TOKEN"] = ' + repr(token) + '\n'
        try:
            nbformat.write(notebook, transport / 'run.ipynb')
            (transport / 'kernel-metadata.json').write_bytes((path / 'kernel-metadata.json').read_bytes())
            result = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(transport),
                                     '--accelerator', 'NvidiaTeslaT4', '--timeout', '21600'],
                                     capture_output=True, text=True, timeout=180)
            log = (result.stdout + result.stderr).replace(token, '[redacted]')
            (path / 'submission.log').write_text(log, encoding='utf-8')
            print(log)
            result.check_returncode()
            (path / 'submission-accepted.json').write_text(json.dumps({'kernel': job['kernel'],
                'accepted_at': datetime.now(timezone.utc).isoformat()}, indent=2), encoding='utf-8')
        finally:
            if (transport / 'run.ipynb').exists():
                (transport / 'run.ipynb').unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('prepare')
    command.add_argument('--model', choices=['tiny', '8b'], required=True)
    command.add_argument('--shard-size', type=int, default=750)
    for name in ('verify', 'submit', 'status'):
        sub.add_parser(name).add_argument('job', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(ROOT, args.model, args.shard_size)
    elif args.command == 'verify':
        job = verify_job(args.job)
        print(json.dumps({'kernel': job['kernel'], 'rows': len(job['examples']), 'verified': True}))
    elif args.command == 'submit':
        submit(args.job)
    else:
        job = verify_job(args.job)
        print(json.dumps({'kernel': job['kernel'], 'status': kernel_status(job['kernel'])}))


if __name__ == '__main__':
    main()
