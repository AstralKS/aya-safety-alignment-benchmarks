"""Package exact remote SimCSE/GTR jobs; no local model inference."""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import v2_kaggle_runner as hashes

ROOT = Path(__file__).resolve().parents[1]
TEMP = Path(r'C:\Users\AstralK\AppData\Local\Temp\opencode')
SOURCES = {'selfaware': 'scripts/v2_selfaware.py', 'factscore': 'scripts/v2_factscore_retrieval.py'}
REPO = 'ASTRALK/aya-baselines-20261111'
FROZEN_EVIDENCE_SHA256 = '29b3c11c3beb319051c1f2588d12985edf25537927ab80569f9fc930275f7fb1'


def unique_records(records):
    by_id = {}
    for record in records:
        identifier = record.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError('Duplicate/invalid generated response ID')
        by_id[identifier] = record
    return by_id


def strict_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


class CollectedOnlyPath(type(ROOT)):
    """Prevent collect's download/write branch even if its receipt disappears."""
    def exists(self, **kwargs):
        exists = super().exists(**kwargs)
        if self.name == 'collected.json' and not exists:
            raise ValueError('Collected receipt disappeared; input loading cannot download')
        return exists

    def open(self, mode='r', *args, **kwargs):
        if any(flag in mode for flag in 'wax+'):
            raise ValueError('Input loading cannot write collection artifacts')
        return super().open(mode, *args, **kwargs)


def selfaware_inputs(root):
    from prepare_v2_data import verify
    verify(root)
    evaluation = [json.loads(line) for line in (root / 'data/v2/evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = {row['id']: row for row in evaluation if row['dataset'] == 'selfaware'}
    if len(selected) != 3369:
        raise ValueError('SelfAware source count changed')
    result = []
    for model, model_id in (('tiny', 'CohereLabs/tiny-aya-global'), ('8b', 'CohereLabs/aya-expanse-8b')):
        responses = [json.loads(line) for line in (root / 'runs/v2' / model / 'responses.jsonl').read_text(encoding='utf-8').splitlines()]
        aggregate = unique_records(responses)
        by_id = unique_records([row for row in responses if row['dataset'] == 'selfaware'])
        if len(by_id) != 3369 or set(by_id) != set(selected):
            raise ValueError('SelfAware generation is not complete for ' + model)
        from launch_v2 import verify_job
        from v2_generation_queue import collect
        validated = set()
        for job_path in (root / 'runs/v2' / model / 'jobs').iterdir():
            job = verify_job(job_path)
            if not any(example['id'] in aggregate for example in job['examples']):
                continue
            # collect() is strictly offline only when its existing receipt is present.
            # Never let an input loader initiate or finish an uncollected download.
            if not (job_path / 'collected.json').is_file():
                raise ValueError('Aggregate references an uncollected generation job')
            group = collect(CollectedOnlyPath(job_path), job)
            for record in group:
                identifier = record['id']
                if (identifier in validated or identifier not in aggregate
                        or not strict_equal(aggregate[identifier], record)):
                    raise ValueError('Aggregate differs from completed verified generation output')
            validated.update(record['id'] for record in group)
        if validated != set(aggregate):
            raise ValueError('Aggregate responses lack collected job provenance')
        for identifier, row in selected.items():
            record = by_id[identifier]
            if record.get('status') != 'ok' or record.get('model_id') != model_id or record.get('revision') != hashes.PINS[model_id]:
                raise ValueError('Unvalidated SelfAware generation')
            result.append({'id': identifier, 'model_id': model_id, 'response': record['response'],
                           'response_sha256': hashes.digest(record['response'].encode()),
                           'answerable': row['metadata']['answerable'], 'answer': row['metadata']['answer'],
                           'source': row['metadata']['source'], 'question_id': row['metadata']['question_id']})
    return result


def rooted_source(root, relative):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or '..' in Path(relative).parts):
        raise ValueError('Extraction source must be a root-relative isolated GLM path')
    path = (root / relative).resolve()
    if not path.is_relative_to((root / 'scoring/v2-glm').resolve()):
        raise ValueError('Extraction source escapes the isolated GLM namespace')
    return path


def saved_extraction_outcome(root, run_directory, job, scoring):
    """Pure journal parsing; deliberately never instantiate/evaluate a JobStore."""
    folder = rooted_source(root, 'scoring/v2-glm/request-journals/' + job['cache_key'])
    pending = {'cache_key': job['cache_key'], 'ok': False, 'label': None}
    if not folder.exists():
        return pending
    contract = scoring.client.read_json((folder / 'job.json').read_bytes())
    if not strict_equal(contract, scoring.request_contract(job)):
        raise ValueError('Saved extraction request/cache contract differs from approved derivation')
    name = scoring.digest({'key': job['cache_key'], 'lineage': job['lineage']}) + '.json'
    request_path = rooted_source(root, (run_directory / 'requests' / name).relative_to(root.resolve()).as_posix())
    if not strict_equal(scoring.client.read_json(request_path.read_bytes()), job):
        raise ValueError('Saved extraction job has different approved Aya/source/answer lineage')
    outcomes = []
    for path in (folder / 'attempts').glob('*.response.json'):
        if path.resolve().parent != (folder / 'attempts').resolve():
            raise ValueError('Extraction journal escapes its approved request cache')
        raw = path.read_bytes()
        journal = scoring.client.read_json(raw)
        if (journal.get('request_sha256') != job['request_sha256']
                or journal.get('attempt_id') != path.name.removesuffix('.response.json')):
            raise ValueError('Extraction journal belongs to a different request/attempt')
        outcome = {**pending, 'journal_path': str(path.resolve()), 'journal_sha256': hashes.digest(raw)}
        if journal.get('kind') == 'http_response' and journal.get('http_status') == 200:
            body = base64.b64decode(journal['raw_body_base64'], validate=True)
            if hashes.digest(body) != journal.get('raw_body_sha256'):
                raise ValueError('Saved extraction raw response hash mismatch')
            parsed = scoring.client.inspect_response(body, 'factscore_bios',
                lambda text: scoring.parse_final(text, job['parser']))
            outcome.update(ok=parsed['ok'], label=parsed['label'])
        outcomes.append(outcome)
    return next((outcome for outcome in outcomes if outcome['ok']), outcomes[-1] if outcomes else pending)


def rederive_factscore_payload(root, payload, frozen):
    # Lazy import avoids a compute/scoring import cycle in callers and tests.
    import v2_glm_scoring as scoring
    root = root.resolve()
    packet_sha = payload.get('extraction_packet_sha256')
    packet, run_directory = None, None
    for path in (root / 'scoring/v2-glm/results').glob('*/snapshots/extract-*.json'):
        path = rooted_source(root, path.relative_to(root).as_posix())
        candidate = scoring.client.read_json(path.read_bytes())
        if hashes.digest(scoring.json_bytes(candidate)) == packet_sha:
            packet, run_directory = candidate, path.parent.parent
            break
    if packet is None or not isinstance(packet.get('responses'), list):
        raise ValueError('Retrieval payload has no bound original GLM extraction snapshot')
    available, binding = scoring.load_available(root, ('tiny', '8b'))
    by_identity = {(short, row['id']): (row, answer, short) for row, answer, short in available
                   if row['dataset'] == 'factscore_bios'}
    selected, identities = [], set()
    for response in packet['responses']:
        identity = (response.get('aya_model'), response.get('id'))
        if identity not in by_identity or identity in identities:
            raise ValueError('Extraction biography/model is not an approved available Aya response')
        identities.add(identity)
        selected.append(by_identity[identity])
    catalog = scoring.SourceCatalog(root)
    extraction_jobs = [job for row, answer, short in selected
                       for job in scoring.extraction_jobs(row, answer, short, catalog)]
    outcomes = [saved_extraction_outcome(root, run_directory, job, scoring) for job in extraction_jobs]
    # extraction_packet is a pure formatter/raw-receipt reader, not the scoring
    # validate_extraction_packet helper (which evaluates and settles a JobStore).
    expected_packet = scoring.extraction_packet(selected, extraction_jobs, outcomes, binding, root=root)
    if not strict_equal(packet, expected_packet):
        raise ValueError('Extraction snapshot differs from approved source/response/request/GLM journals')
    facts = []
    for response in expected_packet['responses']:
        if not response['completed']:
            continue
        if response['topic'] not in frozen:
            raise ValueError('Extraction subject is outside the approved frozen inventory')
        source = frozen[response['topic']]
        text_file = next(file for file in source['files'] if file['path'].endswith('/text.txt'))
        for fact in response['facts']:
            facts.append({**fact, 'fact': fact['text'], 'response_sha256': fact['answer_sha256'],
                          'source_text_sha256': text_file['sha256'], 'source_text_file': text_file['path'],
                          'title': source['title'], 'revision': source['revision'], 'oldid_url': source['url']})
    topics = {fact['topic'] for fact in facts}
    evidence = []
    for topic, source in frozen.items():
        if topic not in topics:
            continue
        text_file = next(file for file in source['files'] if file['path'].endswith('/text.txt'))
        raw = (root / text_file['path']).read_bytes()
        if hashes.digest(raw) != text_file['sha256']:
            raise ValueError('Frozen Wikipedia text hash mismatch')
        evidence.append({'topic': topic, 'text': raw.decode('utf-8'), 'text_sha256': text_file['sha256'],
                         'revision': source['revision']})
    facts_raw = b''.join(scoring.client.canonical(fact) for fact in facts)
    facts_sha = hashes.digest(facts_raw)
    # Mirror the producer's retrieval-ready schema without calling its write-once
    # export routine: validation must not create exports, snapshots or ledger work.
    expected = {'kind': 'glm-factscore-retrieval-ready-v1',
        'extraction_packet_sha256': hashes.digest(scoring.json_bytes(expected_packet)),
        'evidence_manifest_sha256': FROZEN_EVIDENCE_SHA256, 'retrieval_model': scoring.GTR_MODEL,
        'provenance': {'evidence_manifest_sha256': FROZEN_EVIDENCE_SHA256,
            'facts_file': {'path': 'scoring/v2-glm/facts-exports/facts-' + facts_sha + '.jsonl', 'sha256': facts_sha}},
        'retrieval_revision': scoring.GTR_REVISION, 'tokenizer_model': scoring.TOKENIZER_MODEL,
        'tokenizer_revision': scoring.TOKENIZER_REVISION, 'passage_tokens': 256, 'passages_per_fact': 5,
        'evidence': evidence, 'facts': facts,
        'required_retrieval_input': 'Controller v2_factscore_retrieval.run(payload): retrieval.jsonl plus completed manifest.json, pinned remote GTR/title+fact/inner product and RoBERTa 256-token subject-local chunks. No local model/tokenizer loading.'}
    if not strict_equal(payload, expected):
        raise ValueError('Retrieval payload is not the exact rederived GLM retrieval-ready snapshot')


def validate_factscore_payload(root, payload):
    from v2_factscore_retrieval import validate_inputs
    validate_inputs(payload)
    provenance = payload.get('provenance', {})
    manifest_path = root / 'data/v2/factscore-evidence-manifest.json'
    manifest_raw = manifest_path.read_bytes()
    if (hashes.digest(manifest_raw) != FROZEN_EVIDENCE_SHA256
            or provenance.get('evidence_manifest_sha256') != FROZEN_EVIDENCE_SHA256):
        raise ValueError('Missing frozen evidence manifest binding')
    report = json.loads(manifest_raw)
    if report.get('ready') is not True or report.get('count') != 180 or len(report.get('records', [])) != 180:
        raise ValueError('Frozen evidence is not the approved 180-subject inventory')
    frozen = {record['topic']: record for record in report['records']}
    if len(frozen) != 180:
        raise ValueError('Duplicate frozen evidence subjects')
    for record in payload['evidence']:
        if record['topic'] not in frozen:
            raise ValueError('Retrieval subject is not in frozen approved evidence')
        original = frozen[record['topic']]
        text_file = next(file for file in original['files'] if file['path'].endswith('/text.txt'))
        raw = (root / text_file['path']).read_bytes()
        if (hashes.digest(raw) != text_file['sha256'] or record['text_sha256'] != text_file['sha256']
                or record['text'] != raw.decode('utf-8') or record['revision'] != original['revision']):
            raise ValueError('Retrieval evidence differs from verified frozen Wikipedia')
    source = provenance.get('facts_file', {})
    source_path = rooted_source(root, source.get('path', ''))
    raw = source_path.read_bytes()
    if hashes.digest(raw) != source.get('sha256'):
        raise ValueError('Missing/tampered independent GLM facts export')
    exported = [json.loads(line) for line in raw.splitlines()]
    if not strict_equal(exported, payload['facts']):
        raise ValueError('Retrieval facts differ from independent GLM export')
    rederive_factscore_payload(root, payload, frozen)


def prepare(root, kind, payload, rebuild=False):
    import nbformat
    if kind not in SOURCES:
        raise ValueError('Unknown remote computation')
    if kind == 'factscore':
        validate_factscore_payload(root, payload)
    elif not isinstance(payload, list) or not payload:
        raise ValueError('No SelfAware inputs')
    elif payload != selfaware_inputs(root):
        raise ValueError('SelfAware payload differs from validated source/response derivation')
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    identifier = kind + '-' + hashes.digest(raw)[:12]
    path = root / 'runs/v2/compute' / identifier
    if rebuild:
        # Kaggle rejects SaveKernel bodies above ~1 MB, so the reviewed transport now streams the
        # unchanged inputs.json through the private HF dataset instead of embedding it in the notebook.
        job = json.loads((path / 'job.json').read_bytes())
        if job['input_sha256'] != hashes.digest(raw) or (path / 'inputs.json').read_bytes() != raw:
            raise ValueError('Rebuild payload differs from the preserved package')
    else:
        path.mkdir(parents=True, exist_ok=False)
    source = (root / SOURCES[kind]).read_text(encoding='utf-8')
    bootstrap = ('import os, subprocess, sys\nfrom kaggle_secrets import UserSecretsClient\n'
                 'os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")\n'
                 'subprocess.run([sys.executable, "-m", "pip", "install", "-q", '
                 '"transformers==4.57.6", "sentence-transformers==3.4.1", "huggingface-hub==0.36.0", '
                 '"sentencepiece==0.2.1", "scikit-learn"], check=True)\n'
                 'subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "librosa", "torchcodec"], '
                 'stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n')
    code = source.split("if __name__ == '__main__':")[0]
    code += (f'\nimport hashlib, json, os\nfrom huggingface_hub import hf_hub_download\n'
             f'_inputs = hf_hub_download(repo_id={REPO!r}, repo_type="dataset", token=os.environ["HF_TOKEN"],\n'
             f'                           filename="v2/compute/{identifier}/inputs.json")\n'
             f'_raw = open(_inputs, "rb").read()\n'
             f'if hashlib.sha256(_raw).hexdigest() != {hashes.digest(raw)!r}:\n'
             f'    raise SystemExit("Remote compute inputs changed")\n'
             f'run(json.loads(_raw), token=os.environ["HF_TOKEN"])\n')
    checkpoint = ('from huggingface_hub import HfApi\n'
                  f'HfApi(token=os.environ["HF_TOKEN"]).upload_folder(repo_id={REPO!r}, '
                  f'repo_type="dataset", folder_path="/kaggle/working", path_in_repo="v2/compute/{identifier}", '
                  'allow_patterns=["*.json", "*.jsonl"])\n')
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Aya v2 remote ' + kind),
        nbformat.v4.new_code_cell(bootstrap), nbformat.v4.new_code_cell(code), nbformat.v4.new_code_cell(checkpoint)],
        metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}})
    nbformat.validate(notebook)
    nbformat.write(notebook, path / 'run.ipynb')
    (path / 'inputs.json').write_bytes(raw)
    metadata = {'id': 'astralks/aya-v2-compute-' + identifier, 'title': 'Aya v2 compute-' + identifier,
                'code_file': 'run.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [], 'competition_sources': [],
                'kernel_sources': [], 'model_sources': []}
    (path / 'kernel-metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    receipt = {'kind': kind, 'kernel': metadata['id'], 'input_sha256': hashes.digest(raw),
               'source_path': SOURCES[kind], 'source_sha256': hashes.digest(source.encode()),
               'notebook_sha256': hashes.digest((path / 'run.ipynb').read_bytes()),
               'metadata_sha256': hashes.digest((path / 'kernel-metadata.json').read_bytes()),
               'transport': 'hf-dataset-inputs-v1'}
    (path / 'job.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return path


def verify(path, root=ROOT):
    job = json.loads((path / 'job.json').read_bytes())
    if job['kind'] not in SOURCES or job['source_path'] != SOURCES[job['kind']]:
        raise ValueError('Unexpected compute source')
    for filename, key in (('inputs.json', 'input_sha256'), ('run.ipynb', 'notebook_sha256'),
                          ('kernel-metadata.json', 'metadata_sha256')):
        if hashes.digest((path / filename).read_bytes()) != job[key]:
            raise ValueError('Remote compute package changed: ' + filename)
    if hashes.digest((root / job['source_path']).read_text(encoding='utf-8').encode()) != job['source_sha256']:
        raise ValueError('Reviewed compute source changed')
    metadata = json.loads((path / 'kernel-metadata.json').read_bytes())
    if metadata.get('is_private') is not True or metadata['id'] != job['kernel']:
        raise ValueError('Remote computation privacy/identity changed')
    return job


def push_missing_remote(metadata_id):
    """True only when Kaggle positively reports the kernel is absent; unknown outcomes never retry."""
    result = subprocess.run(['kaggle', 'kernels', 'status', metadata_id], capture_output=True,
                            text=True, timeout=60, encoding='utf-8', errors='replace')
    text = (result.stdout or '') + (result.stderr or '')
    return 'Cannot access kernel' in text and 'Permission' in text


def submit(path):
    import nbformat
    from huggingface_hub import get_token
    job = verify(path)
    token = os.environ.get('HF_TOKEN') or get_token()
    if not token:
        raise RuntimeError('Missing HF credential')
    with (path / 'submission-started.json').open('x', encoding='utf-8') as stream:
        json.dump({'kernel': job['kernel']}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Kaggle rejects >~1 MB notebook bodies, so the immutable inputs travel through the private
    # dataset repo and the kernel verifies their SHA-256 before use.
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    remote_name = f'v2/compute/{path.name}/inputs.json'
    api.upload_file(path_or_fileobj=str(path / 'inputs.json'), path_in_repo=remote_name,
                    repo_id=REPO, repo_type='dataset')
    fetched = hf_hub_download(repo_id=REPO, repo_type='dataset', filename=remote_name, token=token,
                              force_download=True)
    if hashes.digest(Path(fetched).read_bytes()) != job['input_sha256']:
        raise RuntimeError('Uploaded remote compute inputs differ from the reviewed payload')
    with tempfile.TemporaryDirectory(prefix='aya-v2-compute-', dir=TEMP) as directory:
        transport = Path(directory)
        notebook = nbformat.read(path / 'run.ipynb', as_version=4)
        notebook.cells[1].source = notebook.cells[1].source.replace(
            'from kaggle_secrets import UserSecretsClient\nos.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")',
            'os.environ["HF_TOKEN"] = ' + repr(token))
        try:
            nbformat.write(notebook, transport / 'run.ipynb')
            (transport / 'kernel-metadata.json').write_bytes((path / 'kernel-metadata.json').read_bytes())
            for attempt in range(3):
                result = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(transport),
                                         '--accelerator', 'NvidiaTeslaT4', '--timeout', '21600'],
                                         capture_output=True, text=True, timeout=180, encoding='utf-8', errors='replace')
                log = (result.stdout + result.stderr).replace(token, '[redacted]')
                (path / 'submission.log').write_text(log, encoding='utf-8')
                print(log)
                if result.returncode == 0:
                    return
                # A push aborted by the network only retries when Kaggle confirms nothing was created.
                if attempt == 2 or not push_missing_remote(job['kernel']):
                    result.check_returncode()
                time.sleep(20)
        finally:
            if (transport / 'run.ipynb').exists():
                (transport / 'run.ipynb').unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--kind', choices=list(SOURCES), required=True)
    prepare_parser.add_argument('--inputs', type=Path)
    for name in ('verify', 'submit'):
        sub.add_parser(name).add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.kind == 'selfaware':
            payload = selfaware_inputs(ROOT)
        else:
            if args.inputs is None:
                parser.error('FActScore retrieval requires --inputs')
            payload = json.loads(args.inputs.read_bytes())
        print(json.dumps({'prepared_directory': str(prepare(ROOT, args.kind, payload))}))
    elif args.command == 'verify':
        print(json.dumps(verify(args.directory)))
    else:
        submit(args.directory)
