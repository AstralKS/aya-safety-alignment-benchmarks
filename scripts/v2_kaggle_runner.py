"""Pinned v2 inference. Model imports and computation are Kaggle-only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

PINS = {
    'CohereLabs/tiny-aya-global': 'af89d219b53ed9b13b8a4645f9c8028973510324',
    'CohereLabs/aya-expanse-8b': '5062468bf9bc0c6035fd64e06274333ec127d980',
}
FIELDS = {'id', 'dataset', 'component', 'question', 'system_prompt', 'max_new_tokens'}
CAPS = {'hallulens_precisewikiqa': 256, 'hallulens_nonexistentrefusal': 512,
        'selfaware': 256, 'factscore_bios': 1536}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload_hash(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':')).encode('utf-8'))


def validate_examples(examples):
    if not isinstance(examples, list) or not examples:
        raise ValueError('Empty generation payload')
    ids = set()
    for row in examples:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError('Generation fields differ from reference-free allowlist')
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
            raise ValueError('Empty or duplicate generation ID')
        ids.add(row['id'])
        if row['dataset'] not in CAPS or type(row['max_new_tokens']) is not int:
            raise ValueError('Unexpected dataset/token limit')
        if row['max_new_tokens'] != CAPS[row['dataset']]:
            raise ValueError('Changed generation token contract')
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError('Empty question')
        if row['system_prompt'] != '':
            raise ValueError('New system prompt must remain empty')
        if row['component'] is not None and not isinstance(row['component'], str):
            raise ValueError('Invalid component')


def validate_config(config, examples):
    validate_examples(examples)
    if config.get('model_id') not in PINS or config.get('revision') != PINS[config['model_id']]:
        raise ValueError('Aya revision differs from approved pin')
    if config.get('input_sha256') != payload_hash(examples):
        raise ValueError('Generation payload hash mismatch')
    for key in ('chat_template_sha256', 'selected_template_sha256'):
        value = config.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing native template identity')
    if not isinstance(config.get('run_name'), str) or not config['run_name']:
        raise ValueError('Missing run identity')


def record_identity(config, example, text):
    return {**{k: example[k] for k in ('id', 'dataset', 'component', 'max_new_tokens')},
            'model_id': config['model_id'], 'revision': config['revision'],
            'input_sha256': config['input_sha256'],
            'chat_template_sha256': config['chat_template_sha256'],
            'selected_template_sha256': config['selected_template_sha256'],
            'example_sha256': payload_hash(example),
            'prompt_sha256': digest(text.encode('utf-8'))}


def validate_records(records, config, rendered):
    ids = set()
    for record in records:
        if not isinstance(record, dict) or record.get('id') not in rendered or record['id'] in ids:
            raise ValueError('Invalid or duplicate resume ID')
        example, text, count = rendered[record['id']]
        expected = record_identity(config, example, text)
        if any(record.get(k) != value for k, value in expected.items()):
            raise ValueError('Resume model/payload/prompt provenance mismatch')
        if record.get('status') != 'ok' or not isinstance(record.get('response'), str) or not record['response'].strip():
            raise ValueError('Resume contains unsuccessful/empty answers')
        if record.get('finish_reason') not in ('eos', 'length'):
            raise ValueError('Invalid resume finish reason')
        if type(record.get('input_tokens')) is not int or record['input_tokens'] != count:
            raise ValueError('Resume input token count mismatch')
        output = record.get('output_tokens')
        if type(output) is not int or not 1 <= output <= example['max_new_tokens']:
            raise ValueError('Resume output token count mismatch')
        ids.add(record['id'])
    return ids


def completion_state(records, total):
    successful = sum(record.get('status') == 'ok' for record in records)
    errors = len(records) - successful
    stage = ('completed_with_errors' if errors else 'completed') if len(records) == total else 'incomplete'
    return {'stage': stage, 'successful': successful, 'errors': errors,
            'accounted': len(records), 'total': total}


def run(config, examples, resume_records=None):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Remote-only runner: execute on a Kaggle GPU notebook')
    validate_config(config, examples)
    os.environ.update(HF_HOME='/kaggle/temp/huggingface', TOKENIZERS_PARALLELISM='false',
                      USE_TF='0', USE_FLAX='0', HF_HUB_DISABLE_PROGRESS_BARS='1',
                      HF_HUB_DOWNLOAD_TIMEOUT='120', HF_HUB_ETAG_TIMEOUT='60',
                      PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    output = Path('/kaggle/working/baseline')
    output.mkdir(exist_ok=True)

    def now():
        return datetime.now(timezone.utc).isoformat()

    def write(name, value):
        path = output / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    write('progress.json', {'stage': 'installing', 'config': config, 'created_at': now()})
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.6',
                    'accelerate==1.12.0', 'huggingface-hub==0.36.0', 'sentencepiece==0.2.1'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'librosa', 'torchcodec'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
    from huggingface_hub import HfApi
    from importlib.metadata import version

    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU unavailable; no local/CPU fallback')
    set_seed(20261111)
    token = os.environ.get('HF_TOKEN')
    if not token:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
    api = HfApi(token=token)
    last_checkpoint = 0.0

    def checkpoint(force=False):
        nonlocal last_checkpoint
        if not config.get('checkpoint_repo') or (not force and time.monotonic() - last_checkpoint < 300):
            return
        last_checkpoint = time.monotonic()
        try:
            api.upload_folder(repo_id=config['checkpoint_repo'], repo_type='dataset',
                              folder_path=str(output), path_in_repo='v2/' + config['run_name'],
                              commit_message='Save v2 generation progress',
                              allow_patterns=['*.json', '*.jsonl', '*.txt'])
        except Exception as exc:
            print('Checkpoint deferred:', type(exc).__name__, flush=True)

    model_id, revision = config['model_id'], config['revision']
    started = time.monotonic()
    gpu_info = [{'index': i, 'name': torch.cuda.get_device_name(i),
                 'total_gib': torch.cuda.get_device_properties(i).total_memory / 2**30}
                for i in range(torch.cuda.device_count())]
    write('progress.json', {'stage': 'loading_model', 'updated_at': now(), 'gpu_info': gpu_info})
    checkpoint(True)
    model_config = AutoConfig.from_pretrained(model_id, revision=revision, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    if digest(str(tokenizer.chat_template).encode()) != config['chat_template_sha256']:
        raise ValueError('Pinned native template metadata changed')
    template = tokenizer.get_chat_template()
    if digest(template.encode()) != config['selected_template_sha256']:
        raise ValueError('Selected native default template changed')
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id != config['pad_token_id']:
        raise ValueError('Pinned pad token differs')
    max_memory = {i: f'{int(torch.cuda.mem_get_info(i)[0] / 2**30) - 3}GiB'
                  for i in range(torch.cuda.device_count())}
    placement = 'balanced' if model_id == 'CohereLabs/aya-expanse-8b' else {'': 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, token=token,
                config=model_config, dtype=torch.float16, device_map=placement,
                max_memory=max_memory, low_cpu_mem_usage=True, attn_implementation='sdpa').eval()
    if getattr(model.config, '_commit_hash', revision) != revision:
        raise ValueError('Resolved model revision differs')
    if any(str(d) in ('cpu', 'disk') for d in model.hf_device_map.values()):
        raise RuntimeError('CPU/disk offload is forbidden')
    if model.generation_config.eos_token_id != config['eos_token_id']:
        raise ValueError('Pinned EOS token differs')
    if any(parameter.dtype != torch.float16 for parameter in model.parameters()):
        raise ValueError('Weights are not all FP16')
    device = model.get_input_embeddings().weight.device
    manifest = {'config': config, 'model_id': model_id, 'resolved_revision': revision,
                'weight_dtype': 'float16', 'quantization': None, 'gpu_info': gpu_info,
                'device_map': {k: str(v) for k, v in model.hf_device_map.items()},
                'python': sys.version, 'torch': torch.__version__,
                'transformers': version('transformers'), 'accelerate': version('accelerate'),
                'model_load_seconds': time.monotonic() - started, 'sample_count': len(examples),
                'chat_template_sha256': config['chat_template_sha256'],
                'selected_template_sha256': config['selected_template_sha256'],
                'started_at': now(), 'resumed_records': len(resume_records or [])}
    (output / 'chat_template.txt').write_text(str(tokenizer.chat_template), encoding='utf-8')
    write('manifest.json', manifest)
    rendered = {}
    for example in examples:
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': example['question']}],
                                             tokenize=False, add_generation_prompt=True)
        rendered[example['id']] = (example, text, len(tokenizer.encode(text, add_special_tokens=False)))
    path = output / 'responses.jsonl'
    records = []
    if path.exists():
        if resume_records:
            raise RuntimeError('Refusing to merge runtime and supplied resume records')
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    elif resume_records:
        records = list(resume_records)
    completed = validate_records(records, config, rendered)
    if records and not path.exists():
        with path.open('x', encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    pending = sorted((value for key, value in rendered.items() if key not in completed),
                     key=lambda x: (x[0]['max_new_tokens'], x[2], x[0]['id']))
    eos = config['eos_token_id']
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    context_limit = int(model.config.max_position_embeddings)
    run_start = time.monotonic()

    def save(record):
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        records.append(record)

    def infer(batch):
        if not batch:
            return
        limit = batch[0][0]['max_new_tokens']
        inputs = tokenizer([row[1] for row in batch], padding=True, return_tensors='pt',
                           add_special_tokens=False, return_token_type_ids=False).to(device)
        width = inputs['input_ids'].shape[1]
        t = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                       num_beams=1, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        elapsed = time.monotonic() - t
        output_ids = generated[:, width:].tolist()
        if len(output_ids) != len(batch):
            raise RuntimeError('Generation output count mismatch')
        # Decode every row before appending so an OOM cannot duplicate a partly saved batch.
        decoded = []
        for (example, text, count), ids in zip(batch, output_ids):
            stop = next((i for i, value in enumerate(ids) if value in eos_ids), None)
            kept = ids if stop is None else ids[:stop + 1]
            response = tokenizer.decode(kept, skip_special_tokens=True).strip()
            decoded.append({**record_identity(config, example, text), 'response': response,
                            'input_tokens': count, 'output_tokens': len(kept),
                            'finish_reason': 'eos' if stop is not None else 'length',
                            'status': 'ok' if response else 'empty_response',
                            'batch_size': len(batch), 'batch_seconds': elapsed, 'completed_at': now()})
        del inputs, generated
        for record in decoded:
            save(record)

    cursor = 0
    while cursor < len(pending):
        limit = pending[cursor][0]['max_new_tokens']
        size = (4 if limit <= 512 else 1) if model_id == 'CohereLabs/aya-expanse-8b' else (8 if limit <= 512 else 2)
        batch = []
        while cursor < len(pending) and len(batch) < size and pending[cursor][0]['max_new_tokens'] == limit:
            example, text, count = pending[cursor]
            cursor += 1
            if count + limit > context_limit:
                save({**record_identity(config, example, text), 'status': 'context_overflow',
                      'input_tokens': count, 'completed_at': now()})
            else:
                batch.append((example, text, count))
        try:
            infer(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for item in batch:
                infer([item])
        state = {**completion_state(records, len(examples)), 'stage': 'generating',
                 'elapsed_seconds': time.monotonic() - run_start, 'updated_at': now()}
        write('progress.json', state)
        print(json.dumps(state), flush=True)
        checkpoint()
    state = completion_state(records, len(examples))
    manifest.update(**state, finished_at=now(), generation_seconds=time.monotonic() - run_start,
                    peak_gpu_allocated_gib=[torch.cuda.max_memory_allocated(i) / 2**30
                                            for i in range(torch.cuda.device_count())])
    write('manifest.json', manifest)
    write('progress.json', {**state, 'updated_at': now()})
    checkpoint(True)
    print('V2_GENERATION_' + state['stage'].upper(), json.dumps(state), flush=True)
    if state['stage'] != 'completed':
        raise RuntimeError('Generation finished with errors; inspect saved records')


if __name__ == '__main__':
    raise SystemExit('Package with launch_v2.py; model execution is remote only')
