"""Shared inference implementation. Runs only inside Kaggle; no local model loading."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def run(config: dict, examples: list[dict], resume_records: list[dict] | None = None):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Remote-only runner: execute on a Kaggle GPU notebook.')
    os.environ['HF_HOME'] = '/kaggle/temp/huggingface'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    os.environ['HF_HUB_ETAG_TIMEOUT'] = '60'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    output = Path('/kaggle/working/baseline')
    output.mkdir(exist_ok=True)
    if resume_records:
        expected_ids={r['id'] for r in examples}
        if len({r['id'] for r in resume_records})!=len(resume_records):
            raise ValueError('Duplicate resume IDs')
        if any(r['id'] not in expected_ids or r.get('model_id')!=config['model_id'] or r.get('status')!='ok' for r in resume_records):
            raise ValueError('Resume records do not match this run')
        resume_path=output/'responses.jsonl'
        if resume_path.exists():
            raise RuntimeError('Refusing to overwrite responses already present in this runtime')
        resume_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in resume_records),encoding='utf-8')

    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    def now():
        return datetime.now(timezone.utc).isoformat()

    write('progress.json', {'stage': 'installing', 'created_at': now(), 'config': config})
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.6', 'accelerate==1.12.0', 'huggingface-hub==0.36.0', 'sentencepiece==0.2.1'], check=True)
    # These unrelated optional packages can have incompatible binary builds in Kaggle's base image.
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'librosa', 'torchcodec'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
    from huggingface_hub import HfApi
    from importlib.metadata import version

    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU unavailable; CPU inference is disabled.')
    set_seed(20261111)
    token = os.environ.get('HF_TOKEN')
    if not token:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
    model_id = config['model_id']
    revision = config['revision']
    api = HfApi(token=token)
    last_checkpoint = 0.0

    def checkpoint(force=False):
        nonlocal last_checkpoint
        if not config.get('checkpoint_repo') or (not force and time.monotonic()-last_checkpoint < 600):
            return
        last_checkpoint = time.monotonic()
        try:
            api.upload_folder(repo_id=config['checkpoint_repo'], repo_type='dataset', folder_path=str(output), path_in_repo=config['run_name'], commit_message='Save baseline progress', allow_patterns=['*.json', '*.jsonl', '*.txt'])
            print('Remote checkpoint saved', flush=True)
        except Exception as exc:
            print('Remote checkpoint deferred:', type(exc).__name__, flush=True)

    gpu_info = [{'index': i, 'name': torch.cuda.get_device_name(i), 'total_gib': torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(torch.cuda.device_count())]
    write('progress.json', {'stage': 'loading_model', 'updated_at': now(), 'model_id': model_id, 'gpu_info': gpu_info})
    checkpoint(True)
    started = time.monotonic()
    model_config = AutoConfig.from_pretrained(model_id, revision=revision, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    if not tokenizer.chat_template:
        raise RuntimeError('Official chat template missing')
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_memory = {i: f'{int(torch.cuda.mem_get_info(i)[0]/2**30)-3}GiB' for i in range(torch.cuda.device_count())}
    placement = 'balanced' if '8b' in model_id else {'': 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, token=token, config=model_config, dtype=torch.float16, device_map=placement, max_memory=max_memory, low_cpu_mem_usage=True, attn_implementation='sdpa').eval()
    device = model.get_input_embeddings().weight.device
    if any(str(d) in ['cpu', 'disk'] for d in getattr(model, 'hf_device_map', {}).values()):
        raise RuntimeError('Unexpected CPU/disk offload: request sufficient GPU memory')
    manifest = {'config': config, 'started_at': now(), 'model_id': model_id, 'resolved_revision': getattr(model.config, '_commit_hash', revision), 'weight_dtype': 'float16', 'quantization': None, 'gpu_info': gpu_info, 'device_map': {k: str(v) for k,v in getattr(model, 'hf_device_map', {}).items()}, 'python': sys.version, 'torch': torch.__version__, 'transformers': version('transformers'), 'accelerate': version('accelerate'), 'model_load_seconds': time.monotonic()-started, 'chat_template_sha256': hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(), 'eos_token_id': model.generation_config.eos_token_id, 'pad_token_id': tokenizer.pad_token_id, 'sample_count': len(examples), 'resumed_records': len(resume_records or [])}
    write('manifest.json', manifest)
    (output / 'chat_template.txt').write_text(str(tokenizer.chat_template), encoding='utf-8')
    write('progress.json', {'stage': 'model_loaded', 'updated_at': now(), 'model_id': model_id, 'gpu_info': gpu_info})
    checkpoint(True)
    print('Loaded', model_id, 'on', gpu_info, flush=True)
    path = output / 'responses.jsonl'
    completed = set()
    if path.exists():
        completed = {json.loads(s)['id'] for s in path.read_text(encoding='utf-8').split('\n') if s.strip()}
    rendered = []
    for example in examples:
        if example['id'] in completed:
            continue
        messages = []
        if example['system_prompt']:
            messages.append({'role': 'system', 'content': example['system_prompt']})
        messages.append({'role': 'user', 'content': example['question']})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(text, add_special_tokens=False))
        rendered.append((example, text, count))
    rendered.sort(key=lambda x: (x[0]['max_new_tokens'], x[2], x[0]['id']))
    context_limit = int(model.config.max_position_embeddings)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    run_start = time.monotonic()
    completed_count = len(completed)

    def save_record(record):
        nonlocal completed_count
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
            f.flush()
        completed_count += 1

    def infer_batch(batch):
        if not batch:
            return
        limit = batch[0][0]['max_new_tokens']
        for item, text, count in batch:
            if count+limit > context_limit:
                raise ValueError(f'Context overflow for {item["id"]}: {count}+{limit}>{context_limit}')
        inputs = tokenizer([x[1] for x in batch], padding=True, return_tensors='pt', add_special_tokens=False, return_token_type_ids=False).to(device)
        width = inputs['input_ids'].shape[1]
        t = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=limit, do_sample=False, num_beams=1, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        elapsed = time.monotonic()-t
        output_ids = generated[:, width:].tolist()
        for (item, text, count), ids in zip(batch, output_ids):
            stop = next((i for i, token_id in enumerate(ids) if token_id in eos_ids), None)
            kept = ids if stop is None else ids[:stop+1]
            response = tokenizer.decode(kept, skip_special_tokens=True).strip()
            record = {'id': item['id'], 'dataset': item['dataset'], 'component': item['component'], 'model_id': model_id, 'response': response, 'input_tokens': count, 'output_tokens': len(kept), 'finish_reason': 'eos' if stop is not None else 'length', 'max_new_tokens': limit, 'batch_size': len(batch), 'batch_seconds': elapsed, 'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'status': 'ok', 'completed_at': now()}
            save_record(record)
        del inputs, generated

    cursor = 0
    while cursor < len(rendered):
        limit = rendered[cursor][0]['max_new_tokens']
        batch_size = (4 if limit<=512 else 1) if '8b' in model_id else (8 if limit<=512 else 2)
        batch = []
        while cursor < len(rendered) and len(batch)<batch_size and rendered[cursor][0]['max_new_tokens']==limit:
            item = rendered[cursor]
            cursor += 1
            if item[2]+limit>context_limit:
                save_record({'id': item[0]['id'], 'dataset': item[0]['dataset'], 'component': item[0]['component'], 'model_id': model_id, 'status': 'context_overflow', 'input_tokens': item[2], 'max_new_tokens': limit})
            else:
                batch.append(item)
        try:
            infer_batch(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for item in batch:
                infer_batch([item])
        progress = {'stage': 'generating', 'completed': completed_count, 'total': len(examples), 'elapsed_seconds': time.monotonic()-run_start, 'updated_at': now(), 'last_ids': [x[0]['id'] for x in batch]}
        write('progress.json', progress)
        print(json.dumps(progress), flush=True)
        checkpoint()
    manifest.update(finished_at=now(), generation_seconds=time.monotonic()-run_start, completed_count=completed_count, peak_gpu_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())])
    write('manifest.json', manifest)
    write('progress.json', {'stage': 'completed', 'completed': completed_count, 'total': len(examples), 'updated_at': now()})
    checkpoint(True)
    print('BASELINE_GENERATION_COMPLETE', completed_count, flush=True)


if __name__ == '__main__':
    raise SystemExit('Use launch.py to package this runner for Kaggle.')
