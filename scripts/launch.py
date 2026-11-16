"""Build a private Kaggle notebook; keep exported notebooks credential-free."""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import shutil
import subprocess
import tempfile

from huggingface_hub import HfApi, get_token
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['tiny', '8b'], required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--resume', action='store_true',help='Seed a fresh runtime with downloaded successful responses from this run')
    args = parser.parse_args()
    model_id = 'CohereLabs/tiny-aya-global' if args.model=='tiny' else 'CohereLabs/aya-expanse-8b'
    token = get_token()
    api = HfApi(token=token)
    revision = api.model_info(model_id).sha
    rows = [json.loads(s) for s in (ROOT/'data/evaluation.jsonl').read_text(encoding='utf-8').split('\n') if s.strip()]
    if args.pilot:
        pilot = [r for r in rows if r['dataset']=='simpleqa_verified'][:6]
        for component in ['bbq','gsm8k_abstain','mmlu_math_abstain']:
            for label in [False,True]:
                pilot.append(next(r for r in rows if r['dataset']=='abstention_bench' and r['component']==component and r['should_abstain'] is label))
        for component in ['longfact_augmented_topics','longfact_augmented_citations','longfact_augmented_biography','longfact_augmented_legal']:
            pilot.append(next(r for r in rows if r['dataset']=='longfact_pp' and r['component']==component))
        rows = pilot
    # References, desired abstention labels and annotation notes never enter the inference notebook.
    examples = [{k:r[k] for k in ['id','dataset','component','question','system_prompt','max_new_tokens']} for r in rows]
    run_name = f"{args.model}-{'pilot' if args.pilot else 'baseline'}"
    slug = f'aya-{run_name}-20261111'
    run_dir = ROOT/'runs'/run_name
    resume_records=[]
    previous_launch=json.loads((run_dir/'launch.json').read_text(encoding='utf-8')) if (run_dir/'launch.json').exists() else None
    if args.resume:
        if not previous_launch:
            raise RuntimeError('Cannot resume a run without saved launch metadata')
        status=subprocess.run(['kaggle','kernels','status',previous_launch['kernel']],capture_output=True,text=True,timeout=60)
        status_text=(status.stdout+status.stderr).lower()
        if status.returncode or not any(s in status_text for s in ['kernelworkerstatus.complete','kernelworkerstatus.error','kernelworkerstatus.cancel']):
            raise RuntimeError('Resume requires a confirmed terminal Kaggle run; download its latest checkpoint first')
        paths=list((run_dir/'output').rglob('responses.jsonl')) if (run_dir/'output').exists() else []
        if (run_dir/'checkpoint/responses.jsonl').exists():
            paths.append(run_dir/'checkpoint/responses.jsonl')
        if paths:
            resume_records=max(([json.loads(line) for line in p.read_text(encoding='utf-8').split('\n') if line.strip()] for p in paths),key=len)
            resume_records=[r for r in resume_records if r.get('status')=='ok']
        revision=previous_launch['config']['revision']
        if set(previous_launch['sample_ids'])!={r['id'] for r in examples}:
            raise ValueError('Resume sample manifest differs from original launch')
    elif previous_launch and not args.pilot:
        raise RuntimeError('Baseline already submitted; use --resume after a terminal run and checkpoint download')
    if (run_dir/'launch.json').exists():
        archive=run_dir/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archive.mkdir(parents=True,exist_ok=True)
        for name in ['launch.json','launch.log','output','checkpoint']:
            source=run_dir/name
            if source.is_dir():
                shutil.copytree(source,archive/name)
            elif source.exists():
                shutil.copy2(source,archive/name)
        old_notebook=ROOT/'notebooks'/f'{run_name}.ipynb'
        if old_notebook.exists():
            shutil.copy2(old_notebook,archive/'submitted-notebook-sanitized.ipynb')
    checkpoint_repo = 'ASTRALK/aya-baselines-20261111'
    try:
        api.create_repo(checkpoint_repo, repo_type='dataset', private=True, exist_ok=True)
    except Exception as exc:
        print('HF checkpoint repository unavailable:',type(exc).__name__)
        checkpoint_repo = None
    data_bytes = json.dumps(examples,ensure_ascii=False).encode()
    config = {'model_id':model_id,'revision':revision,'run_name':run_name,'checkpoint_repo':checkpoint_repo,'input_sha256':hashlib.sha256(data_bytes).hexdigest()}
    if args.resume and previous_launch['config']['input_sha256']!=config['input_sha256']:
        raise ValueError('Resume question/instruction payload differs from original launch')
    config['resumed_records']=len(resume_records)
    encoded = base64.b64encode(gzip.compress(data_bytes)).decode()
    runner = (ROOT/'scripts/kaggle_runner.py').read_text(encoding='utf-8').split("if __name__ == '__main__':")[0]
    # Keep __future__ import first by placing authentication in its own notebook cell.
    bootstrap = 'import os\n# Enable Kaggle secret HF_TOKEN when running this saved notebook manually.\nfrom kaggle_secrets import UserSecretsClient\nos.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")\n'
    resume_encoded=base64.b64encode(gzip.compress(json.dumps(resume_records,ensure_ascii=False).encode())).decode()
    main_cell = runner + f'\nrun({config!r}, json.loads(gzip.decompress(base64.b64decode({encoded!r}))), json.loads(gzip.decompress(base64.b64decode({resume_encoded!r}))))\n'
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(f'# Aya factuality baseline: {run_name}\nPrivate remote execution. Full protocol is in the local baselines folder.'), nbformat.v4.new_code_cell(bootstrap), nbformat.v4.new_code_cell(main_cell)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    nbformat.validate(notebook)
    saved = ROOT/'notebooks'/f'{run_name}.ipynb'
    saved.parent.mkdir(exist_ok=True)
    nbformat.write(notebook,saved)
    metadata = {'id':f'astralks/{slug}','title':slug,'code_file':'run.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    temp = Path(tempfile.mkdtemp(prefix='aya-kaggle-',dir=Path.home()/'AppData/Local/Temp/opencode'))
    # User explicitly authorized injecting this credential into private unattended submissions.
    notebook.cells[1].source = 'import os\nos.environ["HF_TOKEN"] = '+repr(token)+'\n'
    nbformat.write(notebook,temp/'run.ipynb')
    (temp/'kernel-metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    run_dir = ROOT/'runs'/run_name
    run_dir.mkdir(parents=True,exist_ok=True)
    (run_dir/'launch.json').write_text(json.dumps({'kernel':metadata['id'],'config':config,'sample_ids':[r['id'] for r in examples],'saved_notebook':str(saved.relative_to(ROOT))},indent=2),encoding='utf-8')
    result = subprocess.run(['kaggle','kernels','push','-p',str(temp),'--accelerator','NvidiaTeslaT4','--timeout','21600'],capture_output=True,text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr.replace(token,'[redacted]'))
    (run_dir/'launch.log').write_text((result.stdout+result.stderr).replace(token,'[redacted]'),encoding='utf-8')
    # The credential-bearing local transport notebook is not a deliverable.
    (temp/'run.ipynb').unlink()
    result.check_returncode()


if __name__=='__main__':
    main()
