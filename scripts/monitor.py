"""One monitoring window. Automatic shell completion returns control to the assistant."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time

import requests
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[1]


def inspect(run_names):
    token = get_token()
    summaries = []
    for name in run_names:
        directory = ROOT/'runs'/name
        launch = json.loads((directory/'launch.json').read_text(encoding='utf-8'))
        result = subprocess.run(['kaggle','kernels','status',launch['kernel']],capture_output=True,text=True,timeout=60)
        status = (result.stdout+result.stderr).replace(token,'[redacted]').strip()
        summary = {'run':name,'checked_at':datetime.now(timezone.utc).isoformat(),'kaggle_status':status}
        repo = launch['config'].get('checkpoint_repo')
        if repo:
            for filename in ['progress.json','manifest.json','responses.jsonl','chat_template.txt']:
                url = f'https://huggingface.co/datasets/{repo}/resolve/main/{name}/{filename}'
                try:
                    r = requests.get(url,headers={'Authorization':'Bearer '+token},timeout=(15,60))
                    if r.status_code==200:
                        target = directory/'checkpoint'/filename
                        target.parent.mkdir(exist_ok=True)
                        target.write_bytes(r.content)
                        if filename=='progress.json':
                            summary['progress']=r.json()
                    elif r.status_code not in [404]:
                        summary['checkpoint_http_status']=r.status_code
                except requests.RequestException as exc:
                    summary['checkpoint_error']=type(exc).__name__
        terminal = any(word in status.lower() for word in ['complete','error','cancel','failed'])
        if terminal:
            out = directory/'output'
            out.mkdir(exist_ok=True)
            completed = subprocess.run(['kaggle','kernels','output',launch['kernel'],'-p',str(out),'--force'],capture_output=True,text=True,timeout=120)
            summary['output_download']=(completed.stdout+completed.stderr).replace(token,'[redacted]').strip()
            for p in out.rglob('*'):
                if p.is_file() and p.suffix in ['.txt','.json','.jsonl','.log','.ipynb','.py','.html']:
                    try:
                        s=p.read_text(encoding='utf-8')
                        if token in s:
                            p.write_text(s.replace(token,'[HF_TOKEN_REDACTED]'),encoding='utf-8')
                    except UnicodeError:
                        pass
        summaries.append(summary)
    with (ROOT/'runs'/'monitor.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(summaries,ensure_ascii=False)+'\n')
    print(json.dumps(summaries,ensure_ascii=True,indent=2),flush=True)
    return summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('runs',nargs='+')
    parser.add_argument('--minutes',type=float,default=0)
    args=parser.parse_args()
    if args.minutes:
        print(f'Monitor scheduled for {args.minutes:g} minutes; remote jobs continue independently.',flush=True)
        threading.Event().wait(args.minutes*60)
    for attempt in range(5):
        try:
            inspect(args.runs)
            print('CONTINUE_BASELINE_WORK: review results, resolve errors, score completed responses, or schedule the next 15-minute check.',flush=True)
            return
        except (requests.RequestException,subprocess.TimeoutExpired) as exc:
            print('Transient monitor error:',type(exc).__name__,flush=True)
            if attempt==4:
                raise
            threading.Event().wait(min(60,2**attempt*5))


if __name__=='__main__':
    main()
