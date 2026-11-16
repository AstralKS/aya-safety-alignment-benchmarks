"""Small cached HF router client with an explicit shared spending ceiling."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import requests
from huggingface_hub import get_token
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'scoring'/'api_cache'
CACHE.mkdir(parents=True,exist_ok=True)
MODEL = 'Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra'
PRICES = {MODEL:(0.09,0.55),'deepseek-ai/DeepSeek-V3.2:deepinfra':(0.26,0.38),
          'openai/gpt-oss-120b:deepinfra':(0.037,0.17)}


def completion(prompt, *, max_tokens=128, json_output=False, model=MODEL):
    input_price,output_price=PRICES[model]
    payload = {'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':max_tokens}
    if json_output:
        payload['response_format']={'type':'json_object'}
    key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    cache=CACHE/(key+'.json')
    with FileLock(str(cache)+'.lock',timeout=600):
        if cache.exists():
            return json.loads(cache.read_text(encoding='utf-8'))
        token=get_token()
        for attempt in range(5):
            # Conservative reservation includes each retry; no reliance on missing provider billing fields.
            # UTF-8 bytes upper-bound input tokens. Add message overhead and a 2x
            # margin over the selected model's discovered DeepInfra token prices.
            upper=((len(prompt.encode('utf-8'))+256)*input_price*2+max_tokens*output_price*2)/1_000_000
            ledger_path=ROOT/'scoring'/'budget.json'
            with FileLock(str(ledger_path)+'.lock',timeout=30):
                ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'reserved_upper_usd':0.0,'reported_cost_usd':0.00000483,'requests':0,'ceiling_usd':18.0}
                if ledger['reserved_upper_usd']+upper>ledger['ceiling_usd']:
                    raise RuntimeError('Conservative HF evaluation spending ceiling reached')
                ledger['reserved_upper_usd']+=upper
                ledger['requests']+=1
                ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
            started=time.monotonic()
            try:
                response=requests.post('https://router.huggingface.co/v1/chat/completions',headers={'Authorization':'Bearer '+token},json=payload,timeout=(20,180))
                if response.status_code in [429,500,502,503,504]:
                    raise requests.HTTPError(f'Transient HTTP {response.status_code}')
                response.raise_for_status()
                body=response.json()
                choice=body['choices'][0]
                result={'cache_key':key,'request':payload,'response':body,'text':choice['message'].get('content',''),'finish_reason':choice.get('finish_reason'),'elapsed_seconds':time.monotonic()-started,'created_at':datetime.now(timezone.utc).isoformat()}
                cache.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                with FileLock(str(ledger_path)+'.lock',timeout=30):
                    ledger=json.loads(ledger_path.read_text())
                    usage=body.get('usage',{})
                    reported=usage.get('estimated_cost',0) or 0
                    ledger['reported_cost_usd']+=reported
                    # Settle completed requests at a conservative token-count cost,
                    # retaining full reservations for failed/unknown-charge attempts.
                    if isinstance(usage.get('prompt_tokens'),int) and isinstance(usage.get('completion_tokens'),int):
                        settled=max(reported,(usage['prompt_tokens']*input_price*2+usage['completion_tokens']*output_price*2)/1_000_000)
                        ledger['reserved_upper_usd']+=settled-upper
                        ledger['settled_requests']=ledger.get('settled_requests',0)+1
                    ledger_path.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
                return result
            except requests.RequestException as exc:
                error={'cache_key':key,'attempt':attempt+1,'error':type(exc).__name__,'created_at':datetime.now(timezone.utc).isoformat()}
                if getattr(exc,'response',None) is not None:
                    error.update(status=exc.response.status_code,body=exc.response.text[:2000].replace(token,'[redacted]'))
                (CACHE/f'{key}.attempt-{attempt+1}.error.json').write_text(json.dumps(error,indent=2),encoding='utf-8')
                if attempt==4:
                    raise
                threading.Event().wait(min(60,3*2**attempt))
        raise RuntimeError('Unreachable')


def object_completion(prompt, **kwargs):
    result=completion(prompt,json_output=True,**kwargs)
    if result['finish_reason']!='stop':
        raise ValueError('Judge output incomplete: '+str(result['finish_reason']))
    text=result['text'].strip()
    if text.startswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(text)
