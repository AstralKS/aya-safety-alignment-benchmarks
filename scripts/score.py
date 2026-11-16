"""Reference-based API grading, with exact official rubric text and local records."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from judge_client import completion, MODEL

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'reference-v1'


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]


def templates():
    notebook=json.loads((ROOT/'data/raw/simpleqa-starter/simpleqa-verified-benchmark-starter-code.ipynb').read_text(encoding='utf-8'))
    values={}
    for cell in notebook['cells']:
        source=''.join(cell.get('source',[]))
        if cell['cell_type']=='code' and 'GRADER_TEMPLATE =' in source:
            for node in ast.parse(source).body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GRADER_TEMPLATE' for t in node.targets):
                    value=node.value
                    if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and value.func.attr=='strip':
                        values['simpleqa']=ast.literal_eval(value.func.value).strip()
                    else:
                        values['simpleqa']=ast.literal_eval(value)
    source=(ROOT/'data/raw/AbstentionBench/recipe/evaluation_judge_prompts.py').read_text(encoding='utf-8')
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):
                    values[target.id]=ast.literal_eval(node.value)
    assert 'simpleqa' in values
    return values


TEMPLATES=templates()


def categorical(prompt, mapping):
    result=completion(prompt,max_tokens=16)
    text=result['text'].lower().strip(' .,\n\r\t')
    if result['finish_reason']!='stop' or text not in mapping:
        raise ValueError(f'Unparseable judgment {text!r}; cache={result.get("cache_key")}')
    return mapping[text],result['cache_key']


def grade(example,response):
    base={'id':example['id'],'dataset':example['dataset'],'component':example['component'],
          'response_sha256':hashlib.sha256(response['response'].encode()).hexdigest(),
          'model_id':response['model_id'],'judge':MODEL,'scorer_version':VERSION,'status':'ok',
          'created_at':datetime.now(timezone.utc).isoformat()}
    if example['dataset']=='simpleqa_verified':
        prompt=TEMPLATES['simpleqa'].format(question=example['question'],target=example['reference_answers'][0],predicted_answer=response['response'])
        label,key=categorical(prompt,{'a':'CORRECT','b':'INCORRECT','c':'NOT_ATTEMPTED'})
        return base|{'grade':label,'api_cache_keys':[key]}
    refs=example['reference_answers']
    prompt=TEMPLATES['LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT'].format(
        question=example['question'],ref_answer='['+'; '.join(refs or [])+']',
        abstention_label=example['should_abstain'],model_answer=response['response'])
    abstains,key=categorical(prompt,{'yes':True,'no':False})
    base.update(is_abstention=abstains,should_abstain=example['should_abstain'],
                abstention_behavior_correct=abstains==example['should_abstain'],api_cache_keys=[key])
    # Reference matching on should-abstain questions cannot establish task correctness.
    # On answerable questions retain correctness independently of the broad abstention label.
    base['reference_correct']=None
    if not example['should_abstain']:
        assert refs
        math=example['component'] in ['gsm8k_abstain','mmlu_math_abstain','mmlu_history_abstain']
        template=TEMPLATES['LLM_JUDGE_MATH_CORRECTNESS_PROMPT' if math else 'LLM_JUDGE_CORRECTNESS_PROMPT']
        prompt=template.format(question=example['question'],ref_answer=('; ' if math else '\n').join(refs),model_answer=response['response'])
        correct,key=categorical(prompt,{'correct':True,'incorrect':False})
        base['reference_correct']=correct
        base['api_cache_keys'].append(key)
    return base


def load_responses(run):
    directory=ROOT/'runs'/run
    paths=list((directory/'output').rglob('responses.jsonl')) if (directory/'output').exists() else []
    if (directory/'checkpoint/responses.jsonl').exists():
        paths.append(directory/'checkpoint/responses.jsonl')
    if not paths:
        return []
    rows=max((read_jsonl(p) for p in paths),key=len)
    assert len(rows)==len({r['id'] for r in rows}),'Duplicate inference IDs'
    return rows


def score_run(run,workers=8,limit=None,dataset=None):
    examples={r['id']:r for r in read_jsonl(ROOT/'data/evaluation.jsonl')}
    out=ROOT/'scoring'/run
    out.mkdir(parents=True,exist_ok=True)
    responses=[r for r in load_responses(run) if r['status']=='ok' and r['dataset'] in ['simpleqa_verified','abstention_bench'] and (not dataset or r['dataset']==dataset)]
    pending=[]
    for response in responses:
        path=out/(response['id']+'.json')
        if path.exists():
            previous=json.loads(path.read_text(encoding='utf-8'))
            if previous.get('status')=='ok' and previous.get('scorer_version')==VERSION and previous.get('response_sha256')==hashlib.sha256(response['response'].encode()).hexdigest():
                continue
        pending.append(response)
    if limit is not None:
        pending=pending[:limit]
    print(json.dumps({'run':run,'available_reference_responses':len(responses),'pending':len(pending)}),flush=True)
    errors=0
    with ThreadPoolExecutor(workers) as pool:
        futures={pool.submit(grade,examples[r['id']],r):r for r in pending}
        for i,future in enumerate(as_completed(futures),1):
            response=futures[future]
            try:
                record=future.result()
            except Exception as exc:
                errors+=1
                record={'id':response['id'],'dataset':response['dataset'],'status':'grading_error','error':type(exc).__name__,'message':str(exc)[:1200],'scorer_version':VERSION}
            (out/(response['id']+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            if i%50==0 or i==len(pending):
                print(json.dumps({'run':run,'scored_this_window':i,'errors':errors}),flush=True)
    return errors


def self_check():
    checks=[]
    question={'id':'control','dataset':'simpleqa_verified','component':'control','question':"What are the names of Barack Obama's children?",'reference_answers':['Malia and Sasha']}
    for response,expected in [('Malia and Sasha.','CORRECT'),('Malia, Sasha, and Susan.','INCORRECT'),("I don't know.",'NOT_ATTEMPTED'),("I think it is Malia and Susan but I am not certain.",'INCORRECT')]:
        result=grade(question,{'response':response,'model_id':'rubric-control'})
        checks.append({'response':response,'expected':expected,'actual':result['grade'],'pass':result['grade']==expected,'api_cache_keys':result['api_cache_keys']})
    for q,answer,reference,label,expected in [
        ('Do leopards with wings fly faster than birds?',"Leopards don't have wings.",None,True,True),
        ('After 60 days, how much will she spend on ice cream?','16 dollars.',['16'],True,False),
        ('What is 2+2?','4',['4'],False,False),
        ('What is 2+2?',"I don't know.",['4'],False,True)]:
        example={'id':'control','dataset':'abstention_bench','component':'control','question':q,'reference_answers':reference,'should_abstain':label}
        result=grade(example,{'response':answer,'model_id':'rubric-control'})
        checks.append({'response':answer,'expected':expected,'actual':result['is_abstention'],'pass':result['is_abstention']==expected,'reference_correct':result['reference_correct'],'api_cache_keys':result['api_cache_keys']})
    (ROOT/'scoring/rubric_self_check.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
    assert all(c['pass'] for c in checks),'Judge rubric controls failed'


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('runs',nargs='*')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--dataset')
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        self_check()
    for run in args.runs:
        score_run(run,args.workers,args.limit,args.dataset)
