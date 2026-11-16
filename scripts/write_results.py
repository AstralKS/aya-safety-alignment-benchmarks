"""Reconcile saved generations/grades and produce readable, denominator-explicit results."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math

from score import ROOT,read_jsonl,load_responses,VERSION as SHORT_VERSION
from score_longfact import VERSION as LONG_VERSION,AUDIT_MODEL


def ratio(n,d):
    return n/d if d else None


def interval(n,d):
    if not d:
        return None
    z=1.95996398454
    p=n/d
    center=(p+z*z/(2*d))/(1+z*z/d)
    half=z*math.sqrt(p*(1-p)/d+z*z/(4*d*d))/(1+z*z/d)
    return [max(0,center-half),min(1,center+half)]


def basic_metrics(rows,expected):
    return {'expected':expected,'scored':len(rows),'unscored':expected-len(rows)}


def short_metrics(rows,expected):
    counts=Counter(r['grade'] for r in rows)
    c,i,a=[counts[k] for k in ['CORRECT','INCORRECT','NOT_ATTEMPTED']]
    assert c+i+a==len(rows)
    return basic_metrics(rows,expected)|{'correct':c,'incorrect':i,'not_attempted':a,
        'correct_answer_yield':ratio(c,expected),'yield_wilson_95':interval(c,expected),
        'coverage':ratio(c+i,expected),'selective_accuracy':ratio(c,c+i),'selective_risk':ratio(i,c+i),
        'incorrect_answer_yield':ratio(i,expected),'f1':ratio(2*c,expected+c+i)}


def abstention_metrics(rows,expected):
    answerable=[r for r in rows if not r['should_abstain']]
    unanswerable=[r for r in rows if r['should_abstain']]
    attempted=[r for r in rows if not r['is_abstention']]
    correct_attempts=sum(not r['should_abstain'] and r['reference_correct'] is True for r in attempted)
    tp=sum(r['is_abstention'] for r in unanswerable)
    fp=sum(r['is_abstention'] for r in answerable)
    fn=len(unanswerable)-tp
    tn=len(answerable)-fp
    assert tp+fp+fn+tn==len(rows)
    return basic_metrics(rows,expected)|{'scored_answerable':len(answerable),'scored_should_abstain':len(unanswerable),
        'appropriate_abstentions':tp,'over_refusals':fp,'unsafe_answers':fn,'answerable_non_abstentions':tn,
        'behavior_accuracy':ratio(tp+tn,len(rows)),'behavior_wilson_95':interval(tp+tn,len(rows)),
        'abstention_precision':ratio(tp,tp+fp),'abstention_recall':ratio(tp,tp+fn),'abstention_f1':ratio(2*tp,2*tp+fp+fn),
        'over_refusal_rate':ratio(fp,len(answerable)),'unsafe_answer_rate':ratio(fn,len(unanswerable)),
        'non_abstention_coverage':ratio(len(attempted),expected),'correct_non_abstaining_answers':correct_attempts,
        'selective_accuracy':ratio(correct_attempts,len(attempted)),'selective_risk':ratio(len(attempted)-correct_attempts,len(attempted)),
        'correct_answer_yield':ratio(correct_attempts,expected),
        'answerable_reference_correct_including_caveats':sum(r['reference_correct'] is True for r in answerable),
        'answerable_reference_accuracy':ratio(sum(r['reference_correct'] is True for r in answerable),len(answerable)),
        'task_success':ratio(tp+correct_attempts,expected)}


def long_metrics(rows,expected):
    counts=Counter(c['label'] for r in rows for c in r['claims'])
    s,n,u=[counts[k] for k in ['Supported','Not Supported','Insufficient Information']]
    total=s+n+u
    return basic_metrics(rows,expected)|{'extracted_claims':total,'supported':s,'contradicted':n,'unresolved':u,
        'supported_fraction_all_claims':ratio(s,total),'contradiction_fraction_all_claims':ratio(n,total),
        'unresolved_fraction_all_claims':ratio(u,total),'evidence_resolution_coverage':ratio(s+n,total),
        'supported_fraction_resolved_only':ratio(s,s+n),
        'responses_with_contradiction':sum(any(c['label']=='Not Supported' for c in r['claims']) for r in rows),
        'responses_with_unresolved_claim':sum(any(c['label']=='Insufficient Information' for c in r['claims']) for r in rows),
        'responses_without_extracted_factual_claims':sum(not r['claims'] for r in rows),
        'response_types':dict(Counter(r['response_type'] for r in rows)),
        'retrieval_error_count':sum(len(r['retrieval_errors']) for r in rows),
        'invalid_evidence_citations':sum('citation_validation_error' in c for r in rows for c in r['claims']),
        'entailment_audit_disagreements':sum('audit_disagreement' in c for r in rows for c in r['claims']),
        'assistant_reviewed_candidates':sum(r.get('evidence_review',{}).get('reviewed_candidates',0) for r in rows),
        'assistant_review_downgrades':sum('pre_review_label' in c and c['pre_review_label']!=c['label'] for r in rows for c in r['claims']),
        'method':'Alternative claim-level evaluation; not an official entity-level LongFact++ score.'}


def run_metrics(run,examples):
    launch_path=ROOT/'runs'/run/'launch.json'
    if not launch_path.exists():
        return {'status':'not_launched'}
    launch=json.loads(launch_path.read_text(encoding='utf-8'))
    expected=set(launch['sample_ids'])
    assert expected==set(examples),'Main run does not match frozen evaluation IDs'
    responses=load_responses(run)
    assert all(r['id'] in expected for r in responses)
    lookup={r['id']:r for r in responses}
    grades=[]
    grading_errors=[]
    for path in (ROOT/'scoring'/run).glob('*.json'):
        grade=json.loads(path.read_text(encoding='utf-8'))
        if grade['id'] not in lookup:
            continue
        if grade['status']!='ok':
            grading_errors.append(grade)
            continue
        required_version=LONG_VERSION if grade['dataset']=='longfact_pp' else SHORT_VERSION
        if grade.get('scorer_version')!=required_version or grade.get('retrieval_errors'):
            grading_errors.append({'id':grade['id'],'dataset':grade['dataset'],'status':'requires_rescoring',
                'saved_version':grade.get('scorer_version'),'required_version':required_version,
                'retrieval_errors':grade.get('retrieval_errors',[])})
            continue
        response=lookup[grade['id']]
        assert grade['response_sha256']==hashlib.sha256(response['response'].encode()).hexdigest(),'Stale grade'
        if grade['dataset']=='longfact_pp':
            assert all(c.get('evidence_review',{}).get('label')==c['label'] for c in grade['claims'] if c['label']!='Insufficient Information'),'Unreviewed resolved LongFact label'
            if grade.get('evidence_review'):
                receipt=grade['evidence_review']
                assert hashlib.sha256((ROOT/receipt['path']).read_bytes()).hexdigest()==receipt['sha256'],'Changed evidence-review receipt'
        grades.append(grade)
    assert len(grades)==len({r['id'] for r in grades})
    datasets={}
    for dataset,func in [('simpleqa_verified',short_metrics),('abstention_bench',abstention_metrics),('longfact_pp',long_metrics)]:
        selected=[r for r in grades if r['dataset']==dataset]
        total=sum(e['dataset']==dataset for e in examples.values())
        metrics=func(selected,total)
        metrics['by_component']={component:func([r for r in selected if r['component']==component],sum(e['dataset']==dataset and e['component']==component for e in examples.values())) for component in sorted({e['component'] for e in examples.values() if e['dataset']==dataset})}
        generated=[r for r in responses if r['dataset']==dataset]
        metrics['generated']=len(generated)
        metrics['generation_errors']=sum(r['status']!='ok' for r in generated)
        metrics['truncated']=sum(r.get('finish_reason')=='length' for r in generated)
        metrics['empty_responses']=sum(r.get('response','').strip()=='' for r in generated if r['status']=='ok')
        metrics['output_tokens']=sum(r.get('output_tokens',0) for r in generated)
        datasets[dataset]=metrics
    manifests=list((ROOT/'runs'/run/'output').rglob('manifest.json'))
    if not manifests and (ROOT/'runs'/run/'checkpoint/manifest.json').exists():
        manifests=[ROOT/'runs'/run/'checkpoint/manifest.json']
    manifest=json.loads(manifests[0].read_text(encoding='utf-8')) if manifests else None
    runtime=None
    if manifest:
        seconds=manifest.get('model_load_seconds',0)+manifest.get('generation_seconds',0)
        runtime={'model_load_seconds':manifest.get('model_load_seconds'),'generation_seconds':manifest.get('generation_seconds'),
            'allocated_gpus':len(manifest.get('gpu_info',[])),
            'load_and_generation_device_hours_estimate':seconds*len(manifest.get('gpu_info',[]))/3600,
            'peak_gpu_allocated_gib':manifest.get('peak_gpu_allocated_gib'),
            'note':'Device-hour estimate counts allocated GPUs during load/generation only; excludes setup, queueing and prior attempts. Not a Kaggle billing/quota measurement.'}
    return {'status':'complete' if len(grades)==len(expected) and all(r['status']=='ok' for r in responses) else 'in_progress',
        'model_id':launch['config']['model_id'],'revision':launch['config']['revision'],'expected':len(expected),
        'generated':len(responses),'graded':len(grades),'missing_generation_ids':sorted(expected-set(lookup)),
        'grading_errors':grading_errors,'datasets':datasets,'runtime_manifest':manifest,'runtime_summary':runtime}


def pct(value):
    return '—' if value is None else f'{100*value:.1f}%'


def main():
    examples={r['id']:r for r in read_jsonl(ROOT/'data/evaluation.jsonl')}
    data_hash=hashlib.sha256((ROOT/'data/evaluation.jsonl').read_bytes()).hexdigest()
    assert data_hash=='b4aaadab39cc65ee5356ce25528afaea48e992bbc3da6d61450f317c7292c549'
    metrics={run:run_metrics(run,examples) for run in ['tiny-baseline','8b-baseline']}
    complete=all(r['status']=='complete' for r in metrics.values())
    ledger=ROOT/'scoring/budget.json'
    budget=json.loads(ledger.read_text()) if ledger.exists() else {}
    result={'created_at':datetime.now(timezone.utc).isoformat(),'complete':complete,'evaluation_sha256':data_hash,'runs':metrics,'judge_budget':budget}
    (ROOT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Aya factuality baselines','',f'**Status: {"all fixed evaluation items generated and scored" if complete else "in progress; counts below are provisional"}.**',
        '',f'Updated: {result["created_at"]}. Each model has 1,000 SimpleQA Verified, 978 AbstentionBench subset, and 100 LongFact++ held-out subset questions.','',
        '## SimpleQA Verified — full 1,000-question evaluation','',
        '| Model | Scored | Correct | Incorrect | Not attempted | Correct yield | Coverage | Accuracy when attempted |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,run in metrics.items():
        if 'datasets' not in run:continue
        m=run['datasets']['simpleqa_verified']
        lines.append(f'| {name} | {m["scored"]}/1000 | {m["correct"]} | {m["incorrect"]} | {m["not_attempted"]} | {pct(m["correct_answer_yield"])} | {pct(m["coverage"])} | {pct(m["selective_accuracy"])} |')
    lines+=['','SimpleQA grades the requested reference target. It does not certify the factual accuracy of every extra sentence in an answer. Targeted judge reviews and potential grading disagreements are recorded in `AUDIT.md`.','',
        '## AbstentionBench — balanced 978-question subset','',
        '| Model | Scored | Appropriate abstentions / 489 | Over-refusals / 489 | Answered should-abstain / 489 | Behavior accuracy | Task success |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,run in metrics.items():
        if 'datasets' not in run:continue
        m=run['datasets']['abstention_bench']
        lines.append(f'| {name} | {m["scored"]}/978 | {m["appropriate_abstentions"]} | {m["over_refusals"]} | {m["unsafe_answers"]} | {pct(m["behavior_accuracy"])} | {pct(m["task_success"])} |')
    lines+=['','Behavior accuracy checks whether the model abstains when it should, regardless of whether attempted answers are correct. Task success additionally requires non-abstaining answers on answerable questions to match references. Source-defined should-abstain labels include ambiguity and false premises; answering these questions is a desired-behavior error, not automatically a verified factual hallucination. The official broad abstention rubric counts uncertainty and premise corrections; it differs from SimpleQA’s not-attempted label. Component details, precision/recall/F1, selective risk, and reference correctness are in `metrics.json`.','',
        '**WorldSense sensitivity (primary grades unchanged):** interpreting only bare option-3 answers as source-native abstentions changes behavior accuracy from 67.28% to 67.79% for Tiny and 71.06% to 71.88% for 8B, and task success from 46.83% to 48.88% / 54.81% to 56.44%. This applies on both desired-label strata and excludes longer explanations. Exact affected IDs and denominators: `scoring/worldsense-option3-sensitivity.json`; diagnostic controls and other semantic concerns: `AUDIT.md`.','',
        '## LongFact++ — alternative evidence-backed claim evaluation','',
        '| Model | Responses scored | Claims | Supported | Contradicted | Unresolved | Evidence resolution |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,run in metrics.items():
        if 'datasets' not in run:continue
        m=run['datasets']['longfact_pp']
        lines.append(f'| {name} | {m["scored"]}/100 | {m["extracted_claims"]} | {m["supported"]} | {m["contradicted"]} | {m["unresolved"]} | {pct(m["evidence_resolution_coverage"])} |')
    lines+=['','This is **not the official LongFact++ entity-level score**. Fresh claims are extracted from Aya responses and checked against cached Wikipedia passages and Crossref publication metadata. “Contradicted” requires positive contrary evidence; missing evidence is unresolved. Exact evidence quotes are checked mechanically, then a separate short-context check must confirm that those quotes entail the assigned label. Disagreements become unresolved. Claim granularity is model-extracted and may include compound assertions. Retrieval and judge errors are recorded. Resolved-only precision is not factual accuracy over all claims. The original annotations for other models are never used as Aya labels.','',
        '## Execution and scoring record','',
        '- Official Aya weights and chat templates; FP16, no quantization, greedy decoding. Inference runs on Kaggle T4 GPUs.',
        '- Generation ceilings: 256 / 512 / 1,536 tokens for SimpleQA / AbstentionBench / LongFact++. Truncated responses remain in evaluation.',
        '- Short-answer rubrics are copied from official releases; the substitute judge is `Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra`. These are local benchmark evaluations, not official leaderboard submissions.',
        f'- LongFact scorer `{LONG_VERSION}` uses Qwen for extraction/primary verification and `{AUDIT_MODEL}` for single-claim sufficiency/incompatibility checks. Every remaining resolved claim also receives a direct quote-based assistant review, which may confirm or downgrade to unresolved. The legacy receipt identity is GPT-6 Astra; current reviews additionally disclose GPT-6.1 Sol as the actual reviewing model. This is not human annotation or a blinded independent study. Failed retrieval or pending evidence review prevents acceptance as scored.',
        '- Raw generations and runtime manifests: `runs/`. Per-item grades: `scoring/<run>/`. Full judge requests/responses: `scoring/api_cache/`. Retrieved text/URLs: `scoring/evidence_cache/` and per-item `longfact_details/`.',
        f'- HF provider-reported judge cost: ${budget.get("reported_cost_usd",0):.4f}; conservative reserved budget: ${budget.get("reserved_upper_usd",0):.4f}, ceiling ${budget.get("ceiling_usd",18):.2f}.',
        '', '| Model | Generated | Graded | Truncated by dataset (SimpleQA / Abstention / LongFact) |','|---|---:|---:|---|']
    for name,run in metrics.items():
        if 'datasets' not in run:continue
        trunc=' / '.join(str(run['datasets'][d]['truncated']) for d in ['simpleqa_verified','abstention_bench','longfact_pp'])
        lines.append(f'| {name} | {run["generated"]}/2078 | {run["graded"]}/2078 | {trunc} |')
    lines+=['','### Remote compute','']
    for name,run in metrics.items():
        runtime=run.get('runtime_summary')
        if runtime and runtime.get('generation_seconds') is not None:
            lines.append(f'- {name}: model loading {runtime["model_load_seconds"]:.1f}s; generation {runtime["generation_seconds"]/60:.1f} minutes; {runtime["allocated_gpus"]} allocated T4 GPUs; load-plus-generation device-hour estimate {runtime["load_and_generation_device_hours_estimate"]:.3f}; peak allocated GPU memory {runtime["peak_gpu_allocated_gib"]} GiB.')
    lines.append('Device-hour estimates exclude setup, queueing and earlier attempts, and are not Kaggle quota measurements. Earlier attempt logs are retained separately.')
    lines+=['','## Experimental interpretation','']
    if complete:
        saturated=all(run['datasets']['simpleqa_verified']['correct_answer_yield']>=.95 for run in metrics.values())
        lines.append('SimpleQA correct-answer yield meets 95% on both models; investigate a harder existing factuality benchmark before optimizing this score.' if saturated else 'The measured short-form results do not establish ≥95% factual accuracy at high coverage on both models. Preserve this fixed evaluation and proceed to F: activation-based hallucination/abstention detection, with separate training/validation questions and risk–coverage evaluation. Use causal localization (C) after establishing predictive signal. LongFact evidence gaps require further annotation before treating every unresolved claim as a training label.')
    else:
        lines.append('Wait for complete matched outputs and grading before selecting the next experiment.')
    lines+=['','See `DATASET_INSPECTION.md` for exact selection, source versions, prompt deviations, and limitations. Bootstrap/claim independence assumptions are not made; Wilson intervals in JSON apply only to prompt-level binomial summaries.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'complete':complete,'runs':{k:{f:v.get(f) for f in ['status','generated','graded']} for k,v in metrics.items()},'budget':budget},indent=2))


if __name__=='__main__':
    main()
