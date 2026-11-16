"""Saved regression cases for errors discovered during LongFact pilot auditing."""
from __future__ import annotations
from datetime import datetime, timezone
import json

from score_longfact import ROOT, VERSION, AUDIT_MODEL, audit_entailment


def main():
    # Reuse the original frozen claim/quote/title tuples. Per-response source IDs
    # can change when a later retrieval retry adds previously missing sources.
    fixture=json.loads((ROOT/'scoring/entailment-regression-longfact-evidence-claims-v3.json').read_text(encoding='utf-8'))
    claims=fixture['claims']
    sources=fixture['sources']
    controls=[{k:c[k] for k in ['name','claim_id','expected']} for c in fixture['controls']]
    expected={c['claim_id']:c['expected'] for c in controls}
    judgments=[{'claim_id':j['claim_id'],'evidence':j['evidence'],
        'label':expected[j['claim_id']] if expected[j['claim_id']]!='Insufficient Information' else 'Supported'}
        for j in fixture['judgments']]
    audited=audit_entailment(claims,judgments,sources)
    by_id={j['claim_id']:j for j in audited}
    for control in controls:
        actual=by_id[control['claim_id']]['entailment_audit']
        control.update(actual=actual['label'],passed=actual['label']==control['expected'],reason=actual['reason'])
    keys={j['entailment_audit']['api_cache_key'] for j in audited}
    calls=[json.loads((ROOT/'scoring/api_cache'/f'{key}.json').read_text(encoding='utf-8')) for key in keys]
    result={'created_at':datetime.now(timezone.utc).isoformat(),'scorer_version':VERSION,'judge':AUDIT_MODEL,
        'passed':all(c['passed'] for c in controls),'controls':controls,'claims':claims,'judgments':audited,'sources':sources,
        'usage':[{'cache_key':c['cache_key'],'usage':c['response'].get('usage'),'finish_reason':c['finish_reason']} for c in calls]}
    (ROOT/'scoring'/f'entailment-regression-{VERSION}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','judge','controls','usage']},indent=2),flush=True)
    assert result['passed'],'Quote-only entailment regression failed'


if __name__=='__main__':
    main()
