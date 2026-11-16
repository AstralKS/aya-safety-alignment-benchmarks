"""Export evidence for assistant review; apply complete, hash-bound review records."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import shutil

from score import ROOT,read_jsonl
from score_longfact import VERSION


def review_run(run,apply=False,limit=3):
    folder=ROOT/'scoring'/run
    questions={r['id']:r['question'] for r in read_jsonl(ROOT/'data/evaluation.jsonl')}
    pending=[]
    for path in sorted(folder.glob('longfact-*.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        if record.get('status')=='needs_evidence_review' and record.get('scorer_version')==VERSION:
            pending.append((path,record))
    print(json.dumps({'run':run,'awaiting_review':len(pending)}))
    for path,record in pending[:limit] if not apply else pending:
        selected=[c for c in record['claims'] if c['label']!='Insufficient Information']
        review_path=folder/'reviews'/(record['id']+'.json')
        if apply:
            if not review_path.exists():
                continue
            review=json.loads(review_path.read_text(encoding='utf-8'))
            assert review['response_sha256']==record['response_sha256'],'Review response hash mismatch'
            assert review['scorer_version']==VERSION,'Review scorer version mismatch'
            assert set(review['judgments'])=={str(c['claim_id']) for c in selected},'Review must cover every resolved candidate exactly'
            assert review['reviewer']=='GPT-6 Astra (assistant; not human annotation)'
            for claim in selected:
                decision=review['judgments'][str(claim['claim_id'])]
                assert decision['label'] in [claim['label'],'Insufficient Information'],'Review may confirm or conservatively downgrade only'
                assert isinstance(decision['reason'],str) and decision['reason'].strip()
                claim['pre_review_label']=claim['label']
                claim['label']=decision['label']
                claim['evidence_review']=decision|{'reviewer':review['reviewer']}
            history=folder/'history'
            history.mkdir(exist_ok=True)
            shutil.copy2(path,history/(path.stem+'-before-review-'+str(path.stat().st_mtime_ns)+'.json'))
            record.update(status='ok',claim_counts=dict(Counter(c['label'] for c in record['claims'])),
                evidence_review={'path':str(review_path.relative_to(ROOT)),
                    'sha256':hashlib.sha256(review_path.read_bytes()).hexdigest(),
                    'reviewer':review['reviewer'],'reviewed_candidates':len(selected),'applied_at':datetime.now(timezone.utc).isoformat()})
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'id':record['id'],'status':record['status'],'claims':record['claim_counts']}))
        else:
            data=json.loads((folder/'longfact_details'/record['id']/'sources.json').read_text(encoding='utf-8'))
            sources={s['source_id']:s for s in data['sources']}
            print(json.dumps({'id':record['id'],'response_sha256':record['response_sha256'],'scorer_version':VERSION,
                'question':questions[record['id']],
                'candidates':[{'claim_id':c['claim_id'],'claim':c['claim'],'candidate_label':c['label'],
                    'evidence':[{'title':sources[e['source_id']]['title'],'url':sources[e['source_id']]['url'],'quote':e['quote']} for e in c['evidence']]}
                    for c in selected]},ensure_ascii=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--limit',type=int,default=3)
    args=parser.parse_args()
    review_run(args.run,args.apply,args.limit)
