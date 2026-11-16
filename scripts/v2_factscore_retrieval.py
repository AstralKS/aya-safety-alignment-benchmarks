"""Frozen subject-local FActScore GTR retrieval; all tokenization/embeddings remote."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL = 'sentence-transformers/gtr-t5-large'
REVISION = '51aab2cd10b205ba0f3f1ebaf1d4d5460bbf728d'
TOKENIZER_MODEL = 'FacebookAI/roberta-large'
TOKENIZER_REVISION = '722cf37b1afa9454edce342e7895e588b6ff1d59'


def token_chunks(tokens):
    return [tokens[start:start + 256] for start in range(0, len(tokens), 256)]


def passage_inputs(topic, passages):
    return [topic + ' ' + text.replace('<s>', '').replace('</s>', '') for text in passages]


def query_input(topic, fact):
    return topic + ' ' + fact.strip()


def validate_inputs(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('evidence'), list) or not isinstance(payload.get('facts'), list):
        raise ValueError('Invalid retrieval input')
    topics = set()
    for record in payload['evidence']:
        if not isinstance(record.get('topic'), str) or not record['topic'] or record['topic'] in topics:
            raise ValueError('Empty/duplicate evidence subject')
        topics.add(record['topic'])
        if not isinstance(record.get('text'), str) or not record['text'].strip():
            raise ValueError('Empty frozen evidence')
        if hashlib.sha256(record['text'].encode()).hexdigest() != record.get('text_sha256'):
            raise ValueError('Frozen evidence text hash mismatch')
        if type(record.get('revision')) is not int or record['revision'] < 1:
            raise ValueError('Missing frozen revision')
    identities = set()
    for fact in payload['facts']:
        identity = (fact.get('aya_model', fact.get('model_id')), fact.get('fact_id'))
        if identity in identities or not isinstance(fact.get('fact_id'), str) or not fact['fact_id']:
            raise ValueError('Duplicate/empty fact identity')
        identities.add(identity)
        if fact.get('topic') not in topics or not isinstance(fact.get('fact'), str) or not fact['fact'].strip():
            raise ValueError('Missing fact/evidence subject binding')


def run(payload, token=None):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Remote-only FActScore retrieval: execute on Kaggle')
    validate_inputs(payload)
    import torch
    import numpy as np
    from transformers import RobertaTokenizer
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError('Remote GTR requires GPU')
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_MODEL, revision=TOKENIZER_REVISION, token=token)
    encoder = SentenceTransformer(MODEL, revision=REVISION, token=token, device='cuda').eval()
    output = Path('/kaggle/working/factscore-retrieval-v2')
    output.mkdir(exist_ok=True)
    path = output / 'retrieval.jsonl'
    if path.exists():
        raise RuntimeError('Refusing to overwrite or implicitly replay retrieval')
    evidence = {record['topic']: record for record in payload['evidence']}
    by_topic = {}
    for fact in payload['facts']:
        by_topic.setdefault(fact['topic'], []).append(fact)
    with path.open('x', encoding='utf-8') as stream:
        for topic, facts in by_topic.items():
            record = evidence[topic]
            # Official DocDB str input becomes one tokenized string, split at 256.
            ids = tokenizer(record['text'])['input_ids']
            passages = [tokenizer.decode(chunk) for chunk in token_chunks(ids)
                        if any(value not in (0, 2) for value in chunk)]
            if not passages:
                raise ValueError('No frozen passages after official token chunking')
            vectors = encoder.encode(passage_inputs(topic, passages), batch_size=32, device=encoder.device)
            queries = encoder.encode([query_input(topic, fact['fact']) for fact in facts],
                                     batch_size=32, device=encoder.device)
            for fact, query in zip(facts, queries, strict=True):
                scores = np.inner(query, vectors)
                indices = np.argsort(-scores)[:5]
                result = {**fact, 'status': 'ok', 'evidence_revision': record['revision'],
                          'evidence_text_sha256': record['text_sha256'],
                          'retrieval_model': MODEL, 'retrieval_revision': REVISION,
                          'tokenizer_model': TOKENIZER_MODEL, 'tokenizer_revision': TOKENIZER_REVISION,
                          'query': query_input(topic, fact['fact']),
                          'passages': [{'title': topic, 'text': passages[int(index)],
                                        'passage_index': int(index), 'score': float(scores[int(index)]),
                                        'text_sha256': hashlib.sha256(passages[int(index)].encode()).hexdigest()}
                                       for index in indices]}
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
    (output / 'manifest.json').write_text(json.dumps({'stage': 'completed', 'facts': len(payload['facts']),
        'model': MODEL, 'revision': REVISION, 'tokenizer': TOKENIZER_MODEL,
        'tokenizer_revision': TOKENIZER_REVISION, 'passage_tokens': 256, 'passages_per_fact': 5,
        'passage_scope': 'original subject only', 'similarity': 'np.inner; model default pooling/normalization',
        'input_sha256': hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()},
        indent=2), encoding='utf-8')
    print(json.dumps({'stage': 'completed', 'facts': len(payload['facts'])}))


if __name__ == '__main__':
    raise SystemExit('Package run() for Kaggle; no local model/tokenizer loading')
