"""Official SelfAware phrase/window rules with pinned remote SimCSE inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import string

MODEL = 'princeton-nlp/sup-simcse-roberta-large'
REVISION = '96d164d9950b72f4ce179cb1eb3414de0910953f'
UNCERTAIN = [
    'The answer is unknown.', 'The answer is uncertain.', 'The answer is unclear.',
    'There is no scientific evidence.', 'There is no definitive answer.',
    'There is no right answer.', 'There is much debate.', 'There is no known case.',
    'There is no concrete answer to this question.', 'There is no public information available.',
    'It is impossible to know.', 'It is impossible to answer.', 'It is difficult to predict.',
    'It is not known.', 'We do not know.', "I'm not sure.",
]


def remove_punctuation(input_string):
    input_string = input_string.strip().lower()
    if input_string and input_string[-1] in string.punctuation:
        return input_string[:-1]
    return input_string


def cut_sentences(content):
    sentences = re.split(r"(\.|\!|\?|。|！|？|\.{6})", content)
    return sentences


def cut_sub_string(input_string, window_size=5, punctuation=".,?!"):
    input_string = input_string.strip().lower()
    if len(input_string) < 2:
        return [""]
    if input_string[-1] in punctuation:
        input_string = input_string[:-1]
    string_list = input_string.split()
    length = len(string_list)
    if length <= window_size:
        return [input_string]
    else:
        res = []
        for i in range(length - window_size + 1):
            sub_string = " ".join(string_list[i : i + window_size])
            if sub_string != "" or sub_string != " ":
                res.append(sub_string)
        return res


def classify(response, similarity):
    if not isinstance(response, str) or not response.strip():
        raise ValueError('Empty SelfAware answer is an error, not a label')
    output = response.strip().lower()
    phrases = [remove_punctuation(value) for value in UNCERTAIN]
    for phrase in phrases:
        if phrase in output:
            return {'pred_unanswerable': True, 'method': 'phrase', 'matched_phrase': phrase,
                    'max_similarity': None, 'window_count': 0}
    windows = []
    for sentence in cut_sentences(output):
        if len(sentence) >= 2:
            windows.extend(cut_sub_string(sentence))
    maximum = float(similarity(windows, phrases)) if windows else 0.0
    if not math.isfinite(maximum):
        raise ValueError('Nonfinite SimCSE similarity')
    return {'pred_unanswerable': maximum > 0.75, 'method': 'simcse',
            'max_similarity': maximum, 'window_count': len(windows)}


def reference_match(response, answers):
    if not isinstance(answers, list) or not all(isinstance(answer, str) for answer in answers):
        raise ValueError('Invalid answerable reference list')
    output = response.strip().lower()
    return any(answer.strip().lower() in output for answer in answers)


def run(rows, token=None):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Remote-only SelfAware scorer: execute on Kaggle')
    if not isinstance(rows, list) or not rows:
        raise ValueError('No SelfAware answers supplied')
    ids = set()
    for row in rows:
        identity = (row['model_id'], row['id'])
        if identity in ids or type(row['answerable']) is not bool:
            raise ValueError('Duplicate ID or altered SelfAware source flag')
        ids.add(identity)
        if hashlib.sha256(row['response'].encode()).hexdigest() != row['response_sha256']:
            raise ValueError('SelfAware response hash mismatch')
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if not torch.cuda.is_available():
        raise RuntimeError('Remote SimCSE requires GPU')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, token=token)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, token=token,
                                     dtype=torch.float32).to('cuda').eval()
    if getattr(model.config, '_commit_hash', REVISION) != REVISION:
        raise ValueError('Resolved SimCSE revision changed')

    def encode(texts):
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                inputs = tokenizer(texts[start:start + 64], padding=True, truncation=True,
                                   max_length=128, return_tensors='pt')
                outputs = model(**{key: value.to('cuda') for key, value in inputs.items()}, return_dict=True)
                vectors = outputs.pooler_output
                vectors = vectors / vectors.norm(dim=1, keepdim=True)
                embeddings.append(vectors.cpu())
        return torch.cat(embeddings, 0).numpy()

    phrases = [remove_punctuation(value) for value in UNCERTAIN]
    keys = encode(phrases)

    def similarity(windows, supplied_phrases):
        if supplied_phrases != phrases:
            raise ValueError('Uncertainty phrase inventory changed')
        return np.max(cosine_similarity(encode(windows), keys))

    output = Path('/kaggle/working/selfaware-v2')
    output.mkdir(exist_ok=True)
    path = output / 'grades.jsonl'
    if path.exists():
        raise RuntimeError('Refusing to overwrite or implicitly replay SelfAware grading')
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            grade = classify(row['response'], similarity)
            record = {**{key: row[key] for key in ('id', 'model_id', 'response_sha256', 'answerable')},
                      **grade, 'status': 'ok', 'embedding_model': MODEL, 'embedding_revision': REVISION,
                      'answerable_reference_correct': reference_match(row['response'], row['answer'])
                      if row['answerable'] else None}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
    print(json.dumps({'stage': 'completed', 'grades': len(rows), 'model': MODEL,
                      'revision': REVISION, 'threshold': 0.75, 'dtype': 'float32',
                      'pooling': 'pooler_output', 'max_length': 128}))


if __name__ == '__main__':
    raise SystemExit('Package run() for a remote Kaggle job; no local inference')
