"""Small local data downloads and inspection; never loads a language model."""
from __future__ import annotations

import hashlib
import json
import csv
import random
import os
import sys
import types
from collections import Counter
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
REVISIONS = {
    'obalcells/longfact-augmented-prompts': '3b6c1ed74a27fc01d551fc09228a196ad68f8a75',
    'obalcells/longfact-augmented-annotations': '63174c7ea9111adb6f4c4ea61895af3e3740d997',
    'facebook/AbstentionBench': 'af06080e2cbfff73f02a7c33135e5567d6f7ce6a',
    'google/simpleqa-verified': '0dc97e0d28d8233463e005cdc4475cc2a13ba2dc',
}
SESSION = requests.Session()
SESSION.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))


def fetch(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        r = SESSION.get(url, timeout=(20, 120))
        r.raise_for_status()
        path.write_bytes(r.content)
    content = path.read_bytes()
    return {'url': url, 'path': str(path.relative_to(ROOT)), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


def download():
    records = []
    files = {
        'obalcells/longfact-augmented-prompts': ['README.md', 'data/train-00000-of-00001.parquet'],
        'obalcells/longfact-augmented-annotations': ['README.md'] + [f'Meta-Llama-3.1-8B-Instruct/{split}-00000-of-00001.parquet' for split in ['train', 'validation', 'test']],
        'facebook/AbstentionBench': ['README.md', 'AbstentionBench.py', 'data.py', 'subsampling-indices.json', 'UMWP_indices_answerable.json', 'kuq_new_categories.csv'],
        'google/simpleqa-verified': ['README.md', 'simpleqa_verified.csv'],
    }
    for repo, names in files.items():
        for name in names:
            url = f'https://huggingface.co/datasets/{repo}/resolve/{REVISIONS[repo]}/{name}'
            record = fetch(url, RAW / repo.split('/')[-1] / name)
            records.append(record)
            print(json.dumps({'downloaded': record['path'], 'bytes': record['bytes']}), flush=True)
    github = {
        'facebookresearch/AbstentionBench': ('e29184174c69bc95139b8c33dcca09aacab9442a', ['README.md', 'recipe/system_prompt.py', 'recipe/evaluation_judge_prompts.py', 'recipe/evaluation.py', 'data/fast-subset-indices.json']),
        'obalcells/hallucination_probes': ('295450e2c24db7ecac1229e21783d2ebd08f9582', ['annotation_pipeline/entity_annotation.prompt', 'annotation_pipeline/README.md']),
    }
    for repo, (rev, names) in github.items():
        for name in names:
            records.append(fetch(f'https://raw.githubusercontent.com/{repo}/{rev}/{name}', RAW / repo.split('/')[-1] / name))
    (RAW.parent / 'downloads.json').write_text(json.dumps({'revisions': REVISIONS, 'files': records}, indent=2), encoding='utf-8')


def prepare_abstention():
    """Use reviewed official transforms, omitting their unused PyTorch base class."""
    source_path = RAW / 'AbstentionBench' / 'data.py'
    source = source_path.read_text(encoding='utf-8')
    source = source.replace('import torch\n', '').replace('class AbstentionDataset(ABC, torch.utils.data.Dataset):', 'class AbstentionDataset(ABC):')
    module = types.ModuleType('pinned_abstention_data')
    sys.modules[module.__name__] = module
    exec(compile(source, str(source_path), 'exec'), module.__dict__)
    upstream = RAW / 'abstention_sources'
    upstream.mkdir(exist_ok=True)
    os.chdir(upstream)
    official_indices = json.loads((RAW / 'AbstentionBench' / 'subsampling-indices.json').read_text())
    specs = [
        ('big_bench_disambiguate', 'BigBenchDisambiguateDataset', {}),
        ('falseqa', 'FalseQADataset', {}),
        ('gsm8k_abstain', 'GSM8K', {}),
        ('mmlu_math_abstain', 'MMLUMath', {}),
        ('mmlu_history_abstain', 'MMLUHistory', {}),
        ('bbq', 'BBQDataset', {}),
        ('mediq', 'MediQDataset', {}),
        ('known_unknown_questions', 'KUQDataset', {'category_map_path': str(RAW / 'AbstentionBench' / 'kuq_new_categories.csv')}),
        ('squad2', 'Squad2Dataset', {}),
        ('world_sense', 'WorldSenseDataset', {}),
    ]
    summary = {}
    for name, cls, kwargs in specs:
        dest = RAW.parent / 'abstention_components' / f'{name}.jsonl'
        dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            rows = [json.loads(s) for s in dest.read_text(encoding='utf-8').split('\n') if s.strip()]
        else:
            try:
                random.seed(20261111)
                obj = getattr(module, cls)(**kwargs)
                indices = official_indices.get(cls, list(range(len(obj))))
                rows = []
                for idx in indices:
                    item = obj[idx].model_dump()
                    item.update(component=name, source_index=idx)
                    rows.append(item)
                dest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
            except Exception as exc:
                summary[name] = {'error': type(exc).__name__, 'message': str(exc)[:700]}
                print(json.dumps({name: summary[name]}), flush=True)
                continue
        summary[name] = {'rows': len(rows), 'should_abstain': dict(Counter(str(r['should_abstain']) for r in rows)), 'answerable_without_reference': sum(not r['should_abstain'] and not r['reference_answers'] for r in rows)}
        print(json.dumps({name: summary[name]}), flush=True)
    (RAW.parent / 'abstention_inspection.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def assemble():
    import pyarrow.parquet as pq
    data = RAW.parent
    rows = []
    simple = list(csv.DictReader((RAW / 'simpleqa-verified' / 'simpleqa_verified.csv').open(encoding='utf-8')))
    for r in simple:
        rows.append({'id': 'simpleqa-'+r['original_index'], 'dataset': 'simpleqa_verified', 'component': r['topic'], 'question': r['problem'], 'reference_answers': [r['answer']], 'should_abstain': False, 'metadata': r, 'system_prompt': '', 'max_new_tokens': 256})
    annotated = pq.read_table(RAW / 'longfact-augmented-annotations' / 'Meta-Llama-3.1-8B-Instruct' / 'test-00000-of-00001.parquet').to_pylist()
    seen_questions = set()
    exclusions = []
    # Exclude overlap with the chosen annotation configuration's training/development questions.
    training_questions = set()
    for split in ['train', 'validation']:
        for r in pq.read_table(RAW / 'longfact-augmented-annotations' / 'Meta-Llama-3.1-8B-Instruct' / f'{split}-00000-of-00001.parquet', columns=['conversation']).to_pylist():
            training_questions.add(next(m['content'].strip() for m in r['conversation'] if m['role']=='user'))
    allocations = {'longfact_augmented_topics': 65, 'longfact_augmented_citations': 20, 'longfact_augmented_biography': 3, 'longfact_augmented_legal': 12}
    for component, count in allocations.items():
        candidates = []
        for i, r in enumerate(annotated):
            if r['subset'] != component:
                continue
            q = next(m['content'].strip() for m in r['conversation'] if m['role']=='user')
            if q in training_questions or q in seen_questions:
                exclusions.append({'source_index': i, 'reason': 'question_overlap'})
                continue
            seen_questions.add(q)
            candidates.append((i, r, q))
        candidates.sort(key=lambda x: hashlib.sha256(f'20261111:{x[0]}:{x[2]}'.encode()).hexdigest())
        for i, r, q in candidates[:count]:
            rows.append({'id': f'longfact-test-{i}', 'dataset': 'longfact_pp', 'component': component, 'question': q, 'reference_answers': None, 'should_abstain': None, 'metadata': {'source_index': i, 'source_config': 'Meta-Llama-3.1-8B-Instruct', 'source_split': 'test'}, 'system_prompt': 'You are a helpful assistant. Answer the question in detail, including specific names, dates, numbers, events, and examples when relevant.', 'max_new_tokens': 1536})
    official_prompt = (RAW / 'AbstentionBench' / 'recipe' / 'system_prompt.py').read_text(encoding='utf-8')
    import ast
    node = next(n for n in ast.parse(official_prompt).body if isinstance(n, ast.Assign))
    abstention_prompt = ast.literal_eval(node.value)
    # Do not assert an invented December 2023 model cutoff. No stale-information component is included.
    abstention_prompt = '\n\n'.join(p for p in abstention_prompt.split('\n\n') if not p.startswith('Your knowledge base was last updated'))
    selected_by_component = {}
    for path in sorted((data / 'abstention_components').glob('*.jsonl')):
        candidates = [json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s.strip()]
        chosen = []
        for label in [False, True]:
            pool = [r for r in candidates if r['should_abstain'] is label]
            pool.sort(key=lambda r: hashlib.sha256(f"20261111:{path.stem}:{r['source_index']}".encode()).hexdigest())
            # Remove exact within-component duplicate questions before selection.
            unique = {}
            for r in pool:
                unique.setdefault(r['question'].strip(), r)
            chosen.extend(list(unique.values())[:50])
        selected_by_component[path.stem] = len(chosen)
        for r in chosen:
            rows.append({'id': f"abstention-{path.stem}-{r['source_index']}", 'dataset': 'abstention_bench', 'component': path.stem, 'question': r['question'], 'reference_answers': r['reference_answers'], 'should_abstain': r['should_abstain'], 'metadata': r['metadata'] | {'source_index': r['source_index']}, 'system_prompt': abstention_prompt, 'max_new_tokens': 512})
    assert len({r['id'] for r in rows}) == len(rows)
    dest = data / 'evaluation.jsonl'
    dest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    manifest = {'selection_seed': 20261111, 'counts': dict(Counter(r['dataset'] for r in rows)), 'abstention_components': selected_by_component, 'longfact_allocations': allocations, 'longfact_overlap_exclusions': exclusions, 'evaluation_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(), 'revisions': REVISIONS, 'decoding': {'do_sample': False, 'num_beams': 1}, 'abstention_prompt_deviation': 'Official prompt with its fixed December 2023 cutoff paragraph removed; no FreshQA/stale snapshot tasks.', 'grader_plan': 'Official reference-grading instructions with a disclosed alternative HF-hosted judge; LongFact++ requires fresh retrieved evidence and is not an official Claude scorer reproduction.'}
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    examples = {name: [r for r in rows if r['dataset']==name][:3] for name in manifest['counts']}
    (data / 'inspection_examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv)>1 else 'download'
    {'download': download, 'abstention': prepare_abstention, 'assemble': assemble}[command]()
