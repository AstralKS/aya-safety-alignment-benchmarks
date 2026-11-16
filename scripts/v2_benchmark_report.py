"""Reconcile all six complete baselines and write benchmark.md from saved results only."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import v2_benchmark_metrics as metrics
import v2_glm_scoring as scoring
import v2_selfaware as selfaware
from v2_glm_budget import Ledger, atomic_replace, json_bytes

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('tiny', '8b')
COUNTS = {**scoring.EXPECTED, 'selfaware': 3369}
TASK_METRICS = {
    'simpleqa_verified': metrics.simpleqa,
    'abstention_bench': metrics.abstention,
    'hallulens_precisewikiqa': metrics.precise_wiki,
    'hallulens_nonexistentrefusal': metrics.nonexistent,
}



def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_artifact(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()), 'Result artifact escapes the baseline workspace')
    raw = path.read_bytes()
    return raw, {'path': path.relative_to(root.resolve()).as_posix(), 'sha256': sha(raw)}


def journal_label(root, outcome, job):
    require(outcome.get('ok') is True and outcome.get('cache_key') == job['cache_key'], 'Missing or mismatched GLM result')
    require(outcome.get('lineage') == job['lineage'] and outcome.get('stage') == job['stage'], 'GLM result lineage changed')
    raw, _ = read_artifact(root, outcome['journal_path'])
    require(sha(raw) == outcome['journal_sha256'], 'GLM result journal changed')
    journal = scoring.client.read_json(raw)
    require(journal.get('kind') == 'http_response' and journal.get('http_status') == 200
            and journal.get('request_sha256') == job['request_sha256'], 'Invalid GLM HTTP/request receipt')
    body = base64.b64decode(journal['raw_body_base64'], validate=True)
    require(sha(body) == journal['raw_body_sha256'], 'GLM response body changed')
    parsed = scoring.client.inspect_response(body, job['lineage']['dataset'],
        lambda text: scoring.parse_final(text, job['parser']))
    require(parsed['ok'] and parsed['label'] == outcome['label'], 'Saved GLM result differs from strict journal parsing')
    return parsed['label']


def fixed_table(root, packet, available, catalog, tasks=scoring.FIXED_TASKS):
    require(packet.get('kind') == 'glm-fixed-benchmark-results-v1' and packet.get('halted') is False,
            'Fixed-benchmark execution is incomplete or halted')
    by_identity = {}
    for outcome in packet['results']:
        lineage = outcome['lineage']
        identity = (lineage['aya_model'], lineage['id'], outcome['stage'])
        require(identity not in by_identity, 'Duplicate fixed-benchmark result identity')
        by_identity[identity] = outcome
    table, consumed = {}, set()
    for short in MODELS:
        grouped = {task: [] for task in tasks}
        for row, answer, model in available:
            if model != short or row['dataset'] not in grouped:
                continue
            labels = {}
            jobs = scoring.fixed_jobs(row, answer, short, catalog)
            for job in jobs:
                identity = (short, row['id'], job['stage'])
                require(identity in by_identity, f'Missing fixed result: {identity}')
                labels[job['stage']] = journal_label(root, by_identity[identity], job)
                consumed.add(identity)
            task = row['dataset']
            if task == 'simpleqa_verified':
                grouped[task].append(labels['simpleqa'])
            elif task == 'hallulens_nonexistentrefusal':
                grouped[task].append(labels['does_believe'])
            elif task == 'hallulens_precisewikiqa':
                grouped[task].append({'abstained': labels['abstention'], 'correctness': labels['correctness']})
            else:
                record = {'should_abstain': row['should_abstain'], 'abstained': labels['abstention'] == 'Yes'}
                if row['should_abstain'] is False and labels['abstention'] == 'No':
                    job = scoring.fixed_jobs(row, answer, short, catalog, abstention='No')[-1]
                    identity = (short, row['id'], 'correctness')
                    require(identity in by_identity, f'Missing conditional correctness: {identity}')
                    record['correct'] = journal_label(root, by_identity[identity], job) == 'correct'
                    consumed.add(identity)
                grouped[task].append(record)
        for task, records in grouped.items():
            require(len(records) == COUNTS[task], f'Incomplete {task}/{short}: {len(records)}')
        table[short] = {task: TASK_METRICS[task](grouped[task]) for task in tasks}
    require(consumed == set(by_identity), 'Extra or ineligible fixed-benchmark judgments')
    return table


def selfaware_table(records, available):
    expected = {(short, row['id']): (row, answer) for row, answer, short in available if row['dataset'] == 'selfaware'}
    observed, grouped = set(), {short: [] for short in MODELS}
    model_names = {model: short for short, model in scoring.client.AYA_MODELS.items()}
    for record in records:
        identity = (model_names.get(record.get('model_id')), record.get('id'))
        require(identity in expected and identity not in observed, 'Unknown or duplicate SelfAware grade')
        observed.add(identity)
        row, answer = expected[identity]
        require(record.get('status') == 'ok' and record.get('embedding_model') == selfaware.MODEL
                and record.get('embedding_revision') == selfaware.REVISION, 'SelfAware scorer model/status changed')
        require(record.get('response_sha256') == sha(answer.encode())
                and record.get('answerable') is row['metadata']['answerable'], 'SelfAware source/answer binding changed')
        if row['metadata']['answerable']:
            require(record.get('answerable_reference_correct') is selfaware.reference_match(answer, row['metadata']['answer']),
                    'SelfAware reference-substring metric changed')
        else:
            require(record.get('answerable_reference_correct') is None, 'Unexpected unanswerable reference grade')
        method = record.get('method')
        require(method in ('phrase', 'simcse'), 'Unknown SelfAware classification method')
        if method == 'phrase':
            exact = selfaware.classify(answer, lambda *_: (_ for _ in ()).throw(ValueError('Missing remote SimCSE score')))
            require(all(record.get(key) == value for key, value in exact.items()), 'SelfAware phrase rule changed')
        else:
            value = record.get('max_similarity')
            require(isinstance(value, (int, float)) and not isinstance(value, bool), 'Missing remote SelfAware similarity')
            exact = selfaware.classify(answer, lambda *_: value)
            require(all(record.get(key) == item for key, item in exact.items()), 'SelfAware threshold/window rule changed')
        grouped[identity[0]].append(record)
    require(observed == set(expected) and len(observed) == 6738, 'SelfAware grades are incomplete')
    return {short: metrics.selfaware(grouped[short]) for short in MODELS}


def verify_selfaware_receipt(root, path):
    from v2_compute_jobs import selfaware_inputs, verify
    root, path = Path(root).resolve(), Path(path).resolve()
    base = (root / 'runs/v2/compute').resolve()
    require(path.is_relative_to(base), 'SelfAware result lacks an isolated remote compute job')
    relative = path.relative_to(base)
    require(len(relative.parts) >= 4 and relative.parts[1] == 'output', 'Unexpected SelfAware result location')
    job_path = base / relative.parts[0]
    marker = job_path / 'collected.json'
    require(marker.is_file(), 'SelfAware result lacks a collected remote output receipt')
    job = verify(job_path, root)
    require(job['kind'] == 'selfaware', 'SelfAware grades belong to a different compute stage')
    inputs = (job_path / 'inputs.json').read_bytes()
    require(json.loads(inputs) == selfaware_inputs(root), 'Remote SelfAware job used different source/answer inputs')
    receipt = scoring.client.read_json(marker.read_bytes())
    expected = {'kind': 'selfaware', 'kernel': job['kernel'], 'result_path': path.relative_to(root).as_posix(),
                'result_sha256': sha(path.read_bytes()), 'input_sha256': job['input_sha256']}
    require(receipt == expected and sha(inputs) == receipt['input_sha256'], 'SelfAware remote input/output receipt mismatch')
    return marker


def bind_biography_fact(fact, identity, topic, answer_sha256, outcome):
    lineage = outcome['lineage']
    original = lineage['fact']
    require((lineage['aya_model'], lineage['id']) == identity
            and (original['aya_model'], original['id']) == identity
            and original['topic'] == topic and original['answer_sha256'] == answer_sha256,
            'Support judgment belongs to another biography/model/answer')
    require(all(key in original and type(original[key]) is type(value) and original[key] == value
                for key, value in fact.items() if key not in ('is_supported', 'verification_cache_key')),
            'Biography fact identity/content/extraction lineage differs from its support judgment')
    require(fact.get('verification_cache_key') == outcome['cache_key'], 'Biography support cache binding changed')


def factscore_table(root, packet, available, ready, retrieval, catalog):
    require(packet.get('kind') == 'glm-factscore-verification-results-v1' and packet.get('halted') is False,
            'FActScore verification is incomplete or halted')
    evidence = scoring.EvidenceCatalog(root)
    jobs = scoring.remote_verification_jobs(ready, retrieval, evidence, available)
    outcomes = {}
    for outcome in packet['results']:
        identity = outcome['lineage']['fact']['fact_id']
        require(identity not in outcomes, 'Duplicate FActScore support result')
        outcomes[identity] = outcome
    labels = {}
    for job in jobs:
        fact_id = job['lineage']['fact']['fact_id']
        require(fact_id in outcomes, 'Missing FActScore support result')
        labels[fact_id] = journal_label(root, outcomes[fact_id], job) == 'True'
    require(set(labels) == set(outcomes), 'Unexpected FActScore support result')
    expected = {(short, row['id']): (row, answer) for row, answer, short in available if row['dataset'] == 'factscore_bios'}
    observed, consumed, grouped = set(), set(), {short: [] for short in MODELS}
    for response in packet['responses']:
        identity = (response['aya_model'], response['id'])
        require(identity in expected and identity not in observed, 'Unknown or duplicate biography result')
        observed.add(identity)
        row, answer = expected[identity]
        require(response.get('completed') is True and isinstance(response.get('facts'), list)
                and response['topic'] == row['metadata']['topic'] and response['answer_sha256'] == sha(answer.encode()),
                'Biography extraction is incomplete or rebound to another answer')
        # Reconcile every sentence against fresh GLM extraction, including empty-fact responses.
        extract_jobs = scoring.extraction_jobs(row, answer, identity[0], catalog)
        sentences = response.get('sentences', [])
        require(len(sentences) == len(extract_jobs), 'Biography sentence inventory changed')
        expected_atoms = []
        for sentence, job in zip(sentences, extract_jobs):
            require(sentence['cache_key'] == job['cache_key'] and sentence['sentence'] == job['parser']['sentence']
                    and sentence['start'] == job['lineage']['sentence_start']
                    and sentence['end'] == job['lineage']['sentence_end'], 'Biography extraction sentence changed')
            journal_path = root / 'scoring/v2-glm/request-journals' / job['cache_key'] / 'attempts'
            matches = []
            for path in journal_path.glob('*.response.json'):
                raw = path.read_bytes()
                if sha(raw) != sentence['journal_sha256']:
                    continue
                journal = scoring.client.read_json(raw)
                require(journal.get('request_sha256') == job['request_sha256'] and journal.get('http_status') == 200,
                        'Invalid biography extraction request receipt')
                body = base64.b64decode(journal['raw_body_base64'], validate=True)
                require(sha(body) == journal['raw_body_sha256'], 'Biography extraction body changed')
                parsed = scoring.client.inspect_response(body, 'factscore_bios', lambda text: scoring.parse_final(text, job['parser']))
                require(parsed['ok'] and parsed['label']['facts'] == sentence['facts'], 'Biography atomic extraction changed')
                matches.append(parsed['label']['facts'])
            require(len(matches) == 1, 'Biography extraction lacks a unique successful GLM journal')
            expected_atoms.extend((job['lineage']['sentence_index'], index, text) for index, text in enumerate(matches[0]))
        actual_atoms = [(fact['sentence_index'], fact['fact_index'], fact['text']) for fact in response['facts']]
        require(expected_atoms == actual_atoms, 'Biography facts differ from fresh GLM extraction')
        supports = []
        for fact in response['facts']:
            fact_id = fact['fact_id']
            require(fact_id in labels and fact_id not in consumed and fact.get('is_supported') is labels[fact_id],
                    'Missing, duplicate or changed biography support label')
            bind_biography_fact(fact, identity, row['metadata']['topic'], sha(answer.encode()), outcomes[fact_id])
            consumed.add(fact_id)
            supports.append(labels[fact_id])
        grouped[identity[0]].append({'supported': supports})
    require(observed == set(expected) and len(observed) == 360 and consumed == set(labels), 'FActScore results are incomplete')
    return {short: metrics.factscore(grouped[short]) for short in MODELS}


def percent(value):
    return 'N/A' if value is None else f'{100 * value:.2f}%'


def render(table, accounting, sources, generation):
    def cells(task, key):
        return ' | '.join(percent(table[short][task][key]) for short in MODELS)
    lines = ['# Aya baseline benchmarks', '',
        '**Complete:** six benchmarks, 9,527 evaluated rows per model (7,549 new answers + 1,978 reused).', '',
        '| Benchmark | Metric | Tiny Aya Global | Aya Expanse 8B | Rows/model |',
        '|---|---|---:|---:|---:|',
        f'| SimpleQA Verified | Correct-answer yield ↑ | {cells("simpleqa_verified", "correct_answer_yield")} | 1,000 |',
        f'| SimpleQA Verified | Answer coverage | {cells("simpleqa_verified", "answer_coverage")} | 1,000 |',
        f'| SimpleQA Verified | Accuracy when attempted ↑ | {cells("simpleqa_verified", "accuracy_when_attempted")} | 1,000 |',
        f'| HalluLens PreciseWikiQA | Hallucination when not abstained ↓ | {cells("hallulens_precisewikiqa", "hallucination_rate_when_not_abstained")} | 1,000 |',
        f'| HalluLens PreciseWikiQA | False refusal ↓ | {cells("hallulens_precisewikiqa", "false_refusal_rate")} | 1,000 |',
        f'| HalluLens PreciseWikiQA | Correct-answer rate ↑ | {cells("hallulens_precisewikiqa", "correct_answer_rate")} | 1,000 |',
        f'| HalluLens NonExistentRefusal | False acceptance ↓ | {cells("hallulens_nonexistentrefusal", "false_acceptance_rate")} | 3,000 |',
        f'| AbstentionBench | Behavior accuracy ↑ | {cells("abstention_bench", "behavior_accuracy")} | 978 |',
        f'| AbstentionBench | Joint task success ↑ | {cells("abstention_bench", "joint_task_success")} | 978 |',
        f'| SelfAware | Unknown-label F1 ↑ | {cells("selfaware", "f1")} | 3,369 |',
        f'| SelfAware | Answerable reference accuracy ↑ | {cells("selfaware", "answerable_reference_accuracy")} | 2,338 |',
        f'| FActScore biographies | Macro supported-fact precision ↑ | {cells("factscore_bios", "macro_supported_fact_precision")} | 180 |',
        f'| FActScore biographies | Gamma=10 penalized macro ↑ | {cells("factscore_bios", "gamma_10_penalized_macro")} | 180 |',
        f'| FActScore biographies | Responding fraction | {cells("factscore_bios", "responding_fraction")} | 180 |', '',
        '## Denominators and counts', '',
        '| Count | Tiny | 8B |', '|---|---:|---:|']
    count_rows = [('SimpleQA correct / incorrect / not attempted', 'simpleqa_verified', ('correct', 'incorrect', 'not_attempted')),
                  ('PreciseWikiQA non-abstained / hallucinated', 'hallulens_precisewikiqa', ('non_abstained',)),
                  ('NonExistentRefusal false acceptances', 'hallulens_nonexistentrefusal', ('false_acceptances',)),
                  ('AbstentionBench behavior successes / joint successes', 'abstention_bench', ('behavior_correct', 'joint_successes')),
                  ('SelfAware TP / FP / FN', 'selfaware', ('tp', 'fp', 'fn')),
                  ('FActScore responding / no-fact biographies', 'factscore_bios', ('responding_biographies', 'no_fact_responses')),
                  ('FActScore supported / total facts', 'factscore_bios', ('supported_facts', 'total_facts'))]
    for label, task, keys in count_rows:
        values = [' / '.join(str(table[short][task][key]) for key in keys) for short in MODELS]
        if task == 'hallulens_precisewikiqa':
            values = [value + ' / ' + str(table[short][task]['incorrect_non_abstained'] + table[short][task]['unverifiable_non_abstained'])
                      for value, short in zip(values, MODELS)]
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'All required answers and judgments reconciled; no missing or failed judgments are silently graded.', '',
        '## Methods and limitations', '',
        '- GLM-5.3-Flash via HF/DeepInfra is the sole factual judge and biography fact extractor. Greedy judge decoding, low reasoning effort, 2,048-token limit; strict final-label/JSON parsing.',
        '- Aya uses pinned native chat templates, FP16, greedy one-pass decoding, no reflection instruction. All model inference and embedding/retrieval computation ran remotely.',
        '- These are disclosed local variants, not leaderboard reproductions. No new human or assistant factual adjudication; published judge-validation numbers do not transfer to GLM.',
        '- HalluLens uses pinned third-party, sample-validated data. Keys are not individually human-verified; screened entities are presumed nonexistent. NonExistentRefusal contains 2,000 Mixed + 1,000 Generated rows. PreciseWikiQA repairs only the contradictory final Yes/No correctness instruction; incorrect and unverifiable both count as hallucination among non-abstained answers.',
        '- AbstentionBench retains the balanced 489 answerable / 489 should-abstain subset. Correctness is judged only for answerable, non-abstained responses; joint success requires correct answers or appropriate abstention.',
        '- SelfAware retains all 3,369 released rows, including the ID3149 flag discrepancy (2,338 answerable / 1,031 unanswerable). Official phrase-first rules, five-word windows, pinned SimCSE FP32 pooler output, max length 128, strict cosine >0.75; answerable accuracy uses the official case-insensitive reference substring rule.',
        '- FActScore retains 180 original subjects, excluding exactly William Post, Carlos Alfonso and Jagan. Fresh GLM sentence-level atomic extraction uses pinned upstream demonstrations with disclosed stdlib segmentation/strict JSON rather than spaCy postprocessing. Evidence is frozen Wikipedia oldid pages, not the original 2023 database; pinned remote GTR selects up to five subject-local RoBERTa 256-token chunks per fact. No-fact responses are excluded from macro precision and disclosed in coverage; gamma=10 is reported separately.',
        '- TruthfulQA is dropped; LongFact++ remains extra/probe-training data. F/C remains deferred.', '',
        '## Cost and provenance', '',
        f'- GLM token-priced estimate: **${accounting["token_priced_estimates_usd"]}**; provider estimates: **${accounting["provider_estimates_usd"]}** (not invoices).',
        f'- Shared inclusive accounted/reserved total: **${accounting["shared_inclusive_reserved_upper_usd"]} / $18**; remaining conservative headroom: **${accounting["remaining_shared_headroom_usd"]}**. Legacy commitments were preserved, not added twice or released.',
        f'- GLM attempts: {accounting["attempts_reserved"]} reserved, {accounting["attempts_settled"]} settled; unknown v2 reserves: ${accounting["unknown_reserves_usd"]}. Remote Kaggle compute is separate quota, not included as a paid GLM charge.',
        '- Machine-readable metrics and full artifact hashes: `scoring/v2-glm/benchmark-results.json`.', '']
    for source in sources:
        lines.append(f'- `{source["path"]}` — SHA-256 `{source["sha256"]}`')
    lines += ['', '### Generation truncations (retained, not regenerated)', '',
              '| Model | EOS | Token limit |', '|---|---:|---:|']
    for short in MODELS:
        lines.append(f'| {short} | {generation[short]["eos"]} | {generation[short]["length"]} |')
    return '\n'.join(lines) + '\n'


def publish(root, result, markdown):
    destination = root / 'benchmark.md'
    machine = root / 'scoring/v2-glm/benchmark-results.json'
    if machine.exists():
        raw = machine.read_bytes()
        original = scoring.client.read_json(raw)
        normalized = scoring.client.read_json(json_bytes(result))
        comparable = lambda value: {key: item for key, item in value.items() if key != 'verified_at'}
        require(comparable(original) == comparable(normalized), 'Existing benchmark checkpoint belongs to different results')
        result = json.loads(raw)
    else:
        require(not destination.exists(), 'Refusing to overwrite an unfamiliar benchmark report')
        atomic_replace(machine, json_bytes(result))
    content = markdown.encode('utf-8')
    if destination.exists():
        require(destination.read_bytes() == content, 'Existing benchmark report differs from verified results')
    else:
        atomic_replace(destination, content)
    return result


def build(root, fixed_path, selfaware_path, factscore_path, ready_path, retrieval_path, write=False):
    root = Path(root).resolve()
    available, binding = scoring.load_available(root, MODELS)
    inventory = Counter((short, row['dataset']) for row, _, short in available)
    require(inventory == Counter({(short, task): count for short in MODELS for task, count in COUNTS.items()}),
            'Cannot write a complete benchmark before all 9,527 answers/model are available')
    packets, sources = [], []
    for path in (fixed_path, selfaware_path, factscore_path, ready_path, retrieval_path):
        raw, source = read_artifact(root, path)
        sources.append(source)
        packets.append([json.loads(line) for line in raw.splitlines()] if Path(path).suffix == '.jsonl' else scoring.client.read_json(raw))
    fixed, selfaware_grades, factscore, ready, _ = packets
    for packet in (fixed, factscore):
        require(packet.get('v2_inputs') == binding, 'Result snapshot belongs to different frozen evaluation inputs')
    retrieval = scoring.read_remote_retrieval(retrieval_path, ready)
    _, source = read_artifact(root, Path(retrieval_path).with_name('manifest.json'))
    sources.append(source)
    from v2_compute_jobs import validate_factscore_payload
    validate_factscore_payload(root, ready)
    catalog = scoring.SourceCatalog(root)
    table = fixed_table(root, fixed, available, catalog)
    marker = verify_selfaware_receipt(root, selfaware_path)
    _, source = read_artifact(root, marker)
    sources.append(source)
    self_table = selfaware_table(selfaware_grades, available)
    fact_table = factscore_table(root, factscore, available, ready, retrieval, catalog)
    for short in MODELS:
        table[short]['selfaware'] = self_table[short]
        table[short]['factscore_bios'] = fact_table[short]
    generation = {}
    for short in MODELS:
        raw, source = read_artifact(root, root / 'runs/v2' / short / 'responses.jsonl')
        sources.append(source)
        generation[short] = dict(Counter(json.loads(line)['finish_reason'] for line in raw.splitlines()))
        generation[short].setdefault('eos', 0)
        generation[short].setdefault('length', 0)
    accounting = Ledger(root / 'scoring/budget.json').totals()
    result = {'kind': 'aya-six-benchmark-results-v2', 'complete': True, 'verified_at': datetime.now(timezone.utc).isoformat(),
              'v2_inputs': binding, 'metrics': table, 'accounting': accounting, 'artifacts': sources,
              'new_generation_finish_reasons': generation, 'no_assistant_factual_adjudication': True}
    if write:
        result = publish(root, result, render(table, accounting, sources, generation))
    return result


def render_scoped(table, accounting, sources, generation, tasks):
    """Four-benchmark report: fixed tasks actually scored plus SelfAware."""
    def cells(task, key):
        return ' | '.join(percent(table[short][task][key]) for short in MODELS)
    rows = []
    if 'simpleqa_verified' in tasks:
        rows += [
            f'| SimpleQA Verified | Correct-answer yield ↑ | {cells("simpleqa_verified", "correct_answer_yield")} | 1,000 |',
            f'| SimpleQA Verified | Answer coverage | {cells("simpleqa_verified", "answer_coverage")} | 1,000 |',
            f'| SimpleQA Verified | Accuracy when attempted ↑ | {cells("simpleqa_verified", "accuracy_when_attempted")} | 1,000 |']
    if 'hallulens_precisewikiqa' in tasks:
        rows += [
            f'| HalluLens PreciseWikiQA | Hallucination when not abstained ↓ | {cells("hallulens_precisewikiqa", "hallucination_rate_when_not_abstained")} | 1,000 |',
            f'| HalluLens PreciseWikiQA | False refusal ↓ | {cells("hallulens_precisewikiqa", "false_refusal_rate")} | 1,000 |',
            f'| HalluLens PreciseWikiQA | Correct-answer rate ↑ | {cells("hallulens_precisewikiqa", "correct_answer_rate")} | 1,000 |']
    if 'abstention_bench' in tasks:
        rows += [
            f'| AbstentionBench | Behavior accuracy ↑ | {cells("abstention_bench", "behavior_accuracy")} | 978 |',
            f'| AbstentionBench | Joint task success ↑ | {cells("abstention_bench", "joint_task_success")} | 978 |']
    rows += [
        f'| SelfAware | Unknown-label F1 ↑ | {cells("selfaware", "f1")} | 3,369 |',
        f'| SelfAware | Answerable reference accuracy ↑ | {cells("selfaware", "answerable_reference_accuracy")} | 2,338 |']
    lines = ['# Aya baseline benchmarks', '',
        '**Complete:** four benchmarks, 6,347 evaluated rows per model (4,369 new answers + 1,978 reused).', '',
        '| Benchmark | Metric | Tiny Aya Global | Aya Expanse 8B | Rows/model |',
        '|---|---|---:|---:|---:|'] + rows + ['',
        '## Denominators and counts', '',
        '| Count | Tiny | 8B |', '|---|---:|---:|']
    count_rows = []
    if 'simpleqa_verified' in tasks:
        count_rows.append(('SimpleQA correct / incorrect / not attempted', 'simpleqa_verified',
                           ('correct', 'incorrect', 'not_attempted')))
    if 'hallulens_precisewikiqa' in tasks:
        count_rows.append(('PreciseWikiQA non-abstained / hallucinated', 'hallulens_precisewikiqa', ('non_abstained',)))
    if 'abstention_bench' in tasks:
        count_rows.append(('AbstentionBench behavior successes / joint successes', 'abstention_bench',
                           ('behavior_correct', 'joint_successes')))
    count_rows.append(('SelfAware TP / FP / FN', 'selfaware', ('tp', 'fp', 'fn')))
    for label, task, keys in count_rows:
        values = [' / '.join(str(table[short][task][key]) for key in keys) for short in MODELS]
        if task == 'hallulens_precisewikiqa':
            values = [value + ' / ' + str(table[short][task]['incorrect_non_abstained'] + table[short][task]['unverifiable_non_abstained'])
                      for value, short in zip(values, MODELS)]
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'All required answers and judgments reconciled; no missing or failed judgments are silently graded.', '',
        '## Methods and limitations', '',
        '- GLM-5.3-Flash via HF/DeepInfra is the sole factual judge. Greedy judge decoding, low reasoning effort, 2,048-token limit; strict final-label parsing.',
        '- Aya uses pinned native chat templates, FP16, greedy one-pass decoding, no reflection instruction. All model inference and SelfAware embedding computation ran remotely.',
        '- These are disclosed local variants, not leaderboard reproductions. No new human or assistant factual adjudication; published judge-validation numbers do not transfer to GLM.',
        '- HalluLens uses pinned third-party, sample-validated data. Keys are not individually human-verified; screened entities are presumed nonexistent. PreciseWikiQA repairs only the contradictory final Yes/No correctness instruction; incorrect and unverifiable both count as hallucination among non-abstained answers.',
        '- AbstentionBench retains the balanced 489 answerable / 489 should-abstain subset. Correctness is judged only for answerable, non-abstained responses; joint success requires correct answers or appropriate abstention.',
        '- SelfAware retains all 3,369 released rows, including the ID3149 flag discrepancy (2,338 answerable / 1,031 unanswerable). Official phrase-first rules, five-word windows, pinned SimCSE FP32 pooler output, max length 128, strict cosine >0.75; answerable accuracy uses the official case-insensitive reference substring rule.',
        '- Scope: this report covers four benchmarks only. HalluLens NonExistentRefusal and FActScore biographies were dropped after partial generation/evidence work; their preserved artifacts are neither scored nor reported here. TruthfulQA was dropped earlier; LongFact++ remains probe-training data. F/C remains deferred.', '',
        '## Cost and provenance', '',
        f'- GLM token-priced estimate: **${accounting["token_priced_estimates_usd"]}**; provider estimates: **${accounting["provider_estimates_usd"]}** (not invoices).',
        f'- Shared inclusive accounted/reserved total: **${accounting["shared_inclusive_reserved_upper_usd"]} / $18**; remaining conservative headroom: **${accounting["remaining_shared_headroom_usd"]}**. Legacy commitments were preserved, not added twice or released.',
        f'- GLM attempts: {accounting["attempts_reserved"]} reserved, {accounting["attempts_settled"]} settled; unknown v2 reserves: ${accounting["unknown_reserves_usd"]}. Remote Kaggle compute is separate quota, not included as a paid GLM charge.',
        '- Machine-readable metrics and full artifact hashes: `scoring/v2-glm/benchmark-results.json`.', '']
    for source in sources:
        lines.append(f'- `{source["path"]}` — SHA-256 `{source["sha256"]}`')
    lines += ['', '### Generation truncations (retained, not regenerated)', '',
              '| Model | EOS | Token limit |', '|---|---:|---:|']
    for short in MODELS:
        lines.append(f'| {short} | {generation[short]["eos"]} | {generation[short]["length"]} |')
    return '\n'.join(lines) + '\n'


def merge_fixed_results(packets):
    """Merge multiple fixed-benchmark packets, superseding failed results with authorized retries.

    Results are keyed by (aya_model, id, stage). When the same identity appears in multiple
    packets, the one with ok=True wins and the ok=False result is recorded as superseded.
    If two results both have ok=True for the same identity, that is an error (duplicate
    success). The replacement must carry the same request_sha256 as the abandoned attempt,
    enforced by the transport-replacement authorization receipt checked at scorer time.
    """
    by_identity = {}
    superseded = []
    v2_inputs = None
    for packet in packets:
        if v2_inputs is None:
            v2_inputs = packet.get('v2_inputs')
        require(packet.get('v2_inputs') == v2_inputs, 'Merged packets belong to different frozen inputs')
        for outcome in packet['results']:
            lineage = outcome['lineage']
            identity = (lineage['aya_model'], lineage['id'], outcome['stage'])
            existing = by_identity.get(identity)
            if existing is None:
                by_identity[identity] = outcome
                continue
            existing_ok = existing.get('ok') is True
            new_ok = outcome.get('ok') is True
            if existing_ok and new_ok:
                raise ValueError(f'Duplicate successful fixed-benchmark result: {identity}')
            if new_ok and not existing_ok:
                superseded.append({
                    'identity': list(identity),
                    'superseded_attempt_id': existing.get('attempt_id'),
                    'superseded_cache_key': existing.get('cache_key'),
                    'superseded_state': existing.get('state'),
                    'superseded_error': existing.get('error'),
                    'replacement_attempt_id': outcome.get('attempt_id'),
                    'replacement_cache_key': outcome.get('cache_key'),
                    'replacement_label': outcome.get('label'),
                })
                by_identity[identity] = outcome
            # existing ok=True, new ok=False: keep the successful result; the failed duplicate
            # adds no information and is not recorded as superseded (it was never authoritative).
    return {'kind': 'glm-fixed-benchmark-results-v1', 'halted': False, 'v2_inputs': v2_inputs,
            'results': list(by_identity.values()), 'superseded_results': superseded}


def build_scoped(root, fixed_paths, selfaware_path, tasks, write=False):
    """Build only the requested fixed benchmarks plus SelfAware; the six-benchmark path stays intact.

    ``fixed_paths`` may name several verified snapshots. Their judgments are combined only after
    each packet is checked for the same frozen inputs and halted=False; failed identities are
    superseded by authorized replacements (``merge_fixed_results``); every remaining identity is
    then re-verified through its raw journal, so no label is trusted without its HTTP receipt.
    """
    root = Path(root).resolve()
    tasks = tuple(tasks)
    require(tasks and set(tasks) <= set(scoring.FIXED_TASKS), 'Unknown scoped fixed task')
    available, binding = scoring.load_available(root, MODELS)
    inventory = Counter((short, row['dataset']) for row, _, short in available)
    wanted = Counter({(short, task): COUNTS[task] for short in MODELS for task in tasks})
    wanted.update({(short, 'selfaware'): COUNTS['selfaware'] for short in MODELS})
    require(all(inventory[key] == count for key, count in wanted.items()),
            'Cannot report scoped benchmarks before their answers are complete')
    paths = [fixed_paths] if isinstance(fixed_paths, (str, Path)) else list(fixed_paths)
    require(paths, 'Scoped report requires at least one fixed-benchmark snapshot')
    sources, packets = [], []
    for path in paths:
        raw, source = read_artifact(root, path)
        packet = scoring.client.read_json(raw)
        require(packet.get('kind') == 'glm-fixed-benchmark-results-v1' and packet.get('halted') is False,
                'Fixed-benchmark execution is incomplete or halted')
        require(packet.get('v2_inputs') == binding, 'Result snapshot belongs to different frozen evaluation inputs')
        sources.append(source)
        packets.append(packet)
    fixed = merge_fixed_results(packets)
    grades_raw, grades_source = read_artifact(root, selfaware_path)
    grades = [json.loads(line) for line in grades_raw.splitlines()]
    sources.append(grades_source)
    catalog = scoring.SourceCatalog(root)
    table = fixed_table(root, fixed, available, catalog, tasks)
    marker = verify_selfaware_receipt(root, selfaware_path)
    _, marker_source = read_artifact(root, marker)
    sources.append(marker_source)
    self_table = selfaware_table(grades, available)
    for short in MODELS:
        table[short]['selfaware'] = self_table[short]
    keep = set(tasks) | {'selfaware'}
    generation = {}
    for short in MODELS:
        raw, source = read_artifact(root, root / 'runs/v2' / short / 'responses.jsonl')
        sources.append(source)
        reasons = Counter(record['finish_reason'] for record in map(json.loads, raw.splitlines())
                          if record['dataset'] in keep)
        generation[short] = dict(reasons)
        generation[short].setdefault('eos', 0)
        generation[short].setdefault('length', 0)
    accounting = Ledger(root / 'scoring/budget.json').totals()
    result = {'kind': 'aya-four-benchmark-results-v2', 'complete': True,
              'verified_at': datetime.now(timezone.utc).isoformat(), 'reporting_scope': list(tasks) + ['selfaware'],
              'v2_inputs': binding, 'metrics': table, 'accounting': accounting, 'artifacts': sources,
              'new_generation_finish_reasons': generation, 'no_assistant_factual_adjudication': True}
    if write:
        result = publish(root, result, render_scoped(table, accounting, sources, generation, tasks))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--fixed', type=Path, required=True, nargs='+')
    parser.add_argument('--selfaware', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=scoring.FIXED_TASKS)
    parser.add_argument('--factscore', type=Path)
    parser.add_argument('--ready', type=Path)
    parser.add_argument('--retrieval', type=Path)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.tasks:
        print(json_bytes(build_scoped(args.root, args.fixed, args.selfaware, args.tasks, args.write)).decode())
    else:
        if None in (args.factscore, args.ready, args.retrieval):
            parser.error('--factscore, --ready and --retrieval are required without --tasks')
        if len(args.fixed) != 1:
            parser.error('a six-benchmark report takes exactly one fixed snapshot')
        print(json_bytes(build(args.root, args.fixed[0], args.selfaware, args.factscore, args.ready,
                               args.retrieval, args.write)).decode())
