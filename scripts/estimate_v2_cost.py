"""Reproducible planning scenarios only; no inference, credentials or ledger edits."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICE = Decimal('0.15')
OUTPUT_PRICE = Decimal('0.50')
MILLION = Decimal(1_000_000)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def calculate(root):
    evaluation = root/'data/v2/evaluation.jsonl'
    rows = [json.loads(line) for line in evaluation.read_text(encoding='utf-8').splitlines()]
    counts = Counter(r['dataset'] for r in rows)
    assert len(rows) == 9527 and counts['factscore_bios'] == 180
    answerable = sum(r['dataset'] == 'abstention_bench' and r['should_abstain'] is False for r in rows)
    assert answerable == 489
    ledger_path = root/'scoring/budget.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'), parse_float=Decimal)
    headroom = ledger['ceiling_usd'] - ledger['reserved_upper_usd']
    price_receipt = root/'scoring/v2-glm/preflight/pricing/receipt.json'
    sources = json.loads(price_receipt.read_text(encoding='utf-8'))
    for item in sources['files']:
        data = (root/item['path']).read_bytes()
        if sha(data) != item['sha256'] or len(data) != item['bytes']:
            raise ValueError(f'Pricing source changed: {item["path"]}')

    # Average input allowances, not measured GLM token counts. Include prompts and
    # Aya text; e.g. SimpleQA rubric 7,125 chars, precise abstention 4,868 chars.
    fixed = [
        ('simpleqa_autorater', 2*counts['simpleqa_verified'], 2200),
        ('precisewikiqa_abstention', 2*counts['hallulens_precisewikiqa'], 1800),
        ('precisewikiqa_correctness_all_responses', 2*counts['hallulens_precisewikiqa'], 600),
        ('nonexistent_mixed', 2*2000, 1400),
        ('nonexistent_generated', 2*1000, 1000),
        ('abstention_classifier', 2*counts['abstention_bench'], 2200),
        ('abstention_correctness_maximum', 2*answerable, 1200),
    ]
    scenarios = [
        {'name': 'compact', 'sentences_per_bio': 6, 'facts_per_bio': 20,
         'label_output_tokens': 32, 'extraction_output_tokens': 128,
         'extraction_input_tokens': 1200, 'fact_input_tokens': 1600, 'fixed_input_multiplier': 1},
        {'name': 'planning', 'sentences_per_bio': 12, 'facts_per_bio': 40,
         'label_output_tokens': 64, 'extraction_output_tokens': 192,
         'extraction_input_tokens': 1200, 'fact_input_tokens': 1600, 'fixed_input_multiplier': 1},
        {'name': 'long_reasoning_stress', 'sentences_per_bio': 24, 'facts_per_bio': 80,
         'label_output_tokens': 512, 'extraction_output_tokens': 768,
         'extraction_input_tokens': 1600, 'fact_input_tokens': 2200, 'fixed_input_multiplier': 1.25},
    ]
    results = []
    for scenario in scenarios:
        stages = []
        def stage(name, calls, input_tokens, output_tokens):
            input_total = calls*input_tokens
            output_total = calls*output_tokens
            cost = (Decimal(input_total)*INPUT_PRICE + Decimal(output_total)*OUTPUT_PRICE)/MILLION
            stages.append({'stage': name, 'calls_both_models': calls,
                           'average_input_tokens': input_tokens, 'average_billed_output_tokens': output_tokens,
                           'input_tokens_total': input_total, 'output_tokens_total': output_total,
                           'usd': float(cost)})
        for name, calls, allowance in fixed:
            stage(name, calls, int(allowance*scenario['fixed_input_multiplier']), scenario['label_output_tokens'])
        bios = 2*counts['factscore_bios']
        stage('factscore_sentence_extraction', bios*scenario['sentences_per_bio'],
              scenario['extraction_input_tokens'], scenario['extraction_output_tokens'])
        stage('factscore_per_fact_verification', bios*scenario['facts_per_bio'],
              scenario['fact_input_tokens'], scenario['label_output_tokens'])
        total = sum(Decimal(str(s['usd'])) for s in stages)
        retry = total*Decimal('1.5')
        results.append({'assumptions': scenario, 'stages': stages,
                        'calls_both_models_before_retries': sum(s['calls_both_models'] for s in stages),
                        'input_tokens_total': sum(s['input_tokens_total'] for s in stages),
                        'billed_output_tokens_total': sum(s['output_tokens_total'] for s in stages),
                        'glm_usd': float(total), 'glm_with_50_percent_retries_usd': float(retry),
                        'headroom_after_glm_and_retries_usd': float(headroom-retry)})
    return {
        'status': 'revised planning scenarios, not a full-run quote or spending authorization',
        'judge': 'zai-org/GLM-5.3-Flash:deepinfra',
        'pricing_usd_per_million': {'input': float(INPUT_PRICE), 'output_including_reasoning': float(OUTPUT_PRICE)},
        'pricing_source_receipt_sha256': sha(price_receipt.read_bytes()),
        'evaluation_sha256': sha(evaluation.read_bytes()),
        'total_items_per_model': len(rows), 'new_answers_per_model': 7549, 'reused_answers_per_model': 1978,
        'fixed_judge_calls_both_models': 13956,
        'additional_abstention_correctness_calls_range_both_models': [0, 2*answerable],
        'budget': {'legacy_ledger_sha256': sha(ledger_path.read_bytes()),
                   'reported_usd': float(ledger['reported_cost_usd']),
                   'reserved_inclusive_usd': float(ledger['reserved_upper_usd']),
                   'combined_ceiling_usd': float(ledger['ceiling_usd']),
                   'conservative_available_usd': float(headroom),
                   'reported_is_not_added_again_to_reserved': True},
        'scenarios': results,
        'uncertainties': [
            'Input allowances are planning averages, not GLM-tokenized prompts or hard bounds; no cache or credit discount assumed.',
            'Output allowances include any billed reasoning tokens; actual behavior, disabling reasoning, usage reporting and parse success require a pilot.',
            'All 360 biographies are budgeted as fact-bearing; facts/sentences are not yet observed. No truncation or fact dropping to fit the estimate.',
            'Seven fixed FActScore demonstrations total3291 characters; eighth ranges209..730, before new sentence/chat framing.1200 input tokens is an allowance, not measured usage.',
            'Verification allowance includes five up-to256-token Wikipedia passages plus claim/framing. Tokenizers differ; evidence failures are not unsupported-fact labels.',
            'SelfAware hosted hf-inference bills compute time times hardware rate, not a published per-token tariff. Pooling, window count, latency and hardware cost remain untested.',
            'If exact SimCSE and GTR retrieval run within free Kaggle quota, their billed API cost is zero; quota/runtime have not been verified. No paid endpoint provisioned.',
            'Aya generation is planned on free remote Kaggle GPU quota; no dollar quote or guarantee of available GPU hours.',
            'The planning case plus50% retries exceeds conservative headroom even before hosted embeddings. Budget fit is NOT established.',
        ],
        'paid_calls_made': 0,
        'next_gate': 'Present scenarios; implement shared-ceiling accounting and a small bounded pilot before bulk paid execution; re-estimate from real token/fact/window counts.',
    }


def write_report(root, report):
    """Write only a missing, identical, or explicitly recognized historical output."""
    output = (json.dumps(report, indent=2) + '\n').encode('utf-8')
    path = root/'scoring/stack-cost-comparison-20261111.json'
    historical = root/'scoring/v2-glm/preflight/approved-as-received-stack-cost-comparison-20261111.json'
    if path.exists() and path.read_bytes() != output:
        if not historical.is_file() or path.read_bytes() != historical.read_bytes():
            raise ValueError('Unexpected existing estimate; refusing to overwrite')
    path.write_bytes(output)
    return {'saved': str(path), 'sha256': sha(output),
            'scenarios': [{k: s[k] for k in ['glm_usd', 'glm_with_50_percent_retries_usd']} for s in report['scenarios']]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Replace only the preserved obsolete estimate or identical output.')
    args = parser.parse_args()
    report = calculate(ROOT)
    output = (json.dumps(report, indent=2) + '\n').encode('utf-8')
    if args.write:
        print(json.dumps(write_report(ROOT, report), indent=2))
    else:
        print(output.decode('utf-8'), end='')
