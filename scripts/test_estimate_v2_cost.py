"""Cost arithmetic and destination guards, using only isolated offline fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

import estimate_v2_cost as estimator

TEMP = Path(r'C:\Users\AstralK\AppData\Local\Temp\opencode')


class CostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP, prefix='v2-cost-tests-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        counts = {'simpleqa_verified': 1000, 'abstention_bench': 978,
                  'hallulens_precisewikiqa': 1000, 'hallulens_nonexistentrefusal': 3000,
                  'selfaware': 3369, 'factscore_bios': 180}
        rows = [{'dataset': dataset, 'should_abstain': dataset != 'abstention_bench' or i >= 489}
                for dataset, count in counts.items() for i in range(count)]
        self.write('data/v2/evaluation.jsonl', b''.join((json.dumps(row) + '\n').encode() for row in rows))
        self.write('scoring/budget.json', json.dumps({'ceiling_usd': 18, 'reserved_upper_usd': 5.877233978,
                                                     'reported_cost_usd': 2.303337630}).encode())
        self.write('scoring/v2-glm/preflight/pricing/receipt.json', b'{"files": []}')

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_counts_bio_calls_and_shared_budget_unchanged(self):
        report = estimator.calculate(self.root)
        self.assertEqual(report['total_items_per_model'], 9527)
        self.assertEqual(report['new_answers_per_model'], 7549)
        self.assertEqual(report['reused_answers_per_model'], 1978)
        self.assertTrue(any('360 biographies' in text for text in report['uncertainties']))
        self.assertEqual(report['budget']['conservative_available_usd'], 12.122766022)
        self.assertEqual(report['paid_calls_made'], 0)
        for scenario in report['scenarios']:
            stages = {s['stage']: s for s in scenario['stages']}
            self.assertEqual(stages['factscore_sentence_extraction']['calls_both_models'],
                             360 * scenario['assumptions']['sentences_per_bio'])
            self.assertEqual(stages['factscore_per_fact_verification']['calls_both_models'],
                             360 * scenario['assumptions']['facts_per_bio'])
            self.assertAlmostEqual(scenario['glm_with_50_percent_retries_usd'], scenario['glm_usd'] * 1.5)

    def test_write_guard_rejects_unrecognized_estimate_and_accepts_missing_or_identical(self):
        report = estimator.calculate(self.root)
        path = self.write('scoring/stack-cost-comparison-20261111.json', b'controller-owned estimate')
        self.write('scoring/v2-glm/preflight/approved-as-received-stack-cost-comparison-20261111.json', b'recognized historical estimate')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            estimator.write_report(self.root, report)
        self.assertEqual(path.read_bytes(), b'controller-owned estimate')
        path.unlink()
        estimator.write_report(self.root, report)
        expected = path.read_bytes()
        estimator.write_report(self.root, report)
        self.assertEqual(path.read_bytes(), expected)


if __name__ == '__main__':
    unittest.main()
