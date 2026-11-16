import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import v2_benchmark_report as report
import v2_selfaware as selfaware


class ReportTests(unittest.TestCase):
    def test_report_publication_resumes_machine_only_checkpoint_without_overwrite(self):
        with tempfile.TemporaryDirectory(dir=r'C:\Users\AstralK\AppData\Local\Temp\opencode') as directory:
            root = Path(directory)
            (root / 'scoring/v2-glm').mkdir(parents=True)
            first = {'kind': 'aya-six-benchmark-results-v2', 'complete': True, 'verified_at': 'original', 'metrics': {'fixture': 0.1}}
            path = root / 'scoring/v2-glm/benchmark-results.json'
            path.write_bytes(report.json_bytes(first))
            raw = path.read_bytes()
            result = report.publish(root, {**first, 'verified_at': 'later'}, '# Actual results\n')
            self.assertEqual(result, first)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual((root / 'benchmark.md').read_text(), '# Actual results\n')
            report.publish(root, first, '# Actual results\n')
            with self.assertRaises(ValueError):
                report.publish(root, {**first, 'metrics': {'fixture': 2}}, '# Changed results\n')

    def test_incomplete_inventory_cannot_write_or_read_result_files(self):
        with patch.object(report.scoring, 'load_available', return_value=([], {})), \
             patch.object(report, 'read_artifact') as read, self.assertRaisesRegex(ValueError, '9,527'):
            report.build(report.ROOT, 'fixed', 'selfaware', 'facts', 'ready', 'retrieval', write=True)
        read.assert_not_called()

    def test_no_fabricated_selfaware_grade_or_altered_threshold(self):
        row = {'id': 'fixture', 'dataset': 'selfaware', 'metadata': {'answerable': False, 'answer': None}}
        available = [(row, 'The answer is unknown.', 'tiny')]
        grade = {**selfaware.classify(available[0][1], lambda *_: 0), 'id': row['id'],
                 'model_id': report.scoring.client.AYA_MODELS['tiny'], 'response_sha256': report.sha(available[0][1].encode()),
                 'answerable': False, 'answerable_reference_correct': None, 'status': 'ok',
                 'embedding_model': selfaware.MODEL, 'embedding_revision': selfaware.REVISION}
        for changed in ({**grade, 'pred_unanswerable': False}, {**grade, 'response_sha256': 'wrong'},
                        {**grade, 'embedding_revision': 'HEAD'}):
            with self.assertRaises(ValueError):
                report.selfaware_table([changed], available)

    def test_unknown_fixed_packet_rejected(self):
        with self.assertRaises(ValueError):
            report.fixed_table(report.ROOT, {'kind': 'historical-grades', 'halted': False}, [], None)

    def test_selfaware_report_rejects_an_unreceipted_result_file(self):
        with tempfile.TemporaryDirectory(dir=r'C:\Users\AstralK\AppData\Local\Temp\opencode') as directory:
            root = Path(directory)
            path = root / 'invented-grades.jsonl'
            path.write_text('{}\n')
            with self.assertRaises(ValueError):
                report.verify_selfaware_receipt(root, path)

    def test_swapped_fact_id_cannot_bind_support_to_another_biography(self):
        fact = {'fact_id': 'foreign', 'text': 'Same fact.', 'sentence_index': 0, 'fact_index': 0,
                'aya_model': 'tiny', 'id': 'bio-one', 'topic': 'One', 'answer_sha256': 'a'}
        lineage = {'aya_model': 'tiny', 'id': 'bio-two', 'fact': {**fact, 'id': 'bio-two', 'topic': 'Two', 'answer_sha256': 'b'}}
        outcome = {'lineage': lineage, 'cache_key': 'c'}
        with self.assertRaises(ValueError):
            report.bind_biography_fact(fact, ('tiny', 'bio-one'), 'One', 'a', outcome)

    def test_scoped_report_requires_complete_selected_answers(self):
        with patch.object(report.scoring, 'load_available', return_value=([], {})), \
             patch.object(report, 'read_artifact') as read, self.assertRaisesRegex(ValueError, 'scoped benchmarks'):
            report.build_scoped(report.ROOT, 'fixed', 'selfaware', ('simpleqa_verified',))
        read.assert_not_called()

    def test_scoped_fixed_table_rejects_unselected_benchmark_judgments(self):
        row = {'id': 'q1', 'dataset': 'simpleqa_verified', 'metadata': {}}
        available = [(row, 'answer', 'tiny')]
        job = {'stage': 'simpleqa', 'cache_key': 'k', 'request_sha256': 'r'}
        packet = {'kind': 'glm-fixed-benchmark-results-v1', 'halted': False, 'results': [
            {'lineage': {'aya_model': 'tiny', 'id': 'q1'}, 'stage': 'simpleqa', 'cache_key': 'k'},
            {'lineage': {'aya_model': 'tiny', 'id': 'dropped'}, 'stage': 'does_believe', 'cache_key': 'x'}]}
        with patch.object(report, 'MODELS', ('tiny',)), \
             patch.dict(report.COUNTS, {'simpleqa_verified': 1}, clear=False), \
             patch.object(report.scoring, 'fixed_jobs', return_value=[job]), \
             patch.object(report, 'journal_label', return_value='A'):
            with self.assertRaisesRegex(ValueError, 'Extra or ineligible'):
                report.fixed_table(report.ROOT, packet, available, None, tasks=('simpleqa_verified',))

    def test_duplicate_judgment_across_snapshots_rejected(self):
        outcome = {'lineage': {'aya_model': 'tiny', 'id': 'q1'}, 'stage': 'simpleqa', 'cache_key': 'k'}
        packet = {'kind': 'glm-fixed-benchmark-results-v1', 'halted': False, 'results': [outcome, dict(outcome)]}
        with self.assertRaisesRegex(ValueError, 'Duplicate fixed-benchmark result identity'):
            report.fixed_table(report.ROOT, packet, [], None, tasks=('simpleqa_verified',))


if __name__ == '__main__':
    unittest.main()
