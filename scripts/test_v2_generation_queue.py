import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import v2_generation_queue as queue
import v2_kaggle_runner as runner

TEMP = r'C:\Users\AstralK\AppData\Local\Temp\opencode'


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'v2-selfaware-1', 'dataset': 'selfaware', 'component': 'fixture',
                    'question': 'Question?', 'system_prompt': '', 'max_new_tokens': 256}
        self.config = {'model_id': 'CohereLabs/tiny-aya-global',
                       'revision': runner.PINS['CohereLabs/tiny-aya-global'],
                       'input_sha256': runner.payload_hash([self.row]),
                       'chat_template_sha256': 'a' * 64, 'selected_template_sha256': 'b' * 64,
                       'run_name': 'fixture'}
        self.record = {**runner.record_identity(self.config, self.row, 'rendered'),
                       'response': 'Answer', 'status': 'ok', 'finish_reason': 'eos',
                       'input_tokens': 10, 'output_tokens': 20}
        self.job = {'config': self.config, 'examples': [self.row], 'kernel': 'owner/fixture'}

    def test_downloaded_records_require_exact_contract(self):
        queue.validate_output_records([self.record], self.job, complete=True)
        for key, value in [('revision', 'HEAD'), ('input_sha256', 'wrong'),
                           ('dataset', 'other'), ('example_sha256', 'wrong'),
                           ('status', 'context_overflow'), ('response', ''),
                           ('finish_reason', 'wrong'), ('prompt_sha256', 'bad'),
                           ('input_tokens', True), ('output_tokens', 257)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                queue.validate_output_records([{**self.record, key: value}], self.job, complete=True)
        with self.assertRaises(ValueError):
            queue.validate_output_records([self.record, self.record], self.job, complete=True)
        with self.assertRaises(ValueError):
            queue.validate_output_records([], self.job, complete=True)

    def test_aggregation_is_sorted_unique_and_idempotent(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            root = Path(directory)
            queue.aggregate(root, 'tiny', [[self.record]])
            path = root / 'runs/v2/tiny/responses.jsonl'
            content = path.read_bytes()
            self.assertEqual(json.loads(content)['id'], self.row['id'])
            queue.aggregate(root, 'tiny', [[self.record]])
            self.assertEqual(path.read_bytes(), content)
            with self.assertRaises(ValueError):
                queue.aggregate(root, 'tiny', [[self.record], [self.record]])

    def test_per_model_active_job_prevents_second_submission(self):
        self.assertIsNone(queue.choose_next([('a', 'running'), ('b', 'unsubmitted')]))
        self.assertEqual(queue.choose_next([('a', 'complete'), ('b', 'unsubmitted')]), 'b')
        with self.assertRaises(RuntimeError):
            queue.choose_next([('a', 'error'), ('b', 'unsubmitted')])

    def test_runtime_manifest_requires_gpu_and_pinned_library_versions(self):
        manifest = {'config': self.config, 'resolved_revision': self.config['revision'],
                    'model_id': self.config['model_id'], 'weight_dtype': 'float16',
                    'quantization': None, 'stage': 'completed', 'successful': 1, 'errors': 0,
                    'sample_count': 1, 'transformers': '4.57.6', 'accelerate': '1.12.0',
                    'gpu_info': [{'index': 0, 'name': 'Tesla T4'}], 'device_map': {'': '0'},
                    'chat_template_sha256': 'a' * 64, 'selected_template_sha256': 'b' * 64}
        queue.validate_runtime_manifest(manifest, self.job)
        for key, value in [('device_map', {'': 'meta'}), ('device_map', {'': 'mps'}),
                           ('sample_count', 99), ('transformers', 'HEAD'), ('accelerate', 'HEAD')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                queue.validate_runtime_manifest({**manifest, key: value}, self.job)

    def test_validated_partial_outputs_published_before_failed_job_stops_queue(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            root = Path(directory)
            good, bad = root / 'good', root / 'bad'
            for path in (good, bad):
                path.mkdir()
                (path / 'submission-started.json').write_text('{}')
            (bad / 'failed-output').mkdir()
            with patch.object(queue, 'ordered_jobs', return_value=[good, bad]), \
                 patch.object(queue.launch, 'verify_job', return_value=self.job), \
                 patch.object(queue.launch, 'kernel_status', side_effect=['complete', 'error']), \
                 patch.object(queue, 'collect', return_value=[self.record]), \
                 self.assertRaises(RuntimeError):
                queue.tick(root)
            self.assertEqual(json.loads((root / 'runs/v2/tiny/responses.jsonl').read_bytes())['id'], self.row['id'])


if __name__ == '__main__':
    unittest.main()
