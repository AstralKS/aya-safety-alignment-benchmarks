"""Packaging tests only; no remote calls or model imports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch_v2 as launch
import v2_kaggle_runner as runner

TEMP = r'C:\Users\AstralK\AppData\Local\Temp\opencode'


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'id': f'v2-selfaware-{i}', 'dataset': 'selfaware',
                      'component': 'fixture', 'question': f'Question {i}?',
                      'system_prompt': '', 'max_new_tokens': 256} for i in range(8)]
        self.model = {'model_id': 'CohereLabs/tiny-aya-global',
                      'revision': runner.PINS['CohereLabs/tiny-aya-global'],
                      'chat_template': {'sha256': 'a' * 64, 'selected_template_sha256': 'b' * 64},
                      'eos_token_id': 3, 'pad_token_id': 0}

    def test_shards_preserve_every_id_and_exact_prompt(self):
        jobs = launch.make_jobs('tiny', self.model, self.rows, 3, 'c' * 64, 'd' * 64)
        self.assertEqual([len(j['examples']) for j in jobs], [3, 3, 2])
        self.assertEqual([r for j in jobs for r in j['examples']], self.rows)
        for job in jobs:
            runner.validate_config(job['config'], job['examples'])
            self.assertEqual(job['config']['revision'], self.model['revision'])
        self.assertEqual(jobs, launch.make_jobs('tiny', self.model, self.rows, 3, 'c' * 64, 'd' * 64))

    def test_reference_leak_is_rejected(self):
        with self.assertRaises(ValueError):
            launch.make_jobs('tiny', self.model, [{**self.rows[0], 'answer': 'gold'}], 3, 'c' * 64, 'd' * 64)

    def test_write_once_credential_free_notebook(self):
        job = launch.make_jobs('tiny', self.model, self.rows, 3, 'c' * 64, 'd' * 64)[0]
        source = Path(runner.__file__).read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            path = launch.write_job(Path(directory), job, source)
            launch.verify_job(path)
            notebook = json.loads((path / 'run.ipynb').read_text(encoding='utf-8'))
            self.assertIn('UserSecretsClient', ''.join(notebook['cells'][1]['source']))
            with self.assertRaises(FileExistsError):
                launch.write_job(Path(directory), job, source)
            manifest = json.loads((path / 'job.json').read_bytes())
            manifest['config']['revision'] = 'HEAD'
            (path / 'job.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                launch.verify_job(path)

    def test_status_inspection_is_not_submission(self):
        with patch.object(launch.subprocess, 'run') as call:
            call.return_value.returncode = 0
            call.return_value.stdout = 'KernelWorkerStatus.COMPLETE'
            call.return_value.stderr = ''
            self.assertEqual(launch.kernel_status('owner/job'), 'complete')
            self.assertEqual(call.call_args.args[0][:3], ['kaggle', 'kernels', 'status'])


if __name__ == '__main__':
    unittest.main()
