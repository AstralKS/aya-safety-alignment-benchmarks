"""Offline validation tests: never import a model or submit a notebook."""
import copy
import unittest

import v2_kaggle_runner as runner


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.example = {'id': 'v2-selfaware-1', 'dataset': 'selfaware',
                        'component': 'fixture', 'question': 'Question?',
                        'system_prompt': '', 'max_new_tokens': 256}
        self.config = {
            'model_id': 'CohereLabs/tiny-aya-global',
            'revision': 'af89d219b53ed9b13b8a4645f9c8028973510324',
            'chat_template_sha256': 'a' * 64, 'selected_template_sha256': 'b' * 64,
            'eos_token_id': 3, 'pad_token_id': 0, 'run_name': 'fixture',
            'input_sha256': runner.payload_hash([self.example]),
        }
        self.record = {
            **runner.record_identity(self.config, self.example, 'rendered'),
            'status': 'ok', 'response': 'An answer', 'input_tokens': 10,
            'output_tokens': 20, 'finish_reason': 'eos',
        }

    def test_generation_allowlist_and_empty_system(self):
        runner.validate_examples([self.example])
        for key, value in [('answer', 'leaked'), ('reference', 'leaked')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_examples([{**self.example, key: value}])
        with self.assertRaises(ValueError):
            runner.validate_examples([{**self.example, 'system_prompt': 'Reflect first'}])

    def test_no_duplicate_ids_or_changed_token_contract(self):
        with self.assertRaises(ValueError):
            runner.validate_examples([self.example, self.example])
        with self.assertRaises(ValueError):
            runner.validate_examples([{**self.example, 'max_new_tokens': 512}])

    def test_checkpoint_revision_and_payload_are_pinned(self):
        runner.validate_config(self.config, [self.example])
        with self.assertRaises(ValueError):
            runner.validate_config({**self.config, 'revision': 'HEAD'}, [self.example])
        with self.assertRaises(ValueError):
            runner.validate_config(self.config, [{**self.example, 'question': 'Changed'}])

    def test_resume_requires_exact_prompt_metadata_and_success(self):
        rendered = {self.example['id']: (self.example, 'rendered', 10)}
        self.assertEqual(runner.validate_records([self.record], self.config, rendered),
                         {self.example['id']})
        for key, value in [('revision', 'wrong'), ('model_id', 'wrong'),
                           ('dataset', 'wrong'), ('component', 'wrong'),
                           ('prompt_sha256', 'wrong'), ('example_sha256', 'wrong'),
                           ('input_sha256', 'wrong'), ('max_new_tokens', 1),
                           ('status', 'context_overflow'), ('response', ''),
                           ('finish_reason', 'wrong'), ('input_tokens', True),
                           ('output_tokens', 999)]:
            record = copy.deepcopy(self.record)
            record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_records([record], self.config, rendered)
        with self.assertRaises(ValueError):
            runner.validate_records([self.record, self.record], self.config, rendered)

    def test_errors_do_not_report_successful_completion(self):
        self.assertEqual(runner.completion_state([self.record], 1)['stage'], 'completed')
        state = runner.completion_state([{'status': 'context_overflow'}], 1)
        self.assertEqual(state['stage'], 'completed_with_errors')
        self.assertEqual(state['successful'], 0)
        self.assertEqual(state['errors'], 1)
        self.assertEqual(runner.completion_state([], 1)['stage'], 'incomplete')

    def test_remote_guard_precedes_model_import(self):
        with self.assertRaisesRegex(RuntimeError, 'Remote-only'):
            runner.run(self.config, [self.example])


if __name__ == '__main__':
    unittest.main()
