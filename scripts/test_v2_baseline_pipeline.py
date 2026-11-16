from pathlib import Path
from contextlib import ExitStack
import tempfile
import unittest
from unittest.mock import patch

import v2_baseline_pipeline as pipeline

TEMP = r'C:\Users\AstralK\AppData\Local\Temp\opencode'


class PipelineTests(unittest.TestCase):
    def test_fingerprint_changes_only_for_scored_answers(self):
        row = {'id': 'fixed', 'dataset': 'simpleqa_verified', '_input_row_sha256': 'a'}
        first = [(row, 'answer', 'tiny')]
        self.assertEqual(pipeline.source_fingerprint(first, pipeline.scoring.FIXED_TASKS),
                         pipeline.source_fingerprint(first + [({**row, 'id': 'self', 'dataset': 'selfaware'}, 'extra', 'tiny')], pipeline.scoring.FIXED_TASKS))
        self.assertNotEqual(pipeline.source_fingerprint(first, pipeline.scoring.FIXED_TASKS),
                            pipeline.source_fingerprint([(row, 'changed', 'tiny')], pipeline.scoring.FIXED_TASKS))

    def test_missing_generation_state_never_assumes_free_gpu_slot(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            self.assertEqual(pipeline.active_generation(Path(directory)), 2)

    def test_checkpoint_is_atomic_and_restart_keeps_stage_state(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            root = Path(directory)
            (root / 'runs/v2').mkdir(parents=True)
            state = pipeline.load_state(root)
            state['factscore']['extractions'] = 'saved.json'
            pipeline.checkpoint(root, state)
            self.assertEqual(pipeline.load_state(root)['factscore']['extractions'], 'saved.json')

    def test_unresolved_grades_cannot_be_silently_completed(self):
        for packet in ({'halted': True, 'results': []}, {'halted': False, 'results': [{'ok': False, 'state': 'unknown'}]}):
            with patch.object(Path, 'read_bytes', return_value=pipeline.json_bytes(packet)), self.assertRaises(ValueError):
                pipeline.require_usable_result(pipeline.ROOT, 'snapshot.json')


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / 'runs/v2').mkdir(parents=True)
        self.binding = {'fixture': 'frozen-inputs'}
        self.available = self.rows({'simpleqa_verified': 1})
        self.state = pipeline.load_state(self.root)
        self.fixed = self.save('fixed.json', {'halted': False, 'results': []})
        self.state['fixed'] = {'snapshot': 'fixed.json',
                              'source_fingerprint': pipeline.source_fingerprint(self.available, pipeline.scoring.FIXED_TASKS)}
        self.save('runs/v2/generation-progress.json', {'models': {'tiny': {'jobs': []}, '8b': {'jobs': []}}})
        mocks = ExitStack()
        self.addCleanup(mocks.close)
        self.load_available = mocks.enter_context(patch.object(pipeline.scoring, 'load_available',
            side_effect=lambda *_: (self.available, self.binding)))
        self.score_fixed = mocks.enter_context(patch.object(pipeline.scoring, 'score_fixed',
            return_value={'snapshot': str(self.fixed), 'halted': False}))
        self.factscore_stage = mocks.enter_context(patch.object(pipeline.scoring, 'factscore_stage'))
        self.advance_compute = mocks.enter_context(patch.object(pipeline, 'advance_compute', return_value=None))
        mocks.enter_context(patch.object(pipeline.compute, 'selfaware_inputs', return_value={}))
        self.ledger = mocks.enter_context(patch.object(pipeline, 'Ledger'))
        self.ledger.return_value.totals.return_value = {'shared_inclusive_reserved_upper_usd': '5.877233978'}

    def save(self, relative, packet):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pipeline.json_bytes(packet))
        return path

    def rows(self, counts):
        return [({'id': task + '-' + str(index), 'dataset': task, '_input_row_sha256': 'fixture'}, 'answer', short)
                for short in ('tiny', '8b') for task, count in counts.items() for index in range(count)]

    def extraction(self, completed=True):
        return {'kind': 'glm-factscore-extractions-v1', 'v2_inputs': self.binding,
                'responses': [{'aya_model': short, 'id': 'factscore_bios-' + str(index), 'completed': completed,
                               'sentences': [], 'facts': [] if completed else None}
                              for short in ('tiny', '8b') for index in range(180)]}

    def saved_factscore(self, extraction=None, verification=None):
        self.save('extract.json', self.extraction() if extraction is None else extraction)
        self.save('ready.json', {'kind': 'glm-factscore-retrieval-ready-v1'})
        self.state['factscore'].update(extractions='extract.json', ready='ready.json')
        if verification is not None:
            self.save('verify.json', verification)
            self.state['factscore']['verification'] = 'verify.json'

    def assert_no_paid_work(self):
        self.score_fixed.assert_not_called()
        self.factscore_stage.assert_not_called()
        self.advance_compute.assert_not_called()

    def completed_state(self):
        self.saved_factscore(verification={'halted': False, 'results': []})
        self.state.update(complete=True, report='benchmark.md')
        self.state['compute'] = {
            'selfaware': {'receipt': {'result_path': 'selfaware/grades.jsonl'}},
            'factscore': {'receipt': {'result_path': 'factscore/retrieval.jsonl'}}}
        pipeline.checkpoint(self.root, self.state)
        compiled = {'kind': 'aya-six-benchmark-results-v2', 'complete': True, 'verified_at': 'new-verification',
                    'metrics': {}, 'accounting': {}, 'artifacts': [], 'new_generation_finish_reasons': {}}
        self.save('scoring/v2-glm/benchmark-results.json', {**compiled, 'verified_at': 'original-publication'})
        (self.root / 'benchmark.md').write_bytes(b'verified markdown\n')
        return compiled

    def test_failed_saved_fixed_snapshot_blocks_changed_fingerprint_before_all_paid_work(self):
        self.available = self.rows({'simpleqa_verified': 2, 'selfaware': 3369})
        next_snapshot = self.save('next-fixed.json', {'halted': False, 'results': []})
        self.score_fixed.return_value = {'snapshot': str(next_snapshot), 'halted': False}
        for packet in ({'halted': True, 'results': []},
                       {'halted': False, 'results': [{'ok': False, 'state': 'unknown'}]}):
            with self.subTest(packet=packet):
                self.score_fixed.reset_mock()
                self.advance_compute.reset_mock()
                self.save('fixed.json', packet)
                self.state['fixed'] = {'snapshot': 'fixed.json', 'source_fingerprint': 'prior-answers'}
                pipeline.checkpoint(self.root, self.state)
                progress = self.root / 'runs/v2/baseline-pipeline-progress.json'
                before_progress = progress.read_bytes()
                before_snapshot = self.fixed.read_bytes()
                restarted = pipeline.load_state(self.root)
                before_state = pipeline.json_bytes(restarted)
                with self.assertRaises(ValueError):
                    pipeline.tick(self.root, restarted)
                self.assert_no_paid_work()
                self.assertEqual(pipeline.json_bytes(restarted), before_state)
                self.assertEqual(progress.read_bytes(), before_progress)
                self.assertEqual(self.fixed.read_bytes(), before_snapshot)
                self.ledger.assert_not_called()

    def test_fresh_run_without_fixed_snapshot_still_scores_initial_answers(self):
        self.state['fixed'] = {}
        result = pipeline.tick(self.root, self.state)
        self.score_fixed.assert_called_once_with(self.root, pipeline.RUN_ID, dispatch=True)
        self.assertEqual(result['fixed']['snapshot'], 'fixed.json')
        self.assertEqual(result['fixed']['source_fingerprint'],
                         pipeline.source_fingerprint(self.available, pipeline.scoring.FIXED_TASKS))
        self.factscore_stage.assert_not_called()
        self.advance_compute.assert_not_called()
        self.ledger.assert_not_called()

    def test_persisted_incomplete_extraction_blocks_all_paid_work_after_restart(self):
        self.available = self.rows({'factscore_bios': 180, 'selfaware': 3369, 'simpleqa_verified': 1})
        self.saved_factscore(extraction=self.extraction(False))
        self.state['fixed']['source_fingerprint'] = 'changed'
        pipeline.checkpoint(self.root, self.state)
        before = (self.root / 'extract.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Atomic extraction is incomplete'):
            pipeline.tick(self.root, pipeline.load_state(self.root))
        self.assert_no_paid_work()
        self.assertEqual((self.root / 'extract.json').read_bytes(), before)

    def test_new_failed_extraction_pointer_is_preserved_but_never_retried_on_restart(self):
        self.available = self.rows({'factscore_bios': 180})
        self.state['fixed']['source_fingerprint'] = pipeline.source_fingerprint(self.available, pipeline.scoring.FIXED_TASKS)
        extraction = self.save('extract.json', self.extraction(False))
        self.factscore_stage.return_value = {'snapshot': str(extraction), 'halted': False}
        with self.assertRaisesRegex(ValueError, 'Atomic extraction is incomplete'):
            pipeline.tick(self.root, self.state)
        restarted = pipeline.load_state(self.root)
        self.assertEqual(restarted['factscore']['extractions'], 'extract.json')
        self.factscore_stage.reset_mock()
        with self.assertRaisesRegex(ValueError, 'Atomic extraction is incomplete'):
            pipeline.tick(self.root, restarted)
        self.assert_no_paid_work()

    def test_halted_extraction_with_all_completed_responses_stays_blocked_on_restart(self):
        self.available = self.rows({'factscore_bios': 180})
        self.state['fixed']['source_fingerprint'] = pipeline.source_fingerprint(self.available, pipeline.scoring.FIXED_TASKS)
        extraction = self.save('extract.json', self.extraction())
        before = extraction.read_bytes()
        self.factscore_stage.return_value = {'snapshot': str(extraction), 'halted': True}
        with self.assertRaisesRegex(ValueError, 'Atomic extraction is incomplete'):
            pipeline.tick(self.root, self.state)
        self.factscore_stage.assert_called_once_with(self.root, pipeline.RUN_ID + '-facts', 'extract', dispatch=True)
        self.score_fixed.assert_not_called()
        self.advance_compute.assert_not_called()
        restarted = pipeline.load_state(self.root)
        self.factscore_stage.reset_mock()
        self.available = self.rows({'factscore_bios': 180, 'selfaware': 3369, 'simpleqa_verified': 1})
        restarted['fixed']['source_fingerprint'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Atomic extraction is incomplete'):
            pipeline.tick(self.root, restarted)
        self.assert_no_paid_work()
        self.assertEqual(restarted['factscore']['extractions'], 'extract.json')
        self.assertIs(restarted['factscore']['extraction_halted'], True)
        self.assertEqual(extraction.read_bytes(), before)
        self.ledger.assert_not_called()

    def test_extraction_response_count_and_strict_completion_are_restart_barriers(self):
        for mutation in ('missing', 'truthy-not-true'):
            with self.subTest(mutation=mutation):
                packet = self.extraction()
                if mutation == 'missing':
                    packet['responses'].pop()
                else:
                    packet['responses'][0]['completed'] = 'unknown'
                self.saved_factscore(extraction=packet)
                with self.assertRaisesRegex(ValueError, 'Atomic extraction is incomplete'):
                    pipeline.tick(self.root, self.state)
        self.assert_no_paid_work()

    def test_persisted_failed_verification_blocks_report_and_other_paid_work(self):
        self.available = self.rows(pipeline.report.COUNTS)
        self.state['fixed']['source_fingerprint'] = 'changed'
        for packet in ({'halted': True, 'results': []},
                       {'halted': False, 'results': [{'ok': False, 'state': 'unknown'}]}):
            with self.subTest(packet=packet), patch.object(pipeline.report, 'build') as build:
                self.saved_factscore(verification=packet)
                pipeline.checkpoint(self.root, self.state)
                with self.assertRaises(ValueError):
                    pipeline.tick(self.root, pipeline.load_state(self.root))
                self.assert_no_paid_work()
                build.assert_not_called()

    def test_new_failed_verification_is_not_skipped_on_restart(self):
        self.available = self.rows({'factscore_bios': 180})
        self.state['fixed']['source_fingerprint'] = pipeline.source_fingerprint(self.available, pipeline.scoring.FIXED_TASKS)
        self.saved_factscore()
        verification = self.save('verify.json', {'halted': False, 'results': [{'ok': False, 'state': 'invalid_label'}]})
        self.factscore_stage.return_value = {'snapshot': str(verification), 'halted': False}
        self.advance_compute.return_value = {'result_path': 'retrieval.jsonl'}
        with self.assertRaisesRegex(ValueError, 'unresolved judgments'):
            pipeline.tick(self.root, self.state)
        restarted = pipeline.load_state(self.root)
        self.assertEqual(restarted['factscore']['verification'], 'verify.json')
        self.factscore_stage.reset_mock()
        self.advance_compute.reset_mock()
        with self.assertRaisesRegex(ValueError, 'unresolved judgments'):
            pipeline.tick(self.root, restarted)
        self.assert_no_paid_work()

    def test_valid_extraction_can_advance_without_repeating_extraction(self):
        self.available = self.rows({'factscore_bios': 180})
        self.state['fixed']['source_fingerprint'] = pipeline.source_fingerprint(self.available, pipeline.scoring.FIXED_TASKS)
        self.saved_factscore()
        pipeline.tick(self.root, self.state)
        self.factscore_stage.assert_not_called()
        self.advance_compute.assert_called_once()

    def test_generation_failure_before_selfaware_readiness_is_recorded_before_paid_work(self):
        for status in ('error', 'cancel'):
            with self.subTest(status=status):
                self.state['fixed']['source_fingerprint'] = 'changed'
                pipeline.checkpoint(self.root, self.state)
                self.save('runs/v2/generation-progress.json', {'models': {'tiny': {'jobs': [{'state': status}]}}})
                with patch.object(pipeline.time, 'sleep', side_effect=AssertionError('Failure must not wait')):
                    with self.assertRaisesRegex(ValueError, 'Generation failed') as failure:
                        pipeline.run(self.root)
                saved = pipeline.load_state(self.root)
                self.assertEqual(saved['error']['type'], type(failure.exception).__name__)
                self.assertIn('Generation failed', saved['error']['message'])
                self.assert_no_paid_work()
                self.load_available.assert_not_called()
                self.assertFalse((self.root / 'runs/v2/baseline-pipeline.guard').exists())

    def test_completed_state_identity_is_checked_before_report_build(self):
        self.completed_state()
        for field in ('kind', 'run_id'):
            with self.subTest(field=field), patch.object(pipeline.report, 'build') as build:
                pipeline.checkpoint(self.root, {**self.state, field: 'unfamiliar'})
                with self.assertRaisesRegex(ValueError, 'Unexpected pipeline state'):
                    pipeline.run(self.root)
                build.assert_not_called()
                self.assert_no_paid_work()

    def test_completed_restart_rebuilds_read_only_and_accepts_only_timestamp_difference(self):
        compiled = self.completed_state()
        with patch.object(pipeline.report, 'build', return_value=compiled) as build, \
                patch.object(pipeline.report, 'render', return_value='verified markdown\n'):
            self.assertTrue(pipeline.run(self.root)['complete'])
        build.assert_called_once_with(self.root, self.fixed, self.root / 'selfaware/grades.jsonl',
            self.root / 'verify.json', self.root / 'ready.json', self.root / 'factscore/retrieval.jsonl', write=False)
        self.assert_no_paid_work()
        self.ledger.assert_not_called()

    def test_completed_restart_rejects_incomplete_saved_inventory_using_report_builder(self):
        self.completed_state()
        with self.assertRaisesRegex(ValueError, 'all 9,527 answers/model'):
            pipeline.run(self.root)
        self.assert_no_paid_work()

    def test_completed_restart_propagates_missing_source_or_receipt_validation(self):
        self.completed_state()
        for message in ('Missing saved GLM source', 'Missing collected remote output receipt'):
            with self.subTest(message=message), patch.object(pipeline.report, 'build', side_effect=ValueError(message)) as build:
                with self.assertRaisesRegex(ValueError, message):
                    pipeline.run(self.root)
                self.assertEqual(build.call_args.kwargs, {'write': False})
                self.assert_no_paid_work()

    def test_completed_restart_rejects_changed_published_machine_or_markdown(self):
        for artifact in ('machine', 'markdown', 'missing-machine'):
            with self.subTest(artifact=artifact):
                compiled = self.completed_state()
                machine = self.root / 'scoring/v2-glm/benchmark-results.json'
                if artifact == 'machine':
                    self.save('scoring/v2-glm/benchmark-results.json', {**compiled, 'metrics': {'changed': True}})
                elif artifact == 'markdown':
                    (self.root / 'benchmark.md').write_bytes(b'unverified markdown\n')
                else:
                    machine.unlink()
                before = (self.root / 'benchmark.md').read_bytes()
                with patch.object(pipeline.report, 'build', return_value=compiled), \
                        patch.object(pipeline.report, 'render', return_value='verified markdown\n'):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        pipeline.run(self.root)
                self.assertEqual((self.root / 'benchmark.md').read_bytes(), before)
                self.assert_no_paid_work()

    def test_state_load_exception_cleans_up_owned_guard(self):
        with patch.object(pipeline, 'load_state', side_effect=ValueError('Malformed state')):
            with self.assertRaisesRegex(ValueError, 'Malformed state'):
                pipeline.run(self.root)
        self.assertFalse((self.root / 'runs/v2/baseline-pipeline.guard').exists())
        self.assert_no_paid_work()

    def test_cleanup_never_removes_replaced_guard(self):
        guard = self.root / 'runs/v2/baseline-pipeline.guard'
        replacement = pipeline.json_bytes({'pid': 'other-owner'})

        def replace_guard_then_fail(*_):
            guard.write_bytes(replacement)
            raise ValueError('Malformed state')

        with patch.object(pipeline, 'load_state', side_effect=replace_guard_then_fail):
            with self.assertRaisesRegex(ValueError, 'Malformed state'):
                pipeline.run(self.root)
        self.assertEqual(guard.read_bytes(), replacement)

    def test_successful_return_never_removes_replaced_guard(self):
        compiled = self.completed_state()
        guard = self.root / 'runs/v2/baseline-pipeline.guard'
        replacement = pipeline.json_bytes({'pid': 'other-owner'})

        def replace_guard_then_load(*_):
            guard.write_bytes(replacement)
            return self.state

        with patch.object(pipeline, 'load_state', side_effect=replace_guard_then_load), \
                patch.object(pipeline.report, 'build', return_value=compiled), \
                patch.object(pipeline.report, 'render', return_value='verified markdown\n'):
            pipeline.run(self.root)
        self.assertEqual(guard.read_bytes(), replacement)

    def test_existing_guard_is_not_removed_when_acquisition_fails(self):
        guard = self.save('runs/v2/baseline-pipeline.guard', {'pid': 'other-owner'})
        before = guard.read_bytes()
        with self.assertRaises(FileExistsError):
            pipeline.run(self.root)
        self.assertEqual(guard.read_bytes(), before)
        self.assert_no_paid_work()


if __name__ == '__main__':
    unittest.main()
