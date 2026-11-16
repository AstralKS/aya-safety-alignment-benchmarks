import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import v2_compute_jobs as jobs
import launch_v2 as launch
import prepare_v2_data
import v2_generation_queue as queue
import v2_glm_scoring as scoring

TEMP = r'C:\Users\AstralK\AppData\Local\Temp\opencode'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding='utf-8')


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
                            for record in records), encoding='utf-8')


class SelfAwareProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        rows = [{'id': f'v2-selfaware-{index}', 'dataset': 'selfaware', 'component': 'fixture',
                 'question': f'Question {index}?', 'system_prompt': '', 'max_new_tokens': 256,
                 'metadata': {'answerable': False, 'answer': 'unknown', 'source': 'fixture',
                              'question_id': index}} for index in range(3369)]
        write_jsonl(self.root / 'data/v2/evaluation.jsonl', rows)
        self.verified_jobs = {}
        self.aggregates = {}
        for short, model in scoring.client.AYA_MODELS.items():
            examples = [{key: row[key] for key in ('id', 'dataset', 'component', 'question',
                        'system_prompt', 'max_new_tokens')} for row in rows]
            extra = {**examples[0], 'id': 'v2-other-0', 'dataset': 'factscore_bios'}
            groups = []
            for name, selected in (('selfaware', examples), ('other', [extra])):
                path = self.root / 'runs/v2' / short / 'jobs' / name
                config = {'model_id': model, 'revision': jobs.hashes.PINS[model],
                          'input_sha256': jobs.hashes.payload_hash(selected),
                          'chat_template_sha256': 'a' * 64, 'selected_template_sha256': 'b' * 64}
                job = {'config': config, 'examples': selected, 'kernel': 'owner/' + short + name}
                records = [{**jobs.hashes.record_identity(config, example, 'rendered'),
                            'response': 'I do not know.', 'status': 'ok', 'finish_reason': 'eos',
                            'input_tokens': 10, 'output_tokens': 8, 'batch_seconds': 0.125}
                           for example in selected]
                manifest = {'config': config, 'resolved_revision': config['revision'], 'model_id': model,
                            'weight_dtype': 'float16', 'quantization': None, 'stage': 'completed',
                            'successful': len(selected), 'errors': 0, 'sample_count': len(selected),
                            'transformers': '4.57.6', 'accelerate': '1.12.0',
                            'gpu_info': [{'index': 0}], 'device_map': {'': 'cuda:0'},
                            'chat_template_sha256': 'a' * 64, 'selected_template_sha256': 'b' * 64}
                write_jsonl(path / 'output/responses.jsonl', records)
                write_json(path / 'output/manifest.json', manifest)
                self.verified_jobs[path] = job
                self.receipt(path)
                groups.extend(records)
            self.aggregates[short] = groups
            self.publish(short)
        for mock in (patch.object(prepare_v2_data, 'verify'),
                     patch.object(launch, 'verify_job', side_effect=lambda path: self.verified_jobs[path]),
                     patch.object(queue.subprocess, 'run', side_effect=AssertionError('No download allowed'))):
            mock.start()
            self.addCleanup(mock.stop)

    def receipt(self, path):
        write_json(path / 'collected.json', {'kernel': self.verified_jobs[path]['kernel'],
            'responses_sha256': jobs.hashes.digest((path / 'output/responses.jsonl').read_bytes()),
            'runtime_manifest_sha256': jobs.hashes.digest((path / 'output/manifest.json').read_bytes()),
            'answers': len(self.verified_jobs[path]['examples'])})

    def publish(self, short):
        write_jsonl(self.root / 'runs/v2' / short / 'responses.jsonl', self.aggregates[short])

    def test_verified_collected_selfaware_is_read_only_and_complete(self):
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        with patch.object(queue, 'collect', wraps=queue.collect) as collect:
            result = jobs.selfaware_inputs(self.root)
        self.assertEqual(len(result), 6738)
        self.assertEqual(collect.call_count, 4)
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}, before)

    def test_aggregate_text_and_other_valid_record_fields_cannot_be_tampered(self):
        original = deepcopy(self.aggregates['tiny'][0])
        for key, value in (('response', 'A substituted answer.'), ('output_tokens', 9),
                           ('unverified_extra', 'injected')):
            with self.subTest(key=key):
                self.aggregates['tiny'][0] = {**original, key: value}
                self.publish('tiny')
                with self.assertRaises(ValueError):
                    jobs.selfaware_inputs(self.root)

    def test_all_aggregate_records_not_only_selfaware_need_collected_binding(self):
        self.aggregates['tiny'][-1]['response'] = 'Changed biography.'
        self.publish('tiny')
        with self.assertRaises(ValueError):
            jobs.selfaware_inputs(self.root)

    def test_uncollected_job_is_rejected_without_calling_collect_or_downloading(self):
        path = self.root / 'runs/v2/tiny/jobs/selfaware'
        (path / 'collected.json').unlink()
        with patch.object(queue, 'collect', wraps=queue.collect) as collect, self.assertRaises(ValueError):
            jobs.selfaware_inputs(self.root)
        self.assertNotIn(path, [call.args[0] for call in collect.call_args_list])

    def test_receipt_disappearing_during_collection_cannot_trigger_a_download(self):
        original = queue.collect
        def disappear(path, job):
            (Path(path) / 'collected.json').unlink()
            return original(path, job)
        with patch.object(queue, 'collect', side_effect=disappear), self.assertRaises(ValueError):
            jobs.selfaware_inputs(self.root)

    def test_collected_response_and_runtime_hashes_are_required(self):
        path = self.root / 'runs/v2/tiny/jobs/selfaware'
        original = json.loads((path / 'collected.json').read_bytes())
        for key in ('responses_sha256', 'runtime_manifest_sha256'):
            with self.subTest(key=key):
                write_json(path / 'collected.json', {**original, key: '0' * 64})
                with self.assertRaises(ValueError):
                    jobs.selfaware_inputs(self.root)

    def test_rehashed_wrong_runtime_contract_is_rejected(self):
        path = self.root / 'runs/v2/tiny/jobs/selfaware'
        manifest = json.loads((path / 'output/manifest.json').read_bytes())
        manifest['weight_dtype'] = 'int8'
        write_json(path / 'output/manifest.json', manifest)
        self.receipt(path)
        with self.assertRaises(ValueError):
            jobs.selfaware_inputs(self.root)


class FactScoreProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = scoring.SourceCatalog(jobs.ROOT)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'scoring/v2-glm/results/compute-fixture'
        self.binding = {'manifest_sha256': 'c' * 64, 'evaluation_sha256': 'd' * 64,
                        'reuse_sha256': 'e' * 64, 'expected_counts_per_model': scoring.EXPECTED}
        self.available = []
        subjects = {}
        for name in ('Alice', 'Bob'):
            text = name + ' is a writer. ' + name + ' lives in Rome.'
            relative = 'data/v2/factscore_evidence/' + name.lower() + '/text.txt'
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
            subjects[name] = {'topic': name, 'title': name, 'revision': 123,
                'url': 'https://en.wikipedia.org/w/index.php?oldid=123', 'text': text,
                'text_sha256': jobs.hashes.digest(text.encode()), 'text_file': relative}
            row = {'id': 'v2-factscore-' + name.lower(), 'dataset': 'factscore_bios',
                   'question': 'Tell me about ' + name + '.', 'metadata': {'topic': name},
                   '_input_row_sha256': 'f' * 64, '_v2_manifest_sha256': 'c' * 64,
                   '_response_file': 'runs/v2/tiny/responses.jsonl', '_response_row_sha256': 'a' * 64}
            self.available.append((row, text, 'tiny'))
        row = {**self.available[0][0], '_response_file': 'runs/v2/8b/responses.jsonl'}
        self.available.append((row, 'Alice is a poet.', '8b'))
        manifest = {'ready': True, 'count': 180, 'records': [
            {'topic': source['topic'], 'title': source['title'], 'revision': source['revision'],
             'url': source['url'], 'files': [{'path': source['text_file'], 'sha256': source['text_sha256']}]}
            for source in subjects.values()] + [
            {'topic': f'fixture-{index}', 'revision': 1, 'files': []} for index in range(178)]}
        manifest_path = self.root / 'data/v2/factscore-evidence-manifest.json'
        write_json(manifest_path, manifest)
        manifest_sha = jobs.hashes.digest(manifest_path.read_bytes())
        from types import SimpleNamespace
        self.evidence = SimpleNamespace(subjects=subjects, manifest_sha256=manifest_sha)
        all_jobs = [job for row, answer, short in self.available
                    for job in scoring.extraction_jobs(row, answer, short, self.catalog)]
        outcomes = []
        for number, job in enumerate(all_jobs):
            relative = ('scoring/v2-glm/request-journals/' + job['cache_key'] +
                        f'/attempts/fixture-{number}.response.json')
            journal_path = self.root / relative
            label = {'sentence': job['parser']['sentence'], 'facts': [job['parser']['sentence']]}
            body = scoring.client.canonical({'model': scoring.client.MODEL, 'choices': [
                {'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant',
                    'content': json.dumps(label)}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}})
            write_json(journal_path, {'kind': 'http_response', 'http_status': 200,
                'attempt_id': f'fixture-{number}', 'request_sha256': job['request_sha256'],
                'raw_body_base64': base64.b64encode(body).decode(),
                'raw_body_sha256': jobs.hashes.digest(body)})
            write_json(journal_path.parent.parent / 'job.json', scoring.request_contract(job))
            request_name = scoring.digest({'key': job['cache_key'], 'lineage': job['lineage']}) + '.json'
            write_json(self.run / 'requests' / request_name, job)
            outcomes.append({'cache_key': job['cache_key'], 'ok': True, 'label': label,
                'journal_path': str(journal_path), 'journal_sha256': jobs.hashes.digest(journal_path.read_bytes())})
        self.packet = scoring.extraction_packet(self.available, all_jobs, outcomes, self.binding, root=self.root)
        self.packet_path = self.run / 'snapshots/extract-fixture.json'
        self.ready_path = self.run / 'snapshots/retrieval-ready-fixture.json'
        write_json(self.packet_path, self.packet)
        self.ready = scoring.retrieval_ready(self.packet, self.evidence, root=self.root)
        write_json(self.ready_path, self.ready)
        for mock in (patch.object(jobs, 'FROZEN_EVIDENCE_SHA256', manifest_sha),
                     patch.object(scoring, 'load_available', return_value=(self.available, self.binding)),
                     patch.object(scoring, 'SourceCatalog', return_value=self.catalog),
                     patch.object(scoring, 'JobStore', side_effect=AssertionError('No JobStore allowed')),
                     patch.object(scoring.client, 'http_request', side_effect=AssertionError('No HTTP allowed'))):
            mock.start()
            self.addCleanup(mock.stop)

    def rewrite_export(self, ready):
        raw = b''.join(scoring.client.canonical(fact) for fact in ready['facts'])
        relative = 'scoring/v2-glm/facts-exports/facts-' + jobs.hashes.digest(raw) + '.jsonl'
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        ready['provenance']['facts_file'] = {'path': relative, 'sha256': jobs.hashes.digest(raw)}
        write_json(self.ready_path, ready)

    def test_exact_glm_export_is_rederived_read_only(self):
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        jobs.validate_factscore_payload(self.root, self.ready)
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}, before)

    def test_self_consistent_fact_reassignment_is_rejected(self):
        for key, value in (('aya_model', '8b'), ('aya_model', 'unapproved'),
                           ('id', 'v2-factscore-bob'), ('topic', 'Bob'),
                           ('answer_sha256', '0' * 64), ('response_sha256', '0' * 64),
                           ('sentence_index', 1), ('fact_index', True),
                           ('extraction_cache_key', '0' * 64)):
            with self.subTest(key=key, value=value):
                ready = deepcopy(self.ready)
                ready['facts'][0][key] = value
                self.rewrite_export(ready)
                with self.assertRaises(ValueError):
                    jobs.validate_factscore_payload(self.root, ready)

    def test_successful_foreign_extraction_journal_cannot_be_assigned_to_a_fact(self):
        ready = deepcopy(self.ready)
        foreign = next(fact for fact in ready['facts'] if fact['topic'] == 'Bob')
        # Keep the claimed text a member of the foreign GLM facts array: mere
        # membership/model checks must not authorize a different biography request.
        journal_path = self.root / foreign['extraction_journal']
        journal = json.loads(journal_path.read_bytes())
        body = json.loads(base64.b64decode(journal['raw_body_base64']))
        extracted = json.loads(body['choices'][0]['message']['content'])
        extracted['facts'] = [ready['facts'][0]['text']]
        body['choices'][0]['message']['content'] = json.dumps(extracted)
        body_raw = scoring.client.canonical(body)
        journal.update(raw_body_base64=base64.b64encode(body_raw).decode(),
                       raw_body_sha256=jobs.hashes.digest(body_raw))
        write_json(journal_path, journal)
        foreign = {**foreign, 'extraction_journal_sha256': jobs.hashes.digest(journal_path.read_bytes()),
                   'extraction_response_sha256': jobs.hashes.digest(body_raw)}
        for key in ('extraction_journal', 'extraction_journal_sha256', 'extraction_response_sha256'):
            ready['facts'][0][key] = foreign[key]
        self.rewrite_export(ready)
        with self.assertRaises(ValueError):
            jobs.validate_factscore_payload(self.root, ready)

    def test_self_consistent_original_snapshot_reassignment_still_requires_source_derivation(self):
        for key, value in (('aya_model', '8b'), ('id', 'v2-factscore-bob'), ('topic', 'Bob'),
                           ('answer_sha256', '0' * 64)):
            with self.subTest(key=key):
                packet = deepcopy(self.packet)
                packet['responses'][0][key] = value
                for fact in packet['responses'][0]['facts']:
                    fact[key] = value
                write_json(self.packet_path, packet)
                ready = deepcopy(self.ready)
                ready['extraction_packet_sha256'] = jobs.hashes.digest(scoring.json_bytes(packet))
                for fact in ready['facts'][:len(packet['responses'][0]['facts'])]:
                    fact[key] = value
                self.rewrite_export(ready)
                with self.assertRaises(ValueError):
                    jobs.validate_factscore_payload(self.root, ready)

    def test_valid_saved_extraction_cannot_follow_a_changed_aya_answer(self):
        row, _, short = self.available[0]
        self.available[0] = (row, 'Alice was born in Paris.', short)
        with self.assertRaises(ValueError):
            jobs.validate_factscore_payload(self.root, self.ready)

    def test_zero_fact_export_requires_real_empty_glm_extractions_without_padding(self):
        all_jobs, outcomes = [], []
        for row, answer, short in self.available:
            all_jobs.extend(scoring.extraction_jobs(row, answer, short, self.catalog))
        for job in all_jobs:
            folder = self.root / 'scoring/v2-glm/request-journals' / job['cache_key'] / 'attempts'
            path = next(folder.glob('*.response.json'))
            journal = json.loads(path.read_bytes())
            body = json.loads(base64.b64decode(journal['raw_body_base64']))
            label = {'sentence': job['parser']['sentence'], 'facts': []}
            body['choices'][0]['message']['content'] = json.dumps(label)
            raw_body = scoring.client.canonical(body)
            journal.update(raw_body_base64=base64.b64encode(raw_body).decode(),
                           raw_body_sha256=jobs.hashes.digest(raw_body))
            write_json(path, journal)
            outcomes.append({'cache_key': job['cache_key'], 'ok': True, 'label': label,
                'journal_path': str(path), 'journal_sha256': jobs.hashes.digest(path.read_bytes())})
        packet = scoring.extraction_packet(self.available, all_jobs, outcomes, self.binding, root=self.root)
        write_json(self.packet_path, packet)
        ready = scoring.retrieval_ready(packet, self.evidence, root=self.root)
        self.assertEqual(ready['facts'], [])
        self.assertEqual(ready['evidence'], [])
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        jobs.validate_factscore_payload(self.root, ready)
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}, before)

    def test_rehashed_journal_request_assignment_and_job_contract_are_rejected(self):
        fact = self.ready['facts'][0]
        journal_path = self.root / fact['extraction_journal']
        journal = json.loads(journal_path.read_bytes())
        journal['request_sha256'] = '0' * 64
        write_json(journal_path, journal)
        ready = deepcopy(self.ready)
        ready['facts'][0]['extraction_journal_sha256'] = jobs.hashes.digest(journal_path.read_bytes())
        self.rewrite_export(ready)
        with self.assertRaises(ValueError):
            jobs.validate_factscore_payload(self.root, ready)

    def test_saved_request_must_have_exact_approved_model_subject_and_answer_lineage(self):
        request = next((self.run / 'requests').glob('*.json'))
        value = json.loads(request.read_bytes())
        value['lineage']['aya_model'] = '8b' if value['lineage']['aya_model'] == 'tiny' else 'tiny'
        write_json(request, value)
        with self.assertRaises(ValueError):
            jobs.validate_factscore_payload(self.root, self.ready)

    def test_tampered_extraction_cache_contract_is_rejected(self):
        fact = self.ready['facts'][0]
        path = self.root / 'scoring/v2-glm/request-journals' / fact['extraction_cache_key'] / 'job.json'
        contract = json.loads(path.read_bytes())
        contract['request_sha256'] = '0' * 64
        write_json(path, contract)
        with self.assertRaises(ValueError):
            jobs.validate_factscore_payload(self.root, self.ready)

    def test_omitted_fact_or_changed_ready_contract_is_rejected(self):
        for key, value in (('facts', self.ready['facts'][1:]), ('retrieval_revision', 'wrong'),
                           ('passage_tokens', 256.0), ('extraction_packet_sha256', '0' * 64)):
            with self.subTest(key=key):
                ready = deepcopy(self.ready)
                ready[key] = value
                self.rewrite_export(ready)
                with self.assertRaises(ValueError):
                    jobs.validate_factscore_payload(self.root, ready)

    def test_frozen_manifest_pin_remains_mandatory(self):
        with patch.object(jobs, 'FROZEN_EVIDENCE_SHA256', '0' * 64), self.assertRaises(ValueError):
            jobs.validate_factscore_payload(self.root, self.ready)


class ComputePackagingTests(unittest.TestCase):
    def fixture_root(self, directory):
        root = Path(directory)
        for relative in jobs.SOURCES.values():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((jobs.ROOT / relative).read_bytes())
        return root

    def test_selfaware_package_is_pinned_credential_free_and_write_once(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            root = self.fixture_root(directory)
            payload = [{'id': 'fixture', 'response': 'Answer'}]
            with patch.object(jobs, 'selfaware_inputs', return_value=payload):
                path = jobs.prepare(root, 'selfaware', payload)
            receipt = jobs.verify(path, root)
            self.assertEqual(receipt['kind'], 'selfaware')
            notebook = json.loads((path / 'run.ipynb').read_bytes())
            self.assertIn('UserSecretsClient', ''.join(notebook['cells'][1]['source']))
            with patch.object(jobs, 'selfaware_inputs', return_value=payload), self.assertRaises(FileExistsError):
                jobs.prepare(root, 'selfaware', payload)

    def test_modified_input_or_source_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            root = self.fixture_root(directory)
            payload = [{'id': 'fixture'}]
            with patch.object(jobs, 'selfaware_inputs', return_value=payload):
                path = jobs.prepare(root, 'selfaware', payload)
            raw = (path / 'inputs.json').read_bytes()
            (path / 'inputs.json').write_bytes(raw + b' ')
            with self.assertRaises(ValueError):
                jobs.verify(path, root)
            (path / 'inputs.json').write_bytes(raw)
            (root / jobs.SOURCES['selfaware']).write_text('# changed')
            with self.assertRaises(ValueError):
                jobs.verify(path, root)

    def test_duplicate_selfaware_ids_cannot_be_silently_collapsed(self):
        with self.assertRaises(ValueError):
            jobs.unique_records([{'id': 'same'}, {'id': 'same'}])

    def test_invented_selfaware_payload_is_rejected_before_packaging(self):
        with patch.object(jobs, 'selfaware_inputs', return_value=[{'id': 'expected'}]), self.assertRaises(ValueError):
            jobs.prepare(Path(TEMP), 'selfaware', [{'id': 'invented'}])

    def test_self_consistent_invented_evidence_has_no_frozen_binding(self):
        payload = {'evidence': [{'topic': 'Invented subject', 'text': 'Invented article',
                                'revision': 123, 'text_sha256': jobs.hashes.digest(b'Invented article')}],
                   'facts': [{'id': 'invented', 'fact_id': 'f0', 'topic': 'Invented subject', 'fact': 'Invented fact'}]}
        with self.assertRaises(ValueError):
            jobs.validate_factscore_payload(jobs.ROOT, payload)


if __name__ == '__main__':
    unittest.main()
