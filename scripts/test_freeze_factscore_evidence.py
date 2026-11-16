"""Offline provenance/failure tests; no HTTP or model inference."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

import freeze_factscore_evidence as freezer
import prepare_v2_data as preparation

TEMP = Path(r'C:\Users\AstralK\AppData\Local\Temp\opencode')
TOPIC = 'Fixture subject'
HTML = '<div><p>' + 'A fixture biography sentence with enough source text. ' * 4 + '</p><table><tr><td>Retained fact</td></tr></table><div class="navbox">NAVIGATION</div></div>'


class FakeResponse:
    def __init__(self, params, payload):
        self.url = freezer.API + '?' + urlencode(params)
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, query, parsed):
        self.headers = {}
        self.query = query
        self.parsed = parsed
        self.calls = []

    def mount(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def get(self, url, *, params, timeout):
        self.calls.append(params)
        return FakeResponse(params, self.query if params['action'] == 'query' else self.parsed)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('requests.sessions.Session.request', side_effect=AssertionError('HTTP forbidden in offline tests')).start()
        self.temp = tempfile.TemporaryDirectory(dir=TEMP, prefix='factscore-evidence-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.query = {'query': {'pages': [{'ns': 0, 'pageid': 10, 'title': TOPIC,
            'revisions': [{'revid': 20, 'timestamp': '2026-11-11T00:00:00Z'}]}]}}
        self.parsed = {'parse': {'pageid': 10, 'revid': 20, 'text': HTML}}
        self.directory = self.root/'data/v2/factscore_evidence'/freezer.digest(TOPIC.encode())[:20]

    def fetch(self, query=None, parsed=None):
        session = FakeSession(query or self.query, parsed or self.parsed)
        with patch.object(freezer.requests, 'Session', return_value=session):
            record = freezer.fetch_one(self.root, TOPIC)
        return record, session

    def test_revision_pinning_resume_and_content_tamper(self):
        record, session = self.fetch()
        self.assertEqual(record['status'], 'ok')
        self.assertEqual(session.calls[1]['oldid'], 20)
        text = (self.directory/'text.txt').read_text(encoding='utf-8')
        self.assertIn('Retained fact', text)
        self.assertNotIn('NAVIGATION', text)
        with patch.object(freezer.requests, 'Session', side_effect=AssertionError('resume must be offline')):
            self.assertEqual(freezer.fetch_one(self.root, TOPIC), record)
        (self.directory/'text.txt').write_text('tampered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            freezer.verify_record(self.root, TOPIC, record)

    def test_missing_disambiguation_and_revision_mismatch_preserve_failure(self):
        missing = {'query': {'pages': [{'title': TOPIC, 'missing': True}]}}
        ambiguous = copy.deepcopy(self.query)
        ambiguous['query']['pages'][0]['pageprops'] = {'disambiguation': ''}
        wrong_revision = copy.deepcopy(self.parsed)
        wrong_revision['parse']['revid'] = 21
        for index, (query, parsed) in enumerate([(missing, self.parsed), (ambiguous, self.parsed), (self.query, wrong_revision)]):
            with self.subTest(index=index):
                self.root = Path(self.temp.name)/str(index)
                directory = self.root/'data/v2/factscore_evidence'/freezer.digest(TOPIC.encode())[:20]
                record, _ = self.fetch(query, parsed)
                self.assertEqual(record['status'], 'retrieval_error')
                self.assertFalse((directory/'record.json').exists())
                self.assertEqual(len(list(directory.glob('error-*.json'))), 1)

    def test_record_identity_fields_rejected(self):
        record, _ = self.fetch()
        for field, wrong in [('revision', 21), ('pageid', 11), ('revision_timestamp', 'wrong'), ('url', 'wrong')]:
            with self.subTest(field=field):
                altered = {**record, field: wrong}
                with self.assertRaises(ValueError):
                    freezer.verify_record(self.root, TOPIC, altered)

    def test_title_change_requires_recorded_redirect(self):
        wrong = copy.deepcopy(self.query)
        wrong['query']['pages'][0]['title'] = 'Another title'
        request = {'url': freezer.API+'?'+urlencode({'action': 'query', 'titles': TOPIC}), 'response': wrong}
        with self.assertRaisesRegex(ValueError, 'recorded redirect'):
            freezer.resolved_page(request, TOPIC)
        wrong['query']['redirects'] = [{'from': TOPIC, 'to': 'Another title'}]
        self.assertEqual(freezer.resolved_page(request, TOPIC)['title'], 'Another title')

    def test_corrupt_resume_is_saved_in_incomplete_report(self):
        record, _ = self.fetch()
        (self.directory/'text.txt').write_text('tampered', encoding='utf-8')
        archive = self.root/'data/v2/raw/FActScore/data.zip'
        freezer.immutable(archive, b'fixture archive; topics() is mocked')
        with patch.object(freezer, 'subject_selection', return_value=([(0, TOPIC)], {})), \
                patch.object(freezer, 'evaluation_binding', return_value={}), \
                patch.object(freezer.requests, 'Session', side_effect=AssertionError('resume must be offline')):
            with self.assertRaisesRegex(RuntimeError, '1 evidence sources unavailable'):
                freezer.freeze(self.root)
        errors = list(self.directory.glob('error-*.json'))
        reports = list((self.root/'data/v2/factscore_evidence').glob('incomplete-*.json'))
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(reports), 1)
        self.assertFalse(json.loads(reports[0].read_text(encoding='utf-8'))['ready'])
        self.assertFalse((self.root/'data/v2/factscore-evidence-manifest.json').exists())
        self.assertEqual(json.loads((self.directory/'record.json').read_text(encoding='utf-8')), record)

    def test_maintenance_and_navigation_removed_from_text(self):
        html = '<p>Factual paragraph.</p><table><tr><td>Factual table.</td></tr></table>'
        html += ''.join(f'<div class="{name}">UI {name}</div>' for name in ['portalbox', 'hatnote', 'ambox'])
        self.assertEqual(freezer.clean_html(html), 'Factual paragraph.\nFactual table.\n')

    def test_write_once_and_failed_publish_is_not_partial(self):
        path = self.root/'artifact.json'
        freezer.immutable(path, b'original')
        freezer.immutable(path, b'original')
        with self.assertRaises(ValueError):
            freezer.immutable(path, b'changed')
        self.assertEqual(path.read_bytes(), b'original')
        failed = self.root/'failed.json'
        with patch.object(freezer.os, 'link', side_effect=OSError('fixture publication failure')):
            with self.assertRaises(OSError):
                freezer.immutable(failed, b'content')
        self.assertFalse(failed.exists())
        self.assertEqual(list(self.root.iterdir()), [path])


class OfflineFinalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=TEMP, prefix='factscore-180-fixture-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.fixture = Path(cls.temp.name)
        for relative in ('data/v2/raw/FActScore/data.zip', 'data/v2/new-sources-receipt.json'):
            path = cls.fixture / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(preparation.DEFAULT_ROOT / relative, path)
        cls.selected, cls.selection = freezer.subject_selection(cls.fixture)
        cls.names = [topic for _, topic in cls.selected]
        cls.rows, _ = preparation._factscore({preparation.FACTSCORE_PATH:
            (cls.fixture / preparation.FACTSCORE_PATH).read_bytes()})
        freezer.immutable(cls.fixture / 'data/v2/evaluation.jsonl', b''.join(preparation._json_line(row) for row in cls.rows))
        freezer.immutable(cls.fixture / 'data/v2/manifest.json', freezer.json_bytes({
            'counts': {'factscore_bios': 180}, 'selection': {'factscore_bios': cls.selection},
            'total_evaluation_rows': 9527, 'new_generation_rows': 7549, 'reused_rows_per_model': 1978,
        }))
        for topic in cls.names:
            query = {'query': {'pages': [{'ns': 0, 'pageid': 10, 'title': topic,
                'revisions': [{'revid': 20, 'timestamp': '2026-11-11T00:00:00Z'}]}]}}
            session = FakeSession(query, {'parse': {'pageid': 10, 'revid': 20, 'text': HTML}})
            with patch.object(freezer.requests, 'Session', return_value=session):
                record = freezer.fetch_one(cls.fixture, topic)
            if record['status'] != 'ok':
                raise AssertionError(record)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP, prefix='factscore-offline-finalize-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'root'
        shutil.copytree(self.fixture, self.root)
        self.manifest_path = self.root / 'data/v2/factscore-evidence-manifest.json'
        self.addCleanup(patch.stopall)
        self.session = patch.object(freezer.requests, 'Session', side_effect=AssertionError('No sessions in offline finalization')).start()
        self.http = patch('requests.sessions.Session.request', side_effect=AssertionError('No HTTP in offline finalization')).start()
        self.fetch = patch.object(freezer, 'fetch_one', side_effect=AssertionError('Offline path must only reuse record.json')).start()

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_offline_finalization_reuses_180_records_and_is_write_once(self):
        before = self.snapshot()
        self.assertEqual(freezer.finalize_offline(self.root)['subjects'], 180)
        report = json.loads(self.manifest_path.read_bytes())
        self.assertEqual(report['count'], 180)
        self.assertEqual(report['selection'], self.selection)
        self.assertEqual(report['evaluation']['selection'], self.selection)
        self.assertEqual(report['evaluation']['factscore_source_indices'], [i for i, _ in self.selected])
        self.assertEqual(report['evaluation']['factscore_ids'], self.selection['selected_ids'])
        after = self.snapshot()
        self.assertEqual({k: v for k, v in after.items() if k != 'data/v2/factscore-evidence-manifest.json'}, before)
        actual_open = Path.open

        def read_only(path, mode='r', *args, **kwargs):
            if any(flag in mode for flag in 'wax+'):
                raise AssertionError('Repeat finalization must not write')
            return actual_open(path, mode, *args, **kwargs)

        with patch.object(Path, 'open', read_only):
            self.assertEqual(freezer.finalize_offline(self.root)['subjects'], 180)
            self.assertEqual(freezer.verify(self.root)['subjects'], 180)
        self.session.assert_not_called()
        self.http.assert_not_called()
        self.fetch.assert_not_called()

    def test_missing_or_corrupt_saved_evidence_fails_with_zero_http_and_no_writes(self):
        directory = self.root / 'data/v2/factscore_evidence' / freezer.digest(self.names[0].encode())[:20]
        cases = [('record.json', None), ('record.json', b'not JSON'),
                 ('text.txt', None), ('text.txt', b'changed'), ('query.json', b'changed'), ('parse.json', b'changed')]
        for filename, content in cases:
            path = directory / filename
            original = path.read_bytes()
            try:
                if content is None:
                    path.unlink()
                else:
                    path.write_bytes(content)
                before = self.snapshot()
                with self.subTest(file=filename, content=content), self.assertRaises((ValueError, OSError)):
                    freezer.finalize_offline(self.root)
                self.assertFalse(self.manifest_path.exists())
                self.assertEqual(self.snapshot(), before)
            finally:
                path.write_bytes(original)
        self.session.assert_not_called()
        self.http.assert_not_called()
        self.fetch.assert_not_called()

    def test_semantically_corrupt_record_and_late_missing_record_never_fetch(self):
        path = self.root / 'data/v2/factscore_evidence' / freezer.digest(self.names[-1].encode())[:20] / 'record.json'
        original = path.read_bytes()
        record = json.loads(original)
        cases = [None, b'[]', b'null', freezer.json_bytes({**record, 'revision': 21}),
                 freezer.json_bytes({k: v for k, v in record.items() if k != 'files'}),
                 freezer.json_bytes({**record, 'files': []})]
        for content in cases:
            try:
                if content is None:
                    path.unlink()
                else:
                    path.write_bytes(content)
                before = self.snapshot()
                with self.subTest(content=content), self.assertRaises((ValueError, OSError)):
                    freezer.finalize_offline(self.root)
                self.assertEqual(self.snapshot(), before)
                self.assertFalse(self.manifest_path.exists())
            finally:
                path.write_bytes(original)
        self.session.assert_not_called()
        self.http.assert_not_called()
        self.fetch.assert_not_called()

    def test_freeze_uses_same_180_subjects_and_selection_as_offline_finalization(self):
        with patch.object(freezer, 'fetch_one', side_effect=freezer.reuse_record) as reuse:
            self.assertEqual(freezer.freeze(self.root)['subjects'], 180)
        self.assertEqual(sorted(call.args[1] for call in reuse.call_args_list), sorted(self.names))
        report = json.loads(self.manifest_path.read_bytes())
        self.assertEqual([record['topic'] for record in report['records']], self.names)
        self.assertEqual(report['selection'], self.selection)
        self.assertEqual(report['evaluation']['selection'], self.selection)
        self.assertEqual(freezer.verify(self.root)['subjects'], 180)
        self.session.assert_not_called()
        self.http.assert_not_called()

    def test_binding_rejects_wrong_subject_prompt_index_id_order_or_inventory(self):
        path = self.root / 'data/v2/evaluation.jsonl'
        binding = freezer.evaluation_binding(self.root, self.names)
        self.assertEqual(binding['sha256'], freezer.digest(path.read_bytes()))
        cases = []
        for location, key, value in [('metadata', 'topic', 'Substitute'), ('metadata', 'source_index', 2),
                                     ('metadata', 'source_index', False), ('metadata', 'source_path', 'other.zip'),
                                     ('metadata', 'zip_member', 'other.txt'), ('row', 'id', 'v2-factscore_bios-2'),
                                     ('row', 'question', 'Changed prompt')]:
            rows = copy.deepcopy(self.rows)
            (rows[0]['metadata'] if location == 'metadata' else rows[0])[key] = value
            cases.append(rows)
        cases.extend([list(reversed(self.rows)), self.rows[:-1], self.rows + [self.rows[0]]])
        for rows in cases:
            path.write_bytes(b''.join(preparation._json_line(row) for row in rows))
            with self.assertRaises(ValueError):
                freezer.evaluation_binding(self.root, self.names)
        with self.assertRaises(ValueError):
            freezer.evaluation_binding(self.root, self.names[:-1])

    def test_binding_and_manifest_selection_mismatches_are_rejected_before_fetch(self):
        path = self.root / 'data/v2/manifest.json'
        document = json.loads(path.read_bytes())
        document['selection']['factscore_bios']['selected_source_indices'][0] = 2
        path.write_bytes(freezer.json_bytes(document))
        before = self.snapshot()
        for action in (freezer.finalize_offline, freezer.freeze):
            with self.assertRaisesRegex(ValueError, 'selection'):
                action(self.root)
        self.assertEqual(self.snapshot(), before)
        self.session.assert_not_called()
        self.http.assert_not_called()
        self.fetch.assert_not_called()

    def test_existing_bad_evidence_manifest_is_not_replaced(self):
        freezer.finalize_offline(self.root)
        good = json.loads(self.manifest_path.read_bytes())
        for field, value in [('count', 183), ('selection', {}), ('evaluation', {}),
                             ('records', list(reversed(good['records']))), ('ready', False)]:
            report = {**good, field: value}
            self.manifest_path.write_bytes(freezer.json_bytes(report))
            before = self.snapshot()
            with self.subTest(field=field), self.assertRaises(ValueError):
                freezer.finalize_offline(self.root)
            self.assertEqual(self.snapshot(), before)

    def test_record_and_binding_json_types_must_match_frozen_bytes(self):
        freezer.finalize_offline(self.root)
        good = json.loads(self.manifest_path.read_bytes())
        for mutation in ('record', 'binding'):
            report = copy.deepcopy(good)
            if mutation == 'record':
                report['records'][0]['revision'] = 20.0
            else:
                report['evaluation']['factscore_source_indices'][0] = False
            self.manifest_path.write_bytes(freezer.json_bytes(report))
            before = self.snapshot()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                freezer.verify(self.root)
            self.assertEqual(self.snapshot(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
