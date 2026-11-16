import hashlib
import unittest

import v2_factscore_retrieval as retrieval


class RetrievalTests(unittest.TestCase):
    def test_chunking_keeps_original_256_token_boundaries(self):
        self.assertEqual(retrieval.token_chunks(list(range(600))),
                         [list(range(256)), list(range(256, 512)), list(range(512, 600))])
        self.assertEqual(retrieval.token_chunks([]), [])

    def test_gtr_inputs_match_official_title_and_fact_query(self):
        self.assertEqual(retrieval.passage_inputs('Subject', ['<s>Some text</s>']), ['Subject Some text'])
        self.assertEqual(retrieval.query_input('Subject', '  A fact.  '), 'Subject A fact.')

    def test_input_evidence_hash_and_same_subject_required(self):
        text = 'Frozen Wikipedia article.'
        payload = {'evidence': [{'topic': 'Subject', 'text': text,
                                'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                'revision': 123}],
                   'facts': [{'id': 'bio-1', 'fact_id': 'bio-1:f0', 'fact': 'A fact.', 'topic': 'Subject'}]}
        retrieval.validate_inputs(payload)
        for changed in ({**payload['evidence'][0], 'text_sha256': 'wrong'},
                        {**payload['evidence'][0], 'topic': 'Other'}):
            with self.assertRaises(ValueError):
                retrieval.validate_inputs({**payload, 'evidence': [changed]})

    def test_remote_guard_precedes_model_import(self):
        with self.assertRaisesRegex(RuntimeError, 'Remote-only'):
            retrieval.run({})


if __name__ == '__main__':
    unittest.main()
