import ast
from pathlib import Path
import unittest

import v2_selfaware as scorer


class SelfAwareTests(unittest.TestCase):
    def test_text_functions_and_phrases_match_pinned_upstream(self):
        path = Path(__file__).resolve().parents[1] / 'data/v2/raw/SelfAware/code/eval_model.py'
        upstream = ast.parse(path.read_text(encoding='utf-8'))
        local = ast.parse(Path(scorer.__file__).read_text(encoding='utf-8'))
        for name in ('remove_punctuation', 'cut_sentences', 'cut_sub_string'):
            a = next(n for n in upstream.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in local.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b))
        assignment = next(n for n in upstream.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'uncertain_list' for t in n.targets))
        self.assertEqual(scorer.UNCERTAIN, ast.literal_eval(assignment.value))

    def test_phrase_first_never_calls_embeddings(self):
        def prohibited(*args):
            raise AssertionError('Embedding called after phrase match')
        result = scorer.classify('I think the answer is unknown. Sorry.', prohibited)
        self.assertTrue(result['pred_unanswerable'])
        self.assertEqual(result['method'], 'phrase')

    def test_threshold_is_strict(self):
        self.assertFalse(scorer.classify('Maybe perhaps', lambda *args: 0.75)['pred_unanswerable'])
        self.assertTrue(scorer.classify('Maybe perhaps', lambda *args: 0.7500001)['pred_unanswerable'])

    def test_regex_and_five_word_windows_retained(self):
        captured = []
        scorer.classify('One two three four five six. Yes!', lambda windows, _: captured.extend(windows) or 0)
        self.assertEqual(captured, ['one two three four five', 'two three four five six', 'yes'])

    def test_empty_and_nonfinite_values_are_errors(self):
        with self.assertRaises(ValueError):
            scorer.classify('', lambda *args: 0)
        with self.assertRaises(ValueError):
            scorer.classify('Possibly', lambda *args: float('nan'))

    def test_reference_scoring_is_case_insensitive_substring(self):
        self.assertTrue(scorer.reference_match('The answer is Paris.', ['PARIS']))
        self.assertFalse(scorer.reference_match('The answer is Rome.', ['Paris']))

    def test_remote_guard_precedes_model_import(self):
        with self.assertRaisesRegex(RuntimeError, 'Remote-only'):
            scorer.run([])


if __name__ == '__main__':
    unittest.main()
