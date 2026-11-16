import math
import unittest

import v2_benchmark_metrics as metrics


class MetricTests(unittest.TestCase):
    def test_simpleqa_coverage_and_conditional_accuracy_are_distinct(self):
        result = metrics.simpleqa(['A', 'B', 'C'], expected=3)
        self.assertEqual(result['correct_answer_yield'], 1 / 3)
        self.assertEqual(result['answer_coverage'], 2 / 3)
        self.assertEqual(result['accuracy_when_attempted'], 0.5)

    def test_precise_wiki_unverifiable_is_hallucination_only_nonabstained(self):
        result = metrics.precise_wiki([
            {'abstained': False, 'correctness': 'CORRECT'},
            {'abstained': False, 'correctness': 'UNVERIFIABLE'},
            {'abstained': True, 'correctness': 'INCORRECT'}], expected=3)
        self.assertEqual(result['hallucination_rate_when_not_abstained'], 0.5)
        self.assertEqual(result['false_refusal_rate'], 1 / 3)

    def test_abstention_joint_success_is_conditional(self):
        result = metrics.abstention([
            {'should_abstain': True, 'abstained': True},
            {'should_abstain': False, 'abstained': False, 'correct': True},
            {'should_abstain': False, 'abstained': False, 'correct': False}], expected=3)
        self.assertEqual(result['behavior_accuracy'], 1)
        self.assertEqual(result['joint_task_success'], 2 / 3)

    def test_missing_grades_are_exposed_not_silently_dropped(self):
        result = metrics.nonexistent([False], expected=3)
        self.assertEqual(result['missing_or_error'], 2)
        result = metrics.abstention([{'should_abstain': False, 'abstained': False}], expected=1)
        self.assertEqual(result['conditional_correctness_missing_or_error'], 1)

    def test_selfaware_f1_preserves_released_flags(self):
        result = metrics.selfaware([
            {'answerable': False, 'pred_unanswerable': True},
            {'answerable': False, 'pred_unanswerable': False},
            {'answerable': True, 'pred_unanswerable': True, 'answerable_reference_correct': True}],
            expected=3, answerable_count=1)
        self.assertEqual(result['f1'], 0.5)
        self.assertEqual(result['answerable_reference_accuracy'], 1)

    def test_factscore_is_macro_and_no_fact_responses_are_excluded(self):
        result = metrics.factscore([{'supported': [True]}, {'supported': [False] * 20}, {'supported': []}], expected=3)
        self.assertEqual(result['macro_supported_fact_precision'], 0.5)
        self.assertEqual(result['responding_fraction'], 2 / 3)
        self.assertEqual(result['gamma_10_penalized_macro'], math.exp(-9) / 2)
        self.assertEqual(result['no_fact_responses'], 1)


if __name__ == '__main__':
    unittest.main()
