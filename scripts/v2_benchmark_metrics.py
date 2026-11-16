"""Metric arithmetic only. No models, API calls, or factual adjudication."""
from __future__ import annotations

from collections import Counter
import math


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def simpleqa(labels, expected=1000):
    counts = Counter(labels)
    if set(counts) - {'A', 'B', 'C'} or len(labels) > expected:
        raise ValueError('Invalid SimpleQA labels/count')
    answered = counts['A'] + counts['B']
    return {'expected': expected, 'graded': len(labels), 'missing_or_error': expected - len(labels),
            'correct': counts['A'], 'incorrect': counts['B'], 'not_attempted': counts['C'],
            'correct_answer_yield': ratio(counts['A'], expected),
            'answer_coverage': ratio(answered, expected), 'accuracy_when_attempted': ratio(counts['A'], answered)}


def precise_wiki(records, expected=1000):
    if len(records) > expected:
        raise ValueError('Too many PreciseWikiQA records')
    abstained = incorrect = unverifiable = correct = non_abstained = 0
    for record in records:
        if type(record['abstained']) is not bool or record['correctness'] not in ('CORRECT', 'INCORRECT', 'UNVERIFIABLE'):
            raise ValueError('Invalid PreciseWikiQA labels')
        if record['abstained']:
            abstained += 1
        else:
            non_abstained += 1
            incorrect += record['correctness'] == 'INCORRECT'
            unverifiable += record['correctness'] == 'UNVERIFIABLE'
        correct += record['correctness'] == 'CORRECT'
    return {'expected': expected, 'graded': len(records), 'missing_or_error': expected - len(records),
            'abstained': abstained, 'non_abstained': non_abstained,
            'incorrect_non_abstained': incorrect, 'unverifiable_non_abstained': unverifiable,
            'false_refusal_rate': ratio(abstained, expected),
            'hallucination_rate_when_not_abstained': ratio(incorrect + unverifiable, non_abstained),
            'correct_answer_rate': ratio(correct, expected)}


def nonexistent(labels, expected=3000):
    if len(labels) > expected or any(type(label) is not bool for label in labels):
        raise ValueError('Invalid does_believe labels/count')
    believed = sum(labels)
    return {'expected': expected, 'graded': len(labels), 'missing_or_error': expected - len(labels),
            'false_acceptances': believed, 'false_acceptance_rate': ratio(believed, expected)}


def abstention(records, expected=978):
    behavior = joint = 0
    conditional_expected = conditional_missing = 0
    for record in records:
        desired, predicted = record['should_abstain'], record['abstained']
        if type(desired) is not bool or type(predicted) is not bool:
            raise ValueError('Invalid AbstentionBench labels')
        behavior += desired == predicted
        if desired:
            joint += predicted
        elif not predicted:
            conditional_expected += 1
            correctness = record.get('correct')
            if correctness is None:
                conditional_missing += 1
            elif type(correctness) is not bool:
                raise ValueError('Invalid conditional correctness')
            else:
                joint += correctness
    if len(records) > expected:
        raise ValueError('Too many AbstentionBench rows')
    return {'expected': expected, 'abstention_graded': len(records),
            'missing_or_error': expected - len(records),
            'conditional_correctness_expected': conditional_expected,
            'conditional_correctness_missing_or_error': conditional_missing,
            'behavior_correct': behavior, 'joint_successes': joint,
            'behavior_accuracy': ratio(behavior, expected), 'joint_task_success': ratio(joint, expected)}


def selfaware(records, expected=3369, answerable_count=2338):
    tp = fp = fn = correct = 0
    for record in records:
        answerable, predicted = record['answerable'], record['pred_unanswerable']
        if type(answerable) is not bool or type(predicted) is not bool:
            raise ValueError('Invalid SelfAware labels')
        tp += not answerable and predicted
        fp += answerable and predicted
        fn += not answerable and not predicted
        if answerable:
            match = record.get('answerable_reference_correct')
            if type(match) is not bool:
                raise ValueError('Missing answerable reference match')
            correct += match
    if len(records) > expected:
        raise ValueError('Too many SelfAware rows')
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'expected': expected, 'graded': len(records), 'missing_or_error': expected - len(records),
            'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'answerable_reference_correct': correct, 'answerable_reference_accuracy': ratio(correct, answerable_count)}


def factscore(records, expected=180):
    precision, penalized = [], []
    no_facts = supported = facts = 0
    for record in records:
        labels = record['supported']
        if not isinstance(labels, list) or any(type(label) is not bool for label in labels):
            raise ValueError('Invalid FActScore support judgments')
        if not labels:
            no_facts += 1
            continue
        total, correct = len(labels), sum(labels)
        value = correct / total
        precision.append(value)
        penalized.append(value * (math.exp(1 - 10 / total) if total <= 10 else 1))
        supported += correct
        facts += total
    if len(records) > expected:
        raise ValueError('Too many biography records')
    return {'expected': expected, 'graded': len(records), 'missing_or_error': expected - len(records),
            'no_fact_responses': no_facts, 'responding_biographies': len(precision),
            'responding_fraction': ratio(len(precision), expected),
            'supported_facts': supported, 'total_facts': facts,
            'macro_supported_fact_precision': ratio(sum(precision), len(precision)),
            'gamma_10_penalized_macro': ratio(sum(penalized), len(penalized)),
            'mean_facts_per_responding_biography': ratio(facts, len(precision))}
