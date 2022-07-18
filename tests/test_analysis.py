import pytest
import os
import cfro
import numpy as np

"""
Sample invokation: python3 -m pytest tests/test_analysis.py --durations=10 -vv

The durations flag will time the tests, which is useful to benchmark the
optimized algo against the original algo, after verifying its correctness.
"""

NUM_ITERATIONS = 1
NUM_CONFIDENCES = 100


def _gen_confidences_helper(num_true_confidences, num_false_confidences):
    true_match_confidences = np.random.rand(num_true_confidences)
    false_match_confidences = np.random.rand(num_false_confidences)
    return true_match_confidences, false_match_confidences


def _gen_confidences(num_confidences):
    return _gen_confidences_helper(num_confidences, num_confidences)


def test_compute_match_rates():
    for i in range(NUM_ITERATIONS):
        true_match_confidences, false_match_confidences = _gen_confidences_helper(
            NUM_CONFIDENCES, NUM_CONFIDENCES + 1
        )

        match_rates = cfro._compute_match_rates(
            true_match_confidences, false_match_confidences
        )
        match_rates_optimized = cfro._compute_match_rates_optimized(
            true_match_confidences, false_match_confidences
        )

        # Check each of [ts, false_match_rate, false_non_match_rate]
        for i in range(3):
            assert np.array_equal(
                match_rates[i], match_rates_optimized[i]
            ), f"Failure at {i}th index"


def test_compute_match_rates_time():
    for i in range(NUM_ITERATIONS):
        cfro._compute_match_rates(*_gen_confidences(NUM_CONFIDENCES))


def test_compute_match_rates_time_faster():
    for i in range(NUM_ITERATIONS):
        cfro._compute_match_rates_optimized(*_gen_confidences(NUM_CONFIDENCES))
