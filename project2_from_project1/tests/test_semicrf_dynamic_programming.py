import math

import pytest
import torch

from project2_from_project1.src.entity_recognition.semicrf import (
    OUTSIDE,
    Segment,
    log_partition,
    path_score,
    segment_marginals,
    viterbi,
)

LABELS = 3


def _segmentations(length, max_segment, position=0):
    if position == length:
        yield []
        return
    for label in range(LABELS):
        sizes = [1] if label == OUTSIDE else range(1, min(max_segment, length - position) + 1)
        for size in sizes:
            for rest in _segmentations(length, max_segment, position + size):
                yield [Segment(position, size, label), *rest]


def _score(path, scores, transitions, start, end):
    total = start[path[0].label] + end[path[-1].label]
    total += sum(scores[item.start, item.size - 1, item.label] for item in path)
    total += sum(transitions[left.label, right.label] for left, right in zip(path, path[1:]))
    return float(total)


def _random_problem(seed, lengths, max_segment=3):
    generator = torch.Generator().manual_seed(seed)
    tokens = max(lengths)
    shape = (len(lengths), tokens, max_segment, LABELS)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float64),
        torch.tensor(lengths),
        torch.randn((LABELS, LABELS), generator=generator, dtype=torch.float64),
        torch.randn(LABELS, generator=generator, dtype=torch.float64),
        torch.randn(LABELS, generator=generator, dtype=torch.float64),
    )


@pytest.mark.parametrize("seed", range(5))
def test_partition_viterbi_and_marginals_match_exhaustive_enumeration(seed):
    lengths = [5, 1, 3, 4]
    scores, lengths_tensor, transitions, start, end = _random_problem(seed, lengths)

    log_z = log_partition(scores, lengths_tensor, transitions, start, end)
    paths, best = viterbi(scores, lengths_tensor, transitions, start, end)
    marginals = segment_marginals(scores, lengths_tensor, transitions, start, end)

    for b, length in enumerate(lengths):
        everything = list(_segmentations(length, 3))
        values = [_score(path, scores[b], transitions, start, end) for path in everything]
        expected_log_z = max(values) + math.log(sum(math.exp(value - max(values)) for value in values))
        assert float(log_z[b]) == pytest.approx(expected_log_z, abs=1e-10)

        best_index = max(range(len(values)), key=values.__getitem__)
        assert paths[b] == everything[best_index]
        assert float(best[b]) == pytest.approx(values[best_index], abs=1e-10)

        expected = torch.zeros_like(scores[b])
        for path, value in zip(everything, values):
            for item in path:
                expected[item.start, item.size - 1, item.label] += math.exp(value - expected_log_z)
        assert torch.allclose(marginals[b], expected, atol=1e-10)


def test_path_score_matches_manual_sum_and_gives_finite_gradients():
    scores, lengths, transitions, start, end = _random_problem(7, [4])
    path = [Segment(0, 1, 0), Segment(1, 3, 2)]
    scores.requires_grad_(True)
    value = path_score(scores, lengths, transitions, start, end, [path])
    assert float(value.detach()[0]) == pytest.approx(_score(path, scores.detach()[0], transitions, start, end))
    (log_partition(scores, lengths, transitions, start, end) - value).sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_outside_segments_are_unit_and_long_entities_respect_the_limit():
    scores, lengths, transitions, start, end = _random_problem(3, [4], max_segment=2)
    scores[0, 0, 1, OUTSIDE] = 1e6  # an invalid multi-token outside segment must never win
    paths, _ = viterbi(scores, lengths, transitions, start, end)
    assert all(item.size == 1 for item in paths[0] if item.label == OUTSIDE)
    assert all(item.size <= 2 for item in paths[0])
    marginals = segment_marginals(scores, lengths, transitions, start, end)
    assert float(marginals[0, 0, 1, OUTSIDE]) == 0.0


def test_large_scores_remain_numerically_stable():
    scores, lengths, transitions, start, end = _random_problem(11, [5, 2])
    log_z = log_partition(scores * 1e3, lengths, transitions * 1e3, start, end)
    assert torch.isfinite(log_z).all()
    marginals = segment_marginals(scores * 1e3, lengths, transitions * 1e3, start, end)
    assert torch.isfinite(marginals).all()
    # Every token is covered by exactly one segment in every segmentation.
    for b, length in enumerate([5, 2]):
        for token in range(length):
            covered = sum(
                marginals[b, i, d - 1].sum()
                for i in range(length) for d in range(1, 4) if i <= token < i + d
            )
            assert float(covered) == pytest.approx(1.0, abs=1e-8)
