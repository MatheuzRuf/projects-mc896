"""Semi-Markov CRF dynamic programming owned by the project.

All functions operate on a batch of sentences padded to ``T`` tokens and on a
dense tensor of segment scores ``S[b, i, d - 1, y]``: the score of a segment
that starts at token ``i``, spans ``d`` tokens and has label ``y``.  Label
``OUTSIDE`` (index 0) only forms unit segments, so each flat entity output has
exactly one segmentation.

A segmentation ``(i_1, d_1, y_1), ..., (i_K, d_K, y_K)`` of a sentence of
length ``n`` scores

    start[y_1] + sum_k S[i_k, d_k, y_k] + sum_k trans[y_{k-1}, y_k] + end[y_K]

The recursion keeps, for every position ``j``:

* ``alpha[j, y]``: log-sum over prefixes ``[0, j)`` whose last segment has label ``y``;
* ``inside[j, y]``: log-sum of the scores entering a segment that starts at ``j``
  with label ``y`` (``start`` when ``j == 0``).

Marginals are gradients of ``log Z`` with respect to ``S``, which is exactly
the forward-backward computation obtained through autograd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

OUTSIDE = 0
NEG = -1e30


def valid_segment_mask(lengths: torch.Tensor, max_tokens: int, max_segment: int, labels: int) -> torch.Tensor:
    """Boolean ``(B, T, D, Y)`` mask of segments that lie inside their sentence."""
    starts = torch.arange(max_tokens).view(1, -1, 1, 1)
    sizes = torch.arange(1, max_segment + 1).view(1, 1, -1, 1)
    inside = (starts + sizes) <= lengths.view(-1, 1, 1, 1)
    label = torch.arange(labels).view(1, 1, 1, -1)
    return inside & ((label != OUTSIDE) | (sizes == 1))


def _masked(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch, tokens, segment, labels = scores.shape
    if lengths.shape != (batch,) or int(lengths.min()) < 1 or int(lengths.max()) > tokens:
        raise ValueError("lengths must be in [1, T] for every sentence")
    return scores.masked_fill(~valid_segment_mask(lengths, tokens, segment, labels), NEG)


def _by_end(scores: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Re-index ``S[b, i, d-1]`` as ``E[j-1][b, d-1]`` for segments ending at ``j = i + d``.

    One gather and one ``unbind`` keep autograd from materializing a full-size
    gradient for every position of the recursion.  Entries with ``d > j`` are
    filled with ``NEG``.
    """
    batch, tokens, segment, labels = scores.shape
    ends = torch.arange(tokens).view(-1, 1)
    sizes = torch.arange(segment).view(1, -1)
    starts = ends - sizes
    index = starts.clamp(min=0).view(1, tokens, segment, 1).expand(batch, tokens, segment, labels)
    gathered = scores.gather(1, index)
    gathered = gathered.masked_fill((starts < 0).view(1, tokens, segment, 1), NEG)
    return gathered.unbind(1)


def _entering(inside: list[torch.Tensor], j: int, segment: int) -> torch.Tensor:
    """Stack ``inside[j - d]`` for ``d = 1 .. min(D, j)`` along dimension 1."""
    return torch.stack(inside[max(0, j - segment):j][::-1], dim=1)


def log_partition(
    scores: torch.Tensor, lengths: torch.Tensor, transitions: torch.Tensor, start: torch.Tensor, end: torch.Tensor,
) -> torch.Tensor:
    """``log Z`` per sentence, computed with ``logsumexp`` for stability."""
    scores = _masked(scores, lengths)
    batch, tokens, segment, labels = scores.shape
    columns = _by_end(scores)
    inside = [start.expand(batch, labels)]
    alphas = []
    for j in range(1, tokens + 1):
        width = min(segment, j)
        alpha = torch.logsumexp(_entering(inside, j, segment) + columns[j - 1][:, :width], dim=1)
        alphas.append(alpha)
        inside.append(torch.logsumexp(alpha.unsqueeze(2) + transitions.unsqueeze(0), dim=1))
    final = torch.stack(alphas, dim=1)[torch.arange(batch), lengths - 1]
    return torch.logsumexp(final + end, dim=1)


@dataclass(frozen=True)
class Segment:
    start: int
    size: int
    label: int


def viterbi(
    scores: torch.Tensor, lengths: torch.Tensor, transitions: torch.Tensor, start: torch.Tensor, end: torch.Tensor,
) -> tuple[list[list[Segment]], torch.Tensor]:
    """Highest-scoring segmentation and its score for each sentence."""
    with torch.no_grad():
        scores = _masked(scores, lengths)
        batch, tokens, segment, labels = scores.shape
        columns = _by_end(scores)
        inside = [start.expand(batch, labels)]
        inside_from: list[torch.Tensor | None] = [None]
        alphas = []
        best_size = []
        for j in range(1, tokens + 1):
            width = min(segment, j)
            alpha, size_index = (_entering(inside, j, segment) + columns[j - 1][:, :width]).max(dim=1)
            alphas.append(alpha)
            best_size.append(size_index + 1)
            value, previous = (alpha.unsqueeze(2) + transitions.unsqueeze(0)).max(dim=1)
            inside.append(value)
            inside_from.append(previous)
        rows = torch.arange(batch)
        final = torch.stack(alphas, dim=1)[rows, lengths - 1] + end
        best, label = final.max(dim=1)
        paths = []
        for b in range(batch):
            j, y = int(lengths[b]), int(label[b])
            path = []
            while j > 0:
                size = int(best_size[j - 1][b, y])
                path.append(Segment(j - size, size, y))
                j -= size
                if j > 0:
                    y = int(inside_from[j][b, y])
            paths.append(path[::-1])
        return paths, best


def segment_marginals(
    scores: torch.Tensor, lengths: torch.Tensor, transitions: torch.Tensor, start: torch.Tensor, end: torch.Tensor,
) -> torch.Tensor:
    """Posterior probability of every segment, ``d log Z / d S``; zero when invalid."""
    leaf = scores.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        total = log_partition(leaf, lengths, transitions.detach(), start.detach(), end.detach()).sum()
        (gradient,) = torch.autograd.grad(total, leaf)
    return gradient.clamp(0.0, 1.0)


def path_score(
    scores: torch.Tensor, lengths: torch.Tensor, transitions: torch.Tensor, start: torch.Tensor, end: torch.Tensor,
    paths: list[list[Segment]],
) -> torch.Tensor:
    """Differentiable score of one given segmentation per sentence."""
    scores = _masked(scores, lengths)
    batch_index, starts, sizes, labels, firsts, lasts = [], [], [], [], [], []
    trans_batch, trans_from, trans_to = [], [], []
    for b, path in enumerate(paths):
        if sum(item.size for item in path) != int(lengths[b]) or path[0].start != 0:
            raise ValueError(f"path {b} does not cover its sentence")
        for previous, current in zip(path, path[1:]):
            if previous.start + previous.size != current.start:
                raise ValueError(f"path {b} is not contiguous")
            trans_batch.append(b)
            trans_from.append(previous.label)
            trans_to.append(current.label)
        for item in path:
            batch_index.append(b)
            starts.append(item.start)
            sizes.append(item.size - 1)
            labels.append(item.label)
        firsts.append(path[0].label)
        lasts.append(path[-1].label)
    batch = len(paths)
    total = torch.zeros(batch, dtype=scores.dtype)
    total = total.index_add(0, torch.tensor(batch_index), scores[batch_index, starts, sizes, labels])
    if trans_batch:
        total = total.index_add(0, torch.tensor(trans_batch), transitions[trans_from, trans_to])
    return total + start[firsts] + end[lasts]


__all__ = ["NEG", "OUTSIDE", "Segment", "log_partition", "path_score", "segment_marginals", "valid_segment_mask", "viterbi"]
