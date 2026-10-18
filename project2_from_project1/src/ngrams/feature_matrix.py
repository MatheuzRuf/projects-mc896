"""Build sparse candidate feature matrices from precomputed n-gram counts."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass

from scipy.sparse import csr_matrix

from project2_from_project1.src.ngram.ngram_extractor import (
    NgramCounts,
    NgramFeature,
    build_ngram_vocabulary,
    vectorize_ngrams,
)


@dataclass(frozen=True)
class CandidateMetadata:
    """Identifiers kept outside the numerical feature matrix."""

    document_id: str
    source_id: str
    target_id: str
    additional_ids: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class NgramCandidateExample:
    """One candidate's n-gram counts, metadata, and optional label."""

    ngram_counts: NgramCounts
    metadata: CandidateMetadata
    label: Hashable | None = None


@dataclass(frozen=True)
class NgramFeatureMatrix:
    """Sparse feature rows with the matching metadata and optional labels."""

    X: csr_matrix
    metadata: tuple[CandidateMetadata, ...]
    y: tuple[Hashable | None, ...] | None
    vocabulary: tuple[NgramFeature, ...]


class NgramFeatureVectorizer:
    """Fit one shared n-gram schema and reuse it for later candidate batches.

    An empty fit batch produces a fitted ``(0, 0)`` matrix. Empty examples
    produce zero rows; transforming an empty batch retains the fitted width.
    """

    def __init__(self) -> None:
        self._vocabulary: tuple[NgramFeature, ...] | None = None

    @property
    def vocabulary(self) -> tuple[NgramFeature, ...]:
        """Return the fitted feature schema."""
        if self._vocabulary is None:
            raise RuntimeError("the n-gram vectorizer has not been fitted")
        return self._vocabulary

    def fit_transform(
        self,
        examples: Iterable[NgramCandidateExample],
    ) -> NgramFeatureMatrix:
        """Learn features only from this batch, then transform it in order."""
        examples = tuple(examples)
        vocabulary = build_ngram_vocabulary(
            example.ngram_counts for example in examples
        )
        result = self._transform(examples, vocabulary)
        self._vocabulary = vocabulary
        return result

    def transform(
        self,
        examples: Iterable[NgramCandidateExample],
    ) -> NgramFeatureMatrix:
        """Transform examples with the fitted schema, ignoring unseen features."""
        return self._transform(tuple(examples), self.vocabulary)

    @staticmethod
    def _transform(
        examples: tuple[NgramCandidateExample, ...],
        vocabulary: tuple[NgramFeature, ...],
    ) -> NgramFeatureMatrix:
        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[int] = []

        for row_index, example in enumerate(examples):
            row = vectorize_ngrams(example.ngram_counts, vocabulary, sparse=True)
            for column_index, value in row.items():
                row_indices.append(row_index)
                column_indices.append(column_index)
                values.append(value)

        matrix = csr_matrix(
            (values, (row_indices, column_indices)),
            shape=(len(examples), len(vocabulary)),
            dtype=int,
        )
        labels = tuple(example.label for example in examples)
        has_labels = any(label is not None for label in labels)
        return NgramFeatureMatrix(
            X=matrix,
            metadata=tuple(example.metadata for example in examples),
            y=labels if has_labels else None,
            vocabulary=vocabulary,
        )
