from collections import Counter

import pytest

from project2_from_project1.src.ngram.feature_matrix import (
    CandidateMetadata,
    NgramCandidateExample,
    NgramFeatureVectorizer,
)
from project2_from_project1.src.ngram.ngram_extractor import vectorize_ngrams


def _counts(unigrams=None, bigrams=None, trigrams=None):
    return (
        Counter(unigrams or {}),
        Counter(bigrams or {}),
        Counter(trigrams or {}),
    )


def _example(document_id, counts, label=None, source_id="source", target_id="target"):
    return NgramCandidateExample(
        ngram_counts=counts,
        metadata=CandidateMetadata(
            document_id=document_id,
            source_id=source_id,
            target_id=target_id,
        ),
        label=label,
    )


def test_fit_transform_shares_columns_preserves_rows_and_keeps_metadata_and_labels_aligned():
    examples = [
        _example("doc-a", _counts({"patient": 2, "pain": 1}), "CAUSES", "a", "b"),
        _example("doc-b", _counts({"pain": 3, "fever": 1}), "NO_RELATION", "c", "d"),
        _example("doc-c", _counts({"patient": 1, "fever": 2}), None, "e", "f"),
    ]
    vectorizer = NgramFeatureVectorizer()

    result = vectorizer.fit_transform(examples)

    assert result.X.shape == (3, 3)
    assert result.X.format == "csr"
    assert result.X.dtype.kind in "iu"
    assert result.X.toarray().tolist() == [
        [0, 1, 2],
        [1, 3, 0],
        [2, 0, 1],
    ]
    assert [item.document_id for item in result.metadata] == [
        "doc-a",
        "doc-b",
        "doc-c",
    ]
    assert [(item.source_id, item.target_id) for item in result.metadata] == [
        ("a", "b"),
        ("c", "d"),
        ("e", "f"),
    ]
    assert result.y == ("CAUSES", "NO_RELATION", None)
    assert result.vocabulary == (
        (1, ("fever",)),
        (1, ("pain",)),
        (1, ("patient",)),
    )
    assert result.vocabulary == vectorizer.vocabulary


def test_fit_transform_preserves_bigram_and_trigram_features():
    example = _example(
        "doc-a",
        _counts(
            {"pain": 2},
            {("pain", "pain"): 3},
            {("pain", "pain", "pain"): 4},
        ),
    )

    result = NgramFeatureVectorizer().fit_transform([example])

    assert result.vocabulary == (
        (1, ("pain",)),
        (2, ("pain", "pain")),
        (3, ("pain", "pain", "pain")),
    )
    assert result.X.toarray().tolist() == [[2, 3, 4]]
    assert result.y is None


def test_transform_uses_training_vocabulary_and_ignores_validation_only_features():
    train = [
        _example("train-a", _counts({"patient": 2})),
        _example("train-b", _counts({"pain": 1})),
    ]
    validation = [
        _example("validation", _counts({"patient": 3, "fever": 5}))
    ]
    vectorizer = NgramFeatureVectorizer()
    vectorizer.fit_transform(train)
    training_vocabulary = vectorizer.vocabulary

    result = vectorizer.transform(validation)

    assert vectorizer.vocabulary == training_vocabulary
    assert result.vocabulary == training_vocabulary
    assert result.X.shape == (1, 2)
    assert result.X.toarray().tolist() == [[0, 3]]


def test_empty_examples_and_empty_collection_have_explicit_sparse_shapes():
    empty_example = _example("empty", _counts())
    vectorizer = NgramFeatureVectorizer()
    fitted = vectorizer.fit_transform([empty_example])

    assert fitted.X.shape == (1, 0)
    assert fitted.X.nnz == 0
    assert fitted.y is None

    validation = vectorizer.transform([])
    assert validation.X.shape == (0, 0)
    assert validation.metadata == ()
    assert validation.y is None

    empty_fit = NgramFeatureVectorizer().fit_transform([])
    assert empty_fit.X.shape == (0, 0)
    assert empty_fit.X.nnz == 0
    assert empty_fit.metadata == ()
    assert empty_fit.y is None

    vectorizer.fit_transform([_example("non-empty", _counts({"pain": 1}))])
    empty_validation = vectorizer.transform([])
    assert empty_validation.X.shape == (0, 1)
    assert empty_validation.metadata == ()
    assert empty_validation.y is None


def test_transform_before_fit_fails_explicitly():
    with pytest.raises(RuntimeError, match="has not been fitted"):
        NgramFeatureVectorizer().transform([])


def test_sparse_matrix_rows_match_stage_one_vectorizer():
    examples = [
        _example("doc-a", _counts({"pain": 2}, {("patient", "pain"): 1})),
        _example("doc-b", _counts({"fever": 3})),
    ]
    vectorizer = NgramFeatureVectorizer()
    result = vectorizer.fit_transform(examples)

    assert result.X.toarray().tolist() == [
        vectorize_ngrams(examples[0].ngram_counts, result.vocabulary),
        vectorize_ngrams(examples[1].ngram_counts, result.vocabulary),
    ]
    assert result.X.nnz == 3


def test_sparse_vectorization_does_not_materialize_zero_columns():
    counts = _counts({"observed": 2})
    vocabulary = (
        (1, ("absent-a",)),
        (1, ("absent-b",)),
        (1, ("observed",)),
    )

    assert vectorize_ngrams(counts, vocabulary, sparse=True) == {2: 2}


def test_additional_candidate_identifiers_remain_in_metadata():
    metadata = CandidateMetadata(
        "doc-a",
        "source",
        "target",
        additional_ids=(("candidate_relation", "CAUSES"), ("candidate_id", "candidate-1")),
    )
    result = NgramFeatureVectorizer().fit_transform(
        [NgramCandidateExample(_counts({"pain": 1}), metadata)]
    )

    assert result.metadata == (metadata,)
    assert result.X.shape == (1, 1)
