from collections import Counter

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from project2_from_project1.src.ngram.feature_matrix import (
    CandidateMetadata,
    NgramCandidateExample,
    NgramFeatureVectorizer,
)
from project2_from_project1.src.ngram.knn_classifier import NgramKNNClassifier
from project2_from_project1.src.ngram.ngram_extractor import (
    build_ngram_vocabulary,
    vectorize_ngrams,
)


def test_classifier_learns_known_classes_and_returns_one_prediction_per_row():
    X_train = np.array([[1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9]])
    y_train = ["alpha", "beta", "alpha", "beta"]
    classifier = NgramKNNClassifier(n_neighbors=1).fit(X_train, y_train)

    predictions = classifier.predict([[0.8, 0.2], [0.2, 0.8]])

    assert predictions.tolist() == ["alpha", "beta"]
    assert len(predictions) == 2


def test_classifier_accepts_sparse_training_and_prediction_matrices():
    X_train = csr_matrix([[1, 0], [0, 1], [0.9, 0.1]])
    classifier = NgramKNNClassifier(n_neighbors=1).fit(
        X_train, ["alpha", "beta", "alpha"]
    )

    assert classifier.predict(csr_matrix([[0.8, 0.2]])).tolist() == ["alpha"]


def test_default_neighbor_count_is_three():
    classifier = NgramKNNClassifier().fit(
        [[1, 0], [0, 1], [0.9, 0.1]],
        ["alpha", "beta", "alpha"],
    )

    assert classifier.n_neighbors == 3
    assert classifier.predict([[1, 0]]).tolist() == ["alpha"]


def test_cosine_vote_tie_uses_first_sorted_class():
    classifier = NgramKNNClassifier(n_neighbors=2).fit(
        [[1, 0], [0, 1]],
        ["zeta", "alpha"],
    )

    assert classifier.predict([[1, 1]]).tolist() == ["alpha"]


@pytest.mark.parametrize("n_neighbors", [0, -1, True, 1.5])
def test_rejects_invalid_neighbor_counts(n_neighbors):
    with pytest.raises(ValueError, match="positive integer"):
        NgramKNNClassifier(n_neighbors=n_neighbors)


def test_rejects_more_neighbors_than_training_rows():
    with pytest.raises(ValueError, match="cannot exceed"):
        NgramKNNClassifier(n_neighbors=3).fit([[1, 0], [0, 1]], ["a", "b"])


@pytest.mark.parametrize(
    ("X_train", "y_train", "message"),
    [
        (csr_matrix((0, 2)), [], "at least one training row"),
        ([[1, 0]], None, "must contain training labels"),
        ([[1, 0], [0, 1]], ["a"], "row count must match"),
        ([[1, 0]], [["a"]], "one-dimensional"),
        (csr_matrix((2, 0)), ["a", "b"], "at least one feature"),
        ([1, 2], ["a", "b"], "two-dimensional"),
    ],
)
def test_rejects_invalid_training_inputs(X_train, y_train, message):
    with pytest.raises(ValueError, match=message):
        NgramKNNClassifier(n_neighbors=1).fit(X_train, y_train)


def test_rejects_missing_training_labels():
    with pytest.raises(ValueError, match="missing labels"):
        NgramKNNClassifier(n_neighbors=1).fit([[1, 0], [0, 1]], ["a", None])


def test_predict_empty_matrix_returns_empty_predictions():
    classifier = NgramKNNClassifier(n_neighbors=1).fit([[1, 0]], ["alpha"])

    predictions = classifier.predict(csr_matrix((0, 2)))

    assert predictions.shape == (0,)


def test_predict_rejects_invalid_dimensions_and_feature_count():
    classifier = NgramKNNClassifier(n_neighbors=1).fit([[1, 0]], ["alpha"])

    with pytest.raises(ValueError, match="two-dimensional"):
        classifier.predict([1, 0])
    with pytest.raises(ValueError, match="must have 2 features"):
        classifier.predict([[1, 0, 1]])


def test_predict_before_fit_fails_explicitly():
    with pytest.raises(RuntimeError, match="has not been fitted"):
        NgramKNNClassifier().predict([[1, 0]])


def test_prediction_does_not_require_or_use_validation_labels():
    classifier = NgramKNNClassifier(n_neighbors=1).fit(
        [[1, 0], [0, 1]], ["alpha", "beta"]
    )

    prediction_without_labels = classifier.predict([[0.9, 0.1]])
    unrelated_validation_labels = ["beta"]
    prediction_with_separate_labels = classifier.predict([[0.9, 0.1]])

    assert unrelated_validation_labels == ["beta"]
    assert np.array_equal(prediction_without_labels, prediction_with_separate_labels)


def test_classifier_accepts_feature_matrix_from_stage_two():
    examples = [
        NgramCandidateExample(
            (Counter({"alpha": 2}), Counter(), Counter()),
            CandidateMetadata("train-a", "s1", "t1"),
            "alpha",
        ),
        NgramCandidateExample(
            (Counter({"beta": 2}), Counter(), Counter()),
            CandidateMetadata("train-b", "s2", "t2"),
            "beta",
        ),
    ]
    feature_matrix = NgramFeatureVectorizer().fit_transform(examples)
    classifier = NgramKNNClassifier(n_neighbors=1).fit(
        feature_matrix.X, feature_matrix.y
    )

    predictions = classifier.predict(feature_matrix.X)

    assert predictions.tolist() == ["alpha", "beta"]


def test_full_in_memory_pipeline_from_ngram_counters_to_predictions():
    train_counts = [
        (Counter({"alpha": 2}), Counter(), Counter()),
        (Counter({"alpha": 1}), Counter(), Counter()),
        (Counter({"beta": 2}), Counter(), Counter()),
        (Counter({"beta": 1}), Counter(), Counter()),
    ]
    held_out_counts = [
        (Counter({"alpha": 3}), Counter(), Counter()),
        (Counter({"beta": 3}), Counter(), Counter()),
    ]

    vocabulary = build_ngram_vocabulary(train_counts)
    assert vectorize_ngrams(train_counts[0], vocabulary) == [2, 0]

    training_examples = [
        NgramCandidateExample(
            counts,
            CandidateMetadata(f"train-{index}", "source", "target"),
            label,
        )
        for index, (counts, label) in enumerate(
            zip(train_counts, ["alpha", "alpha", "beta", "beta"])
        )
    ]
    prediction_examples = [
        NgramCandidateExample(
            counts,
            CandidateMetadata(f"held-out-{index}", "source", "target"),
        )
        for index, counts in enumerate(held_out_counts)
    ]

    vectorizer = NgramFeatureVectorizer()
    training_matrix = vectorizer.fit_transform(training_examples)
    prediction_matrix = vectorizer.transform(prediction_examples)
    classifier = NgramKNNClassifier(n_neighbors=1).fit(
        training_matrix.X, training_matrix.y
    )
    predictions = classifier.predict(prediction_matrix.X)

    assert predictions.tolist() == ["alpha", "beta"]
    assert len(predictions) == len(prediction_examples)
    assert len(prediction_matrix.metadata) == len(predictions)
