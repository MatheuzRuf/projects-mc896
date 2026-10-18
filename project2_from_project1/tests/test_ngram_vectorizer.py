from collections import Counter

from project2_from_project1.src.ngrams.ngrams_extractor import (
    build_ngram_vocabulary,
    vectorize_ngrams,
)


def test_vectorizer_preserves_frequencies_and_distinguishes_ngram_types():
    counts = (
        Counter({"pain": 2}),
        Counter({("pain", "pain"): 3}),
        Counter({("pain", "pain", "pain"): 4}),
    )

    vocabulary = build_ngram_vocabulary([counts])
    vector = vectorize_ngrams(counts, vocabulary)

    assert vocabulary == (
        (1, ("pain",)),
        (2, ("pain", "pain")),
        (3, ("pain", "pain", "pain")),
    )
    assert vector == [2, 3, 4]
    assert vectorize_ngrams(counts, vocabulary, sparse=True) == {
        0: 2,
        1: 3,
        2: 4,
    }


def test_two_counters_use_the_same_deterministic_feature_schema():
    first = (
        Counter({"patient": 2, "pain": 1}),
        Counter({("patient", "pain"): 1}),
        Counter(),
    )
    second = (
        Counter({"pain": 3}),
        Counter(),
        Counter({("patient", "pain", "patient"): 2}),
    )

    vocabulary = build_ngram_vocabulary([first, second])
    first_vector = vectorize_ngrams(first, vocabulary)

    assert vocabulary == build_ngram_vocabulary([second, first])
    assert first_vector == [1, 2, 1, 0]
    assert first_vector == vectorize_ngrams(first, vocabulary)
    assert vectorize_ngrams(second, vocabulary) == [3, 0, 0, 2]


def test_empty_counters_produce_empty_or_zero_vectors():
    empty_counts = (Counter(), Counter(), Counter())

    assert build_ngram_vocabulary([empty_counts]) == ()
    assert vectorize_ngrams(empty_counts, ()) == []

    vocabulary = build_ngram_vocabulary(
        [(Counter({"patient": 1}), Counter(), Counter())]
    )
    assert vectorize_ngrams(empty_counts, vocabulary) == [0]


def test_vectorization_does_not_modify_input_counters():
    counts = (
        Counter({"patient": 2}),
        Counter({("patient", "pain"): 1}),
        Counter(),
    )
    original_counts = tuple(counter.copy() for counter in counts)
    vocabulary = build_ngram_vocabulary([counts])

    vectorize_ngrams(counts, vocabulary)

    assert counts == original_counts
