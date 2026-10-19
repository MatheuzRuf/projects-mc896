import json

import pytest

from project2_from_project1.src.entity_recognition.corpus import EntityRecognitionError
from project2_from_project1.src.entity_recognition.nested_cv import (
    L2_GRID,
    StudyInputs,
    inner_partitions,
    run_nested_study,
    select_l2,
)
from project2_from_project1.tests.test_entity_dictionary import TEST, TRAIN, VALIDATION, _write_inputs


def test_inner_partitions_are_document_disjoint_cover_outer_train_and_are_seeded():
    ids = [f"doc-{index}" for index in range(10)]
    folds = inner_partitions(ids, 2)
    assert len(folds) == 3
    validation = [doc for fold in folds for doc in fold.validation_document_ids]
    assert sorted(validation) == sorted(ids)
    for fold in folds:
        assert not set(fold.train_document_ids) & set(fold.validation_document_ids)
        assert set(fold.train_document_ids) | set(fold.validation_document_ids) == set(ids)
    assert inner_partitions(ids, 2) == folds
    assert inner_partitions(ids, 3) != folds


def test_selection_uses_mean_then_variance_then_strongest_regularization():
    assert select_l2({0.01: [0.5, 0.5, 0.5], 0.1: [0.6, 0.6, 0.6], 1.0: [0.1, 0.1, 0.1], 10.0: [0.2] * 3})["selected_l2"] == 0.1
    tie_mean = select_l2({0.01: [0.4, 0.6, 0.5], 0.1: [0.5, 0.5, 0.5], 1.0: [0.0] * 3, 10.0: [0.0] * 3})
    assert tie_mean["selected_l2"] == 0.1
    full_tie = select_l2({0.01: [0.5] * 3, 0.1: [0.5] * 3, 1.0: [0.5] * 3, 10.0: [0.5] * 3})
    assert full_tie["selected_l2"] == 10.0
    failed = select_l2({0.01: [0.9, None, 0.9], 0.1: [0.2] * 3, 1.0: [0.1] * 3, 10.0: [0.1] * 3})
    assert failed["selected_l2"] == 0.1 and failed["table"]["0.01"]["eligible"] is False
    with pytest.raises(EntityRecognitionError):
        select_l2({0.01: [None] * 3})


def test_nested_study_runs_both_arms_on_the_same_folds_resumes_and_never_reads_test(tmp_path):
    corpus, projection, projection_manifest, partitions = _write_inputs(tmp_path, (*TRAIN, VALIDATION, TEST))
    inputs = StudyInputs(str(corpus), str(projection), str(projection_manifest), str(partitions))
    output = tmp_path / "study"
    summary = run_nested_study(inputs, output, max_iterations=15)

    crf = summary["arms"]["semi-markov-crf"]
    dictionary = summary["arms"]["canonical-dictionary"]
    assert [row["fold"] for row in crf["folds"]] == [row["fold"] for row in dictionary["folds"]] == [1, 2]
    assert all(row["inner_runs"]["total"] == len(L2_GRID) * 3 for row in crf["folds"])
    assert all(row["selected_l2"] in L2_GRID for row in crf["folds"])
    assert crf["aggregate"]["evaluated_folds"] == dictionary["aggregate"]["evaluated_folds"] == 2
    assert summary["test_partition_read"] is False
    serialized = (output / "summary.json").read_text(encoding="utf-8")
    assert '"d5"' not in serialized

    selection = json.loads((output / "crf" / "fold-01" / "selection.json").read_text(encoding="utf-8"))
    outer_train = {"d1", "d2", "d3"}
    for item in selection["inner_partitions"]:
        assert set(item["train_document_ids"]) | set(item["validation_document_ids"]) == outer_train

    inner = output / "crf" / "fold-01" / "inner" / "l2-1" / "inner-1.json"
    stamp = inner.stat().st_mtime_ns
    again = run_nested_study(inputs, output, max_iterations=15)
    assert inner.stat().st_mtime_ns == stamp
    assert again["arms"]["semi-markov-crf"]["aggregate"]["macro_f1"] == crf["aggregate"]["macro_f1"]


def test_grid_is_fixed(tmp_path):
    with pytest.raises(EntityRecognitionError, match="pre-registered"):
        run_nested_study(StudyInputs("a", "b", "c", "d"), tmp_path, l2_grid=(0.5,))
