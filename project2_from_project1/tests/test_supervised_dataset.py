from pathlib import Path

import pytest

from project2_from_project1.src.clinical_documents import (
    AnnotatedRelation,
    CanonicalRelation,
    ClinicalCase,
    ClinicalDocument,
    Mention,
    Patient,
    Span,
    write_jsonl,
)
from project2_from_project1.src.mcrc_suitability import MCRCSuitabilityReport, SuitabilityCriteria, SuitabilityDecision, write_mcrc_suitability_report
from project2_from_project1.src.supervised_dataset import (
    CandidatePolicy,
    PartitionPolicy,
    SupervisedDatasetError,
    build_frozen_supervised_dataset,
    build_frozen_supervised_dataset_from_files,
    generate_candidates,
    gold_outside_window,
    write_frozen_supervised_dataset,
)


def _document(document_id: str, extra_findings: int = 0) -> ClinicalDocument:
    text = "Patient condition factor. finding. No finding." + " finding." * extra_findings
    case_id, patient_id = f"{document_id}:case", f"{document_id}:patient"
    starts = [index for index in range(len(text)) if text.startswith("finding", index)]
    mentions = [
        Mention(f"{document_id}:condition", [Span(text.index("condition"), text.index("condition") + 9)], ["condition"], "Condition"),
        Mention(f"{document_id}:finding", [Span(starts[0], starts[0] + 7)], ["finding"], "Finding"),
        Mention(f"{document_id}:factor", [Span(text.index("factor"), text.index("factor") + 6)], ["factor"], "Factor"),
        Mention(f"{document_id}:modifier", [Span(text.index("No"), text.index("No") + 2)], ["modifier"], "Modifier"),
        Mention(f"{document_id}:negated", [Span(starts[1], starts[1] + 7)], ["finding"], "Finding"),
    ]
    mentions.extend(
        Mention(f"{document_id}:extra:{index}", [Span(start, start + 7)], ["finding"], "Finding")
        for index, start in enumerate(starts[2:])
    )
    condition, finding, factor, modifier, negated = mentions[:5]
    relations = [
        AnnotatedRelation(f"{document_id}:describes", case_id, patient_id, "describes", "DESCRIBES"),
        AnnotatedRelation(f"{document_id}:condition", patient_id, condition.mention_id, "case_has_condition", "HAS_CONDITION"),
        AnnotatedRelation(f"{document_id}:finding", patient_id, finding.mention_id, "case_has_finding", "HAS_FINDING"),
        AnnotatedRelation(f"{document_id}:factor", patient_id, factor.mention_id, "case_has_factor", "HAS_FACTOR"),
        AnnotatedRelation(f"{document_id}:negates", modifier.mention_id, negated.mention_id, "modifier_modifies_finding", "NEGATES"),
        AnnotatedRelation(f"{document_id}:causes", condition.mention_id, finding.mention_id, "condition_causes_finding", "CAUSES"),
    ]
    return ClinicalDocument(document_id, text, [ClinicalCase(case_id)], [Patient(patient_id)], mentions, relations)


def _approved(ids):
    return MCRCSuitabilityReport(SuitabilityDecision.APPROVED, SuitabilityCriteria(), {}, {}, ["approved for test"], ids, "freeze")


def test_candidates_respect_type_compatibility_and_include_adjacent_sentence_gold_relation():
    document = _document("mcrc:one")
    candidates = generate_candidates(document, CandidatePolicy(maximum_sentence_distance=1))
    causes = next(item for item in candidates if item.candidate_relation is CanonicalRelation.CAUSES and item.is_positive)

    assert causes.sentence_distance == 1
    assert all(item.candidate_relation is not CanonicalRelation.DESCRIBES for item in candidates)
    assert {(item.source_type, item.target_type, item.candidate_relation) for item in candidates} >= {
        ("Patient", "Condition", CanonicalRelation.HAS_CONDITION),
        ("Patient", "Finding", CanonicalRelation.HAS_FINDING),
        ("Modifier", "Finding", CanonicalRelation.NEGATES),
        ("Condition", "Finding", CanonicalRelation.CAUSES),
    }


def test_gold_relation_outside_window_is_reported_not_candidate():
    document, policy = _document("mcrc:one"), CandidatePolicy(maximum_sentence_distance=0)
    outside = gold_outside_window(document, policy)
    assert [relation.canonical_label.value for relation in outside] == ["CAUSES"]
    candidates = generate_candidates(document, policy)
    assert all(candidate.gold_relation_id != outside[0].relation_id for candidate in candidates)
    assert gold_outside_window(document, CandidatePolicy(maximum_sentence_distance=None)) == ()


def test_manifest_counts_gold_outside_window():
    documents = [_document(f"mcrc:{index}", extra_findings=10) for index in range(6)]
    dataset = build_frozen_supervised_dataset(
        documents,
        _approved([item.document_id for item in documents]),
        candidate_policy=CandidatePolicy(maximum_sentence_distance=0),
        partition_policy=PartitionPolicy(seed=23, folds=3),
    )
    assert dict(dataset.gold_outside_window)["CAUSES"] == 6


def test_document_grouped_partitions_exclude_non_eligible_and_limit_train_negatives():
    released = [_document(f"mcrc:{index}", extra_findings=10) for index in range(6)]
    blocked = _document("mcrc:blocked", extra_findings=10)
    dataset = build_frozen_supervised_dataset(
        [*released, blocked],
        _approved([item.document_id for item in released]),
        partition_policy=PartitionPolicy(seed=23, folds=3),
    )

    assert len(dataset.test_document_ids) == 1
    assert "mcrc:blocked" not in dataset.eligible_document_ids
    assert {item.document_id for item in dataset.test_candidates} == set(dataset.test_document_ids)
    validation_ids = set()
    for fold in dataset.folds:
        assert not set(fold.train_document_ids) & set(fold.validation_document_ids)
        assert not set(fold.train_document_ids) & set(dataset.test_document_ids)
        assert not set(fold.validation_document_ids) & set(dataset.test_document_ids)
        validation_ids.update(fold.validation_document_ids)
        positives = sum(item.is_positive for item in fold.train_candidates)
        negatives = sum(not item.is_positive for item in fold.train_candidates)
        assert negatives <= 2 * positives
    assert validation_ids | set(dataset.test_document_ids) == set(dataset.eligible_document_ids)


def test_same_input_policy_and_seed_produce_identical_artifacts(tmp_path):
    documents = [_document(f"mcrc:{index}", extra_findings=8) for index in range(5)]
    report = _approved([item.document_id for item in documents])
    policy = PartitionPolicy(seed=7, folds=3)
    first = build_frozen_supervised_dataset(documents, report, partition_policy=policy)
    second = build_frozen_supervised_dataset(documents, report, partition_policy=policy)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_manifest = write_frozen_supervised_dataset(first, first_dir)
    second_manifest = write_frozen_supervised_dataset(second, second_dir)

    assert first_manifest == second_manifest
    assert {path.name: path.read_bytes() for path in first_dir.iterdir()} == {path.name: path.read_bytes() for path in second_dir.iterdir()}


def test_file_entrypoint_requires_approved_report_and_records_input_hash(tmp_path):
    documents = [_document(f"mcrc:{index}") for index in range(4)]
    jsonl, report_path = tmp_path / "mcrc.jsonl", tmp_path / "suitability.json"
    write_jsonl(documents, jsonl)
    write_mcrc_suitability_report(_approved([item.document_id for item in documents]), report_path)

    dataset = build_frozen_supervised_dataset_from_files(jsonl, report_path, partition_policy=PartitionPolicy(folds=2))
    manifest = write_frozen_supervised_dataset(dataset, tmp_path / "output")

    assert manifest["source_jsonl_sha256"]
    assert (tmp_path / "output" / "manifest.json").exists()