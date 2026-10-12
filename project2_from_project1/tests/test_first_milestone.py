import json

from project2_from_project1.src.clinical_documents import (
    AnnotatedRelation,
    ClinicalCase,
    ClinicalDocument,
    Mention,
    Patient,
    Span,
    write_jsonl,
)
from project2_from_project1.src.first_milestone import evaluate_first_milestone, prepare_first_milestone
from project2_from_project1.src.leakage_audit import build_protected_collection, write_protected_collection
from project2_from_project1.src.relation_evaluation import RelationPrediction, load_supervised_candidates, write_relation_predictions


def _document(index: int) -> ClinicalDocument:
    document_id = f"external:{index}"
    text = "Patient condition and factor. finding. No finding."
    case_id, patient_id = f"{document_id}:case", f"{document_id}:patient"
    condition_start = text.index("condition")
    factor_start = text.index("factor")
    finding_starts = [position for position in range(len(text)) if text.startswith("finding", position)]
    no_start = text.index("No")
    condition = Mention(f"{document_id}:condition", [Span(condition_start, condition_start + 9)], ["condition"], "Condition")
    factor = Mention(f"{document_id}:factor", [Span(factor_start, factor_start + 6)], ["factor"], "Factor")
    finding = Mention(f"{document_id}:finding", [Span(finding_starts[0], finding_starts[0] + 7)], ["finding"], "Finding")
    modifier = Mention(f"{document_id}:modifier", [Span(no_start, no_start + 2)], ["modifier"], "Modifier")
    negated = Mention(f"{document_id}:negated", [Span(finding_starts[1], finding_starts[1] + 7)], ["finding"], "Finding")
    return ClinicalDocument(
        document_id,
        text,
        [ClinicalCase(case_id)],
        [Patient(patient_id)],
        [condition, factor, finding, modifier, negated],
        [
            AnnotatedRelation(f"{document_id}:describes", case_id, patient_id, "describes", "DESCRIBES"),
            AnnotatedRelation(f"{document_id}:condition", patient_id, condition.mention_id, "case_has_condition", "HAS_CONDITION"),
            AnnotatedRelation(f"{document_id}:factor", patient_id, factor.mention_id, "case_has_factor", "HAS_FACTOR"),
            AnnotatedRelation(f"{document_id}:finding", patient_id, finding.mention_id, "case_has_finding", "HAS_FINDING"),
            AnnotatedRelation(f"{document_id}:negates", modifier.mention_id, negated.mention_id, "modifier_modifies_finding", "NEGATES"),
            AnnotatedRelation(f"{document_id}:causes", condition.mention_id, finding.mention_id, "condition_causes_finding", "CAUSES"),
        ],
    )


def _prediction(candidate):
    return RelationPrediction(
        document_id=candidate.document_id,
        source_id=candidate.source_id,
        target_id=candidate.target_id,
        native_label=candidate.label.value.lower(),
        canonical_label=candidate.label,
        method="fixture-oracle",
        evidence={"fixture": True},
        sentences=[],
        model_version="fixture-oracle-1",
    )


def test_prepare_and_evaluate_first_milestone_are_auditable_and_reproducible(tmp_path):
    safe_jsonl = tmp_path / "safe.jsonl"
    documents = [_document(index) for index in range(6)]
    write_jsonl(documents, safe_jsonl)
    protected = ClinicalDocument("protected:1", "Completely unrelated protected dermatology narrative.", [], [], [], [], metadata={"article_id": "PMC-protected"})
    protected_path = tmp_path / "protected.json"
    write_protected_collection(build_protected_collection([protected]), protected_path)

    first = prepare_first_milestone(safe_jsonl, protected_path, tmp_path / "first")
    second = prepare_first_milestone(safe_jsonl, protected_path, tmp_path / "second")

    assert first.status == second.status == "approved"
    assert first.dataset_directory is not None
    assert json.loads(first.manifest_path.read_text(encoding="utf-8")) == json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert {
        path.relative_to(first.dataset_directory).as_posix(): path.read_bytes()
        for path in first.dataset_directory.rglob("*") if path.is_file()
    } == {
        path.relative_to(second.dataset_directory).as_posix(): path.read_bytes()
        for path in second.dataset_directory.rglob("*") if path.is_file()
    }
    preparation = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert preparation["frozen_dataset_manifest"]["negative_policy"]["maximum_negatives_per_positive"] == 2
    assert preparation["leakage_report"]["counts"]["released"] == len(documents)
    assert documents[0].text not in first.manifest_path.read_text(encoding="utf-8")

    candidates = load_supervised_candidates(first.dataset_directory / "test.jsonl")
    predictions = tmp_path / "predictions.jsonl"
    write_relation_predictions([_prediction(candidate) for candidate in candidates], predictions)
    evaluation_path = evaluate_first_milestone(first.output_directory, predictions, tmp_path / "evaluation-first.json", seed=17, minimum_positive_support=1)
    evaluate_first_milestone(first.output_directory, predictions, tmp_path / "evaluation-second.json", seed=17, minimum_positive_support=1)
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation_manifest = json.loads((tmp_path / "evaluation-first.manifest.json").read_text(encoding="utf-8"))

    assert report["metadata"]["partition"] == "test"
    assert report["metrics"]["primary_metric"] == "positive_macro_f1"
    assert report["metrics"]["confusion_matrix"]
    assert report["timing_seconds"]["evaluation"] >= 0
    assert evaluation_manifest["data_hashes"] == report["data_hashes"]
    assert evaluation_manifest == json.loads((tmp_path / "evaluation-second.manifest.json").read_text(encoding="utf-8"))
