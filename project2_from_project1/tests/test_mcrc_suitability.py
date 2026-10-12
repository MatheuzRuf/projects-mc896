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
from project2_from_project1.src.leakage_audit import (
    LeakageAuditReport,
    audit_external_corpus,
    build_protected_collection,
    write_leakage_report,
)
from project2_from_project1.src.mcrc_suitability import (
    MCRCSuitabilityError,
    SuitabilityDecision,
    assess_mcrc_suitability,
    assess_mcrc_suitability_from_files,
    load_mcrc_suitability_report,
    write_mcrc_suitability_report,
)


def _complete_document(document_id="mcrc:clean", **metadata):
    text = "Case condition finding factor no finding."
    case_id = f"{document_id}:case"
    patient_id = f"{document_id}:patient"
    mentions = [
        Mention(f"{document_id}:condition", [Span(5, 14)], ["condition"], "Condition"),
        Mention(f"{document_id}:finding", [Span(15, 22)], ["finding"], "Finding"),
        Mention(f"{document_id}:factor", [Span(23, 29)], ["factor"], "Factor"),
        Mention(f"{document_id}:modifier", [Span(30, 32)], ["modifier"], "Modifier"),
        Mention(f"{document_id}:negated", [Span(33, 40)], ["finding"], "Finding"),
    ]
    condition, finding, factor, modifier, negated = mentions
    relations = [
        AnnotatedRelation(f"{document_id}:describes", case_id, patient_id, "document_describes_case", CanonicalRelation.DESCRIBES),
        AnnotatedRelation(f"{document_id}:has-condition", patient_id, condition.mention_id, "case_has_condition", CanonicalRelation.HAS_CONDITION),
        AnnotatedRelation(f"{document_id}:has-finding", patient_id, finding.mention_id, "case_has_finding", CanonicalRelation.HAS_FINDING),
        AnnotatedRelation(f"{document_id}:has-factor", patient_id, factor.mention_id, "case_has_factor", CanonicalRelation.HAS_FACTOR),
        AnnotatedRelation(f"{document_id}:negates", modifier.mention_id, negated.mention_id, "modifier_modifies_finding", CanonicalRelation.NEGATES),
        AnnotatedRelation(f"{document_id}:causes", condition.mention_id, finding.mention_id, "condition_causes_finding", CanonicalRelation.CAUSES),
    ]
    return ClinicalDocument(
        document_id=document_id,
        text=text,
        cases=[ClinicalCase(case_id)],
        patients=[Patient(patient_id)],
        mentions=mentions,
        relations=relations,
        metadata=metadata,
    )


def _protected_document():
    return ClinicalDocument(
        document_id="multicare:protected",
        text="A protected report that has no overlap with the candidate.",
        cases=[], patients=[], mentions=[], relations=[], metadata={"pmid": "999"},
    )


def test_approval_reports_counts_per_native_and_canonical_class_and_only_released_ids(tmp_path):
    released = _complete_document()
    excluded = _complete_document("mcrc:excluded", pmid="999")
    audit = audit_external_corpus([released, excluded], build_protected_collection([_protected_document()]))

    report = assess_mcrc_suitability([released, excluded], audit)

    assert report.decision is SuitabilityDecision.APPROVED
    assert report.documents_for_partitioning == ("mcrc:clean",)
    assert report.populations["before_audit"]["documents"] == 2
    assert report.populations["released"]["relations"] == 6
    assert report.populations["excluded"]["documents"] == 1
    assert report.populations["released"]["native_relation_support"]["condition_causes_finding"] == 1
    assert report.populations["released"]["canonical_relation_support"]["CAUSES"] == 1
    assert report.populations["released"]["canonical_relation_support"]["NO_RELATION"] == 0
    assert report.audit["excluded_document_ids"] == ["mcrc:excluded"]

    output = tmp_path / "suitability.json"
    write_mcrc_suitability_report(report, output)
    assert load_mcrc_suitability_report(output) == report
    assert released.text not in output.read_text(encoding="utf-8")


def test_missing_canonical_coverage_rejects_without_exposing_a_training_split():
    complete = _complete_document()
    incomplete = ClinicalDocument(
        document_id=complete.document_id,
        text=complete.text,
        cases=complete.cases,
        patients=complete.patients,
        mentions=complete.mentions,
        relations=[relation for relation in complete.relations if relation.canonical_label is not CanonicalRelation.CAUSES],
        metadata=complete.metadata,
    )
    audit = audit_external_corpus([incomplete], build_protected_collection([_protected_document()]))

    report = assess_mcrc_suitability([incomplete], audit)

    assert report.decision is SuitabilityDecision.REJECTED
    assert report.eligible_document_ids == ()
    assert "i2b2/n2c2" in report.next_action
    assert any("CAUSES support below minimum" in reason for reason in report.reasons)


def test_file_entrypoint_revalidates_safe_jsonl_and_consumes_complete_audit(tmp_path):
    document = _complete_document()
    protected = _protected_document()
    jsonl = tmp_path / "mcrc.jsonl"
    audit_path = tmp_path / "audit.json"
    write_jsonl([document], jsonl)
    write_leakage_report(audit_external_corpus([document], build_protected_collection([protected])), audit_path)

    report = assess_mcrc_suitability_from_files(jsonl, audit_path)

    assert report.approved
    assert report.documents_for_partitioning == (document.document_id,)


def test_audit_must_cover_exactly_the_converted_corpus():
    document = _complete_document()
    incomplete_audit = LeakageAuditReport(records=(), similarity_threshold=0.85)

    with pytest.raises(MCRCSuitabilityError, match="cover exactly"):
        assess_mcrc_suitability([document], incomplete_audit)
