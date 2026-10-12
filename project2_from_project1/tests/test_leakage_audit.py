from project2_from_project1.src.clinical_documents import ClinicalDocument
from project2_from_project1.src.leakage_audit import (
    AuditDecision,
    TrainingEligibilityError,
    audit_external_corpus,
    build_protected_collection,
    load_leakage_report,
    load_protected_collection,
    release_for_training,
    require_training_eligible,
    write_leakage_report,
    write_protected_collection,
)


def _document(document_id, text, **metadata):
    return ClinicalDocument(
        document_id=document_id,
        text=text,
        cases=[],
        patients=[],
        mentions=[],
        relations=[],
        metadata=metadata,
    )


def test_protected_manifest_never_contains_multicare_text_and_round_trips(tmp_path):
    protected = _document(
        "multicare-case-1",
        "A protected patient had severe abdominal pain after surgery.",
        article_id="PMC111",
        pmcid="PMC111",
        doi="10.1000/PROTECTED",
    )

    collection = build_protected_collection([protected])
    manifest = tmp_path / "multicare.fingerprints.json"
    write_protected_collection(collection, manifest)

    payload = manifest.read_text(encoding="utf-8")
    assert collection.article_count == 1
    assert protected.text not in payload
    assert "PMC111" in payload
    assert load_protected_collection(manifest) == collection


def test_identifier_and_exact_content_matches_are_excluded():
    protected = _document("protected", "The complete protected clinical report.", pmid="12345")
    collection = build_protected_collection([protected])
    same_pmid = _document("external-pmid", "Different article wording.", pmid="12345")
    same_content = _document("external-content", " the COMPLETE protected clinical report. ")

    report = audit_external_corpus([same_pmid, same_content], collection)

    assert [record.decision for record in report.records] == [AuditDecision.EXCLUDED, AuditDecision.EXCLUDED]
    assert [record.rule for record in report.records] == ["identifier_match", "content_hash_match"]


def test_contained_excerpt_is_quarantined_but_unrelated_document_is_released():
    shared = "patient developed progressive shortness of breath with bilateral pleural effusions"
    protected = _document(
        "protected",
        "Introduction with unrelated editorial language. " + shared + " after a recent procedure. Follow-up was recorded.",
    )
    excerpt = _document("external-excerpt", shared)
    unrelated = _document("external-clean", "A dermatology report describes a benign isolated skin lesion.")

    report = audit_external_corpus([excerpt, unrelated], build_protected_collection([protected]), similarity_threshold=0.80)

    assert report.records[0].decision is AuditDecision.QUARANTINED
    assert report.records[0].rule == "text_similarity_threshold"
    assert report.records[0].evidence[0]["comparison"] in {"document", "chunk"}
    assert report.records[1].decision is AuditDecision.RELEASED
    assert report.records[1].rule == "no_overlap"


def test_small_partial_phrase_is_recorded_but_not_quarantined():
    protected = _document("protected", "A patient developed acute chest pain after a prolonged procedure.")
    partial = _document("external-partial", "The chart says patient developed acute chest pain during triage.")

    [record] = audit_external_corpus([partial], build_protected_collection([protected])).records

    assert record.decision is AuditDecision.RELEASED
    assert record.rule == "similarity_below_threshold"
    assert record.evidence[0]["shared_shingles"] == 1


def test_training_release_is_fail_closed_and_report_is_auditable(tmp_path):
    protected = _document("protected", "Cardiac biopsy confirmed a rare condition.", doi="10.2/protected")
    blocked = _document("blocked", "Other wording", doi="https://doi.org/10.2/PROTECTED")
    clean = _document("clean", "A nutrition survey reported a dietary preference.")
    report = audit_external_corpus([blocked, clean], build_protected_collection([protected]))

    assert release_for_training([blocked, clean], report) == [clean]
    try:
        require_training_eligible([blocked], report)
    except TrainingEligibilityError as error:
        assert "excluded" in str(error)
    else:
        raise AssertionError("blocked document was allowed into training")

    output = tmp_path / "leakage-report.json"
    write_leakage_report(report, output)
    assert load_leakage_report(output) == report
    payload = output.read_text(encoding="utf-8")
    assert "identifier_match" in payload
    assert "Other wording" not in payload
