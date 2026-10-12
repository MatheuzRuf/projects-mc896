from pathlib import Path

import pytest

from project2_from_project1.src.clinical_documents import (
    AnnotatedRelation,
    CanonicalRelation,
    ClinicalCase,
    ClinicalDocument,
    ClinicalDocumentValidationError,
    Mention,
    Patient,
    load_jsonl,
    write_jsonl,
)


FIXTURE = Path(__file__).parent / "fixtures" / "clinical_document.jsonl"


def test_complete_fixture_round_trips_losslessly_through_jsonl(tmp_path):
    [document] = load_jsonl(FIXTURE)

    assert document.cases[0].case_id == "case-1"
    assert document.patients[0].patient_id == "patient-1"
    assert document.mentions[0].spans[0].start == 0
    assert document.mentions[0].spans[1].end == 17
    assert document.mentions[1].native_types == ("sign", "symptom")
    assert document.mentions[4].spans[0].start == 22  # nested around no + cough
    assert document.relations[0].canonical_label is CanonicalRelation.DESCRIBES

    output = tmp_path / "documents.jsonl"
    write_jsonl([document], output)

    assert load_jsonl(output) == [document]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"spans": [[25, 32]]}, "exceeds text length"),
        ({"target_id": "unknown"}, "references unknown endpoint"),
    ],
)
def test_invalid_spans_and_references_report_the_problem(replacement, message):
    [document] = load_jsonl(FIXTURE)
    payload = document.to_dict()
    if "spans" in replacement:
        payload["mentions"][3].update(replacement)
    else:
        payload["relations"][3].update(replacement)

    with pytest.raises(ClinicalDocumentValidationError, match=message):
        ClinicalDocument.from_dict(payload)


def test_negated_finding_cannot_also_be_positive():
    [document] = load_jsonl(FIXTURE)

    with pytest.raises(ClinicalDocumentValidationError, match="cannot also receive HAS_FINDING"):
        ClinicalDocument(
            document_id=document.document_id,
            text=document.text,
            cases=document.cases,
            patients=document.patients,
            mentions=document.mentions,
            relations=(*document.relations, AnnotatedRelation(
                relation_id="r-cough-positive",
                source_id="patient-1",
                target_id="m-cough",
                native_label="case_has_finding",
                canonical_label=CanonicalRelation.HAS_FINDING,
            )),
            metadata=document.metadata,
        )


def test_describes_must_keep_case_and_patient_distinct():
    with pytest.raises(ClinicalDocumentValidationError, match="ClinicalCase -> Patient"):
        ClinicalDocument(
            document_id="document-1",
            text="text",
            cases=[ClinicalCase("case-1")],
            patients=[Patient("patient-1")],
            mentions=[],
            relations=[AnnotatedRelation("r-1", "patient-1", "case-1", "describes", "DESCRIBES")],
        )
