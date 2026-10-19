from project2_from_project1.scripts.assemble_external_multicare import _adjudicate, _eligible_support
from project2_from_project1.src.clinical_documents import (
    AnnotatedRelation, ClinicalCase, ClinicalDocument, Mention, Patient, Span,
)


def _vote(role: str, findings: tuple[str, ...], document_id: str = "doc") -> ClinicalDocument:
    offsets = {"fever": (0, 5), "cough": (6, 11)}
    mentions = [Mention(f"{role}:{term}", [Span(*offsets[term])], ["Finding"], "Finding")
                for term in findings]
    relations = [AnnotatedRelation(f"{role}:r:{term}", "patient", f"{role}:{term}",
                                   "HAS_FINDING", "HAS_FINDING") for term in findings]
    return ClinicalDocument(document_id, "fever cough", [ClinicalCase("case")],
                            [Patient("patient")], mentions, relations)


def test_adjudication_preserves_ab_agreements_and_uses_c_for_each_dispute():
    gold, decision = _adjudicate((_vote("a", ("fever", "cough")),
                                  _vote("b", ("fever",)), _vote("c", ("cough",))))

    assert {(item.spans[0].start, item.spans[0].end) for item in gold.mentions} == {(0, 5), (6, 11)}
    assert len(gold.relations) == 2
    assert len(decision["disputed_entities"]) == 1
    assert len(decision["disputed_relations"]) == 1
    assert decision["disputed_entities"][0]["third_vote"] is True


def test_support_counts_only_source_eligible_documents():
    included = _vote("included", ("fever",), "included")
    excluded = _vote("excluded", ("fever", "cough"), "excluded")

    document_count, entities, relations = _eligible_support([included, excluded], {"excluded"})

    assert document_count == 1
    assert entities == {"Finding": 1}
    assert relations == {"HAS_FINDING": 1}
