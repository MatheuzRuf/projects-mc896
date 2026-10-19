from dataclasses import replace

import pytest

from project2_from_project1.src.clinical_documents import (
    AnnotatedRelation,
    ClinicalDocument,
    Mention,
    Patient,
    Span,
)
from project2_from_project1.src.gold_projection import build_projection_plan
from project2_from_project1.src.relation_end_to_end import (
    PATIENT,
    EndToEndEvaluationError,
    build_gold,
    evaluate_end_to_end,
)
from project2_from_project1.src.relation_evaluation import RelationPrediction

TEXT = "No fever. Asthma noted. Cough present. Rash seen. Pain."
NATIVE = {"Condition": "condition", "Finding": "finding", "Modifier": "modifier"}


def _span(surface):
    start = TEXT.index(surface)
    return start, start + len(surface)


def _document():
    mentions = [Mention(name, (Span(*_span(surface)),), (NATIVE[kind],), kind) for name, surface, kind in (
        ("no", "No", "Modifier"), ("fever", "fever", "Finding"), ("asthma", "Asthma", "Condition"),
        ("noted", "noted", "Finding"), ("cough", "Cough", "Finding"), ("rash", "Rash", "Finding"),
        ("pain", "Pain", "Finding"),
    )]
    relations = [AnnotatedRelation(f"r{index}", source, target, "native", label) for index, (source, target, label) in enumerate((
        ("no", "fever", "NEGATES"), ("p", "asthma", "HAS_CONDITION"), ("p", "cough", "HAS_FINDING"),
        ("asthma", "rash", "CAUSES"), ("p", "noted", "HAS_FINDING"), ("p", "rash", "HAS_FINDING"),
        ("p", "pain", "HAS_FINDING"),
    ))]
    return ClinicalDocument("d", TEXT, (), (Patient("p"),), tuple(mentions), tuple(relations))


def _gold():
    document = _document()
    [flat] = build_projection_plan([document]).project(())
    rejected = replace(flat, mentions=tuple(item for item in flat.mentions if item.source_mention_ids != ("noted",)))
    return build_gold([document], [rejected])


def _key(surface, kind):
    return (*_span(surface), kind)


def _prediction(source, target, label):
    return RelationPrediction("d", source, target, label, label, "test", {}, model_version="test-v1")


def test_every_error_is_attributed_to_its_stage():
    gold = _gold()
    resolve = {("d", "P"): PATIENT, ("d", "no"): _key("No", "Modifier"), ("d", "fever"): _key("fever", "Finding"),
               ("d", "asthma-wide"): (*_span("Asthma")[:1], _span("noted")[1], "Condition"),
               ("d", "noted"): _key("noted", "Finding"), ("d", "rash"): _key("Rash", "Finding"),
               ("d", "pain"): _key("Pain", "Finding")}
    recognized = {("d", key) for key in resolve.values() if key != PATIENT}
    candidates = {("d", resolve[("d", "no")], resolve[("d", "fever")], "NEGATES")}
    predictions = [
        _prediction("no", "fever", "NO_RELATION"),            # FN: classifier
        _prediction("P", "pain", "HAS_FINDING"),              # TP
        _prediction("P", "fever", "HAS_FINDING"),             # FP: relation's own error
        _prediction("P", "asthma-wide", "HAS_CONDITION"),     # FP: inherited entity error
        _prediction("no", "noted", "NEGATES"),                # FP: endpoint rejected by projection
    ]

    report = evaluate_end_to_end(gold, ["d"], predictions, resolve, recognized_entities=recognized,
                                 candidates=candidates, minimum_positive_support=2)

    assert report["micro"]["true_positive"] == 1
    assert report["micro"]["support"] == 7 and report["micro"]["predicted"] == 4
    assert report["errors"]["false_negative"] == {
        "endpoint_rejected_by_projection": 1, "outside_candidate_window": 1, "endpoint_boundary_or_type_error": 1,
        "endpoint_not_recognized": 1, "candidate_not_generated": 1, "classified_no_relation": 1,
    }
    assert report["errors"]["false_positive"] == {
        "endpoint_rejected_by_projection": 1, "inherited_entity_error": 1, "relation_false_positive": 1,
    }
    assert report["per_label"]["HAS_FINDING"]["support"] == 4
    assert report["per_label"]["HAS_FINDING"]["conclusion"] == "conclusive"
    assert set(report["inconclusive_labels"]) == {"CAUSES", "HAS_CONDITION", "HAS_FACTOR", "NEGATES"}
    assert report["macro_f1"] == pytest.approx(report["per_label"]["HAS_FINDING"]["f1"] / 5)


def test_gold_ceiling_resolution_has_no_inherited_entity_errors():
    gold = _gold()
    resolve = dict(gold.mention_keys)
    resolve.update({key: PATIENT for key in gold.patient_ids})
    recognized = {(document_id, key) for (document_id, _), key in gold.mention_keys.items()}
    predictions = [_prediction(source, target, label) for source, target, label in (
        ("no", "fever", "NEGATES"), ("p", "pain", "HAS_FINDING"), ("p", "fever", "HAS_FINDING"))]
    report = evaluate_end_to_end(gold, ["d"], predictions, resolve, recognized_entities=recognized, candidates=set())
    assert report["errors"]["false_positive"]["inherited_entity_error"] == 0
    assert report["errors"]["false_negative"]["endpoint_not_recognized"] == 0
    assert report["micro"]["true_positive"] == 2


def test_unresolvable_or_foreign_predictions_fail():
    gold = _gold()
    with pytest.raises(EndToEndEvaluationError, match="unresolvable"):
        evaluate_end_to_end(gold, ["d"], [_prediction("x", "y", "NEGATES")], {}, recognized_entities=(), candidates=())
    with pytest.raises(EndToEndEvaluationError, match="unevaluated"):
        evaluate_end_to_end(gold, ["other"], [_prediction("x", "y", "NEGATES")], {}, recognized_entities=(), candidates=())
