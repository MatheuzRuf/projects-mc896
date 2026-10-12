import json

import pytest

from project2_from_project1.src.clinical_documents import CanonicalRelation
from project2_from_project1.src.relation_evaluation import (
    EvaluationConfig,
    RelationEvaluationError,
    RelationPrediction,
    evaluate_predictions,
    load_relation_predictions,
    load_supervised_candidates,
    write_evaluation_report,
    write_relation_predictions,
)
from project2_from_project1.src.supervised_dataset import SupervisedCandidate


def _candidate(index, label, relation=CanonicalRelation.HAS_FINDING):
    return SupervisedCandidate(
        document_id=f"document-{index}",
        source_id=f"patient-{index}",
        target_id=f"mention-{index}",
        candidate_relation=relation,
        label=label,
        source_type="Patient",
        target_type="Finding",
        sentence_distance=None,
    )


def _prediction(candidate, label, **extra):
    return RelationPrediction(
        document_id=candidate.document_id,
        source_id=candidate.source_id,
        target_id=candidate.target_id,
        native_label="case_has_finding" if label is not CanonicalRelation.NO_RELATION else "no_relation",
        canonical_label=label,
        method="fixture-rule",
        evidence={"fixture": True},
        sentences=[0],
        model_version="fixture-rule-1",
        **extra,
    )


def test_no_relation_dominance_does_not_make_accuracy_the_primary_metric():
    candidates = tuple(_candidate(index, CanonicalRelation.HAS_FINDING if index == 0 else CanonicalRelation.NO_RELATION) for index in range(10))
    report = evaluate_predictions(candidates, (_prediction(item, CanonicalRelation.NO_RELATION) for item in candidates), config=EvaluationConfig(minimum_positive_support=2))

    assert report.metrics["positive_macro_f1"] == 0.0
    assert report.metrics["accuracy"] == {"role": "auxiliary", "value": 0.9}
    assert report.metrics["per_class"]["HAS_FINDING"]["conclusion"] == "inconclusive_insufficient_support"
    assert report.majority_baseline["label"] == "NO_RELATION"


def test_absent_positive_class_is_explicitly_inconclusive():
    candidates = tuple(_candidate(index, CanonicalRelation.NO_RELATION, CanonicalRelation.HAS_FACTOR) for index in range(3))
    report = evaluate_predictions(candidates, (_prediction(item, CanonicalRelation.NO_RELATION) for item in candidates), config=EvaluationConfig(minimum_positive_support=2))

    metric = report.metrics["per_class"]["HAS_FACTOR"]
    assert metric["support"] == 0
    assert metric["conclusion"] == "inconclusive_insufficient_support"
    assert report.metrics["positive_macro_f1"] == 0.0


def test_error_in_a_minority_positive_class_is_visible_in_its_metric():
    candidates = (
        _candidate(0, CanonicalRelation.CAUSES, CanonicalRelation.CAUSES),
        _candidate(1, CanonicalRelation.CAUSES, CanonicalRelation.CAUSES),
        *( _candidate(index, CanonicalRelation.NO_RELATION, CanonicalRelation.CAUSES) for index in range(2, 8)),
    )
    predictions = [_prediction(item, item.label) for item in candidates]
    predictions[1] = _prediction(candidates[1], CanonicalRelation.NO_RELATION)

    report = evaluate_predictions(candidates, predictions, config=EvaluationConfig(minimum_positive_support=2))

    metric = report.metrics["per_class"]["CAUSES"]
    assert metric["recall"] == 0.5
    assert metric["f1"] < 1.0
    assert metric["conclusion"] == "conclusive"
    assert report.metrics["confusion_matrix"]["values"]["CAUSES"]["NO_RELATION"] == 1


def test_prediction_contract_preserves_vote_fraction_without_calling_it_confidence(tmp_path):
    candidate = _candidate(1, CanonicalRelation.HAS_FINDING)
    prediction = _prediction(candidate, CanonicalRelation.HAS_FINDING, neighbor_vote_fraction=0.75)
    path = tmp_path / "predictions.jsonl"

    write_relation_predictions([prediction], path)

    assert load_relation_predictions(path) == [prediction]
    assert json.loads(path.read_text(encoding="utf-8"))["neighbor_vote_fraction"] == 0.75
    with pytest.raises(RelationEvaluationError, match="confidence must be absent or null"):
        _prediction(candidate, CanonicalRelation.HAS_FINDING, confidence=0.75)


def test_evaluation_requires_exactly_one_prediction_per_gold_pair_and_writes_auditable_report(tmp_path):
    candidate = _candidate(1, CanonicalRelation.HAS_FINDING)
    with pytest.raises(RelationEvaluationError, match="missing predictions"):
        evaluate_predictions([candidate], [])

    report = evaluate_predictions([candidate], [_prediction(candidate, CanonicalRelation.HAS_FINDING)], metadata={"partition": "test"})
    output = tmp_path / "report.json"
    write_evaluation_report(report, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["config"]["seed"] == 0
    assert payload["data_hashes"]["gold_candidates_sha256"]
    assert payload["methods"] == ["fixture-rule"]
    assert payload["model_versions"] == ["fixture-rule-1"]
    assert payload["metrics"]["accuracy"]["role"] == "auxiliary"


def test_frozen_candidate_jsonl_loader_uses_the_shared_candidate_contract(tmp_path):
    candidate = _candidate(1, CanonicalRelation.HAS_FINDING)
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps(candidate.to_dict()) + "\n", encoding="utf-8")

    assert load_supervised_candidates(path) == [candidate]
