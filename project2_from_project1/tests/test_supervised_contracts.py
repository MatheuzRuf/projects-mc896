import json

import pytest

from project2_from_project1.src.clinical_documents import CanonicalRelation, Span
from project2_from_project1.src.supervised_contracts import (
    ContractError, EntityPrediction, GraphEdge, GraphNode, RelationCandidate,
    load_contract_jsonl, write_contract_jsonl,
)


def test_synthetic_fixture_round_trips_all_contracts_deterministically(tmp_path):
    # Synthetic document: "Asthma causes cough." Text stays in its source document.
    asthma = EntityPrediction("doc-1", Span(0, 6), "Condition", 0, "dictionary", "1")
    cough = EntityPrediction("doc-1", Span(14, 19), "Finding", 0, "dictionary", "1", 0.75)
    candidate = RelationCandidate("doc-1", asthma.prediction_id, cough.prediction_id,
                                  "Condition", "Finding", CanonicalRelation.CAUSES, 0)
    mention_provenance = {"method": "dictionary", "model_version": "1", "source_ids": [asthma.prediction_id]}
    cough_provenance = {"method": "dictionary", "model_version": "1", "source_ids": [cough.prediction_id]}
    relation_provenance = {"method": "rules", "model_version": "2", "source_ids": [candidate.candidate_id]}
    graph = [
        GraphNode("m1", "doc-1", "mention", "Condition", mention_provenance, asthma.span, 0),
        GraphNode("m2", "doc-1", "mention", "Finding", cough_provenance, cough.span, 0),
        GraphNode("c1", "doc-1", "concept", "Condition", {"method": "projection", "model_version": "1", "source_ids": ["m1"]}, normalized_surface="asthma"),
        GraphNode("c2", "doc-1", "concept", "Finding", {"method": "projection", "model_version": "1", "source_ids": ["m2"]}, normalized_surface="cough"),
        GraphEdge("e1", "doc-1", "mention", "m1", "m2", "CAUSES", relation_provenance),
        GraphEdge("e2", "doc-1", "concept", "c1", "c2", "CAUSES", {"method": "projection", "model_version": "1", "source_ids": ["e1"]}),
    ]
    for kind, items in (("entity", [asthma, cough]), ("candidate", [candidate]), ("graph", graph)):
        first, second = tmp_path / f"{kind}.jsonl", tmp_path / f"{kind}-copy.jsonl"
        write_contract_jsonl(items, first)
        assert load_contract_jsonl(first, kind) == items
        write_contract_jsonl(load_contract_jsonl(first, kind), second)
        assert first.read_bytes() == second.read_bytes()
    entity_json = (tmp_path / "entity.jsonl").read_text(encoding="utf-8")
    assert "Asthma" not in entity_json and "native_types" not in entity_json
    assert "label" not in json.loads((tmp_path / "candidate.jsonl").read_text(encoding="utf-8"))
    assert candidate.candidate_relation is CanonicalRelation.CAUSES
    assert candidate.candidate_relation is not CanonicalRelation.NO_RELATION


@pytest.mark.parametrize("change, field", [
    ({"span": [2, 1]}, "span"),
    ({"canonical_type": "Symptom"}, "canonical_type"),
    ({"schema_version": 2}, "schema_version"),
    ({"prediction_id": "incorrect"}, "prediction_id"),
    ({"text": "private"}, "text"),
    ({"marginal_probability": float("nan")}, "marginal_probability"),
])
def test_invalid_prediction_reports_field_and_line(tmp_path, change, field):
    record = EntityPrediction("doc", Span(0, 4), "Finding", 0, "dictionary", "1").to_dict()
    record.update(change)
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match=rf"line 1:.*{field}"):
        load_contract_jsonl(path, "entity")


def test_candidate_rejects_labels_even_no_relation(tmp_path):
    candidate = RelationCandidate("doc", "patient", "finding", "Patient", "Finding", "HAS_FINDING", None)
    payload = candidate.to_dict()
    payload["label"] = "NO_RELATION"
    path = tmp_path / "candidate.jsonl"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="line 1:.*label"):
        load_contract_jsonl(path, "candidate")
    payload.pop("label")
    payload["candidate_relation"] = "NO_RELATION"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="line 1:.*candidate_relation"):
        load_contract_jsonl(path, "candidate")


def test_unicode_line_separators_inside_json_strings_round_trip(tmp_path):
    prediction = EntityPrediction("doc\u2028\u2029\u0085id", Span(0, 1), "Finding", 0, "dictionary", "1")
    path = tmp_path / "entity.jsonl"
    write_contract_jsonl([prediction], path)
    assert load_contract_jsonl(path, "entity") == [prediction]


def test_oversized_probability_reports_field_and_line(tmp_path):
    payload = EntityPrediction("doc", Span(0, 1), "Finding", 0, "dictionary", "1").to_dict()
    payload["marginal_probability"] = 10 ** 400
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="line 1:.*marginal_probability"):
        load_contract_jsonl(path, "entity")


@pytest.mark.parametrize("kind,item,field", [
    ("entity", EntityPrediction("doc", Span(0, 1), "Finding", 0, "dictionary", "1"), "prediction_id"),
    ("candidate", RelationCandidate("doc", "patient", "finding", "Patient", "Finding", "HAS_FINDING", None), "candidate_id"),
])
def test_duplicate_records_report_identifier_field_and_position(tmp_path, kind, item, field):
    path = tmp_path / "duplicates.jsonl"
    with pytest.raises(ContractError, match=rf"record 2:.*{field}"):
        write_contract_jsonl([item, item], path)
    line = json.dumps(item.to_dict())
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match=rf"line 2:.*{field}"):
        load_contract_jsonl(path, kind)


def test_unsupported_record_type_reports_position(tmp_path):
    prediction = EntityPrediction("doc", Span(0, 1), "Finding", 0, "dictionary", "1")
    with pytest.raises(ContractError, match="record 2:.*record type"):
        write_contract_jsonl([prediction, object()], tmp_path / "invalid.jsonl")


def test_graph_rejects_missing_endpoint_with_record_identity(tmp_path):
    provenance = {"method": "rules", "model_version": "1", "source_ids": ["candidate"]}
    edge = GraphEdge("edge-1", "doc", "mention", "missing", "also-missing", "CAUSES", provenance)
    path = tmp_path / "graph.jsonl"
    path.write_text(json.dumps(edge.to_dict()) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="record 1 edge 'edge-1': source_id"):
        load_contract_jsonl(path, "graph")
