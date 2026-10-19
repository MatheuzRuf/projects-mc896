import csv
import json

from project2_from_project1.src.clinical_documents import AnnotatedRelation, ClinicalDocument, Mention, Patient, Span
from project2_from_project1.src.gold_projection import build_projection_plan
from project2_from_project1.src.relation_candidates import logical_patient_id
from project2_from_project1.src.relation_evaluation import RelationPrediction
from project2_from_project1.src.supervised_contracts import (
    EntityPrediction,
    GraphEdge,
    GraphNode,
    load_contract_jsonl,
    write_contract_jsonl,
)
from project2_from_project1.src.supervised_graph import (
    build_gold_graph,
    build_predicted_graph,
    evaluate_graph,
    export_legacy_csv,
    normalize_surface,
    to_networkx,
)
from project2_from_project1.src.visualization.graph import load_case_graph

TEXT = "Pain  in chest. No pain today. Goiter seen. Goiter resected."


def _span(surface, start=0):
    position = TEXT.index(surface, start)
    return Span(position, position + len(surface))


def _entity(surface, kind, sentence, start=0):
    return EntityPrediction("d", _span(surface, start), kind, sentence, "test-ner", "ner-v1")


ENTITIES = [
    _entity("Pain  in chest", "Finding", 0), _entity("No", "Modifier", 1), _entity("pain", "Finding", 1),
    _entity("Goiter", "Finding", 2), _entity("Goiter", "Condition", 3, 40),
]


def _relation(source, target, label, candidate="candidate:x"):
    return RelationPrediction("d", source, target, label, label, "test-rel", {"candidate_id": candidate},
                              model_version="rel-v1")


def _predicted():
    patient = logical_patient_id("d")
    relations = [
        _relation(patient, ENTITIES[0].prediction_id, "HAS_FINDING", "candidate:a"),
        _relation(ENTITIES[1].prediction_id, ENTITIES[2].prediction_id, "NEGATES", "candidate:b"),
        _relation(patient, ENTITIES[3].prediction_id, "HAS_FINDING", "candidate:c"),
        _relation(patient, ENTITIES[4].prediction_id, "HAS_CONDITION", "candidate:d"),
        _relation(patient, ENTITIES[2].prediction_id, "NO_RELATION", "candidate:e"),
    ]
    return build_predicted_graph({"d": TEXT}, ["d"], ENTITIES, relations)


def test_mentions_keep_identity_and_concepts_are_typed_and_traceable():
    records = _predicted()
    mentions = [item for item in records if isinstance(item, GraphNode) and item.layer == "mention"]
    assert {(item.span, item.canonical_type, item.provenance["method"]) for item in mentions} >= {
        (_span("Goiter"), "Finding", "test-ner"), (_span("Goiter", 40), "Condition", "test-ner"),
    }
    concepts = {(item.canonical_type, item.normalized_surface): item for item in records
                if isinstance(item, GraphNode) and item.layer == "concept"}
    assert ("Finding", "goiter") in concepts and ("Condition", "goiter") in concepts
    assert ("Finding", "pain in chest") in concepts and normalize_surface("Pain  in\nchest") == "pain in chest"
    assert concepts[("Finding", "pain")].provenance["source_ids"] == [ENTITIES[2].prediction_id]
    edges = [item for item in records if isinstance(item, GraphEdge)]
    assert not any(item.canonical_relation.value == "NO_RELATION" for item in edges)
    mention_edges = {item.edge_id: item for item in edges if item.layer == "mention"}
    assert sorted(item.provenance["source_ids"][0] for item in mention_edges.values()) == [
        "candidate:a", "candidate:b", "candidate:c", "candidate:d"]
    for item in edges:
        if item.layer == "concept":
            assert set(item.provenance["source_ids"]) <= set(mention_edges)


def test_jsonl_is_canonical_and_csv_and_networkx_derive_from_it(tmp_path):
    records = _predicted()
    path = tmp_path / "graph.jsonl"
    write_contract_jsonl(records, path)
    reloaded = load_contract_jsonl(path, "graph")
    assert [item.to_dict() for item in reloaded] == [item.to_dict() for item in records]
    view = to_networkx(reloaded)
    assert view.number_of_nodes() == sum(isinstance(item, GraphNode) for item in reloaded)
    assert view.number_of_edges() == sum(isinstance(item, GraphEdge) for item in reloaded)
    some_edge = next(item for item in reloaded if isinstance(item, GraphEdge))
    assert view.edges[some_edge.source_id, some_edge.target_id, some_edge.edge_id]["provenance"]["source_ids"]

    counts = export_legacy_csv(reloaded, {"d": TEXT}, tmp_path / "nodes.csv", tmp_path / "edges.csv")
    case_id, nodes, edges = load_case_graph(tmp_path / "nodes.csv", tmp_path / "edges.csv", "d")
    assert (len(nodes), len(edges)) == (counts["nodes"], counts["edges"])
    with (tmp_path / "nodes.csv").open(encoding="utf-8") as handle:
        row = next(item for item in csv.DictReader(handle) if item["type"] == "Condition")
    attributes = json.loads(row["attributes"])
    assert attributes["layer"] == "concept" and attributes["provenance"]["source_ids"]
    assert attributes["graph_node_id"] in {item.node_id for item in reloaded if isinstance(item, GraphNode)}


def test_concept_metrics_separate_aggregation_gains_from_exact_mentions():
    mentions = tuple(Mention(name, (_span(surface, start),), (native,), kind) for name, surface, start, native, kind in (
        ("chest", "Pain  in chest", 0, "finding", "Finding"), ("no", "No", 0, "modifier", "Modifier"),
        ("pain", "pain", 0, "finding", "Finding"), ("g1", "Goiter", 0, "condition", "Condition"),
        ("g2", "Goiter", 40, "condition", "Condition"),
    ))
    relations = tuple(AnnotatedRelation(f"r{index}", source, target, "native", label) for index, (source, target, label) in enumerate((
        ("p", "chest", "HAS_FINDING"), ("no", "pain", "NEGATES"), ("p", "g1", "HAS_CONDITION"),
    )))
    document = ClinicalDocument("d", TEXT, (), (Patient("p"),), mentions, relations)
    [flat] = build_projection_plan([document]).project(())
    gold, lost = build_gold_graph([document], [flat])
    assert lost == {"CAUSES": 0, "HAS_CONDITION": 0, "HAS_FACTOR": 0, "HAS_FINDING": 0, "NEGATES": 0}

    report = evaluate_graph(gold, _predicted())
    edges = report["concept_edges"]["micro"]
    assert (edges["true_positive"], edges["predicted"], edges["support"]) == (3, 4, 3)
    # The Condition edge only matches at concept level: the prediction used the second occurrence.
    assert report["aggregation"]["concept_edges_correct_only_after_aggregation"] == 1
    assert report["mention_edges_exact_matches"] == 2
    assert report["concept_nodes"]["per_type"]["Condition"]["true_positive"] == 1
    assert report["aggregation"]["same_surface_kept_apart_by_type"] == 1
