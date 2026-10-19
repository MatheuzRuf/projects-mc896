"""Versioned two-layer Knowledge Graph from frozen end-to-end predictions.

The mention layer keeps one node per predicted entity (document, offsets, type,
method, version) and one edge per positive relation prediction.  The concept
layer is derived afterwards: a concept is identified by document, canonical
type and a surface normalized by Unicode NFKC, ``casefold`` and whitespace
consolidation, without stemming, synonyms or ontologies.  Each concept node and
edge lists the mention-layer IDs that support it.

Canonical output is graph JSONL v2 (:mod:`supervised_contracts`).  CSV and
NetworkX views are derived from those records and are never the source of
truth.  Concept-level metrics are secondary to the strict mention and relation
metrics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from project2_from_project1.src.clinical_documents import ClinicalDocument
from project2_from_project1.src.gold_projection import FlatGoldDocument
from project2_from_project1.src.relation_candidates import logical_patient_id
from project2_from_project1.src.relation_evaluation import RelationPrediction
from project2_from_project1.src.supervised_contracts import (
    RELATIONS,
    EntityPrediction,
    GraphEdge,
    GraphNode,
)

GRAPH_PROJECTION_VERSION = "supervised-graph-v1"
CONCEPT_METHOD = "typed-normalized-surface-aggregation"
NORMALIZATION = "unicode-nfkc+casefold+collapse-whitespace"
Record = GraphNode | GraphEdge


def normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _concept_key(node: GraphNode) -> tuple:
    return (node.document_id, node.canonical_type, node.normalized_surface)


def _add_concepts(texts: Mapping[str, str], mention_nodes: Sequence[GraphNode], mention_edges: Sequence[GraphEdge]) -> list[Record]:
    """Aggregate mention nodes and edges into typed concepts with provenance."""
    members: dict[tuple, list[str]] = defaultdict(list)
    concept_of: dict[str, tuple] = {}
    for node in mention_nodes:
        if node.canonical_type == "Patient":
            key = (node.document_id, "Patient", None)
        else:
            key = (node.document_id, node.canonical_type,
                   normalize_surface(texts[node.document_id][node.span.start:node.span.end]))
        members[key].append(node.node_id)
        concept_of[node.node_id] = key
    concept_ids = {key: _id("concept", *key) for key in members}
    records: list[Record] = [
        GraphNode(concept_ids[key], key[0], "concept", key[1],
                  {"method": CONCEPT_METHOD, "model_version": GRAPH_PROJECTION_VERSION, "source_ids": sorted(ids)},
                  normalized_surface=key[2])
        for key, ids in sorted(members.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or ""))
    ]
    edges: dict[tuple, list[str]] = defaultdict(list)
    for edge in mention_edges:
        key = (edge.document_id, concept_ids[concept_of[edge.source_id]], concept_ids[concept_of[edge.target_id]],
               edge.canonical_relation.value)
        edges[key].append(edge.edge_id)
    for (document_id, source, target, label), ids in sorted(edges.items()):
        records.append(GraphEdge(_id("concept-edge", document_id, source, target, label), document_id, "concept",
                                 source, target, label,
                                 {"method": CONCEPT_METHOD, "model_version": GRAPH_PROJECTION_VERSION,
                                  "source_ids": sorted(ids)}))
    return records


def build_predicted_graph(
    texts: Mapping[str, str],
    document_ids: Sequence[str],
    entities: Iterable[EntityPrediction],
    relations: Iterable[RelationPrediction],
) -> list[Record]:
    """Project frozen entity and positive relation predictions into both layers."""
    documents = set(document_ids)
    nodes: list[GraphNode] = []
    known: dict[str, GraphNode] = {}
    for document_id in sorted(documents):
        patient = logical_patient_id(document_id)
        node = GraphNode(patient, document_id, "mention", "Patient",
                         {"method": "logical-patient", "model_version": "sha256-json-document-id-v1", "source_ids": [patient]})
        nodes.append(node)
        known[patient] = node
    for entity in sorted(entities, key=lambda item: (item.document_id, item.span.start, item.span.end, item.canonical_type)):
        if entity.document_id not in documents:
            raise ValueError(f"entity prediction outside the projected documents: {entity.document_id!r}")
        node = GraphNode(entity.prediction_id, entity.document_id, "mention", entity.canonical_type,
                         {"method": entity.method, "model_version": entity.model_version,
                          "source_ids": [entity.prediction_id]},
                         span=entity.span, sentence_index=entity.sentence_index)
        nodes.append(node)
        known[entity.prediction_id] = node
    edges: list[GraphEdge] = []
    for relation in relations:
        label = relation.canonical_label.value
        if label not in RELATIONS:
            continue
        for endpoint in (relation.source_id, relation.target_id):
            if endpoint not in known or known[endpoint].document_id != relation.document_id:
                raise ValueError(f"relation endpoint {endpoint!r} is not a projected mention")
        candidate = relation.evidence.get("candidate_id") if isinstance(relation.evidence, Mapping) else None
        edges.append(GraphEdge(
            _id("edge", relation.document_id, relation.source_id, relation.target_id, label), relation.document_id,
            "mention", relation.source_id, relation.target_id, label,
            {"method": relation.method, "model_version": relation.model_version,
             "source_ids": [candidate or _id("pair", relation.document_id, relation.source_id, relation.target_id)]}))
    edges.sort(key=lambda item: (item.document_id, item.source_id, item.target_id, item.canonical_relation.value))
    return [*nodes, *edges, *_add_concepts(texts, nodes, edges)]


def build_gold_graph(documents: Iterable[ClinicalDocument], projection: Iterable[FlatGoldDocument]) -> tuple[list[Record], dict[str, int]]:
    """Gold graph over the flat projection; relations losing an endpoint are counted, not invented."""
    flat = {item.document_id: item for item in projection}
    texts, nodes, edges = {}, [], []
    lost = Counter()
    for document in sorted(documents, key=lambda item: item.document_id):
        texts[document.document_id] = document.text
        patient = logical_patient_id(document.document_id)
        nodes.append(GraphNode(patient, document.document_id, "mention", "Patient",
                               {"method": "logical-patient", "model_version": "sha256-json-document-id-v1",
                                "source_ids": [item.patient_id for item in document.patients] or [patient]}))
        endpoint = {item.patient_id: patient for item in document.patients}
        for mention in flat[document.document_id].mentions:
            nodes.append(GraphNode(mention.mention_id, document.document_id, "mention", mention.canonical_type,
                                   {"method": "flat-gold", "model_version": "flat-gold-v1",
                                    "source_ids": list(mention.source_mention_ids)},
                                   span=mention.span, sentence_index=mention.sentence_index))
            for source_id in mention.source_mention_ids:
                endpoint[source_id] = mention.mention_id
        grouped: dict[tuple, list[str]] = defaultdict(list)
        for relation in document.relations:
            label = relation.canonical_label.value
            if label not in RELATIONS:
                continue
            if relation.source_id not in endpoint or relation.target_id not in endpoint:
                lost[label] += 1
                continue
            grouped[(endpoint[relation.source_id], endpoint[relation.target_id], label)].append(relation.relation_id)
        for (source, target, label), ids in sorted(grouped.items()):
            edges.append(GraphEdge(_id("edge", document.document_id, source, target, label), document.document_id,
                                   "mention", source, target, label,
                                   {"method": "flat-gold", "model_version": "flat-gold-v1", "source_ids": sorted(ids)}))
    return [*nodes, *edges, *_add_concepts(texts, nodes, edges)], {label: lost[label] for label in sorted(RELATIONS)}


def _prf(true_positive: int, predicted: int, support: int) -> dict[str, Any]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / support if support else 0.0
    return {"f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "precision": precision,
            "predicted": predicted, "recall": recall, "support": support, "true_positive": true_positive}


def _concept_view(records: Sequence[Record]) -> tuple[dict[str, tuple], set[tuple], dict[tuple, list[str]], set[tuple]]:
    concept_nodes = {item.node_id: _concept_key(item) for item in records if isinstance(item, GraphNode) and item.layer == "concept"}
    edge_keys: dict[tuple, list[str]] = {}
    for item in records:
        if isinstance(item, GraphEdge) and item.layer == "concept":
            key = (item.document_id, concept_nodes[item.source_id], concept_nodes[item.target_id], item.canonical_relation.value)
            edge_keys[key] = list(item.provenance["source_ids"])
    mention_nodes = {item.node_id: item for item in records if isinstance(item, GraphNode) and item.layer == "mention"}
    mention_edges = set()
    for item in records:
        if isinstance(item, GraphEdge) and item.layer == "mention":
            def key(node_id: str) -> tuple:
                node = mention_nodes[node_id]
                return ("Patient",) if node.canonical_type == "Patient" else (node.span.start, node.span.end, node.canonical_type)
            mention_edges.add((item.document_id, key(item.source_id), key(item.target_id), item.canonical_relation.value))
    return concept_nodes, set(concept_nodes.values()), edge_keys, mention_edges


def evaluate_graph(gold: Sequence[Record], predicted: Sequence[Record]) -> dict[str, Any]:
    """Secondary exact metrics on concept nodes and edges, plus aggregation diagnostics."""
    _, gold_nodes, gold_edges, gold_mention_edges = _concept_view(gold)
    predicted_ids, predicted_nodes, predicted_edges, predicted_mention_edges = _concept_view(predicted)
    entity_gold = {key for key in gold_nodes if key[1] != "Patient"}
    entity_predicted = {key for key in predicted_nodes if key[1] != "Patient"}
    edge_hits = set(predicted_edges) & set(gold_edges)
    mention_edge_by_id = {item.edge_id: item for item in predicted if isinstance(item, GraphEdge) and item.layer == "mention"}
    mention_node_by_id = {item.node_id: item for item in predicted if isinstance(item, GraphNode) and item.layer == "mention"}

    def mention_key(edge: GraphEdge) -> tuple:
        def key(node_id: str) -> tuple:
            node = mention_node_by_id[node_id]
            return ("Patient",) if node.canonical_type == "Patient" else (node.span.start, node.span.end, node.canonical_type)
        return (edge.document_id, key(edge.source_id), key(edge.target_id), edge.canonical_relation.value)

    only_after_aggregation = sum(
        not any(mention_key(mention_edge_by_id[edge_id]) in gold_mention_edges for edge_id in predicted_edges[key])
        for key in edge_hits
    )
    types = sorted({key[1] for key in entity_gold | entity_predicted})
    labels = sorted(RELATIONS)
    surfaces: dict[tuple, set[str]] = defaultdict(set)
    for document_id, kind, surface in entity_predicted:
        surfaces[(document_id, surface)].add(kind)
    support = Counter(len(ids) for ids in predicted_edges.values())
    return {
        "aggregation": {
            "concept_edges_correct_only_after_aggregation": only_after_aggregation,
            "concept_edges_by_supporting_mention_edges": {str(size): count for size, count in sorted(support.items())},
            "predicted_concept_nodes": len(entity_predicted),
            "predicted_mention_nodes": sum(1 for item in predicted if isinstance(item, GraphNode) and item.layer == "mention" and item.canonical_type != "Patient"),
            "same_surface_kept_apart_by_type": sum(len(kinds) > 1 for kinds in surfaces.values()),
        },
        "concept_edges": {
            "micro": _prf(len(edge_hits), len(predicted_edges), len(gold_edges)),
            "per_label": {label: _prf(sum(key[3] == label for key in edge_hits), sum(key[3] == label for key in predicted_edges),
                                      sum(key[3] == label for key in gold_edges)) for label in labels},
        },
        "concept_nodes": {
            "micro": _prf(len(entity_gold & entity_predicted), len(entity_predicted), len(entity_gold)),
            "per_type": {kind: _prf(sum(key[1] == kind for key in entity_gold & entity_predicted),
                                    sum(key[1] == kind for key in entity_predicted),
                                    sum(key[1] == kind for key in entity_gold)) for kind in types},
        },
        "interpretation": "secondary; strict mention and relation metrics remain primary",
        "mention_edges_exact_matches": len(predicted_mention_edges & gold_mention_edges),
        "schema_version": 1,
    }


# -- derived views ---------------------------------------------------------

def to_networkx(records: Iterable[Record]):
    """MultiDiGraph view; node and edge attributes carry the full contract record."""
    import networkx as nx

    graph = nx.MultiDiGraph(schema_version=2)
    items = list(records)
    for item in items:
        if isinstance(item, GraphNode):
            graph.add_node(item.node_id, **{key: value for key, value in item.to_dict().items() if key != "node_id"})
    for item in items:
        if isinstance(item, GraphEdge):
            graph.add_edge(item.source_id, item.target_id, key=item.edge_id,
                           **{key: value for key, value in item.to_dict().items() if key not in ("source_id", "target_id")})
    return graph


def export_legacy_csv(records: Iterable[Record], texts: Mapping[str, str], nodes_csv: str | Path, edges_csv: str | Path,
                      *, layer: str = "concept") -> dict[str, int]:
    """Legacy ``nodes.csv``/``edges.csv`` for one layer; IDs are prefixed with ``document_id:``.

    ``label`` holds clinical text (the normalized concept surface or the
    mention text), so these files must stay private.
    """
    items = [item for item in records if item.layer == layer]
    node_rows, edge_rows = [], []
    for item in items:
        if isinstance(item, GraphNode):
            if item.canonical_type == "Patient":
                label = "Patient"
            elif item.layer == "concept":
                label = item.normalized_surface
            else:
                label = texts[item.document_id][item.span.start:item.span.end]
            attributes = {"document_id": item.document_id, "graph_node_id": item.node_id, "layer": item.layer,
                          "provenance": item.provenance, "schema_version": 2}
            if item.span is not None:
                attributes.update(span=item.span.to_dict(), sentence_index=item.sentence_index)
            node_rows.append({"node_id": f"{item.document_id}:{item.node_id}", "type": item.canonical_type,
                              "label": label, "attributes": json.dumps(attributes, ensure_ascii=False, sort_keys=True)})
        else:
            edge_rows.append({"edge_id": f"{item.document_id}:{item.edge_id}",
                              "source_id": f"{item.document_id}:{item.source_id}",
                              "target_id": f"{item.document_id}:{item.target_id}",
                              "relation": item.canonical_relation.value,
                              "attributes": json.dumps({"graph_edge_id": item.edge_id, "layer": item.layer,
                                                        "provenance": item.provenance, "schema_version": 2},
                                                       ensure_ascii=False, sort_keys=True)})
    for path, columns, rows in ((Path(nodes_csv), ("node_id", "type", "label", "attributes"), node_rows),
                                (Path(edges_csv), ("edge_id", "source_id", "target_id", "relation", "attributes"), edge_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return {"edges": len(edge_rows), "nodes": len(node_rows)}


__all__ = [
    "GRAPH_PROJECTION_VERSION", "build_gold_graph", "build_predicted_graph", "evaluate_graph", "export_legacy_csv",
    "normalize_surface", "to_networkx",
]
