"""Project every end-to-end arm of the factorial study into the versioned graph."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from project2_from_project1.src.clinical_documents import load_jsonl
from project2_from_project1.src.entity_recognition.corpus import load_document_folds
from project2_from_project1.src.gold_projection import load_flat_gold_jsonl
from project2_from_project1.src.relation_evaluation import load_relation_predictions
from project2_from_project1.src.run_manifest import dependency_versions, git_provenance, sha256_file
from project2_from_project1.src.supervised_contracts import GraphEdge, GraphNode, load_contract_jsonl, write_contract_jsonl
from project2_from_project1.src.supervised_graph import (
    GRAPH_PROJECTION_VERSION,
    NORMALIZATION,
    build_gold_graph,
    build_predicted_graph,
    evaluate_graph,
    export_legacy_csv,
    to_networkx,
)

ARMS = ("dictionary+rules", "dictionary+knn", "crf+rules", "crf+knn")


def _paths(entity_study: Path, factorial: Path, arm: str, fold: int) -> tuple[Path, Path]:
    recognizer, classifier = arm.split("+")
    tag = f"fold-{fold:02d}"
    entity = (entity_study / "dictionary" / tag) if recognizer == "dictionary" else (entity_study / "crf" / tag / "outer")
    relation = (factorial / "rules" / recognizer / tag) if classifier == "rules" else (factorial / "knn" / tag / recognizer)
    return entity, relation


def _write_graph(records: Sequence[GraphNode | GraphEdge], texts: Mapping[str, str], directory: Path) -> dict[str, Any]:
    graph_path = directory / "graph.jsonl"
    write_contract_jsonl(records, graph_path)
    reloaded = load_contract_jsonl(graph_path, "graph")
    if [item.to_dict() for item in reloaded] != [item.to_dict() for item in records]:
        raise ValueError(f"graph JSONL in {directory} does not round-trip")
    view = to_networkx(reloaded)
    nodes = sum(isinstance(item, GraphNode) for item in reloaded)
    edges = sum(isinstance(item, GraphEdge) for item in reloaded)
    if (view.number_of_nodes(), view.number_of_edges()) != (nodes, edges):
        raise ValueError("NetworkX view lost records")
    exports = {layer: export_legacy_csv(reloaded, texts, directory / f"{layer}-nodes.csv", directory / f"{layer}-edges.csv", layer=layer)
               for layer in ("mention", "concept")}
    counts = {layer: {"nodes": sum(isinstance(item, GraphNode) and item.layer == layer for item in reloaded),
                      "edges": sum(isinstance(item, GraphEdge) and item.layer == layer for item in reloaded)}
              for layer in ("mention", "concept")}
    return {"counts": counts, "exports": exports,
            "hashes": {name.name: sha256_file(name) for name in sorted(directory.glob("*")) if name.suffix in (".jsonl", ".csv")}}


def run_graph_study(corpus: str | Path, projection: str | Path, partitions: str | Path, entity_study: str | Path,
                    factorial: str | Path, output_directory: str | Path, *, folds: Sequence[int] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    corpus, projection, partitions, entity_study, factorial, output = map(
        Path, (corpus, projection, partitions, entity_study, factorial, output_directory))
    all_folds = {fold.index: fold for fold in load_document_folds(partitions)}
    indexes = sorted(folds) if folds else sorted(all_folds)
    documents = {item.document_id: item for item in load_jsonl(corpus)}
    texts = {key: value.text for key, value in documents.items()}
    flat = {item.document_id: item for item in load_flat_gold_jsonl(projection)}
    factorial_summary = json.loads((factorial / "summary.json").read_text(encoding="utf-8"))
    rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    gold_rows = []
    for index in indexes:
        fold = all_folds[index]
        ids = sorted(fold.validation_document_ids)
        gold, lost = build_gold_graph([documents[item] for item in ids], [flat[item] for item in ids])
        gold_dir = output / "gold" / f"fold-{index:02d}"
        gold_info = _write_graph(gold, texts, gold_dir)
        gold_manifest = {"fold": index, "graph": gold_info, "relations_lost_by_projection": lost,
                         "role": "reference graph from flat-gold-v1", "schema_version": 1,
                         "validation_document_ids": ids}
        (gold_dir / "manifest.json").write_text(json.dumps(gold_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        gold_rows.append({"fold": index, "counts": gold_info["counts"], "relations_lost_by_projection": lost})
        for arm in ARMS:
            status = next(item["status"] for item in factorial_summary["arms"][arm]["folds"] if item["fold"] == index)
            if status != "completed":
                rows[arm].append({"fold": index, "status": status})
                continue
            entity_dir, relation_dir = _paths(entity_study, factorial, arm, index)
            entities = load_contract_jsonl(entity_dir / "predictions.jsonl", "entity")
            relations = load_relation_predictions(relation_dir / "predictions.jsonl")
            graph = build_predicted_graph(texts, ids, entities, relations)
            directory = output / arm.replace("+", "-") / f"fold-{index:02d}"
            info = _write_graph(graph, texts, directory)
            evaluation = evaluate_graph(gold, graph)
            evaluation["metadata"] = {"arm": arm, "fold": index}
            (directory / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            manifest = {
                "arm": arm, "configuration": {"concept_identity": ["document_id", "canonical_type", "normalized_surface"],
                                              "normalization": NORMALIZATION, "projection_version": GRAPH_PROJECTION_VERSION,
                                              "stemming": False, "synonyms": False, "ontology_linking": False},
                "fold": index, "graph": info,
                "input_hashes": {"corpus_sha256": sha256_file(corpus), "entity_predictions_sha256": sha256_file(entity_dir / "predictions.jsonl"),
                                 "entity_manifest_sha256": sha256_file(entity_dir / "manifest.json"),
                                 "relation_predictions_sha256": sha256_file(relation_dir / "predictions.jsonl"),
                                 "relation_manifest_sha256": sha256_file(relation_dir / "manifest.json"),
                                 "gold_graph_sha256": gold_info["hashes"]["graph.jsonl"]},
                "evaluation_sha256": sha256_file(directory / "evaluation.json"),
                "schema_version": 1, "sensitive_artifacts": ["concept-nodes.csv", "mention-nodes.csv", "graph.jsonl"],
                "git": git_provenance(), "environment": dependency_versions(),
            }
            (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            rows[arm].append({"fold": index, "status": "completed", "evaluation": evaluation,
                              "manifest_sha256": sha256_file(directory / "manifest.json")})

    def spread(values: Sequence[float]) -> dict[str, float | None]:
        return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else None}

    arms = {}
    for arm, items in rows.items():
        evaluations = [item["evaluation"] for item in items if item["status"] == "completed"]
        if not evaluations:
            arms[arm] = {"completed_folds": [], "folds": items}
            continue
        pooled = {}
        for part in ("concept_nodes", "concept_edges"):
            tp = sum(item[part]["micro"]["true_positive"] for item in evaluations)
            predicted = sum(item[part]["micro"]["predicted"] for item in evaluations)
            support = sum(item[part]["micro"]["support"] for item in evaluations)
            precision, recall = (tp / predicted if predicted else 0.0), (tp / support if support else 0.0)
            pooled[part] = {"f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                            "precision": precision, "recall": recall, "predicted": predicted, "support": support, "true_positive": tp}
        arms[arm] = {
            "concept_edge_f1": spread([item["concept_edges"]["micro"]["f1"] for item in evaluations]),
            "concept_node_f1": spread([item["concept_nodes"]["micro"]["f1"] for item in evaluations]),
            "concept_edges_correct_only_after_aggregation": sum(item["aggregation"]["concept_edges_correct_only_after_aggregation"] for item in evaluations),
            "mention_edges_exact_matches": sum(item["mention_edges_exact_matches"] for item in evaluations),
            "pooled": pooled,
            "primary_relation_metric": factorial_summary["arms"][arm]["across_folds"]["macro_f1"],
            "completed_folds": [item["fold"] for item in items if item["status"] == "completed"],
            "folds": [{"fold": item["fold"], "status": item["status"], "manifest_sha256": item.get("manifest_sha256")}
                      for item in items],
        }
    summary = {"arms": arms, "gold": gold_rows, "duration_seconds": round(time.perf_counter() - started, 3),
               "input_hashes": {"corpus_sha256": sha256_file(corpus), "gold_projection_sha256": sha256_file(projection),
                                "factorial_summary_sha256": sha256_file(factorial / "summary.json")},
               "interpretation": "concept metrics are secondary to strict entity and relation metrics",
               "schema_version": 1, "git": git_provenance()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


__all__ = ["ARMS", "run_graph_study"]
