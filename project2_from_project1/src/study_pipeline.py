"""Single public entry point: raw text to Knowledge Graph for every arm.

Stages, in order, each under its own subdirectory of the output:

1. ``gold``      flat gold projection from the lossless corpus and adjudication;
2. ``entities``  nested document CV of the dictionary and the semi-Markov CRF;
3. ``relations`` factorial {dictionary, CRF} x {rules, k-NN} plus the gold ceiling;
4. ``graph``     two-layer graph JSONL v2 with CSV and NetworkX views.

No stage applies a performance gate: every arm stays in the summaries.  A
non-finite CRF run marks only that recognizer-fold and the arms built on it.
The root ``manifest.json`` is written last and lists the commit, dependencies,
configuration, partitions, stage times and the SHA-256 of every child artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from project2_from_project1.src.entity_recognition.corpus import load_document_folds
from project2_from_project1.src.entity_recognition.nested_cv import INNER_FOLDS, L2_GRID, StudyInputs, run_nested_study
from project2_from_project1.src.gold_projection import project_gold_from_files
from project2_from_project1.src.graph_study import run_graph_study
from project2_from_project1.src.relation_factorial import run_factorial_study
from project2_from_project1.src.run_manifest import dependency_versions, git_provenance, sha256_file

STUDY_VERSION = "end-to-end-study-v1"


def run_full_study(
    corpus_jsonl: str | Path,
    adjudication_jsonl: str | Path,
    frozen_dataset: str | Path,
    output_directory: str | Path,
    *,
    folds: Sequence[int] | None = None,
    max_iterations: int = 250,
    workers: int = 1,
    threads_per_worker: int = 1,
    crf_fit: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    corpus, adjudication, frozen, output = map(Path, (corpus_jsonl, adjudication_jsonl, frozen_dataset, output_directory))
    partitions = frozen / "manifest.json"
    selected = sorted(folds) if folds else sorted(fold.index for fold in load_document_folds(partitions))
    timings = {}

    stage = time.perf_counter()
    gold = output / "gold"
    projection_manifest = project_gold_from_files(corpus, adjudication, gold / "gold-projection.jsonl",
                                                  gold / "adjudication-audit.jsonl", gold / "manifest.json")
    timings["gold_projection"] = round(time.perf_counter() - stage, 3)

    stage = time.perf_counter()
    entities = run_nested_study(
        StudyInputs(str(corpus), str(gold / "gold-projection.jsonl"), str(gold / "manifest.json"), str(partitions)),
        output / "entities", folds=selected, max_iterations=max_iterations, workers=workers,
        threads_per_worker=threads_per_worker, fit=crf_fit)
    timings["entity_recognition"] = round(time.perf_counter() - stage, 3)

    stage = time.perf_counter()
    relations = run_factorial_study(corpus, gold / "gold-projection.jsonl", frozen, output / "entities",
                                    output / "relations", folds=selected)
    timings["relations"] = round(time.perf_counter() - stage, 3)

    stage = time.perf_counter()
    graph = run_graph_study(corpus, gold / "gold-projection.jsonl", partitions, output / "entities",
                            output / "relations", output / "graph", folds=selected)
    timings["graph"] = round(time.perf_counter() - stage, 3)

    arms = {}
    for arm, data in relations["arms"].items():
        recognizer = arm.split("+")[0]
        entity_arm = entities["arms"].get({"dictionary": "canonical-dictionary", "crf": "semi-markov-crf"}.get(recognizer, ""), {})
        arms[arm] = {
            "completed_folds": data["completed_folds"],
            "failed_folds": data["failed_folds"],
            "relation_macro_f1": data["across_folds"]["macro_f1"] if data["across_folds"] else None,
            "role": data["role"],
            "entity_macro_f1": entity_arm.get("aggregate", {}).get("macro_f1") if entity_arm else None,
            "concept_edge_f1": graph["arms"].get(arm, {}).get("concept_edge_f1"),
        }
    folds_manifest = json.loads(partitions.read_text(encoding="utf-8"))["partitions"]["folds"]
    root_manifest = output / "manifest.json"
    artifacts = {str(path.relative_to(output)): sha256_file(path)
                 for path in sorted(output.rglob("*")) if path.is_file() and path != root_manifest}
    manifest = {
        "arms": arms,
        "artifact_hashes": artifacts,
        "configuration": {
            "crf": {"inner_folds": INNER_FOLDS, "l2_grid": list(L2_GRID), "max_iterations": max_iterations},
            "folds": selected, "study_version": STUDY_VERSION,
            "threads_per_worker": threads_per_worker, "workers": workers,
        },
        "dependencies": dependency_versions(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "git": git_provenance(),
        "inputs": {"adjudication_sha256": sha256_file(adjudication), "corpus_sha256": sha256_file(corpus),
                   "partition_manifest_sha256": sha256_file(partitions),
                   "projection_output_sha256": projection_manifest["output_sha256"]},
        "interpretation": "exploratory development folds; no performance gate; gold+* arms are diagnostic ceilings",
        "partitions": [item for item in folds_manifest if item["fold"] in selected],
        "schema_version": 1,
        "stage_seconds": timings,
        "stages": {"entities": "entities/summary.json", "gold": "gold/manifest.json", "graph": "graph/summary.json",
                   "relations": "relations/summary.json"},
        "test_partition_read": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                          encoding="utf-8")
    return manifest


__all__ = ["STUDY_VERSION", "run_full_study"]
