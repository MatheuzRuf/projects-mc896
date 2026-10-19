"""Nested document cross-validation comparing the dictionary and the semi-Markov CRF.

For every outer fold of the frozen partitions:

* the outer training documents are split into ``INNER_FOLDS`` inner folds by
  document, with a seeded shuffle;
* the CRF is fitted for each L2 value of the pre-registered grid on every
  inner training split and scored by strict macro-F1 on its inner validation;
* the selected L2 maximizes the mean inner macro-F1, ties going to the lowest
  population variance and then to the strongest regularization;
* the CRF is refitted on the whole outer training partition with that value
  and evaluated on the outer validation fold;
* the dictionary arm runs on the same outer folds.

The consumed MCRC test partition is never read: folds come from
:func:`load_document_folds`, which only exposes train and validation ids.
Inner results are cached as JSON keyed by their inputs and configuration, so
an interrupted study resumes without refitting completed runs.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from project2_from_project1.src.entity_recognition.corpus import DocumentFold, EntityRecognitionError, select_documents
from project2_from_project1.src.entity_recognition.crf_model import NonFiniteTrainingError, SemiMarkovCRF, TrainingConfig
from project2_from_project1.src.entity_recognition.evaluation import evaluate_entities, training_vocabulary
from project2_from_project1.src.entity_recognition.run import (
    prepare_fold,
    run_crf_fold_from_files,
    run_dictionary_fold_from_files,
)
from project2_from_project1.src.run_manifest import dependency_versions, git_provenance, sha256_file, sha256_text

L2_GRID = (0.01, 0.1, 1.0, 10.0)
INNER_FOLDS = 3
INNER_SEED = 42
_TIE_DIGITS = 12


def inner_partitions(train_document_ids: Sequence[str], outer_index: int, folds: int = INNER_FOLDS,
                     seed: int = INNER_SEED) -> tuple[DocumentFold, ...]:
    """Seeded, document-disjoint inner folds covering the outer training ids."""
    ids = sorted(train_document_ids)
    if len(ids) < folds:
        raise EntityRecognitionError(f"outer fold {outer_index} has fewer than {folds} training documents")
    random.Random(f"{seed}:outer:{outer_index}").shuffle(ids)
    buckets = [sorted(ids[position::folds]) for position in range(folds)]
    return tuple(
        DocumentFold(index, tuple(sorted(set(ids) - set(bucket))), tuple(bucket))
        for index, bucket in enumerate(buckets, 1)
    )


def select_l2(scores: Mapping[float, Sequence[float | None]]) -> dict[str, Any]:
    """Pick L2 by mean inner macro-F1, then lowest variance, then strongest L2.

    A value with any failed (non-finite) inner run is ineligible but reported.
    """
    table = {}
    for l2, values in sorted(scores.items()):
        complete = all(value is not None for value in values)
        mean = statistics.fmean(values) if complete else None
        variance = statistics.pvariance(values) if complete else None
        table[l2] = {"eligible": complete, "inner_macro_f1": list(values), "mean": mean, "variance": variance}
    eligible = [l2 for l2, row in table.items() if row["eligible"]]
    if not eligible:
        raise EntityRecognitionError("no L2 value completed every inner fold")
    selected = min(eligible, key=lambda l2: (-round(table[l2]["mean"], _TIE_DIGITS),
                                             round(table[l2]["variance"], _TIE_DIGITS), -l2))
    return {"criteria": ["max_mean_inner_strict_macro_f1", "min_population_variance", "max_l2"],
            "selected_l2": selected, "table": {str(l2): row for l2, row in table.items()}}


@dataclass(frozen=True)
class StudyInputs:
    corpus_jsonl: str
    projection_jsonl: str
    projection_manifest_json: str
    partition_manifest_json: str

    def paths(self) -> tuple[str, str, str, str]:
        return self.corpus_jsonl, self.projection_jsonl, self.projection_manifest_json, self.partition_manifest_json


@dataclass(frozen=True)
class InnerJob:
    inputs: StudyInputs
    outer_index: int
    inner: DocumentFold
    config: TrainingConfig
    output_json: str
    cache_key: str


_PREPARED: dict[tuple[StudyInputs, int], Any] = {}


def _prepared(inputs: StudyInputs, outer_index: int):
    key = (inputs, outer_index)
    if key not in _PREPARED:
        _PREPARED[key] = prepare_fold(*inputs.paths(), outer_index)
    return _PREPARED[key]


def run_inner_job(job: InnerJob) -> dict[str, Any]:
    """Fit one CRF on an inner training split and score its inner validation."""
    output = Path(job.output_json)
    if output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("cache_key") == job.cache_key:
            return cached
    started = time.perf_counter()
    prepared = _prepared(job.inputs, job.outer_index)
    outer_train = set(prepared.fold.train_document_ids)
    if not set(job.inner.train_document_ids) | set(job.inner.validation_document_ids) <= outer_train:
        raise EntityRecognitionError("inner folds must use outer training documents only")
    training = select_documents(prepared.gold, job.inner.train_document_ids, "inner train")
    validation = select_documents(prepared.gold, job.inner.validation_document_ids, "inner validation")
    result: dict[str, Any] = {
        "cache_key": job.cache_key, "config": job.config.to_dict(), "inner_fold": job.inner.index,
        "inner_train_document_ids": list(job.inner.train_document_ids),
        "inner_validation_document_ids": list(job.inner.validation_document_ids),
        "l2": job.config.l2, "outer_fold": job.outer_index, "schema_version": 1,
        "torch_threads": torch.get_num_threads(),
    }
    try:
        model, diagnostics = SemiMarkovCRF.fit(training, job.config)
    except NonFiniteTrainingError as error:
        result.update(status="failed_non_finite", failure={"message": str(error), **error.diagnostic},
                      macro_f1=None, training=None)
    else:
        predictions = model.predict([document.raw() for document in validation])
        surfaces, tokens = training_vocabulary(training)
        evaluation = evaluate_entities(validation, predictions, training_surfaces=surfaces, training_tokens=tokens)
        result.update(
            status="completed", macro_f1=evaluation["strict"]["macro_f1"],
            micro_f1=evaluation["strict"]["micro"]["f1"],
            per_type_f1={name: row["f1"] for name, row in evaluation["strict"]["per_type"].items()},
            training=diagnostics.to_dict(),
        )
    result["duration_seconds"] = round(time.perf_counter() - started, 6)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def _init_worker(threads: int) -> None:
    torch.set_num_threads(threads)


def _map(function: Callable[[Any], Any], jobs: Sequence[Any], workers: int, threads: int) -> list[Any]:
    if workers <= 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(threads,)) as pool:
        return list(pool.map(function, jobs))


@dataclass(frozen=True)
class OuterJob:
    inputs: StudyInputs
    outer_index: int
    config: TrainingConfig
    output_dir: str


def run_outer_crf_job(job: OuterJob) -> dict[str, Any]:
    output = Path(job.output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed" and manifest.get("configuration", {}).get("l2") == job.config.l2 \
                and manifest.get("configuration", {}).get("max_iterations") == job.config.max_iterations \
                and all(manifest["artifact_hashes"].get(name) == value
                        for name, value in prepare_fold(*job.inputs.paths(), job.outer_index).input_hashes.items()):
            return manifest
    try:
        return run_crf_fold_from_files(*job.inputs.paths(), job.outer_index, output, config=job.config)
    except NonFiniteTrainingError:
        return json.loads(manifest_path.read_text(encoding="utf-8"))


def _metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    strict = evaluation["strict"]
    return {
        "errors": evaluation["errors"],
        "macro_f1": strict["macro_f1"],
        "micro": {name: strict["micro"][name] for name in ("f1", "precision", "recall", "predicted", "support", "true_positive")},
        "oov_surface_recall": {name: row["recall"] for name, row in evaluation["diagnostics"]["oov_surface"].items()},
        "per_type_f1": {name: row["f1"] for name, row in strict["per_type"].items()},
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def spread(values: Sequence[float]) -> dict[str, float | None]:
        return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values), "max": max(values)}

    evaluated = [row for row in rows if row.get("metrics")]
    if not evaluated:
        return {"evaluated_folds": 0}
    types = sorted(evaluated[0]["metrics"]["per_type_f1"])
    errors: dict[str, dict[str, int]] = {}
    for row in evaluated:
        for side in ("false_negative", "false_positive"):
            for reason, count in row["metrics"]["errors"][side].items():
                errors.setdefault(side, {}).setdefault(reason, 0)
                errors[side][reason] += count
    return {
        "errors_total": errors,
        "evaluated_folds": len(evaluated),
        "macro_f1": spread([row["metrics"]["macro_f1"] for row in evaluated]),
        "micro_f1": spread([row["metrics"]["micro"]["f1"] for row in evaluated]),
        "micro_precision": spread([row["metrics"]["micro"]["precision"] for row in evaluated]),
        "micro_recall": spread([row["metrics"]["micro"]["recall"] for row in evaluated]),
        "per_type_f1": {name: spread([row["metrics"]["per_type_f1"][name] for row in evaluated]) for name in types},
        "unseen_surface_recall": spread([row["metrics"]["oov_surface_recall"]["unseen_surface"] for row in evaluated]),
    }


def run_nested_study(
    inputs: StudyInputs,
    output_directory: str | Path,
    *,
    folds: Iterable[int] | None = None,
    max_iterations: int = 250,
    workers: int = 1,
    threads_per_worker: int = 1,
    l2_grid: Sequence[float] = L2_GRID,
) -> dict[str, Any]:
    if tuple(l2_grid) != L2_GRID:
        raise EntityRecognitionError(f"the pre-registered L2 grid is {L2_GRID}")
    started = time.perf_counter()
    output = Path(output_directory)
    base = prepare_fold(*inputs.paths(), 1)
    from project2_from_project1.src.entity_recognition.corpus import load_document_folds

    all_folds = {fold.index: fold for fold in load_document_folds(Path(inputs.partition_manifest_json))}
    selected_folds = sorted(folds) if folds is not None else sorted(all_folds)
    hash_key = sha256_text(json.dumps(base.input_hashes, sort_keys=True))

    jobs = []
    plans: dict[int, tuple[DocumentFold, ...]] = {}
    for outer in selected_folds:
        plans[outer] = inner_partitions(all_folds[outer].train_document_ids, outer)
        for l2 in l2_grid:
            config = TrainingConfig(l2=l2, max_iterations=max_iterations)
            for inner in plans[outer]:
                key = sha256_text(json.dumps([hash_key, outer, inner.index, inner.train_document_ids,
                                              inner.validation_document_ids, config.to_dict()], sort_keys=True))
                path = output / "crf" / f"fold-{outer:02d}" / "inner" / f"l2-{l2:g}" / f"inner-{inner.index}.json"
                jobs.append(InnerJob(inputs, outer, inner, config, str(path), key))
    inner_results = _map(run_inner_job, jobs, workers, threads_per_worker)

    selections = {}
    outer_jobs = []
    for outer in selected_folds:
        scores = {l2: [None] * INNER_FOLDS for l2 in l2_grid}
        for result in inner_results:
            if result["outer_fold"] == outer:
                scores[result["l2"]][result["inner_fold"] - 1] = result["macro_f1"]
        selections[outer] = select_l2(scores)
        selection_path = output / "crf" / f"fold-{outer:02d}" / "selection.json"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps({
            **selections[outer],
            "inner_partitions": [{"inner_fold": item.index, "train_document_ids": list(item.train_document_ids),
                                  "validation_document_ids": list(item.validation_document_ids)}
                                 for item in plans[outer]],
            "outer_fold": outer,
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        outer_jobs.append(OuterJob(inputs, outer, TrainingConfig(l2=selections[outer]["selected_l2"],
                                                                 max_iterations=max_iterations),
                                   str(output / "crf" / f"fold-{outer:02d}" / "outer")))
    crf_manifests = _map(run_outer_crf_job, outer_jobs, workers, threads_per_worker)

    dictionary_manifests = [
        run_dictionary_fold_from_files(*inputs.paths(), outer, output / "dictionary" / f"fold-{outer:02d}")
        for outer in selected_folds
    ]

    def row(manifest: Mapping[str, Any], directory: Path, extra: Mapping[str, Any]) -> dict[str, Any]:
        evaluation_path = directory / "evaluation.json"
        metrics = _metrics(json.loads(evaluation_path.read_text(encoding="utf-8"))) if manifest["status"] == "completed" else None
        return {"duration_seconds": manifest["duration_seconds"], "fold": manifest["fold"],
                "manifest_sha256": sha256_file(directory / "manifest.json"), "metrics": metrics,
                "status": manifest["status"], **extra}

    crf_rows = []
    for outer, manifest in zip(selected_folds, crf_manifests):
        inner_for_fold = [result for result in inner_results if result["outer_fold"] == outer]
        crf_rows.append(row(manifest, Path(outer_jobs[selected_folds.index(outer)].output_dir), {
            "converged": manifest.get("converged"),
            "inner_runs": {
                "converged": sum(bool(result["training"] and result["training"]["converged"]) for result in inner_for_fold),
                "failed_non_finite": sum(result["status"] == "failed_non_finite" for result in inner_for_fold),
                "total": len(inner_for_fold),
                "duration_seconds": round(sum(result["duration_seconds"] for result in inner_for_fold), 3),
            },
            "selected_l2": selections[outer]["selected_l2"],
            "training": manifest.get("training"),
        }))
    dictionary_rows = [row(manifest, output / "dictionary" / f"fold-{outer:02d}", {})
                       for outer, manifest in zip(selected_folds, dictionary_manifests)]

    summary = {
        "arms": {
            "canonical-dictionary": {"aggregate": _aggregate(dictionary_rows), "folds": dictionary_rows},
            "semi-markov-crf": {"aggregate": _aggregate(crf_rows), "folds": crf_rows},
        },
        "configuration": {
            "inner_folds": INNER_FOLDS, "inner_seed": INNER_SEED, "l2_grid": list(L2_GRID),
            "max_iterations": max_iterations, "outer_folds": selected_folds,
            "selection": "max mean inner strict macro-F1; then min population variance; then max L2",
            "threads_per_worker": threads_per_worker, "workers": workers,
        },
        "duration_seconds": round(time.perf_counter() - started, 3),
        "environment": dependency_versions(),
        "git": git_provenance(),
        "input_hashes": base.input_hashes,
        "interpretation": "exploratory: development folds only; the consumed MCRC test partition was not read",
        "schema_version": 1,
        "test_partition_read": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                         encoding="utf-8")
    return summary


__all__ = ["INNER_FOLDS", "L2_GRID", "StudyInputs", "inner_partitions", "run_nested_study", "select_l2"]
