"""Cross-validation model selection for the n-gram k-NN classifier."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

from project2_from_project1.src.ngrams.pipeline import run_ngram_knn_from_files
from project2_from_project1.src.relation_evaluation import (
    EvaluationConfig,
    evaluate_predictions,
    load_supervised_candidates,
    write_evaluation_report,
)
from project2_from_project1.src.supervised_dataset import (
    SupervisedCandidate,
    sha256_file,
)


class NgramKNNSelectionError(ValueError):
    """Raised when frozen folds cannot support reproducible model selection."""


def _fold_pairs(frozen: Path) -> list[tuple[str, Path, Path]]:
    result: list[tuple[str, Path, Path]] = []
    for training_path in sorted(frozen.glob("fold-*-train.jsonl")):
        fold_id = training_path.name.removesuffix("-train.jsonl")
        validation_path = frozen / f"{fold_id}-validation.jsonl"
        if not validation_path.is_file():
            raise NgramKNNSelectionError(
                f"missing validation partition for {training_path.name}"
            )
        result.append((fold_id, training_path, validation_path))
    if len(result) < 2:
        raise NgramKNNSelectionError("at least two complete folds are required")
    return result


def _positive_integer_values(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(sorted(set(values)))
    if not result or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in result
    ):
        raise NgramKNNSelectionError(f"{name} must contain positive integers")
    return result


def select_ngram_knn_from_folds(
    corpus_jsonl: str | Path,
    frozen_dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    neighbor_values: Iterable[int] = (1, 3, 5),
    maximum_ngram_orders: Iterable[int] = (1, 2, 3),
    seed: int = 0,
    minimum_positive_support: int = 5,
    model_version: str = "ngram-knn-v1",
) -> dict[str, Any]:
    """Evaluate a parameter grid on every validation fold, never on test.jsonl."""

    neighbor_values = _positive_integer_values(neighbor_values, "neighbor_values")
    maximum_ngram_orders = _positive_integer_values(
        maximum_ngram_orders, "maximum_ngram_orders"
    )
    if any(order not in (1, 2, 3) for order in maximum_ngram_orders):
        raise NgramKNNSelectionError(
            "maximum_ngram_orders may contain only 1, 2, or 3"
        )

    frozen = Path(frozen_dataset_directory)
    fold_pairs = _fold_pairs(frozen)

    output = Path(output_directory)
    configurations: list[dict[str, Any]] = []
    for maximum_order in maximum_ngram_orders:
        for n_neighbors in neighbor_values:
            configuration_name = f"ngrams-1-{maximum_order}_k-{n_neighbors}"
            configuration_directory = output / configuration_name
            folds: list[dict[str, Any]] = []
            for fold_id, training_path, validation_path in fold_pairs:
                predictions_path = (
                    configuration_directory / f"{fold_id}.predictions.jsonl"
                )
                evaluation_path = (
                    configuration_directory / f"{fold_id}.evaluation.json"
                )
                run = run_ngram_knn_from_files(
                    corpus_jsonl,
                    training_path,
                    validation_path,
                    predictions_path,
                    n_neighbors=n_neighbors,
                    maximum_ngram_order=maximum_order,
                    model_version=model_version,
                )
                report = evaluate_predictions(
                    load_supervised_candidates(validation_path),
                    run.predictions,
                    config=EvaluationConfig(
                        seed=seed,
                        minimum_positive_support=minimum_positive_support,
                    ),
                    metadata={
                        "fold": fold_id,
                        "maximum_ngram_order": maximum_order,
                        "n_neighbors": n_neighbors,
                        "selection_partition": "validation",
                    },
                )
                write_evaluation_report(report, evaluation_path)
                folds.append(
                    {
                        "classification_count": report.metrics["classification_count"],
                        "fold": fold_id,
                        "per_class": report.metrics["per_class"],
                        "positive_macro_f1": report.metrics["positive_macro_f1"],
                        "vocabulary_size": len(run.vocabulary),
                    }
                )

            scores = [float(fold["positive_macro_f1"]) for fold in folds]
            configurations.append(
                {
                    "folds": folds,
                    "maximum_ngram_order": maximum_order,
                    "mean_positive_macro_f1": fmean(scores),
                    "n_neighbors": n_neighbors,
                    "population_std_positive_macro_f1": pstdev(scores),
                }
            )

    best = max(
        configurations,
        key=lambda item: (
            item["mean_positive_macro_f1"],
            -item["population_std_positive_macro_f1"],
            -item["maximum_ngram_order"],
            -item["n_neighbors"],
        ),
    )
    summary = {
        "best_configuration": {
            key: best[key]
            for key in (
                "maximum_ngram_order",
                "mean_positive_macro_f1",
                "n_neighbors",
                "population_std_positive_macro_f1",
            )
        },
        "configurations": configurations,
        "folds": [fold_id for fold_id, _, _ in fold_pairs],
        "metric": "positive_macro_f1",
        "model_version": model_version,
        "schema_version": 1,
        "seed": seed,
        "test_partition_used": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "model-selection-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_final_training_partition(
    frozen_dataset_directory: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    maximum_negatives_per_positive: int = 2,
) -> dict[str, Any]:
    """Build final development training data from disjoint validation folds."""

    if (
        isinstance(maximum_negatives_per_positive, bool)
        or not isinstance(maximum_negatives_per_positive, int)
        or maximum_negatives_per_positive < 0
    ):
        raise NgramKNNSelectionError(
            "maximum_negatives_per_positive must be a non-negative integer"
        )
    fold_pairs = _fold_pairs(Path(frozen_dataset_directory))
    candidates: list[SupervisedCandidate] = []
    seen: set[SupervisedCandidate] = set()
    seen_document_ids: set[str] = set()
    for fold_id, _, validation_path in fold_pairs:
        fold_candidates = load_supervised_candidates(validation_path)
        fold_document_ids = {candidate.document_id for candidate in fold_candidates}
        overlap = seen_document_ids.intersection(fold_document_ids)
        if overlap:
            raise NgramKNNSelectionError(
                f"validation folds overlap in {fold_id}: {sorted(overlap)[0]!r}"
            )
        seen_document_ids.update(fold_document_ids)
        for candidate in fold_candidates:
            if candidate in seen:
                raise NgramKNNSelectionError(
                    f"duplicate candidate across validation folds: {candidate!r}"
                )
            seen.add(candidate)
            candidates.append(candidate)

    positives = [candidate for candidate in candidates if candidate.is_positive]
    negatives = [candidate for candidate in candidates if not candidate.is_positive]
    negative_limit = maximum_negatives_per_positive * len(positives)
    if len(negatives) > negative_limit:
        negatives = random.Random(f"{seed}:final:negatives").sample(
            negatives, negative_limit
        )
    selected = sorted((*positives, *negatives))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(
                candidate.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for candidate in selected
        ),
        encoding="utf-8",
    )
    manifest = {
        "candidate_count": len(selected),
        "document_count": len(seen_document_ids),
        "maximum_negatives_per_positive": maximum_negatives_per_positive,
        "negative_count": len(negatives),
        "output_sha256": sha256_file(output),
        "positive_count": len(positives),
        "seed": seed,
        "source": "union_of_disjoint_validation_folds",
        "source_folds": [fold_id for fold_id, _, _ in fold_pairs],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def finalize_selected_ngram_knn(
    corpus_jsonl: str | Path,
    frozen_dataset_directory: str | Path,
    selection_summary_path: str | Path,
    output_directory: str | Path,
    *,
    maximum_negatives_per_positive: int = 2,
    minimum_positive_support: int = 5,
) -> dict[str, Any]:
    """Fit the selected model on development data and evaluate test once."""

    selection = json.loads(Path(selection_summary_path).read_text(encoding="utf-8"))
    if selection.get("test_partition_used") is not False:
        raise NgramKNNSelectionError(
            "selection summary must confirm that the test partition was not used"
        )
    best = selection.get("best_configuration")
    if not isinstance(best, dict):
        raise NgramKNNSelectionError("selection summary has no best configuration")
    seed = selection.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise NgramKNNSelectionError("selection summary has an invalid seed")
    n_neighbors = best.get("n_neighbors")
    maximum_order = best.get("maximum_ngram_order")
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int):
        raise NgramKNNSelectionError("selected n_neighbors is invalid")
    if maximum_order not in (1, 2, 3):
        raise NgramKNNSelectionError("selected maximum_ngram_order is invalid")

    frozen = Path(frozen_dataset_directory)
    test_path = frozen / "test.jsonl"
    if not test_path.is_file():
        raise NgramKNNSelectionError("frozen test.jsonl is missing")
    output = Path(output_directory)
    training_path = output / "final-development-train.jsonl"
    training_manifest = build_final_training_partition(
        frozen,
        training_path,
        seed=seed,
        maximum_negatives_per_positive=maximum_negatives_per_positive,
    )
    predictions_path = output / "final-test.predictions.jsonl"
    run = run_ngram_knn_from_files(
        corpus_jsonl,
        training_path,
        test_path,
        predictions_path,
        n_neighbors=n_neighbors,
        maximum_ngram_order=maximum_order,
        model_version=selection.get("model_version", "ngram-knn-v1"),
    )
    report = evaluate_predictions(
        load_supervised_candidates(test_path),
        run.predictions,
        config=EvaluationConfig(
            seed=seed,
            minimum_positive_support=minimum_positive_support,
        ),
        metadata={
            "maximum_ngram_order": maximum_order,
            "n_neighbors": n_neighbors,
            "partition": "fixed-test",
            "selection_summary_sha256": sha256_file(selection_summary_path),
        },
    )
    evaluation_path = output / "final-test.evaluation.json"
    write_evaluation_report(report, evaluation_path)
    summary = {
        "evaluation_path": str(evaluation_path),
        "maximum_ngram_order": maximum_order,
        "n_neighbors": n_neighbors,
        "positive_macro_f1": report.metrics["positive_macro_f1"],
        "predictions_path": str(predictions_path),
        "test_partition_used_once_after_selection": True,
        "training_manifest": training_manifest,
    }
    (output / "final-run-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "NgramKNNSelectionError",
    "build_final_training_partition",
    "finalize_selected_ngram_knn",
    "select_ngram_knn_from_folds",
]
