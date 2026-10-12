"""Reproducible orchestration for the first supervised-relation milestone.

The lower-level modules deliberately keep conversion, leakage auditing,
suitability, partitioning, and evaluation independently usable.  This module
is the narrow, documented seam that runs the approved path end to end without
ever reopening a raw distribution or placing clinical text in its reports.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import time
from typing import Any, Mapping, Sequence

from project2_from_project1.src.clinical_documents import CanonicalRelation, load_jsonl
from project2_from_project1.src.leakage_audit import (
    audit_external_corpus,
    load_protected_collection,
    write_leakage_report,
)
from project2_from_project1.src.mcrc_suitability import (
    MCRCSuitabilityReport,
    SuitabilityCriteria,
    SuitabilityDecision,
    assess_mcrc_suitability,
    write_mcrc_suitability_report,
)
from project2_from_project1.src.relation_evaluation import (
    EvaluationConfig,
    evaluate_predictions,
    load_relation_predictions,
    load_supervised_candidates,
    write_evaluation_report,
)
from project2_from_project1.src.supervised_dataset import (
    CandidatePolicy,
    PartitionPolicy,
    build_frozen_supervised_dataset,
    sha256_file,
    write_frozen_supervised_dataset,
)


class FirstMilestoneError(ValueError):
    """Raised when a milestone handoff is incomplete or unsafe."""


_EVALUATION_PARTITION = re.compile(r"(?:test|fold-\d{2}-validation)")


def _write_json(path: Path, value: Mapping[str, Any], *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent, separators=None if indent else (",", ":")) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FirstMilestoneError(f"cannot read {name}: {error}") from error
    if not isinstance(value, dict):
        raise FirstMilestoneError(f"{name} must be a JSON object")
    return value


def _versions() -> dict[str, str]:
    return {
        "first_milestone_schema": "1",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def _artifact_hashes(directory: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in sorted(names)}


@dataclass(frozen=True)
class PreparationResult:
    """Locations and explicit outcome from the preparation interface."""

    status: str
    output_directory: Path
    manifest_path: Path
    run_report_path: Path
    leakage_report_path: Path
    suitability_report_path: Path
    dataset_directory: Path | None


def prepare_first_milestone(
    safe_jsonl: str | Path,
    protected_collection: str | Path,
    output_directory: str | Path,
    *,
    candidate_policy: CandidatePolicy | None = None,
    partition_policy: PartitionPolicy | None = None,
    suitability_criteria: SuitabilityCriteria | None = None,
    similarity_threshold: float = 0.85,
) -> PreparationResult:
    """Audit a safe corpus, decide suitability, and freeze approved partitions.

    A rejected corpus is represented by deterministic reports with
    ``status == 'rejected'`` and never receives a dataset directory.  This is
    intentional: callers can preserve the audit evidence without accidentally
    treating a failed preparation as permission to train.
    """

    source_path = Path(safe_jsonl)
    protected_path = Path(protected_collection)
    output = Path(output_directory)
    candidate_policy = candidate_policy or CandidatePolicy()
    partition_policy = partition_policy or PartitionPolicy()
    suitability_criteria = suitability_criteria or SuitabilityCriteria()
    if not isinstance(candidate_policy, CandidatePolicy):
        raise TypeError("candidate_policy must be a CandidatePolicy")
    if not isinstance(partition_policy, PartitionPolicy):
        raise TypeError("partition_policy must be a PartitionPolicy")
    if not isinstance(suitability_criteria, SuitabilityCriteria):
        raise TypeError("suitability_criteria must be a SuitabilityCriteria")

    output.mkdir(parents=True, exist_ok=True)
    leakage_path = output / "leakage-report.json"
    suitability_path = output / "suitability-report.json"
    manifest_path = output / "preparation-manifest.json"
    run_report_path = output / "preparation-run-report.json"
    started = time.perf_counter()

    audit_started = time.perf_counter()
    documents = load_jsonl(source_path)
    collection = load_protected_collection(protected_path)
    leakage_report = audit_external_corpus(documents, collection, similarity_threshold=similarity_threshold)
    write_leakage_report(leakage_report, leakage_path)
    audit_elapsed = time.perf_counter() - audit_started

    suitability_started = time.perf_counter()
    suitability_report = assess_mcrc_suitability(documents, leakage_report, criteria=suitability_criteria)
    write_mcrc_suitability_report(suitability_report, suitability_path)
    suitability_elapsed = time.perf_counter() - suitability_started

    artifact_names = ["leakage-report.json", "suitability-report.json"]
    dataset_directory: Path | None = None
    dataset_manifest: Mapping[str, Any] | None = None
    freeze_elapsed = 0.0
    if suitability_report.decision is SuitabilityDecision.APPROVED:
        freeze_started = time.perf_counter()
        dataset_directory = output / "frozen-dataset"
        dataset = build_frozen_supervised_dataset(
            documents,
            suitability_report,
            candidate_policy=candidate_policy,
            partition_policy=partition_policy,
            source_jsonl_sha256=sha256_file(source_path),
        )
        dataset_manifest = write_frozen_supervised_dataset(dataset, dataset_directory)
        freeze_elapsed = time.perf_counter() - freeze_started
        artifact_names.extend(f"frozen-dataset/{name}" for name in sorted(dataset_manifest["artifact_hashes"]))
        artifact_names.append("frozen-dataset/manifest.json")

    # This manifest is deliberately timing-free, so it is byte-for-byte
    # reproducible for identical safe inputs and policies.  The adjacent run
    # report retains observed durations, which are naturally non-deterministic.
    artifact_hashes = {
        name: sha256_file(output / name)
        for name in sorted(artifact_names)
    }
    status = suitability_report.decision.value
    manifest: dict[str, Any] = {
        "artifact_hashes": artifact_hashes,
        "candidate_policy": candidate_policy.to_dict(),
        "input_hashes": {
            "protected_collection_sha256": sha256_file(protected_path),
            "safe_jsonl_sha256": sha256_file(source_path),
        },
        "leakage_report": leakage_report.to_dict(),
        "partition_policy": partition_policy.to_dict(),
        "schema_version": 1,
        "status": status,
        "suitability_report": suitability_report.to_dict(),
        "versions": _versions(),
    }
    if dataset_manifest is not None:
        manifest["frozen_dataset_manifest"] = dataset_manifest
    _write_json(manifest_path, manifest)

    run_report = {
        "deterministic_manifest_sha256": sha256_file(manifest_path),
        "outputs": {"preparation_manifest": manifest_path.name, "status": status},
        "schema_version": 1,
        "status": status,
        "timing_seconds": {
            "audit": audit_elapsed,
            "freeze": freeze_elapsed,
            "preparation_total": time.perf_counter() - started,
            "suitability": suitability_elapsed,
        },
    }
    _write_json(run_report_path, run_report, indent=2)
    return PreparationResult(status, output, manifest_path, run_report_path, leakage_path, suitability_path, dataset_directory)


def evaluate_first_milestone(
    prepared_directory: str | Path,
    predictions: str | Path,
    output_report: str | Path,
    *,
    partition: str = "test",
    seed: int = 0,
    minimum_positive_support: int = 5,
) -> Path:
    """Evaluate complete predictions against one prepared test/validation gold set."""

    if not _EVALUATION_PARTITION.fullmatch(partition):
        raise FirstMilestoneError("partition must be 'test' or 'fold-XX-validation'")
    prepared = Path(prepared_directory)
    preparation_manifest = _read_json(prepared / "preparation-manifest.json", "preparation manifest")
    if preparation_manifest.get("status") != SuitabilityDecision.APPROVED.value:
        raise FirstMilestoneError("prepared directory is not an approved frozen dataset")
    candidate_path = prepared / "frozen-dataset" / f"{partition}.jsonl"
    dataset_manifest_path = prepared / "frozen-dataset" / "manifest.json"
    if not candidate_path.is_file() or not dataset_manifest_path.is_file():
        raise FirstMilestoneError(f"prepared gold partition is unavailable: {candidate_path.name}")

    output = Path(output_report)
    config = EvaluationConfig(seed=seed, minimum_positive_support=minimum_positive_support)
    report = evaluate_predictions(
        load_supervised_candidates(candidate_path),
        load_relation_predictions(predictions),
        config=config,
        metadata={
            "evaluation_scope": "gold_entities_relation_classification",
            "partition": partition,
            "preparation_manifest_sha256": sha256_file(prepared / "preparation-manifest.json"),
            "prepared_dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        },
    )
    write_evaluation_report(report, output)

    # The deterministic companion makes the evaluation inputs, metrics and
    # provenance comparable across runs while the main report keeps elapsed
    # time as an observed measurement.
    evaluation_manifest_path = output.with_name(f"{output.stem}.manifest.json")
    report_value = report.to_dict()
    evaluation_manifest = {
        "config": report_value["config"],
        "data_hashes": report_value["data_hashes"],
        "majority_baseline": report_value["majority_baseline"],
        "methods": report_value["methods"],
        "metrics": report_value["metrics"],
        "model_versions": report_value["model_versions"],
        "partition": partition,
        "prediction_file_sha256": sha256_file(predictions),
        "report_timing_excluded_from_deterministic_manifest": True,
        "schema_version": 1,
        "versions": report_value["versions"],
    }
    _write_json(evaluation_manifest_path, evaluation_manifest)
    return output


__all__ = [
    "FirstMilestoneError",
    "PreparationResult",
    "evaluate_first_milestone",
    "prepare_first_milestone",
]
