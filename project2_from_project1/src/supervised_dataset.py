"""Freeze a reproducible, document-grouped relation-extraction dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from project2_from_project1.src.clinical_documents import AnnotatedRelation, CanonicalRelation, ClinicalDocument, Mention, load_jsonl
from project2_from_project1.src.mcrc_suitability import MCRCSuitabilityReport, SuitabilityDecision, load_mcrc_suitability_report


class SupervisedDatasetError(ValueError):
    """Raised when a frozen supervised dataset would be incomplete or unsafe."""


_SUPERVISED_RELATIONS = (
    CanonicalRelation.HAS_CONDITION,
    CanonicalRelation.HAS_FINDING,
    CanonicalRelation.HAS_FACTOR,
    CanonicalRelation.NEGATES,
    CanonicalRelation.CAUSES,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'\]\)]*)\s+|\n+")


@dataclass(frozen=True)
class CandidatePolicy:
    """Construct compatible typed pairs; the sentence window is inclusive."""

    maximum_sentence_distance: int | None = 1

    def __post_init__(self) -> None:
        value = self.maximum_sentence_distance
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise SupervisedDatasetError("maximum_sentence_distance must be a non-negative integer or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible_relation_types": [label.value for label in _SUPERVISED_RELATIONS],
            "maximum_sentence_distance": self.maximum_sentence_distance,
            "sentence_window_inclusive": True,
        }


@dataclass(frozen=True)
class PartitionPolicy:
    """Deterministic document-level holdout and cross-validation settings."""

    seed: int = 0
    test_fraction: float = 0.2
    folds: int = 5
    maximum_train_negatives_per_positive: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SupervisedDatasetError("seed must be an integer")
        if isinstance(self.test_fraction, bool) or not isinstance(self.test_fraction, (float, int)) or not 0 < self.test_fraction < 1:
            raise SupervisedDatasetError("test_fraction must be in (0, 1)")
        if isinstance(self.folds, bool) or not isinstance(self.folds, int) or self.folds < 2:
            raise SupervisedDatasetError("folds must be an integer of at least 2")
        value = self.maximum_train_negatives_per_positive
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SupervisedDatasetError("maximum_train_negatives_per_positive must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds_requested": self.folds,
            "maximum_train_negatives_per_positive": self.maximum_train_negatives_per_positive,
            "seed": self.seed,
            "test_fraction": float(self.test_fraction),
        }


@dataclass(frozen=True, order=True)
class SupervisedCandidate:
    """One typed relation decision, labelled positive or ``NO_RELATION``."""

    document_id: str
    source_id: str
    target_id: str
    candidate_relation: CanonicalRelation
    label: CanonicalRelation
    source_type: str
    target_type: str
    sentence_distance: int | None
    gold_relation_id: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_relation not in _SUPERVISED_RELATIONS:
            raise SupervisedDatasetError("candidate_relation must be a supervised positive relation")
        if self.label not in (*_SUPERVISED_RELATIONS, CanonicalRelation.NO_RELATION):
            raise SupervisedDatasetError("label must be a supervised relation or NO_RELATION")
        if self.label is not CanonicalRelation.NO_RELATION and self.label is not self.candidate_relation:
            raise SupervisedDatasetError("a positive candidate label must equal candidate_relation")
        values = (self.document_id, self.source_id, self.target_id, self.source_type, self.target_type)
        if not all(isinstance(value, str) and value for value in values):
            raise SupervisedDatasetError("candidate identifiers and types must be non-empty strings")
        if self.sentence_distance is not None and (not isinstance(self.sentence_distance, int) or self.sentence_distance < 0):
            raise SupervisedDatasetError("sentence_distance must be a non-negative integer or None")

    @property
    def is_positive(self) -> bool:
        return self.label is not CanonicalRelation.NO_RELATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_relation": self.candidate_relation.value,
            "document_id": self.document_id,
            "gold_relation_id": self.gold_relation_id,
            "label": self.label.value,
            "sentence_distance": self.sentence_distance,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "target_id": self.target_id,
            "target_type": self.target_type,
        }


def _check_disjoint(left: Sequence[str], right: Sequence[str], left_name: str, right_name: str) -> None:
    if set(left) & set(right):
        raise SupervisedDatasetError(f"{left_name} and {right_name} document partitions overlap")


def _check_candidate_membership(candidates: Sequence[SupervisedCandidate], ids: Sequence[str], name: str) -> None:
    unexpected = sorted({item.document_id for item in candidates if item.document_id not in set(ids)})
    if unexpected:
        raise SupervisedDatasetError(f"{name} candidates include documents outside their partition: {', '.join(unexpected)}")


@dataclass(frozen=True)
class CrossValidationFold:
    index: int
    train_document_ids: tuple[str, ...]
    validation_document_ids: tuple[str, ...]
    train_candidates: tuple[SupervisedCandidate, ...]
    validation_candidates: tuple[SupervisedCandidate, ...]

    def __post_init__(self) -> None:
        if self.index < 1:
            raise SupervisedDatasetError("fold index must start at 1")
        _check_disjoint(self.train_document_ids, self.validation_document_ids, "train", "validation")
        _check_candidate_membership(self.train_candidates, self.train_document_ids, "train")
        _check_candidate_membership(self.validation_candidates, self.validation_document_ids, "validation")


def _counts(candidates: Sequence[SupervisedCandidate]) -> dict[str, Any]:
    positive = sum(item.is_positive for item in candidates)
    labels = {label.value: 0 for label in (*_SUPERVISED_RELATIONS, CanonicalRelation.NO_RELATION)}
    for item in candidates:
        labels[item.label.value] += 1
    return {"by_label": labels, "candidates": len(candidates), "negative_candidates": len(candidates) - positive, "positive_candidates": positive}


@dataclass(frozen=True)
class FrozenSupervisedDataset:
    test_document_ids: tuple[str, ...]
    test_candidates: tuple[SupervisedCandidate, ...]
    folds: tuple[CrossValidationFold, ...]
    candidate_policy: CandidatePolicy
    partition_policy: PartitionPolicy
    eligible_document_ids: tuple[str, ...]
    source_jsonl_sha256: str | None = None
    gold_outside_window: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if len(self.folds) < 2:
            raise SupervisedDatasetError("at least two cross-validation folds are required")
        if len(set(self.eligible_document_ids)) != len(self.eligible_document_ids):
            raise SupervisedDatasetError("eligible_document_ids must be unique")
        _check_candidate_membership(self.test_candidates, self.test_document_ids, "test")
        eligible = set(self.eligible_document_ids)
        if set(self.test_document_ids) - eligible:
            raise SupervisedDatasetError("test partition contains a document that is not eligible")
        validation_union: set[str] = set()
        for fold in self.folds:
            if set(fold.train_document_ids) & set(self.test_document_ids) or set(fold.validation_document_ids) & set(self.test_document_ids):
                raise SupervisedDatasetError("test documents must not occur in a training or validation fold")
            if validation_union & set(fold.validation_document_ids):
                raise SupervisedDatasetError("a document may validate in exactly one cross-validation fold")
            validation_union.update(fold.validation_document_ids)
        if eligible != set(self.test_document_ids) | validation_union:
            raise SupervisedDatasetError("partitions must cover every eligible document exactly once outside training reuse")

    def manifest(self, artifact_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
        return {
            "artifact_hashes": dict(sorted((artifact_hashes or {}).items())),
            "candidate_policy": self.candidate_policy.to_dict(),
            "counts": {
                "folds": {f"fold-{item.index:02d}": {"train": _counts(item.train_candidates), "validation": _counts(item.validation_candidates)} for item in self.folds},
                "test": _counts(self.test_candidates),
            },
            "eligible_document_ids": list(self.eligible_document_ids),
            "gold_outside_window": dict(self.gold_outside_window),
            "negative_policy": {
                "applies_to": "train",
                "maximum_negatives_per_positive": self.partition_policy.maximum_train_negatives_per_positive,
                "selection": "seeded_uniform_sample_without_replacement",
                "validation_and_test": "all_eligible_candidates",
            },
            "partitions": {
                "folds": [{"fold": item.index, "train_document_ids": list(item.train_document_ids), "validation_document_ids": list(item.validation_document_ids)} for item in self.folds],
                "test_document_ids": list(self.test_document_ids),
            },
            "partition_policy": {**self.partition_policy.to_dict(), "folds_effective": len(self.folds), "grouping_unit": "document_id"},
            "schema_version": 1,
            "source_jsonl_sha256": self.source_jsonl_sha256,
        }


def _sentence_indices(text: str, mentions: Sequence[Mention]) -> dict[str, int]:
    starts = [0, *(match.end() for match in _SENTENCE_BOUNDARY.finditer(text))]
    return {mention.mention_id: sum(boundary <= mention.spans[0].start for boundary in starts) - 1 for mention in mentions}


def _compatible_pairs(document: ClinicalDocument, policy: CandidatePolicy) -> list[tuple[str, str, CanonicalRelation, str, str, int | None]]:
    sentences = _sentence_indices(document.text, document.mentions)
    pairs: list[tuple[str, str, CanonicalRelation, str, str, int | None]] = []
    patient_labels = {"Condition": CanonicalRelation.HAS_CONDITION, "Finding": CanonicalRelation.HAS_FINDING, "Factor": CanonicalRelation.HAS_FACTOR}
    for patient in document.patients:
        for mention in document.mentions:
            if relation := patient_labels.get(mention.canonical_type):
                pairs.append((patient.patient_id, mention.mention_id, relation, "Patient", mention.canonical_type, None))
    for source in document.mentions:
        for target in document.mentions:
            relation = None
            if source.canonical_type == "Modifier" and target.canonical_type == "Finding":
                relation = CanonicalRelation.NEGATES
            elif source.canonical_type == "Condition" and target.canonical_type == "Finding":
                relation = CanonicalRelation.CAUSES
            if relation is None:
                continue
            distance = abs(sentences[source.mention_id] - sentences[target.mention_id])
            if policy.maximum_sentence_distance is None or distance <= policy.maximum_sentence_distance:
                pairs.append((source.mention_id, target.mention_id, relation, source.canonical_type, target.canonical_type, distance))
    return sorted(set(pairs))


def gold_outside_window(document: ClinicalDocument, policy: CandidatePolicy | None = None) -> tuple[AnnotatedRelation, ...]:
    """Return supervised gold relations that are type-compatible but lie beyond the sentence window."""

    policy = policy or CandidatePolicy()
    if policy.maximum_sentence_distance is None:
        return ()
    windowed = {(source, target, label) for source, target, label, *_ in _compatible_pairs(document, policy)}
    unrestricted = {(source, target, label) for source, target, label, *_ in _compatible_pairs(document, CandidatePolicy(None))}
    beyond = unrestricted - windowed
    return tuple(
        relation
        for relation in document.relations
        if relation.canonical_label in _SUPERVISED_RELATIONS and (relation.source_id, relation.target_id, relation.canonical_label) in beyond
    )


def generate_candidates(document: ClinicalDocument, policy: CandidatePolicy | None = None) -> tuple[SupervisedCandidate, ...]:
    """Generate all compatible candidates in the window; gold beyond it is reported by gold_outside_window, never hidden."""

    if not isinstance(document, ClinicalDocument):
        raise TypeError("generate_candidates accepts a ClinicalDocument")
    policy = policy or CandidatePolicy()
    if not isinstance(policy, CandidatePolicy):
        raise TypeError("policy must be a CandidatePolicy")
    compatible = _compatible_pairs(document, policy)
    compatible_keys = {(source, target, label) for source, target, label, *_ in compatible}
    outside_window = {relation.relation_id for relation in gold_outside_window(document, policy)}
    gold: dict[tuple[str, str, CanonicalRelation], str] = {}
    for relation in document.relations:
        if relation.canonical_label not in _SUPERVISED_RELATIONS or relation.relation_id in outside_window:
            continue
        key = (relation.source_id, relation.target_id, relation.canonical_label)
        if key not in compatible_keys:
            raise SupervisedDatasetError(f"gold relation {relation.relation_id!r} in {document.document_id!r} is outside the candidate compatibility or sentence window")
        if key in gold:
            raise SupervisedDatasetError(f"duplicate supervised gold relation for candidate {key!r} in {document.document_id!r}")
        gold[key] = relation.relation_id
    result = []
    for source, target, candidate_label, source_type, target_type, distance in compatible:
        relation_id = gold.get((source, target, candidate_label))
        result.append(SupervisedCandidate(document.document_id, source, target, candidate_label, candidate_label if relation_id else CanonicalRelation.NO_RELATION, source_type, target_type, distance, relation_id))
    return tuple(sorted(result))


def _partition_ids(ids: Sequence[str], policy: PartitionPolicy) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    ids = tuple(sorted(ids))
    if len(ids) < 3:
        raise SupervisedDatasetError("at least three eligible documents are required for a fixed test set and cross-validation")
    test_count = min(max(1, int(len(ids) * float(policy.test_fraction) + 0.5)), len(ids) - 2)
    shuffled = list(ids)
    random.Random(policy.seed).shuffle(shuffled)
    test_ids = tuple(sorted(shuffled[:test_count]))
    buckets = [[] for _ in range(min(policy.folds, len(ids) - test_count))]
    for index, document_id in enumerate(shuffled[test_count:]):
        buckets[index % len(buckets)].append(document_id)
    return test_ids, tuple(tuple(sorted(bucket)) for bucket in buckets)


def _sample_train_negatives(candidates: Sequence[SupervisedCandidate], policy: PartitionPolicy, fold: int) -> tuple[SupervisedCandidate, ...]:
    positive = [item for item in candidates if item.is_positive]
    negative = [item for item in candidates if not item.is_positive]
    limit = policy.maximum_train_negatives_per_positive * len(positive)
    if len(negative) > limit:
        negative = random.Random(f"{policy.seed}:fold:{fold}:negatives").sample(negative, limit)
    return tuple(sorted((*positive, *negative)))


def build_frozen_supervised_dataset(
    documents: Iterable[ClinicalDocument], suitability_report: MCRCSuitabilityReport, *, candidate_policy: CandidatePolicy | None = None,
    partition_policy: PartitionPolicy | None = None, source_jsonl_sha256: str | None = None,
) -> FrozenSupervisedDataset:
    """Build the approved subset only, split by document and seeded deterministically."""

    if not isinstance(suitability_report, MCRCSuitabilityReport):
        raise TypeError("suitability_report must be an MCRCSuitabilityReport")
    if suitability_report.decision is not SuitabilityDecision.APPROVED:
        raise SupervisedDatasetError("cannot freeze a supervised dataset from a rejected suitability report")
    candidate_policy = candidate_policy or CandidatePolicy()
    partition_policy = partition_policy or PartitionPolicy()
    if not isinstance(candidate_policy, CandidatePolicy) or not isinstance(partition_policy, PartitionPolicy):
        raise TypeError("candidate_policy and partition_policy must have their documented types")
    by_id: dict[str, ClinicalDocument] = {}
    for document in documents:
        if not isinstance(document, ClinicalDocument):
            raise TypeError("documents must contain ClinicalDocument instances")
        if document.document_id in by_id:
            raise SupervisedDatasetError(f"duplicate document id {document.document_id!r}")
        by_id[document.document_id] = document
    eligible = tuple(sorted(suitability_report.documents_for_partitioning))
    missing = sorted(set(eligible) - set(by_id))
    if missing:
        raise SupervisedDatasetError("approved suitability report references documents absent from corpus: " + ", ".join(missing))
    candidates = {document_id: generate_candidates(by_id[document_id], candidate_policy) for document_id in eligible}
    outside = {label.value: 0 for label in _SUPERVISED_RELATIONS}
    for document_id in eligible:
        for relation in gold_outside_window(by_id[document_id], candidate_policy):
            outside[relation.canonical_label.value] += 1
    test_ids, validation_buckets = _partition_ids(eligible, partition_policy)
    test = tuple(sorted(item for document_id in test_ids for item in candidates[document_id]))
    remaining = set(eligible) - set(test_ids)
    folds = []
    for index, validation_ids in enumerate(validation_buckets, start=1):
        train_ids = tuple(sorted(remaining - set(validation_ids)))
        train = tuple(item for document_id in train_ids for item in candidates[document_id])
        validation = tuple(sorted(item for document_id in validation_ids for item in candidates[document_id]))
        folds.append(CrossValidationFold(index, train_ids, validation_ids, _sample_train_negatives(train, partition_policy, index), validation))
    return FrozenSupervisedDataset(test_ids, test, tuple(folds), candidate_policy, partition_policy, eligible, source_jsonl_sha256, tuple(sorted(outside.items())))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_candidates(candidates: Sequence[SupervisedCandidate], path: Path) -> str:
    contents = "\n".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for item in candidates)
    path.write_text(contents + ("\n" if contents else ""), encoding="utf-8")
    return sha256_file(path)


def write_frozen_supervised_dataset(dataset: FrozenSupervisedDataset, output_directory: str | Path) -> dict[str, Any]:
    """Write deterministic JSONL partitions plus a text-free manifest."""

    if not isinstance(dataset, FrozenSupervisedDataset):
        raise TypeError("dataset must be a FrozenSupervisedDataset")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {"test.jsonl": _write_candidates(dataset.test_candidates, output / "test.jsonl")}
    for fold in dataset.folds:
        prefix = f"fold-{fold.index:02d}"
        hashes[f"{prefix}-train.jsonl"] = _write_candidates(fold.train_candidates, output / f"{prefix}-train.jsonl")
        hashes[f"{prefix}-validation.jsonl"] = _write_candidates(fold.validation_candidates, output / f"{prefix}-validation.jsonl")
    manifest = dataset.manifest(hashes)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return manifest


def build_frozen_supervised_dataset_from_files(
    converted_jsonl_path: str | Path, suitability_report_path: str | Path, *, candidate_policy: CandidatePolicy | None = None,
    partition_policy: PartitionPolicy | None = None,
) -> FrozenSupervisedDataset:
    return build_frozen_supervised_dataset(
        load_jsonl(converted_jsonl_path), load_mcrc_suitability_report(suitability_report_path), candidate_policy=candidate_policy,
        partition_policy=partition_policy, source_jsonl_sha256=sha256_file(converted_jsonl_path),
    )


__all__ = [
    "CandidatePolicy", "CrossValidationFold", "FrozenSupervisedDataset", "PartitionPolicy", "SupervisedCandidate", "SupervisedDatasetError",
    "build_frozen_supervised_dataset", "build_frozen_supervised_dataset_from_files", "generate_candidates", "gold_outside_window", "sha256_file",
    "write_frozen_supervised_dataset",
]