"""Common prediction contract and auditable relation-classification evaluation.

The first evaluation boundary deliberately consumes gold entity identifiers.  It
therefore measures relation classification only; entity detection is outside of
this milestone.  A prediction is required for every eligible gold candidate so
that an omitted minority class cannot be mistaken for a true negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

from project2_from_project1.src.clinical_documents import CanonicalRelation, ClinicalDocument
from project2_from_project1.src.supervised_dataset import CandidatePolicy, SupervisedCandidate, generate_candidates


class RelationEvaluationError(ValueError):
    """Raised when predictions cannot be compared safely with the gold data."""


_POSITIVE_RELATIONS = frozenset(
    {
        CanonicalRelation.HAS_CONDITION,
        CanonicalRelation.HAS_FINDING,
        CanonicalRelation.HAS_FACTOR,
        CanonicalRelation.NEGATES,
        CanonicalRelation.CAUSES,
    }
)


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelationEvaluationError(f"{field_name} must be a non-empty string")
    return value


def _json_value(value: object, field_name: str) -> Any:
    try:
        # Round-tripping also detaches mutable input evidence from the record.
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise RelationEvaluationError(f"{field_name} must be JSON-serializable") from error


@dataclass(frozen=True)
class RelationPrediction:
    """One extractor decision for a gold-entity pair.

    ``neighbor_vote_fraction`` is provenance about a kNN vote, not calibrated
    confidence.  ``confidence`` is deliberately accepted only as ``None`` to
    make an unsupported calibration claim impossible in this milestone.
    """

    document_id: str
    source_id: str
    target_id: str
    native_label: str
    canonical_label: CanonicalRelation | str
    method: str
    evidence: object = field(default_factory=dict)
    sentences: Sequence[int] = ()
    model_version: str = ""
    neighbor_vote_fraction: float | None = None
    confidence: None = None

    def __post_init__(self) -> None:
        for name in ("document_id", "source_id", "target_id", "native_label", "method", "model_version"):
            object.__setattr__(self, name, _non_empty_string(getattr(self, name), name))
        if self.source_id == self.target_id:
            raise RelationEvaluationError("a prediction cannot point to itself")
        try:
            label = CanonicalRelation(self.canonical_label)
        except (TypeError, ValueError) as error:
            allowed = ", ".join(item.value for item in CanonicalRelation)
            raise RelationEvaluationError(f"unsupported canonical_label {self.canonical_label!r}; expected one of {allowed}") from error
        if label not in (*_POSITIVE_RELATIONS, CanonicalRelation.NO_RELATION):
            raise RelationEvaluationError("predictions may only use supervised labels or NO_RELATION")
        object.__setattr__(self, "canonical_label", label)
        object.__setattr__(self, "evidence", _json_value(self.evidence, "evidence"))
        if isinstance(self.sentences, str) or not isinstance(self.sentences, Sequence):
            raise RelationEvaluationError("sentences must be a sequence of non-negative sentence indices")
        sentences = tuple(self.sentences)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in sentences):
            raise RelationEvaluationError("sentences must contain non-negative integer indices")
        if len(set(sentences)) != len(sentences):
            raise RelationEvaluationError("sentences must not contain duplicates")
        object.__setattr__(self, "sentences", sentences)
        if self.neighbor_vote_fraction is not None:
            if isinstance(self.neighbor_vote_fraction, bool) or not isinstance(self.neighbor_vote_fraction, (int, float)):
                raise RelationEvaluationError("neighbor_vote_fraction must be a number in [0, 1]")
            vote = float(self.neighbor_vote_fraction)
            if not 0.0 <= vote <= 1.0:
                raise RelationEvaluationError("neighbor_vote_fraction must be a number in [0, 1]")
            object.__setattr__(self, "neighbor_vote_fraction", vote)
        if self.confidence is not None:
            raise RelationEvaluationError("confidence must be absent or null until calibration is established")

    @property
    def pair_key(self) -> tuple[str, str, str]:
        return (self.document_id, self.source_id, self.target_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_label": self.canonical_label.value,
            "confidence": None,
            "document_id": self.document_id,
            "evidence": self.evidence,
            "method": self.method,
            "model_version": self.model_version,
            "native_label": self.native_label,
            "neighbor_vote_fraction": self.neighbor_vote_fraction,
            "sentences": list(self.sentences),
            "source_id": self.source_id,
            "target_id": self.target_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RelationPrediction":
        if not isinstance(value, Mapping):
            raise RelationEvaluationError("each prediction must be a JSON object")
        return cls(
            document_id=value.get("document_id"),
            source_id=value.get("source_id"),
            target_id=value.get("target_id"),
            native_label=value.get("native_label"),
            canonical_label=value.get("canonical_label"),
            method=value.get("method"),
            evidence=value.get("evidence", {}),
            sentences=value.get("sentences", ()),
            model_version=value.get("model_version"),
            neighbor_vote_fraction=value.get("neighbor_vote_fraction"),
            confidence=value.get("confidence"),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration that affects the interpretation of an evaluation."""

    seed: int = 0
    minimum_positive_support: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RelationEvaluationError("seed must be an integer")
        if isinstance(self.minimum_positive_support, bool) or not isinstance(self.minimum_positive_support, int) or self.minimum_positive_support < 1:
            raise RelationEvaluationError("minimum_positive_support must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {"minimum_positive_support": self.minimum_positive_support, "seed": self.seed}


def _candidate_key(candidate: SupervisedCandidate) -> tuple[str, str, str]:
    return (candidate.document_id, candidate.source_id, candidate.target_id)


def _serialize_records(records: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for record in records]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _sha256_records(records: Iterable[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_serialize_records(records)).hexdigest()


def _labels(candidates: Sequence[SupervisedCandidate], predictions: Sequence[RelationPrediction]) -> tuple[CanonicalRelation, ...]:
    positives = {item.candidate_relation for item in candidates}
    positives.update(item.label for item in candidates if item.label in _POSITIVE_RELATIONS)
    positives.update(item.canonical_label for item in predictions if item.canonical_label in _POSITIVE_RELATIONS)
    return (CanonicalRelation.NO_RELATION, *tuple(sorted(positives, key=lambda item: item.value)))


def _class_metrics(gold: Sequence[CanonicalRelation], predicted: Sequence[CanonicalRelation], labels: Sequence[CanonicalRelation], minimum_positive_support: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label in labels:
        true_positive = sum(actual is label and observed is label for actual, observed in zip(gold, predicted))
        support = sum(actual is label for actual in gold)
        predicted_count = sum(observed is label for observed in predicted)
        false_positive = predicted_count - true_positive
        false_negative = support - true_positive
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values: dict[str, Any] = {
            "f1": f1,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "precision": precision,
            "predicted": predicted_count,
            "recall": recall,
            "support": support,
            "true_positive": true_positive,
        }
        if label is not CanonicalRelation.NO_RELATION:
            values["conclusion"] = "conclusive" if support >= minimum_positive_support else "inconclusive_insufficient_support"
        result[label.value] = values
    return result


def _metrics(gold: Sequence[CanonicalRelation], predicted: Sequence[CanonicalRelation], labels: Sequence[CanonicalRelation], config: EvaluationConfig) -> dict[str, Any]:
    per_class = _class_metrics(gold, predicted, labels, config.minimum_positive_support)
    positives = [label for label in labels if label is not CanonicalRelation.NO_RELATION]
    macro = sum(per_class[label.value]["f1"] for label in positives) / len(positives) if positives else None
    insufficient = [label.value for label in positives if per_class[label.value]["conclusion"] != "conclusive"]
    matrix = {
        actual.value: {observed.value: sum(gold_item is actual and predicted_item is observed for gold_item, predicted_item in zip(gold, predicted)) for observed in labels}
        for actual in labels
    }
    return {
        "accuracy": {"role": "auxiliary", "value": sum(actual is observed for actual, observed in zip(gold, predicted)) / len(gold) if gold else None},
        "classification_count": len(gold),
        "confusion_matrix": {"columns": [item.value for item in labels], "rows": [item.value for item in labels], "values": matrix},
        "inconclusive_positive_classes": insufficient,
        "per_class": per_class,
        "positive_macro_f1": macro,
        "primary_metric": "positive_macro_f1",
    }


@dataclass(frozen=True)
class EvaluationReport:
    """JSON-ready result of one common-contract evaluation."""

    config: EvaluationConfig
    metrics: Mapping[str, Any]
    majority_baseline: Mapping[str, Any]
    data_hashes: Mapping[str, str]
    model_versions: tuple[str, ...]
    methods: tuple[str, ...]
    timing_seconds: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "data_hashes": dict(sorted(self.data_hashes.items())),
            "metadata": _json_value(self.metadata, "metadata"),
            "methods": list(self.methods),
            "metrics": self.metrics,
            "model_versions": list(self.model_versions),
            "schema_version": 1,
            "timing_seconds": dict(sorted(self.timing_seconds.items())),
            "versions": {"python": platform.python_version()},
            "majority_baseline": self.majority_baseline,
        }


def evaluate_predictions(
    gold_candidates: Iterable[SupervisedCandidate],
    predictions: Iterable[RelationPrediction],
    *,
    config: EvaluationConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Evaluate complete predictions against frozen candidates using gold entities.

    Candidate and prediction keys are ``(document_id, source_id, target_id)``.
    They must match exactly and be unique.  This deliberate strictness prevents
    sparse output from silently turning omitted decisions into ``NO_RELATION``.
    """

    started = time.perf_counter()
    config = config or EvaluationConfig()
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    candidates = tuple(gold_candidates)
    observed = tuple(predictions)
    if not candidates:
        raise RelationEvaluationError("gold_candidates must not be empty")
    if not all(isinstance(item, SupervisedCandidate) for item in candidates):
        raise TypeError("gold_candidates must contain SupervisedCandidate instances")
    if not all(isinstance(item, RelationPrediction) for item in observed):
        raise TypeError("predictions must contain RelationPrediction instances")
    candidate_by_key = {_candidate_key(item): item for item in candidates}
    if len(candidate_by_key) != len(candidates):
        raise RelationEvaluationError("gold candidates must have unique document/source/target pairs")
    prediction_by_key = {item.pair_key: item for item in observed}
    if len(prediction_by_key) != len(observed):
        raise RelationEvaluationError("predictions must have unique document/source/target pairs")
    missing = sorted(set(candidate_by_key) - set(prediction_by_key))
    unexpected = sorted(set(prediction_by_key) - set(candidate_by_key))
    if missing or unexpected:
        pieces = []
        if missing:
            pieces.append("missing predictions for " + ", ".join(repr(item) for item in missing))
        if unexpected:
            pieces.append("predictions outside gold candidates " + ", ".join(repr(item) for item in unexpected))
        raise RelationEvaluationError("; ".join(pieces))
    for key, candidate in candidate_by_key.items():
        label = prediction_by_key[key].canonical_label
        if label not in (candidate.candidate_relation, CanonicalRelation.NO_RELATION):
            raise RelationEvaluationError(
                f"prediction for {key!r} has {label.value}; expected {candidate.candidate_relation.value} or NO_RELATION"
            )
    ordered_keys = sorted(candidate_by_key)
    gold = [candidate_by_key[key].label for key in ordered_keys]
    predicted = [prediction_by_key[key].canonical_label for key in ordered_keys]
    labels = _labels(candidates, observed)
    metric_values = _metrics(gold, predicted, labels, config)
    majority_label = min(labels, key=lambda label: (-sum(item is label for item in gold), label.value))
    majority_predictions = [majority_label] * len(gold)
    baseline = {"label": majority_label.value, "metrics": _metrics(gold, majority_predictions, labels, config)}
    candidate_records = [candidate_by_key[key].to_dict() for key in ordered_keys]
    prediction_records = [prediction_by_key[key].to_dict() for key in ordered_keys]
    elapsed = time.perf_counter() - started
    return EvaluationReport(
        config=config,
        metrics=metric_values,
        majority_baseline=baseline,
        data_hashes={"gold_candidates_sha256": _sha256_records(candidate_records), "predictions_sha256": _sha256_records(prediction_records)},
        model_versions=tuple(sorted({item.model_version for item in observed})),
        methods=tuple(sorted({item.method for item in observed})),
        timing_seconds={"evaluation": elapsed},
        metadata=dict(metadata or {}),
    )


def evaluate_predictions_against_documents(
    documents: Iterable[ClinicalDocument], predictions: Iterable[RelationPrediction], *, candidate_policy: CandidatePolicy | None = None,
    config: EvaluationConfig | None = None, metadata: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Generate candidates from gold documents, then evaluate relation decisions."""

    documents = tuple(documents)
    if not all(isinstance(item, ClinicalDocument) for item in documents):
        raise TypeError("documents must contain ClinicalDocument instances")
    policy = candidate_policy or CandidatePolicy()
    if not isinstance(policy, CandidatePolicy):
        raise TypeError("candidate_policy must be a CandidatePolicy")
    candidates = tuple(candidate for document in documents for candidate in generate_candidates(document, policy))
    merged_metadata = {"candidate_policy": policy.to_dict(), **dict(metadata or {})}
    return evaluate_predictions(candidates, predictions, config=config, metadata=merged_metadata)


def write_relation_predictions(predictions: Iterable[RelationPrediction], path: str | Path) -> None:
    """Write common-contract predictions in deterministic UTF-8 JSONL."""

    items = tuple(predictions)
    if not all(isinstance(item, RelationPrediction) for item in items):
        raise TypeError("predictions must contain RelationPrediction instances")
    keys = [item.pair_key for item in items]
    if len(set(keys)) != len(keys):
        raise RelationEvaluationError("predictions must have unique document/source/target pairs")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_serialize_records(item.to_dict() for item in sorted(items, key=lambda item: item.pair_key)))


def load_relation_predictions(path: str | Path) -> list[RelationPrediction]:
    """Load and validate common-contract prediction JSONL."""

    items: list[RelationPrediction] = []
    keys: set[tuple[str, str, str]] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise RelationEvaluationError(f"prediction JSONL line {line_number} is empty")
        try:
            item = RelationPrediction.from_dict(json.loads(line))
        except json.JSONDecodeError as error:
            raise RelationEvaluationError(f"invalid prediction JSON on line {line_number}: {error.msg}") from error
        except RelationEvaluationError as error:
            raise RelationEvaluationError(f"invalid prediction on JSONL line {line_number}: {error}") from error
        if item.pair_key in keys:
            raise RelationEvaluationError(f"duplicate prediction pair on JSONL line {line_number}: {item.pair_key!r}")
        keys.add(item.pair_key)
        items.append(item)
    return items


def load_supervised_candidates(path: str | Path) -> list[SupervisedCandidate]:
    """Load a frozen candidate JSONL partition for relation evaluation."""

    items: list[SupervisedCandidate] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise RelationEvaluationError(f"candidate JSONL line {line_number} is empty")
        try:
            value = json.loads(line)
            item = SupervisedCandidate(
                document_id=value["document_id"], source_id=value["source_id"], target_id=value["target_id"],
                candidate_relation=CanonicalRelation(value["candidate_relation"]), label=CanonicalRelation(value["label"]), source_type=value["source_type"],
                target_type=value["target_type"], sentence_distance=value.get("sentence_distance"), gold_relation_id=value.get("gold_relation_id"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RelationEvaluationError(f"invalid candidate on JSONL line {line_number}: {error}") from error
        items.append(item)
    return items


def write_evaluation_report(report: EvaluationReport, path: str | Path) -> None:
    """Write a compact, auditable JSON report without clinical text."""

    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "EvaluationConfig",
    "EvaluationReport",
    "RelationEvaluationError",
    "RelationPrediction",
    "evaluate_predictions",
    "evaluate_predictions_against_documents",
    "load_relation_predictions",
    "load_supervised_candidates",
    "write_evaluation_report",
    "write_relation_predictions",
]
