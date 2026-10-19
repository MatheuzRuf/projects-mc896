"""Strict entity evaluation shared by every recognizer.

The primary criterion is exact character offsets *and* canonical type.  Overlap,
OOV and length results are diagnostics: they explain the strict result but do
not replace it.  Reports contain counts only, never clinical text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from project2_from_project1.src.clinical_documents import Span
from project2_from_project1.src.entity_recognition.corpus import (
    EntityRecognitionError,
    GoldEntityDocument,
    normalize_token,
)
from project2_from_project1.src.gold_projection import tokenize_with_offsets
from project2_from_project1.src.supervised_contracts import ENTITY_TYPES, EntityPrediction

TYPES = tuple(sorted(ENTITY_TYPES))
LENGTH_BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10, "6-10"), (11, 40, "11-40"))
PRIMARY_METRIC = "strict_macro_f1"


@dataclass(frozen=True)
class _Item:
    document_id: str
    span: Span
    canonical_type: str
    surface: tuple[str, ...]
    marginal: float | None = None


def _prf(true_positive: int, predicted: int, support: int) -> dict[str, Any]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1, "false_negative": support - true_positive, "false_positive": predicted - true_positive,
        "precision": precision, "predicted": predicted, "recall": recall, "support": support,
        "true_positive": true_positive,
    }


def _overlap(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


def _length_bucket(item: _Item) -> str:
    size = len(item.surface)
    for low, high, name in LENGTH_BUCKETS:
        if low <= size <= high:
            return name
    return f">{LENGTH_BUCKETS[-1][1]}"


def _surface(text: str, span: Span) -> tuple[str, ...]:
    fragment = text[span.start:span.end]
    return tuple(normalize_token(fragment[token.span.start:token.span.end]) for token in tokenize_with_offsets(fragment))


def _stratified(
    gold: Sequence[_Item], predicted: Sequence[_Item], exact: set[tuple[str, Span, str]],
    key: Callable[[_Item], str], strata: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result = {}
    for stratum in strata:
        stratum_gold = [item for item in gold if key(item) == stratum]
        stratum_predicted = [item for item in predicted if key(item) == stratum]
        hits = sum((item.document_id, item.span, item.canonical_type) in exact for item in stratum_gold)
        result[stratum] = _prf(hits, len(stratum_predicted), len(stratum_gold))
    return result


def evaluate_entities(
    gold_documents: Iterable[GoldEntityDocument],
    predictions: Iterable[EntityPrediction],
    *,
    training_surfaces: Iterable[tuple[str, ...]],
    training_tokens: Iterable[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare frozen predictions with the flat gold of the same documents.

    ``training_surfaces`` and ``training_tokens`` must come from the training
    partition only; they define the OOV strata without consulting the
    evaluated partition.
    """
    documents = {document.document_id: document for document in gold_documents}
    seen_surfaces = frozenset(training_surfaces)
    seen_tokens = frozenset(training_tokens)
    gold = [
        _Item(document.document_id, mention.span, mention.canonical_type, document.mention_key(mention))
        for document in documents.values() for mention in document.mentions
    ]
    predicted = []
    identities = set()
    for prediction in predictions:
        document = documents.get(prediction.document_id)
        if document is None:
            raise EntityRecognitionError(f"prediction references unevaluated document {prediction.document_id!r}")
        if prediction.span.end > len(document.text):
            raise EntityRecognitionError(f"prediction {prediction.prediction_id} exceeds its document text")
        identity = (prediction.document_id, prediction.span, prediction.canonical_type)
        if identity in identities:
            raise EntityRecognitionError(f"duplicate prediction {prediction.prediction_id}")
        identities.add(identity)
        predicted.append(_Item(*identity, _surface(document.text, prediction.span), prediction.marginal_probability))

    gold_identities = {(item.document_id, item.span, item.canonical_type) for item in gold}
    exact = gold_identities & identities
    per_type = {
        name: _prf(
            sum(identity[2] == name for identity in exact),
            sum(item.canonical_type == name for item in predicted),
            sum(item.canonical_type == name for item in gold),
        )
        for name in TYPES
    }
    micro = _prf(len(exact), len(predicted), len(gold))

    gold_by_document: dict[str, list[_Item]] = {}
    predicted_by_document: dict[str, list[_Item]] = {}
    for item in gold:
        gold_by_document.setdefault(item.document_id, []).append(item)
    for item in predicted:
        predicted_by_document.setdefault(item.document_id, []).append(item)

    false_positive = {"boundary_and_type_mismatch": 0, "boundary_mismatch": 0, "spurious": 0, "type_mismatch": 0}
    false_positive_by_type = {name: dict.fromkeys(false_positive, 0) for name in TYPES}
    false_negative = {"missing": 0, "overlapped_by_incorrect_prediction": 0}
    false_negative_by_type = {name: dict.fromkeys(false_negative, 0) for name in TYPES}
    lenient_predicted = lenient_gold = 0
    for item in predicted:
        peers = [peer for peer in gold_by_document.get(item.document_id, ()) if _overlap(peer.span, item.span)]
        if any(peer.canonical_type == item.canonical_type for peer in peers):
            lenient_predicted += 1
        if (item.document_id, item.span, item.canonical_type) in exact:
            continue
        if any(peer.span == item.span for peer in peers):
            reason = "type_mismatch"
        elif any(peer.canonical_type == item.canonical_type for peer in peers):
            reason = "boundary_mismatch"
        elif peers:
            reason = "boundary_and_type_mismatch"
        else:
            reason = "spurious"
        false_positive[reason] += 1
        false_positive_by_type[item.canonical_type][reason] += 1
    for item in gold:
        peers = [peer for peer in predicted_by_document.get(item.document_id, ()) if _overlap(peer.span, item.span)]
        if any(peer.canonical_type == item.canonical_type for peer in peers):
            lenient_gold += 1
        if (item.document_id, item.span, item.canonical_type) in exact:
            continue
        reason = "overlapped_by_incorrect_prediction" if peers else "missing"
        false_negative[reason] += 1
        false_negative_by_type[item.canonical_type][reason] += 1

    lenient_precision = lenient_predicted / len(predicted) if predicted else 0.0
    lenient_recall = lenient_gold / len(gold) if gold else 0.0
    lenient_f1 = (
        2 * lenient_precision * lenient_recall / (lenient_precision + lenient_recall)
        if lenient_precision + lenient_recall else 0.0
    )

    def surface_stratum(item: _Item) -> str:
        return "seen_surface" if item.surface in seen_surfaces else "unseen_surface"

    def token_stratum(item: _Item) -> str:
        return "all_tokens_seen" if all(token in seen_tokens for token in item.surface) else "contains_unseen_token"

    length_names = [name for _, _, name in LENGTH_BUCKETS]
    observed_lengths = {_length_bucket(item) for item in (*gold, *predicted)}
    length_names += sorted(observed_lengths - set(length_names))
    return {
        "counts": {
            "documents": len(documents),
            "gold_mentions": len(gold),
            "predicted_mentions": len(predicted),
        },
        "diagnostics": {
            "length_tokens": _stratified(gold, predicted, exact, _length_bucket, length_names),
            "lenient_overlap_same_type": {
                "f1": lenient_f1, "matched_gold": lenient_gold, "matched_predicted": lenient_predicted,
                "precision": lenient_precision, "recall": lenient_recall,
            },
            "marginal_probability": _marginal_diagnostic(predicted, exact),
            "oov_surface": _stratified(gold, predicted, exact, surface_stratum, ("seen_surface", "unseen_surface")),
            "oov_token": _stratified(
                gold, predicted, exact, token_stratum, ("all_tokens_seen", "contains_unseen_token"),
            ),
        },
        "errors": {
            "false_negative": false_negative,
            "false_negative_by_type": false_negative_by_type,
            "false_positive": false_positive,
            "false_positive_by_type": false_positive_by_type,
        },
        "metadata": dict(metadata or {}),
        "primary_metric": PRIMARY_METRIC,
        "schema_version": 1,
        "strict": {
            "macro_f1": sum(per_type[name]["f1"] for name in TYPES) / len(TYPES),
            "micro": micro,
            "per_type": per_type,
        },
    }


MARGINAL_BINS = ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0))


def _marginal_diagnostic(predicted: Sequence[_Item], exact: set[tuple[str, Span, str]]) -> dict[str, Any] | None:
    """Exact-match rate by model posterior; descriptive only, never a calibration claim."""
    scored = [item for item in predicted if item.marginal is not None]
    if not scored:
        return None
    correct = [item.marginal for item in scored if (item.document_id, item.span, item.canonical_type) in exact]
    incorrect = [item.marginal for item in scored if (item.document_id, item.span, item.canonical_type) not in exact]
    bins = {}
    for low, high in MARGINAL_BINS:
        members = [item for item in scored if low <= item.marginal < high or (high == 1.0 and item.marginal == 1.0)]
        hits = sum((item.document_id, item.span, item.canonical_type) in exact for item in members)
        bins[f"[{low}, {high}{']' if high == 1.0 else ')'}"] = {
            "exact_match_rate": hits / len(members) if members else None, "predictions": len(members),
        }
    return {
        "bins": bins,
        "interpretation": "uncalibrated_model_posterior",
        "mean_correct": sum(correct) / len(correct) if correct else None,
        "mean_incorrect": sum(incorrect) / len(incorrect) if incorrect else None,
        "scored_predictions": len(scored),
    }


def training_vocabulary(documents: Iterable[GoldEntityDocument]) -> tuple[frozenset[tuple[str, ...]], frozenset[str]]:
    """Mention surfaces and token forms observed in a training partition."""
    surfaces = set()
    tokens = set()
    for document in documents:
        surfaces.update(document.mention_key(mention) for mention in document.mentions)
        for sentence in document.sentences:
            tokens.update(normalize_token(document.text[token.span.start:token.span.end]) for token in sentence.tokens)
    return frozenset(surfaces), frozenset(tokens)


__all__ = ["LENGTH_BUCKETS", "PRIMARY_METRIC", "evaluate_entities", "training_vocabulary"]
