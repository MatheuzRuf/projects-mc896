"""Strict end-to-end relation evaluation with error attribution.

A relation is identified by document, source key, target key and label, where
an entity key is ``(start, end, canonical_type)`` and the patient is the
logical ``PATIENT`` key.  Gold endpoints are mapped to their ``flat-gold-v1``
mention through ``source_mention_ids``; an endpoint rejected by adjudication
keeps its lossless span and is attributed to the projection.  Every gold
relation of the supervised schema stays in the recall denominator, including
those that no candidate window can reach.

Missed gold relations are attributed, in order, to: the projection, the
candidate window, an inherited entity error (boundary/type or missing), an
absent candidate, or the relation classifier.  Incorrect positive predictions
are attributed to an endpoint the projection rejected, an inherited entity
error, or the relation classifier.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from project2_from_project1.src.clinical_documents import ClinicalDocument
from project2_from_project1.src.gold_projection import FlatGoldDocument, segment_sentences, tokenize_with_offsets
from project2_from_project1.src.relation_evaluation import RelationPrediction
from project2_from_project1.src.supervised_contracts import RELATIONS

PATIENT = ("Patient",)
LABELS = tuple(sorted(RELATIONS))
MAXIMUM_SENTENCE_DISTANCE = 1

EntityKey = tuple  # (start, end, type) or PATIENT
RelationKey = tuple  # (document_id, source key, target key, label)

FALSE_NEGATIVE_REASONS = (
    "endpoint_rejected_by_projection", "outside_candidate_window", "endpoint_boundary_or_type_error",
    "endpoint_not_recognized", "candidate_not_generated", "classified_no_relation",
)
FALSE_POSITIVE_REASONS = ("endpoint_rejected_by_projection", "inherited_entity_error", "relation_false_positive")


class EndToEndEvaluationError(ValueError):
    """Predictions cannot be resolved to evaluable relation keys."""


@dataclass(frozen=True)
class GoldRelations:
    """Gold relation keys and entity keys for the evaluated documents."""

    relations: Mapping[RelationKey, Mapping[str, Any]]
    entities: frozenset[tuple[str, EntityKey]]
    mention_keys: Mapping[tuple[str, str], EntityKey]
    patient_ids: frozenset[tuple[str, str]]
    rejected_entities: frozenset[tuple[str, EntityKey]] = frozenset()


def build_gold(documents: Iterable[ClinicalDocument], projection: Iterable[FlatGoldDocument]) -> GoldRelations:
    flat = {document.document_id: document for document in projection}
    relations: dict[RelationKey, dict[str, Any]] = {}
    entities = set()
    mention_keys: dict[tuple[str, str], EntityKey] = {}
    patient_ids = set()
    rejected = set()
    for document in documents:
        projected = flat.get(document.document_id)
        if projected is None:
            raise EndToEndEvaluationError(f"document {document.document_id!r} is absent from the flat gold")
        flat_by_source = {}
        for mention in projected.mentions:
            key = (mention.span.start, mention.span.end, mention.canonical_type)
            entities.add((document.document_id, key))
            for source_id in mention.source_mention_ids:
                flat_by_source[source_id] = (key, mention.sentence_index)
        sentences = segment_sentences(document.text, tokenize_with_offsets(document.text))
        endpoint: dict[str, tuple[EntityKey, int | None, bool]] = {}
        for patient in document.patients:
            endpoint[patient.patient_id] = (PATIENT, None, True)
            patient_ids.add((document.document_id, patient.patient_id))
        for mention in document.mentions:
            if mention.mention_id in flat_by_source:
                key, sentence = flat_by_source[mention.mention_id]
                endpoint[mention.mention_id] = (key, sentence, True)
            else:
                start, end = mention.spans[0].start, mention.spans[-1].end
                sentence = next((item.sentence_index for item in sentences if item.span.start <= start < item.span.end), None)
                endpoint[mention.mention_id] = ((start, end, mention.canonical_type), sentence, False)
                rejected.add((document.document_id, (start, end, mention.canonical_type)))
            mention_keys[(document.document_id, mention.mention_id)] = endpoint[mention.mention_id][0]
        for relation in document.relations:
            label = relation.canonical_label.value
            if label not in RELATIONS:
                continue
            source, target = endpoint[relation.source_id], endpoint[relation.target_id]
            distance = None
            if source[0] != PATIENT and target[0] != PATIENT:
                distance = None if None in (source[1], target[1]) else abs(source[1] - target[1])
            key = (document.document_id, source[0], target[0], label)
            row = relations.setdefault(key, {"distance": distance, "in_flat_gold": True, "relation_ids": []})
            row["in_flat_gold"] = row["in_flat_gold"] and source[2] and target[2]
            row["relation_ids"].append(relation.relation_id)
    return GoldRelations(relations, frozenset(entities), mention_keys, frozenset(patient_ids),
                         frozenset(rejected - entities))


def _prf(true_positive: int, predicted: int, support: int) -> dict[str, Any]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / support if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "predicted": predicted, "recall": recall, "support": support,
            "true_positive": true_positive}


def _overlaps(key: EntityKey, others: Iterable[EntityKey]) -> bool:
    return any(other != PATIENT and other[0] < key[1] and key[0] < other[1] for other in others)


def evaluate_end_to_end(
    gold: GoldRelations,
    document_ids: Sequence[str],
    predictions: Iterable[RelationPrediction],
    resolve: Mapping[tuple[str, str], EntityKey],
    *,
    recognized_entities: Iterable[tuple[str, EntityKey]],
    candidates: Iterable[RelationKey],
    minimum_positive_support: int = 5,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score positive relation predictions of ``document_ids`` strictly.

    ``resolve`` maps ``(document_id, endpoint_id)`` to an entity key.
    ``recognized_entities`` and ``candidates`` describe what the arm produced
    before relation classification and are used only for error attribution.
    """
    documents = set(document_ids)
    gold_keys = {key: row for key, row in gold.relations.items() if key[0] in documents}
    gold_entities = {item for item in gold.entities if item[0] in documents}
    recognized = set(recognized_entities)
    recognized_by_document: dict[str, set[EntityKey]] = {}
    for document_id, key in recognized:
        recognized_by_document.setdefault(document_id, set()).add(key)
    candidate_keys = set(candidates)

    predicted: set[RelationKey] = set()
    scored = 0
    for prediction in predictions:
        if prediction.document_id not in documents:
            raise EndToEndEvaluationError(f"prediction for unevaluated document {prediction.document_id!r}")
        scored += 1
        label = prediction.canonical_label.value
        if label not in RELATIONS:
            continue
        try:
            source = resolve[(prediction.document_id, prediction.source_id)]
            target = resolve[(prediction.document_id, prediction.target_id)]
        except KeyError as error:
            raise EndToEndEvaluationError(f"unresolvable relation endpoint {error.args[0]!r}") from error
        predicted.add((prediction.document_id, source, target, label))

    true_positive = predicted & set(gold_keys)
    per_label = {
        label: _prf(sum(key[3] == label for key in true_positive), sum(key[3] == label for key in predicted),
                    sum(key[3] == label for key in gold_keys))
        for label in LABELS
    }
    for row in per_label.values():
        row["conclusion"] = "conclusive" if row["support"] >= minimum_positive_support else "inconclusive_insufficient_support"

    def endpoint_reason(document_id: str, key: EntityKey) -> str | None:
        if key == PATIENT or key in recognized_by_document.get(document_id, ()):
            return None
        if _overlaps(key, recognized_by_document.get(document_id, ())):
            return "endpoint_boundary_or_type_error"
        return "endpoint_not_recognized"

    false_negative = Counter()
    false_negative_by_label = {label: Counter() for label in LABELS}
    for key, row in gold_keys.items():
        if key in true_positive:
            continue
        document_id, source, target, label = key
        if not row["in_flat_gold"]:
            reason = "endpoint_rejected_by_projection"
        elif row["distance"] is not None and row["distance"] > MAXIMUM_SENTENCE_DISTANCE:
            reason = "outside_candidate_window"
        else:
            reasons = [endpoint_reason(document_id, item) for item in (source, target)]
            if "endpoint_not_recognized" in reasons:
                reason = "endpoint_not_recognized"
            elif "endpoint_boundary_or_type_error" in reasons:
                reason = "endpoint_boundary_or_type_error"
            elif key not in candidate_keys:
                reason = "candidate_not_generated"
            else:
                reason = "classified_no_relation"
        false_negative[reason] += 1
        false_negative_by_label[label][reason] += 1

    false_positive = Counter()
    false_positive_by_label = {label: Counter() for label in LABELS}
    for key in predicted - true_positive:
        document_id, source, target, label = key
        outside = [item for item in (source, target) if item != PATIENT and (document_id, item) not in gold_entities]
        if any((document_id, item) in gold.rejected_entities for item in outside):
            reason = "endpoint_rejected_by_projection"
        elif outside:
            reason = "inherited_entity_error"
        else:
            reason = "relation_false_positive"
        false_positive[reason] += 1
        false_positive_by_label[label][reason] += 1

    conclusive = [label for label in LABELS if per_label[label]["conclusion"] == "conclusive"]
    return {
        "counts": {"documents": len(documents), "gold_relations": len(gold_keys), "predicted_positive_relations": len(predicted),
                   "relation_predictions": scored, "candidates": len(candidate_keys),
                   "recognized_entities": len(recognized)},
        "errors": {
            "false_negative": {reason: false_negative[reason] for reason in FALSE_NEGATIVE_REASONS},
            "false_negative_by_label": {label: {reason: counts[reason] for reason in FALSE_NEGATIVE_REASONS}
                                        for label, counts in false_negative_by_label.items()},
            "false_positive": {reason: false_positive[reason] for reason in FALSE_POSITIVE_REASONS},
            "false_positive_by_label": {label: {reason: counts[reason] for reason in FALSE_POSITIVE_REASONS}
                                        for label, counts in false_positive_by_label.items()},
        },
        "inconclusive_labels": [label for label in LABELS if label not in conclusive],
        "macro_f1": sum(per_label[label]["f1"] for label in LABELS) / len(LABELS),
        "macro_f1_conclusive_labels": (sum(per_label[label]["f1"] for label in conclusive) / len(conclusive)
                                       if conclusive else None),
        "metadata": dict(metadata or {}),
        "micro": _prf(len(true_positive), len(predicted), len(gold_keys)),
        "minimum_positive_support": minimum_positive_support,
        "per_label": per_label,
        "primary_metric": "strict_end_to_end_macro_f1",
        "schema_version": 1,
    }


__all__ = [
    "EndToEndEvaluationError", "FALSE_NEGATIVE_REASONS", "FALSE_POSITIVE_REASONS", "GoldRelations", "LABELS",
    "PATIENT", "build_gold", "evaluate_end_to_end",
]
