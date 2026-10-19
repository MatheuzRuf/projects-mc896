"""Canonical dictionary baseline rebuilt from the training partition only.

Fitting reads normalized surfaces of flat gold training mentions.  A surface
seen with several types keeps the strictly most frequent one; a tie leaves the
surface *abstained*: it is kept in the artifact for audit but never emitted and
never blocks other matches.  Inference sees :class:`RawDocument` only, matches
whole tokens inside one sentence and prefers longer matches globally.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from project2_from_project1.src.entity_recognition.corpus import (
    EntityRecognitionError,
    GoldEntityDocument,
    RawDocument,
    normalize_token,
)
from project2_from_project1.src.gold_projection import MAX_MENTION_TOKENS
from project2_from_project1.src.run_manifest import sha256_file, sha256_text
from project2_from_project1.src.clinical_documents import Span
from project2_from_project1.src.supervised_contracts import ENTITY_TYPES, EntityPrediction

METHOD = "canonical-dictionary"
MODEL_VERSION = "canonical-dictionary-v1"
NORMALIZATION = "nfkc-casefold-per-token"


@dataclass(frozen=True)
class DictionaryEntry:
    surface: tuple[str, ...]
    type_counts: Mapping[str, int]

    @property
    def resolved_type(self) -> str | None:
        """The strictly most frequent training type, or ``None`` on a tie."""
        ranked = sorted(self.type_counts.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved_type": self.resolved_type,
            "surface_tokens": list(self.surface),
            "type_counts": dict(sorted(self.type_counts.items())),
        }


@dataclass(frozen=True)
class DictionaryMatchStats:
    emitted: int
    discarded_by_longer_overlap: int
    abstained_surface_occurrences: int

    def to_dict(self) -> dict[str, int]:
        return {
            "abstained_surface_occurrences": self.abstained_surface_occurrences,
            "discarded_by_longer_overlap": self.discarded_by_longer_overlap,
            "emitted": self.emitted,
        }


class CanonicalDictionary:
    """Exact token-sequence dictionary with majority type and tie abstention."""

    def __init__(self, entries: Sequence[DictionaryEntry], training_document_ids: Sequence[str]) -> None:
        self.entries = tuple(sorted(entries, key=lambda entry: entry.surface))
        self.training_document_ids = tuple(sorted(training_document_ids))
        if len({entry.surface for entry in self.entries}) != len(self.entries):
            raise EntityRecognitionError("dictionary surfaces must be unique")
        self._trie: dict[str, Any] = {}
        self.maximum_tokens = 0
        for entry in self.entries:
            if not entry.surface or len(entry.surface) > MAX_MENTION_TOKENS:
                raise EntityRecognitionError(f"dictionary surfaces must have 1 to {MAX_MENTION_TOKENS} tokens")
            if set(entry.type_counts) - ENTITY_TYPES or any(count < 1 for count in entry.type_counts.values()):
                raise EntityRecognitionError("dictionary type counts must use canonical types and positive counts")
            node = self._trie
            for token in entry.surface:
                node = node.setdefault(token, {})
            node[None] = entry.resolved_type
            self.maximum_tokens = max(self.maximum_tokens, len(entry.surface))

    @classmethod
    def fit(cls, training_documents: Iterable[GoldEntityDocument]) -> "CanonicalDictionary":
        counts: dict[tuple[str, ...], Counter[str]] = {}
        document_ids = []
        for document in training_documents:
            document_ids.append(document.document_id)
            for mention in document.mentions:
                counts.setdefault(document.mention_key(mention), Counter())[mention.canonical_type] += 1
        if len(set(document_ids)) != len(document_ids):
            raise EntityRecognitionError("training documents must be unique")
        if not counts:
            raise EntityRecognitionError("training partition contains no gold mentions")
        return cls(
            [DictionaryEntry(surface, dict(type_counts)) for surface, type_counts in counts.items()],
            document_ids,
        )

    def _matches(self, keys: Sequence[str]) -> tuple[list[tuple[int, int, str]], int]:
        matches = []
        abstained = 0
        for start in range(len(keys)):
            node = self._trie
            for end in range(start, min(len(keys), start + self.maximum_tokens)):
                node = node.get(keys[end])
                if node is None:
                    break
                if None in node:
                    if node[None] is None:
                        abstained += 1
                    else:
                        matches.append((start, end + 1, node[None]))
        return matches, abstained

    def predict(self, document: RawDocument) -> tuple[tuple[EntityPrediction, ...], DictionaryMatchStats]:
        """Flat, offset-preserving predictions for a gold-free document."""
        if not isinstance(document, RawDocument):
            raise TypeError("the dictionary predicts RawDocument instances only")
        predictions = []
        discarded = abstained_total = 0
        for sentence in document.sentences:
            keys = [normalize_token(document.text[token.span.start:token.span.end]) for token in sentence.tokens]
            matches, abstained = self._matches(keys)
            abstained_total += abstained
            occupied = [False] * len(keys)
            # Longer matches first; equal lengths are resolved left to right.
            for start, end, canonical_type in sorted(matches, key=lambda item: (item[0] - item[1], item[0])):
                if any(occupied[start:end]):
                    discarded += 1
                    continue
                occupied[start:end] = [True] * (end - start)
                predictions.append(
                    EntityPrediction(
                        document.document_id,
                        Span(sentence.tokens[start].span.start, sentence.tokens[end - 1].span.end),
                        canonical_type,
                        sentence.sentence_index,
                        METHOD,
                        MODEL_VERSION,
                        None,
                    )
                )
        predictions.sort(key=lambda item: (item.span.start, item.span.end))
        return tuple(predictions), DictionaryMatchStats(len(predictions), discarded, abstained_total)

    def to_jsonl(self) -> str:
        """Serialize entries.  The surfaces are clinical text: keep the file private."""
        return "".join(
            json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for entry in self.entries
        )

    def summary(self) -> dict[str, Any]:
        """Text-free description suitable for a manifest."""
        resolved = [entry for entry in self.entries if entry.resolved_type is not None]
        by_type = Counter(entry.resolved_type for entry in resolved)
        return {
            "abstained_tied_surfaces": len(self.entries) - len(resolved),
            "ambiguous_resolved_by_majority": sum(len(entry.type_counts) > 1 for entry in resolved),
            "entries": len(self.entries),
            "maximum_surface_tokens": self.maximum_tokens,
            "normalization": NORMALIZATION,
            "resolved_entries_by_type": {name: by_type.get(name, 0) for name in sorted(ENTITY_TYPES)},
            "training_documents": len(self.training_document_ids),
            "training_document_ids_sha256": sha256_text(json.dumps(self.training_document_ids, ensure_ascii=False)),
        }


def predict_documents(
    dictionary: CanonicalDictionary, documents: Iterable[RawDocument],
) -> tuple[tuple[EntityPrediction, ...], DictionaryMatchStats]:
    predictions: list[EntityPrediction] = []
    totals = [0, 0, 0]
    for document in sorted(documents, key=lambda item: item.document_id):
        items, stats = dictionary.predict(document)
        predictions.extend(items)
        totals[0] += stats.emitted
        totals[1] += stats.discarded_by_longer_overlap
        totals[2] += stats.abstained_surface_occurrences
    return tuple(predictions), DictionaryMatchStats(*totals)


def write_dictionary(dictionary: CanonicalDictionary, path: str | Path) -> str:
    content = dictionary.to_jsonl()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8", newline="\n")
    return sha256_file(output)


__all__ = [
    "CanonicalDictionary", "DictionaryEntry", "DictionaryMatchStats", "METHOD", "MODEL_VERSION",
    "predict_documents", "write_dictionary",
]
