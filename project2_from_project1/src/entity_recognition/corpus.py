"""Gold-free inputs, joined gold documents and document folds for recognizers.

Recognizers only ever receive :class:`RawDocument` objects at prediction time:
an identifier and the original text.  Gold mentions travel in
:class:`GoldEntityDocument`, which is consumed by fitting and evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import unicodedata

from project2_from_project1.src.clinical_documents import ClinicalDocument
from project2_from_project1.src.gold_projection import (
    FlatGoldDocument,
    FlatGoldMention,
    OffsetSentence,
    segment_sentences,
    tokenize_with_offsets,
)


class EntityRecognitionError(ValueError):
    """An entity-recognition run cannot be executed safely."""


def normalize_token(text: str) -> str:
    """Conservative lexical identity: Unicode NFKC followed by ``casefold``."""
    return unicodedata.normalize("NFKC", text).casefold()


def surface_key(text: str, sentence: OffsetSentence, token_start: int, token_end: int) -> tuple[str, ...]:
    """Normalized token tuple of ``sentence.tokens[token_start:token_end]``."""
    return tuple(
        normalize_token(text[token.span.start:token.span.end])
        for token in sentence.tokens[token_start:token_end]
    )


@dataclass(frozen=True)
class RawDocument:
    """What a recognizer may read during inference: no annotations."""

    document_id: str
    text: str

    @property
    def sentences(self) -> tuple[OffsetSentence, ...]:
        return segment_sentences(self.text, tokenize_with_offsets(self.text))


@dataclass(frozen=True)
class GoldEntityDocument:
    """A flat gold projection joined to the text of its lossless source."""

    document_id: str
    text: str
    sentences: tuple[OffsetSentence, ...]
    mentions: tuple[FlatGoldMention, ...]

    def raw(self) -> RawDocument:
        return RawDocument(self.document_id, self.text)

    def mention_key(self, mention: FlatGoldMention) -> tuple[str, ...]:
        return surface_key(self.text, self.sentences[mention.sentence_index], mention.token_start, mention.token_end)


def join_gold_projection(
    sources: Iterable[ClinicalDocument], projection: Iterable[FlatGoldDocument],
) -> dict[str, GoldEntityDocument]:
    """Attach source text to each projected document and verify segmentation."""
    texts: dict[str, str] = {}
    for document in sources:
        texts[document.document_id] = document.text
    joined: dict[str, GoldEntityDocument] = {}
    for document in projection:
        if document.document_id not in texts:
            raise EntityRecognitionError(f"projection references unknown source document {document.document_id!r}")
        text = texts[document.document_id]
        if segment_sentences(text, tokenize_with_offsets(text)) != document.sentences:
            raise EntityRecognitionError(
                f"projection segmentation of {document.document_id!r} differs from the current tokenizer"
            )
        joined[document.document_id] = GoldEntityDocument(document.document_id, text, document.sentences, document.mentions)
    return joined


@dataclass(frozen=True)
class DocumentFold:
    """One cross-validation round expressed only as document identifiers."""

    index: int
    train_document_ids: tuple[str, ...]
    validation_document_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 1:
            raise EntityRecognitionError("fold index must be a positive integer")
        if not self.train_document_ids or not self.validation_document_ids:
            raise EntityRecognitionError(f"fold {self.index} must have train and validation documents")
        if set(self.train_document_ids) & set(self.validation_document_ids):
            raise EntityRecognitionError(f"fold {self.index} shares documents between train and validation")


def load_document_folds(manifest: Mapping[str, object] | str | Path) -> tuple[DocumentFold, ...]:
    """Read document partitions from a frozen supervised-dataset manifest.

    Test identifiers are read only to prove that no fold touches them; they are
    never returned to callers.
    """
    if not isinstance(manifest, Mapping):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    try:
        partitions = manifest["partitions"]
        test_ids = set(partitions["test_document_ids"])
        folds = tuple(
            DocumentFold(item["fold"], tuple(item["train_document_ids"]), tuple(item["validation_document_ids"]))
            for item in partitions["folds"]
        )
    except (KeyError, TypeError) as error:
        raise EntityRecognitionError("partition manifest lacks partitions.folds and test_document_ids") from error
    for fold in folds:
        if test_ids & (set(fold.train_document_ids) | set(fold.validation_document_ids)):
            raise EntityRecognitionError(f"fold {fold.index} contains a held-out test document")
    return folds


def select_documents(
    documents: Mapping[str, GoldEntityDocument], document_ids: Sequence[str], partition: str,
) -> tuple[GoldEntityDocument, ...]:
    missing = sorted(set(document_ids) - documents.keys())
    if missing:
        raise EntityRecognitionError(f"{partition} documents are absent from the gold projection: {missing[0]!r}")
    return tuple(documents[document_id] for document_id in sorted(document_ids))


__all__ = [
    "DocumentFold", "EntityRecognitionError", "GoldEntityDocument", "RawDocument", "join_gold_projection",
    "load_document_folds", "normalize_token", "select_documents", "surface_key",
]
