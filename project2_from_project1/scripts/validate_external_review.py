"""Validate one independent MultiCaRe review against the immutable packet."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from project2_from_project1.src.clinical_documents import load_jsonl
from project2_from_project1.src.gold_projection import segment_sentences, tokenize_with_offsets
from project2_from_project1.src.supervised_contracts import ENTITY_TYPES, RELATIONS


RELATION_TYPES = RELATIONS
_HGVS_DOT = re.compile(r"\b[cpgnrm]\.(?=[A-Za-z0-9])", re.IGNORECASE)


def _aligned_mention(text: str, span: object, tokens: object, sentences: object) -> bool:
    for sentence in sentences:
        inside = [token for token in sentence.tokens
                  if token.span.start >= span.start and token.span.end <= span.end]
        if inside and len(inside) <= 40 and inside[0].span.start == span.start and inside[-1].span.end == span.end:
            return True
    # The shared sentence segmenter treats HGVS dots (c.712C>T, p.H238Y)
    # as sentence ends. Accept only those exact internal boundaries here;
    # leave the versioned projection segmenter unchanged.
    inside = [token for token in tokens if token.span.start >= span.start and token.span.end <= span.end]
    if not inside or len(inside) > 40 or inside[0].span.start != span.start or inside[-1].span.end != span.end:
        return False
    internal_boundaries = [sentence.tokens[-1].span.start for sentence in sentences
                           if sentence.tokens and span.start <= sentence.tokens[-1].span.start < span.end - 1]
    protected = {span.start + match.end() - 1 for match in _HGVS_DOT.finditer(text[span.start:span.end])}
    return bool(internal_boundaries) and set(internal_boundaries) <= protected


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(packet_path: Path, review_path: Path, guide_path: Path, provenance_path: Path) -> dict[str, object]:
    packet = {item.document_id: item for item in load_jsonl(packet_path)}
    reviewed = load_jsonl(review_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (provenance.get("reviewer_kind") != "model" or not provenance.get("reviewer_id")
            or not provenance.get("model") or not provenance.get("reviewed_at_utc")):
        raise ValueError("review provenance must identify a model, reviewer and date")
    for key, path in (("packet_sha256", packet_path), ("guide_sha256", guide_path),
                      ("review_sha256", review_path)):
        if provenance.get(key) != _sha256(path):
            raise ValueError(f"review provenance has wrong {key}")
    if provenance.get("document_ids") != [item.document_id for item in reviewed]:
        raise ValueError("review provenance does not cover the review documents in order")
    if not reviewed:
        raise ValueError("empty review")
    if len({item.document_id for item in reviewed}) != len(reviewed):
        raise ValueError("duplicate reviewed document")
    reviewed_ids = [item.document_id for item in reviewed]
    if reviewed_ids != list(packet)[:len(reviewed_ids)]:
        raise ValueError("review must cover a prefix of the fixed case_id order")
    entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    for document in reviewed:
        original = packet.get(document.document_id)
        if original is None or document.text != original.text:
            raise ValueError(f"source text or document ID changed in {document.document_id}")
        if document.cases != original.cases or document.patients != original.patients or document.metadata != original.metadata:
            raise ValueError(f"source provenance changed in {document.document_id}")
        if not document.mentions:
            raise ValueError(f"clinical review has no mentions: {document.document_id}")
        tokens = tokenize_with_offsets(document.text)
        sentences = segment_sentences(document.text, tokens)
        spans = []
        for mention in document.mentions:
            if mention.canonical_type not in ENTITY_TYPES or len(mention.spans) != 1:
                raise ValueError(f"invalid entity type or discontinuous mention: {mention.mention_id}")
            span = mention.spans[0]
            if not document.text[span.start:span.end].strip():
                raise ValueError(f"empty mention text: {mention.mention_id}")
            if not _aligned_mention(document.text, span, tokens, sentences):
                raise ValueError(f"mention not aligned to <=40 tokens in one sentence: {mention.mention_id}")
            spans.append((span.start, span.end, mention.mention_id))
            entity_counts[mention.canonical_type] += 1
        spans.sort()
        for earlier, later in zip(spans, spans[1:]):
            if earlier[1] > later[0]:
                raise ValueError(f"overlapping mentions in {document.document_id}: {earlier[2]}, {later[2]}")
        for relation in document.relations:
            if relation.canonical_label.value not in RELATION_TYPES:
                raise ValueError(f"unsupported gold relation: {relation.relation_id}")
            relation_counts[relation.canonical_label.value] += 1
    return {"reviewer_id": provenance["reviewer_id"], "documents": len(reviewed),
            "entities": dict(entity_counts), "relations": dict(relation_counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "review", "guide", "provenance"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.packet, args.review, args.guide, args.provenance), sort_keys=True))


if __name__ == "__main__":
    main()
