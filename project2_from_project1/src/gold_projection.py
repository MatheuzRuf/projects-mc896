"""Deterministic, text-free flat gold projection for entity recognition.

The source :class:`ClinicalDocument` JSONL remains the lossless authority.  A
projection stores offsets and provenance only, so consumers recover token text
from the source document without copying clinical prose into derived files.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from project2_from_project1.src.clinical_documents import ClinicalDocument, Span, load_jsonl
from project2_from_project1.src.supervised_contracts import ENTITY_TYPES


SCHEMA_VERSION = 1
PROJECTION_VERSION = "flat-gold-v1"
MAX_MENTION_TOKENS = 40
TOKENIZER_VERSION = 2
SENTENCE_SEGMENTER_VERSION = 4
_TOKEN_PATTERN = re.compile(r"[^\W\d_]+|\d+|_|[^\w\s]", re.UNICODE)
_ABBREVIATIONS = frozenset(
    {
        "al", "approx", "ca", "dr", "etc", "fig", "jr", "mr", "mrs", "ms",
        "no", "prof", "sr", "st", "vs",
    }
)
_UNITS = frozenset(
    {
        "cm", "g", "hr", "kg", "l", "mcg", "mg", "min", "ml", "mm", "sec", "ug",
        "\u00b5g", "\u03bcg",
    }
)
_INITIAL_LABEL_CUES = frozenset(
    {"arm", "category", "class", "grade", "group", "item", "level", "option", "part", "phase", "section", "stage", "type"}
)


class AdjudicationError(ValueError):
    """The projection cannot be produced from the supplied review record."""


def _stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _non_empty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdjudicationError(f"{field} must be a non-empty string")
    return value


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdjudicationError(f"{field} must be an object")
    return value


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing:
        raise AdjudicationError(f"{field} is missing field {missing[0]}")
    if extra:
        raise AdjudicationError(f"{field} has unexpected field {extra[0]}")


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AdjudicationError(f"{field} must be an array of strings")
    items = tuple(value)
    if any(not isinstance(item, str) or not item for item in items):
        raise AdjudicationError(f"{field} must contain only non-empty strings")
    return items


@dataclass(frozen=True)
class OffsetToken:
    span: Span

    def to_dict(self) -> list[int]:
        return self.span.to_dict()


@dataclass(frozen=True)
class OffsetSentence:
    sentence_index: int
    span: Span
    tokens: tuple[OffsetToken, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_index": self.sentence_index,
            "span": self.span.to_dict(),
            "tokens": [token.to_dict() for token in self.tokens],
        }


def tokenize_with_offsets(text: str) -> tuple[OffsetToken, ...]:
    """Split Unicode words/numbers and punctuation without modifying *text*."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(OffsetToken(Span(match.start(), match.end())) for match in _TOKEN_PATTERN.finditer(text))


def _is_protected_period(
    text: str,
    tokens: Sequence[OffsetToken],
    index: int,
) -> bool:
    previous = tokens[index - 1] if index else None
    following = tokens[index + 1] if index + 1 < len(tokens) else None
    if previous is None:
        return False
    previous_text = text[previous.span.start:previous.span.end]
    following_text = text[following.span.start:following.span.end] if following else ""
    lower = previous_text.casefold()
    if lower in _ABBREVIATIONS:
        return following is not None
    if lower in _UNITS:
        return bool(
            following_text
            and (following_text == "/" or following_text[0].islower() or following_text[0].isdigit())
        )
    if len(previous_text) == 1 and previous_text.isalpha():
        if following_text and following_text[0].islower():
            return True
        if not following_text:
            return False
        following_is_initial = bool(
            len(following_text) == 1
            and following_text.isalpha()
            and index + 2 < len(tokens)
            and text[tokens[index + 2].span.start:tokens[index + 2].span.end] == "."
        )
        following_is_name = bool(
            len(following_text) > 1
            and following_text[0].isupper()
            and following_text[1:].islower()
        )
        preceding_text = (
            text[tokens[index - 2].span.start:tokens[index - 2].span.end]
            if index >= 2
            else ""
        )
        return following_is_initial or (
            previous_text.isupper()
            and following_is_name
            and preceding_text.casefold() not in _INITIAL_LABEL_CUES
        )
    if previous_text.isdigit() and following_text.isdigit():
        return previous.span.end == tokens[index].span.start and tokens[index].span.end == following.span.start
    return False


def segment_sentences(
    text: str, tokens: Sequence[OffsetToken] | None = None,
) -> tuple[OffsetSentence, ...]:
    """Segment tokenized text while protecting observed abbreviations and units."""
    items = tuple(tokens if tokens is not None else tokenize_with_offsets(text))
    if not items:
        return ()
    boundaries: list[int] = []
    closing = {'"', "'", "’", ")", "]", "}"}
    index = 0
    terminals = {"!", "?", "."}
    while index < len(items):
        token = items[index]
        value = text[token.span.start:token.span.end]
        if value in {"!", "?"} or (
            value == "." and not _is_protected_period(text, items, index)
        ):
            end = index
            while end + 1 < len(items):
                next_value = text[items[end + 1].span.start:items[end + 1].span.end]
                if next_value not in terminals | closing:
                    break
                end += 1
            boundaries.append(end)
            index = end + 1
            continue
        index += 1
    if not boundaries or boundaries[-1] != len(items) - 1:
        boundaries.append(len(items) - 1)
    sentences = []
    start = 0
    for sentence_index, end in enumerate(dict.fromkeys(boundaries)):
        if end < start:
            continue
        sentence_tokens = items[start:end + 1]
        sentences.append(
            OffsetSentence(
                sentence_index,
                Span(sentence_tokens[0].span.start, sentence_tokens[-1].span.end),
                tuple(sentence_tokens),
            )
        )
        start = end + 1
    return tuple(sentences)


@dataclass(frozen=True)
class ProjectionCandidate:
    candidate_id: str
    document_id: str
    span: Span
    canonical_type: str
    source_mention_ids: tuple[str, ...]
    original_spans: tuple[Span, ...]
    transformations: tuple[str, ...]
    issue_reasons: tuple[str, ...]
    sentence_index: int | None
    token_start: int | None
    token_end: int | None

    @property
    def selectable(self) -> bool:
        return not set(self.issue_reasons) & {
            "cross_sentence", "discontinuous", "over_max_tokens", "token_misalignment",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "canonical_type": self.canonical_type,
            "document_id": self.document_id,
            "issue_reasons": list(self.issue_reasons),
            "original_spans": [span.to_dict() for span in self.original_spans],
            "projection_span": self.span.to_dict(),
            "sentence_index": self.sentence_index,
            "source_mention_ids": list(self.source_mention_ids),
            "token_end": self.token_end,
            "token_start": self.token_start,
            "transformations": list(self.transformations),
        }


@dataclass(frozen=True)
class AdjudicationCase:
    case_id: str
    document_id: str
    candidate_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "case_id": self.case_id,
            "document_id": self.document_id,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ReviewerVote:
    reviewer_id: str
    selected_candidate_ids: tuple[str, ...]
    justification: str
    reviewer_kind: str = "human"

    def __post_init__(self) -> None:
        _non_empty(self.reviewer_id, "reviewer_id")
        _non_empty(self.justification, "vote justification")
        if self.reviewer_kind not in {"human", "model"}:
            raise AdjudicationError("reviewer_kind must be human or model")
        if len(set(self.selected_candidate_ids)) != len(self.selected_candidate_ids):
            raise AdjudicationError("a reviewer vote cannot select a candidate twice")

    def to_dict(self) -> dict[str, Any]:
        return {
            "justification": self.justification,
            "reviewer_id": self.reviewer_id,
            "reviewer_kind": self.reviewer_kind,
            "selected_candidate_ids": list(self.selected_candidate_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewerVote":
        value = _require_mapping(value, "vote")
        _require_exact_fields(
            value,
            {"reviewer_id", "reviewer_kind", "selected_candidate_ids", "justification"},
            "vote",
        )
        return cls(
            value.get("reviewer_id"),
            _string_tuple(value.get("selected_candidate_ids"), "vote selected_candidate_ids"),
            value.get("justification"),
            value.get("reviewer_kind"),
        )


@dataclass(frozen=True)
class AdjudicationBinding:
    source_sha256: str
    plan_sha256: str
    projection_version: str
    tokenizer_version: int
    sentence_segmenter_version: int
    max_mention_tokens: int

    def __post_init__(self) -> None:
        for field, value in (
            ("source_sha256", self.source_sha256),
            ("plan_sha256", self.plan_sha256),
            ("projection_version", self.projection_version),
        ):
            _non_empty(value, field)
        for field, value in (("source_sha256", self.source_sha256), ("plan_sha256", self.plan_sha256)):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise AdjudicationError(f"{field} must be a lowercase SHA-256 digest")
        for field, value in (
            ("tokenizer_version", self.tokenizer_version),
            ("sentence_segmenter_version", self.sentence_segmenter_version),
            ("max_mention_tokens", self.max_mention_tokens),
        ):
            if type(value) is not int or value < 1:
                raise AdjudicationError(f"{field} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_mention_tokens": self.max_mention_tokens,
            "plan_sha256": self.plan_sha256,
            "projection_version": self.projection_version,
            "sentence_segmenter_version": self.sentence_segmenter_version,
            "source_sha256": self.source_sha256,
            "tokenizer_version": self.tokenizer_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdjudicationBinding":
        value = _require_mapping(value, "adjudication binding")
        fields = {
            "source_sha256", "plan_sha256", "projection_version", "tokenizer_version",
            "sentence_segmenter_version", "max_mention_tokens",
        }
        _require_exact_fields(value, fields, "adjudication binding")
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True)
class AdjudicationRecord:
    case_id: str
    votes: tuple[ReviewerVote, ...]
    selected_candidate_ids: tuple[str, ...]
    resolution: str
    justification: str
    resolution_reviewer_ids: tuple[str, ...] = ()
    binding: AdjudicationBinding | None = None

    def __post_init__(self) -> None:
        _non_empty(self.case_id, "case_id")
        _non_empty(self.justification, "adjudication justification")
        if self.resolution not in {"consensus", "third_vote"}:
            raise AdjudicationError("resolution must be consensus or third_vote")
        if len(set(self.selected_candidate_ids)) != len(self.selected_candidate_ids):
            raise AdjudicationError("an adjudication cannot select a candidate twice")
        if len(set(self.resolution_reviewer_ids)) != len(self.resolution_reviewer_ids):
            raise AdjudicationError("resolution reviewer ids must be unique")
        for reviewer_id in self.resolution_reviewer_ids:
            _non_empty(reviewer_id, "resolution reviewer id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "binding": self.binding.to_dict() if self.binding is not None else None,
            "justification": self.justification,
            "resolution": self.resolution,
            "resolution_reviewer_ids": list(self.resolution_reviewer_ids),
            "schema_version": SCHEMA_VERSION,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "votes": [vote.to_dict() for vote in self.votes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdjudicationRecord":
        value = _require_mapping(value, "adjudication record")
        _require_exact_fields(
            value,
            {
                "schema_version", "case_id", "votes", "selected_candidate_ids",
                "resolution", "resolution_reviewer_ids", "justification", "binding",
            },
            "adjudication record",
        )
        if type(value.get("schema_version")) is not int or value.get("schema_version") != SCHEMA_VERSION:
            raise AdjudicationError(f"schema_version must be {SCHEMA_VERSION}")
        votes = value.get("votes")
        if isinstance(votes, (str, bytes)) or not isinstance(votes, Sequence):
            raise AdjudicationError("votes must be an array of vote objects")
        return cls(
            value.get("case_id"),
            tuple(ReviewerVote.from_dict(vote) for vote in votes),
            _string_tuple(value.get("selected_candidate_ids"), "selected_candidate_ids"),
            value.get("resolution"),
            value.get("justification"),
            _string_tuple(value.get("resolution_reviewer_ids"), "resolution_reviewer_ids"),
            AdjudicationBinding.from_dict(value.get("binding")),
        )


@dataclass(frozen=True)
class FlatGoldMention:
    document_id: str
    span: Span
    canonical_type: str
    sentence_index: int
    token_start: int
    token_end: int
    source_mention_ids: tuple[str, ...]
    adjudication_case_id: str | None
    transformations: tuple[str, ...]

    @property
    def mention_id(self) -> str:
        return _stable_id("gold", self.document_id, self.span.to_dict(), self.canonical_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjudication_case_id": self.adjudication_case_id,
            "canonical_type": self.canonical_type,
            "mention_id": self.mention_id,
            "source_mention_ids": list(self.source_mention_ids),
            "span": self.span.to_dict(),
            "sentence_index": self.sentence_index,
            "token_end": self.token_end,
            "token_start": self.token_start,
            "transformations": list(self.transformations),
        }


@dataclass(frozen=True)
class FlatGoldDocument:
    document_id: str
    sentences: tuple[OffsetSentence, ...]
    mentions: tuple[FlatGoldMention, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "mentions": [mention.to_dict() for mention in self.mentions],
            "schema_version": SCHEMA_VERSION,
            "sentences": [sentence.to_dict() for sentence in self.sentences],
        }


def _trim_span(text: str, spans: Sequence[Span]) -> tuple[Span, tuple[str, ...]]:
    envelope = Span(spans[0].start, spans[-1].end)
    start, end = envelope.start, envelope.end
    transformations = []
    while start < end and text[start].isspace():
        start += 1
    if start != envelope.start:
        transformations.append("trim_leading_whitespace")
    while end > start and text[end - 1].isspace():
        end -= 1
    if end != envelope.end:
        transformations.append("trim_trailing_whitespace")
    if start == end:
        return envelope, tuple(transformations)
    return Span(start, end), tuple(transformations)


def _alignment(
    span: Span, sentences: Sequence[OffsetSentence], max_tokens: int,
) -> tuple[int | None, int | None, int | None, tuple[str, ...]]:
    containing = [sentence for sentence in sentences if sentence.span.start <= span.start and span.end <= sentence.span.end]
    if not containing:
        return None, None, None, ("cross_sentence",)
    sentence = containing[0]
    starts = {token.span.start: index for index, token in enumerate(sentence.tokens)}
    ends = {token.span.end: index + 1 for index, token in enumerate(sentence.tokens)}
    if span.start not in starts or span.end not in ends:
        return sentence.sentence_index, None, None, ("token_misalignment",)
    token_start, token_end = starts[span.start], ends[span.end]
    if token_end - token_start > max_tokens:
        return sentence.sentence_index, token_start, token_end, ("over_max_tokens",)
    return sentence.sentence_index, token_start, token_end, ()


def _expand_to_token_boundaries(
    span: Span, tokens: Sequence[OffsetToken],
) -> tuple[Span, tuple[str, ...]]:
    start, end = span.start, span.end
    transformations = []
    for token in tokens:
        if token.span.start < start < token.span.end:
            start = token.span.start
            transformations.append("expand_start_to_token_boundary")
            break
    for token in tokens:
        if token.span.start < end < token.span.end:
            end = token.span.end
            transformations.append("expand_end_to_token_boundary")
            break
    return Span(start, end), tuple(transformations)


def _overlap(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


@dataclass(frozen=True)
class ProjectionPlan:
    documents: tuple[ClinicalDocument, ...]
    sentences: Mapping[str, tuple[OffsetSentence, ...]]
    candidates: tuple[ProjectionCandidate, ...]
    cases: tuple[AdjudicationCase, ...]
    input_mentions: int

    def project(self, records: Iterable[AdjudicationRecord]) -> tuple[FlatGoldDocument, ...]:
        records_by_case: dict[str, AdjudicationRecord] = {}
        for record in records:
            if record.case_id in records_by_case:
                raise AdjudicationError(f"duplicate adjudication for {record.case_id}")
            records_by_case[record.case_id] = record
        expected = {case.case_id for case in self.cases}
        missing = sorted(expected - records_by_case.keys())
        extra = sorted(records_by_case.keys() - expected)
        if missing:
            raise AdjudicationError(f"missing adjudication for {missing[0]}")
        if extra:
            raise AdjudicationError(f"unknown adjudication case {extra[0]}")

        candidates = {candidate.candidate_id: candidate for candidate in self.candidates}
        case_for_candidate = {
            candidate_id: case.case_id for case in self.cases for candidate_id in case.candidate_ids
        }
        selected = {candidate.candidate_id for candidate in self.candidates if candidate.candidate_id not in case_for_candidate}
        for case in self.cases:
            record = records_by_case[case.case_id]
            _validate_record(case, record, candidates)
            selected.update(record.selected_candidate_ids)

        result = []
        for document in self.documents:
            mentions = []
            for candidate in self.candidates:
                if candidate.document_id != document.document_id or candidate.candidate_id not in selected:
                    continue
                if not candidate.selectable or None in (candidate.sentence_index, candidate.token_start, candidate.token_end):
                    raise AdjudicationError(f"selected candidate {candidate.candidate_id} is not token aligned")
                mentions.append(
                    FlatGoldMention(
                        document.document_id,
                        candidate.span,
                        candidate.canonical_type,
                        candidate.sentence_index,
                        candidate.token_start,
                        candidate.token_end,
                        candidate.source_mention_ids,
                        case_for_candidate.get(candidate.candidate_id),
                        candidate.transformations,
                    )
                )
            mentions.sort(key=lambda item: (item.span.start, item.span.end, item.canonical_type))
            for left, right in zip(mentions, mentions[1:]):
                if _overlap(left.span, right.span):
                    raise AdjudicationError(
                        f"adjudicated mentions overlap in document {document.document_id}"
                    )
            result.append(FlatGoldDocument(document.document_id, self.sentences[document.document_id], tuple(mentions)))
        return tuple(result)


def _validate_record(
    case: AdjudicationCase,
    record: AdjudicationRecord,
    candidates: Mapping[str, ProjectionCandidate],
) -> None:
    allowed = set(case.candidate_ids)
    outcome = tuple(sorted(record.selected_candidate_ids))
    if not set(outcome) <= allowed:
        raise AdjudicationError(f"adjudication {case.case_id} selects a candidate outside its case")
    for selected_id in outcome:
        if not candidates[selected_id].selectable:
            raise AdjudicationError(f"selected candidate {selected_id} is not token aligned")
    selected_candidates = [candidates[candidate_id] for candidate_id in outcome]
    for index, left in enumerate(selected_candidates):
        for right in selected_candidates[index + 1:]:
            if _overlap(left.span, right.span):
                raise AdjudicationError(f"adjudication {case.case_id} selects overlapping candidates")
    if len({vote.reviewer_id for vote in record.votes}) != len(record.votes):
        raise AdjudicationError(f"adjudication {case.case_id} requires distinct reviewer provenance")
    for vote in record.votes:
        if not set(vote.selected_candidate_ids) <= allowed:
            raise AdjudicationError(f"reviewer {vote.reviewer_id} selected a candidate outside {case.case_id}")
        vote_candidates = [candidates[candidate_id] for candidate_id in vote.selected_candidate_ids]
        if any(not candidate.selectable for candidate in vote_candidates):
            raise AdjudicationError(f"reviewer {vote.reviewer_id} selected an unaligned candidate")
        for index, left in enumerate(vote_candidates):
            if any(_overlap(left.span, right.span) for right in vote_candidates[index + 1:]):
                raise AdjudicationError(f"reviewer {vote.reviewer_id} selected overlapping candidates")
    if len(record.votes) < 2:
        raise AdjudicationError(f"adjudication {case.case_id} requires two independent reviews")
    first = tuple(sorted(record.votes[0].selected_candidate_ids))
    second = tuple(sorted(record.votes[1].selected_candidate_ids))
    initial_reviewers = {record.votes[0].reviewer_id, record.votes[1].reviewer_id}
    if record.resolution == "consensus":
        if len(record.votes) != 2:
            raise AdjudicationError(f"adjudication {case.case_id} consensus requires exactly two votes")
        if set(record.resolution_reviewer_ids) != initial_reviewers:
            raise AdjudicationError(
                f"adjudication {case.case_id} consensus requires both initial reviewers' acknowledgement"
            )
        if first == second and outcome != first:
            raise AdjudicationError(f"adjudication {case.case_id} consensus does not match its reviews")
    else:
        if len(record.votes) != 3:
            raise AdjudicationError(f"adjudication {case.case_id} third vote requires exactly three votes")
        if first == second:
            raise AdjudicationError(
                f"adjudication {case.case_id} requires the initial reviews to disagree before a third vote"
            )
        if record.resolution_reviewer_ids:
            raise AdjudicationError(
                f"adjudication {case.case_id} third vote cannot have consensus acknowledgements"
            )
        third = tuple(sorted(record.votes[2].selected_candidate_ids))
        if outcome != third:
            raise AdjudicationError(f"adjudication {case.case_id} outcome must match its third vote")


def build_projection_plan(
    documents: Iterable[ClinicalDocument], *, max_mention_tokens: int = MAX_MENTION_TOKENS,
) -> ProjectionPlan:
    """Derive candidates and every case that requires explicit adjudication."""
    if isinstance(max_mention_tokens, bool) or not isinstance(max_mention_tokens, int) or max_mention_tokens < 1:
        raise ValueError("max_mention_tokens must be a positive integer")
    ordered_documents = tuple(sorted(documents, key=lambda item: item.document_id))
    if len({document.document_id for document in ordered_documents}) != len(ordered_documents):
        raise AdjudicationError("document identifiers must be unique")
    sentence_map = {}
    candidates: list[ProjectionCandidate] = []
    input_mentions = 0
    for document in ordered_documents:
        tokens = tokenize_with_offsets(document.text)
        sentences = segment_sentences(document.text, tokens)
        sentence_map[document.document_id] = sentences
        grouped: dict[tuple[tuple[Span, ...], str], list[str]] = {}
        for mention in document.mentions:
            input_mentions += 1
            if mention.canonical_type not in ENTITY_TYPES:
                raise AdjudicationError(
                    f"mention {mention.mention_id} has unsupported canonical type {mention.canonical_type}"
                )
            grouped.setdefault((tuple(mention.spans), mention.canonical_type), []).append(mention.mention_id)
        for (original_spans, canonical_type), source_ids in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            span, trim_transformations = _trim_span(document.text, original_spans)
            span, expansion_transformations = _expand_to_token_boundaries(span, tokens)
            transformations = trim_transformations + expansion_transformations
            sentence_index, token_start, token_end, alignment_issues = _alignment(
                span, sentences, max_mention_tokens
            )
            issues = list(alignment_issues)
            if len(original_spans) != 1:
                issues.append("discontinuous")
            if transformations:
                if trim_transformations:
                    issues.append("boundary_trim")
                if expansion_transformations:
                    issues.append("token_boundary_adjustment")
            candidate_id = _stable_id(
                "candidate", document.document_id,
                [item.to_dict() for item in original_spans], canonical_type,
            )
            candidates.append(
                ProjectionCandidate(
                    candidate_id,
                    document.document_id,
                    span,
                    canonical_type,
                    tuple(sorted(source_ids)),
                    original_spans,
                    transformations,
                    tuple(sorted(set(issues))),
                    sentence_index,
                    token_start,
                    token_end,
                )
            )

    cases = []
    by_document: dict[str, list[ProjectionCandidate]] = {}
    for candidate in candidates:
        by_document.setdefault(candidate.document_id, []).append(candidate)
    for document_id, document_candidates in sorted(by_document.items()):
        adjacency = {candidate.candidate_id: set() for candidate in document_candidates}
        by_id = {candidate.candidate_id: candidate for candidate in document_candidates}
        for index, left in enumerate(document_candidates):
            for right in document_candidates[index + 1:]:
                if _overlap(left.span, right.span):
                    adjacency[left.candidate_id].add(right.candidate_id)
                    adjacency[right.candidate_id].add(left.candidate_id)
        visited = set()
        for candidate in document_candidates:
            if candidate.candidate_id in visited:
                continue
            stack = [candidate.candidate_id]
            component = []
            visited.add(candidate.candidate_id)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            reasons = set()
            for candidate_id in component:
                reasons.update(by_id[candidate_id].issue_reasons)
            if len(component) > 1:
                reasons.add("overlap")
                component_candidates = [by_id[candidate_id] for candidate_id in component]
                for index, left in enumerate(component_candidates):
                    for right in component_candidates[index + 1:]:
                        if left.span == right.span and left.canonical_type != right.canonical_type:
                            reasons.add("type_conflict")
            if not reasons:
                continue
            candidate_ids = tuple(sorted(component))
            case_id = _stable_id("adjudication", document_id, candidate_ids, sorted(reasons))
            cases.append(AdjudicationCase(case_id, document_id, candidate_ids, tuple(sorted(reasons))))
    return ProjectionPlan(
        ordered_documents,
        sentence_map,
        tuple(sorted(candidates, key=lambda item: (item.document_id, item.span, item.canonical_type))),
        tuple(sorted(cases, key=lambda item: (item.document_id, item.case_id))),
        input_mentions,
    )


def _plan_sha256(plan: ProjectionPlan) -> str:
    payload = {
        "candidates": [candidate.to_dict() for candidate in plan.candidates],
        "cases": [case.to_dict() for case in plan.cases],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_adjudication_binding(
    plan: ProjectionPlan,
    source_sha256: str,
    *,
    max_mention_tokens: int = MAX_MENTION_TOKENS,
) -> AdjudicationBinding:
    """Bind decisions to one exact source, plan, and segmentation configuration."""
    return AdjudicationBinding(
        source_sha256,
        _plan_sha256(plan),
        PROJECTION_VERSION,
        TOKENIZER_VERSION,
        SENTENCE_SEGMENTER_VERSION,
        max_mention_tokens,
    )


def write_adjudication_jsonl(records: Iterable[AdjudicationRecord], path: str | Path) -> None:
    items = sorted(records, key=lambda item: item.case_id)
    if len({item.case_id for item in items}) != len(items):
        raise AdjudicationError("adjudication case ids must be unique")
    if any(item.binding is None for item in items):
        raise AdjudicationError("every adjudication record must include its source and plan binding")
    output = Path(path)
    _atomic_write_text(
        output,
        "".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in items),
    )


def write_adjudication_review_packet(
    plan: ProjectionPlan,
    binding: AdjudicationBinding,
    path: str | Path,
    *,
    context_characters: int = 160,
) -> None:
    """Write a local, sensitive packet containing the text needed by human reviewers."""
    if type(context_characters) is not int or context_characters < 0:
        raise ValueError("context_characters must be a non-negative integer")
    if binding.plan_sha256 != _plan_sha256(plan):
        raise AdjudicationError("review packet binding does not match its projection plan")
    documents = {document.document_id: document for document in plan.documents}
    candidates = {candidate.candidate_id: candidate for candidate in plan.candidates}
    lines = []
    for case in plan.cases:
        document = documents[case.document_id]
        case_candidates = [candidates[candidate_id] for candidate_id in case.candidate_ids]
        start = max(0, min(candidate.span.start for candidate in case_candidates) - context_characters)
        end = min(len(document.text), max(candidate.span.end for candidate in case_candidates) + context_characters)
        payload_candidates = []
        for candidate in case_candidates:
            payload = candidate.to_dict()
            payload["projection_surface"] = document.text[candidate.span.start:candidate.span.end]
            payload["original_surfaces"] = [
                document.text[span.start:span.end] for span in candidate.original_spans
            ]
            payload_candidates.append(payload)
        payload = {
            "binding": binding.to_dict(),
            "candidates": payload_candidates,
            "case": case.to_dict(),
            "context": {"span": [start, end], "text": document.text[start:end]},
            "schema_version": SCHEMA_VERSION,
        }
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    _atomic_write_text(Path(path), "\n".join(lines) + ("\n" if lines else ""))


def load_adjudication_jsonl(path: str | Path) -> tuple[AdjudicationRecord, ...]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise AdjudicationError(f"adjudication JSONL line {line_number} is empty")
        try:
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise AdjudicationError("record must be an object")
            records.append(AdjudicationRecord.from_dict(value))
        except (json.JSONDecodeError, TypeError, AdjudicationError) as error:
            raise AdjudicationError(f"adjudication JSONL line {line_number}: {error}") from error
    return tuple(records)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projection_text(documents: Sequence[FlatGoldDocument]) -> str:
    return "".join(
        json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for document in documents
    )


def _audit_text(plan: ProjectionPlan, records: Sequence[AdjudicationRecord]) -> str:
    candidates = {candidate.candidate_id: candidate for candidate in plan.candidates}
    by_case = {record.case_id: record for record in records}
    lines = []
    for case in plan.cases:
        payload = {
            "adjudication": by_case[case.case_id].to_dict(),
            "candidates": [candidates[candidate_id].to_dict() for candidate_id in case.candidate_ids],
            "case": case.to_dict(),
            "schema_version": SCHEMA_VERSION,
        }
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines) + ("\n" if lines else "")


def _stage_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        return Path(handle.name)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path = _stage_text(path, content)
    try:
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def project_gold_from_files(
    source_jsonl: str | Path,
    adjudication_jsonl: str | Path,
    projection_jsonl: str | Path,
    audit_jsonl: str | Path,
    manifest_json: str | Path,
    *,
    max_mention_tokens: int = MAX_MENTION_TOKENS,
) -> dict[str, Any]:
    """Build and validate a deterministic projection without rewriting the source."""
    source = Path(source_jsonl).resolve()
    decisions = Path(adjudication_jsonl).resolve()
    projection = Path(projection_jsonl).resolve()
    audit = Path(audit_jsonl).resolve()
    manifest_path = Path(manifest_json).resolve()
    paths = (source, decisions, projection, audit, manifest_path)
    if len(set(paths)) != len(paths):
        raise AdjudicationError("source and derived artifact paths must all be different")
    source_hash = _sha256(source)
    decisions_hash = _sha256(decisions)
    documents = load_jsonl(source)
    plan = build_projection_plan(documents, max_mention_tokens=max_mention_tokens)
    records = load_adjudication_jsonl(decisions)
    expected_binding = build_adjudication_binding(
        plan,
        source_hash,
        max_mention_tokens=max_mention_tokens,
    )
    for record in records:
        if record.binding != expected_binding:
            raise AdjudicationError(
                f"adjudication {record.case_id} is not bound to this source and projection plan"
            )
    projected = plan.project(records)
    if _sha256(source) != source_hash:
        raise AdjudicationError("lossless source changed while building its projection")
    if _sha256(decisions) != decisions_hash:
        raise AdjudicationError("adjudication file changed while building its projection")
    projection_text = _projection_text(projected)
    audit_text = _audit_text(plan, records)
    output_mentions = sum(len(document.mentions) for document in projected)
    selected_candidate_ids = {
        mention_id
        for record in records
        for mention_id in record.selected_candidate_ids
    }
    case_candidate_ids = {
        candidate_id for case in plan.cases for candidate_id in case.candidate_ids
    }
    rejected_candidate_ids = case_candidate_ids - selected_candidate_ids
    rejected = len(rejected_candidate_ids)
    rejection_reasons: dict[str, int] = {}
    rejection_items = []
    candidates_by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    for case in plan.cases:
        case_candidates = [candidates_by_id[candidate_id] for candidate_id in case.candidate_ids]
        for candidate in case_candidates:
            if candidate.candidate_id not in rejected_candidate_ids:
                continue
            reasons = set(candidate.issue_reasons)
            peers = [peer for peer in case_candidates if peer.candidate_id != candidate.candidate_id]
            if any(_overlap(candidate.span, peer.span) for peer in peers):
                reasons.add("overlap")
            if any(
                candidate.span == peer.span and candidate.canonical_type != peer.canonical_type
                for peer in peers
            ):
                reasons.add("type_conflict")
            for reason in reasons:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            rejection_items.append({
                "candidate_id": candidate.candidate_id,
                "case_id": case.case_id,
                "reasons": sorted(reasons),
            })
    transformation_counts: dict[str, int] = {}
    for document in projected:
        for mention in document.mentions:
            for transformation in mention.transformations:
                transformation_counts[transformation] = transformation_counts.get(transformation, 0) + 1
    transformation_counts["exact_duplicates_collapsed"] = plan.input_mentions - len(plan.candidates)
    result = {
        "adjudication_sha256": decisions_hash,
        "adjudication_plan_sha256": expected_binding.plan_sha256,
        "audit_sha256": _sha256_text(audit_text),
        "configuration": {
            "max_mention_tokens": max_mention_tokens,
            "projection_version": PROJECTION_VERSION,
            "sentence_segmenter_version": SENTENCE_SEGMENTER_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
        },
        "counts": {
            "adjudication_cases": len(plan.cases),
            "documents": len(projected),
            "input_mentions": plan.input_mentions,
            "output_mentions": output_mentions,
            "rejected_candidates": rejected,
            "sentences": sum(len(document.sentences) for document in projected),
            "tokens": sum(len(sentence.tokens) for document in projected for sentence in document.sentences),
        },
        "input_sha256": source_hash,
        "output_sha256": _sha256_text(projection_text),
        "rejections": {
            "by_reason": dict(sorted(rejection_reasons.items())),
            "case_ids": sorted({item["case_id"] for item in rejection_items}),
            "candidates": sorted(rejection_items, key=lambda item: item["candidate_id"]),
        },
        "reviews": {
            "consensus": sum(record.resolution == "consensus" for record in records),
            "third_vote": sum(record.resolution == "third_vote" for record in records),
        },
        "schema_version": SCHEMA_VERSION,
        "transformations": dict(sorted(transformation_counts.items())),
    }
    manifest_text = json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, content in (
            (projection, projection_text),
            (audit, audit_text),
            (manifest_path, manifest_text),
        ):
            staged.append((destination, _stage_text(destination, content)))
        if _sha256(source) != source_hash:
            raise AdjudicationError("lossless source changed while publishing its projection")
        if _sha256(decisions) != decisions_hash:
            raise AdjudicationError("adjudication file changed while publishing its projection")
        # The manifest is the generation's commit marker and is published last.
        for destination, temporary_path in staged:
            temporary_path.replace(destination)
    finally:
        for _, temporary_path in staged:
            if temporary_path.exists():
                temporary_path.unlink()
    return result


__all__ = [
    "AdjudicationBinding", "AdjudicationCase", "AdjudicationError", "AdjudicationRecord", "FlatGoldDocument",
    "FlatGoldMention", "MAX_MENTION_TOKENS", "OffsetSentence", "OffsetToken",
    "PROJECTION_VERSION", "ProjectionCandidate", "ProjectionPlan", "ReviewerVote",
    "build_adjudication_binding", "build_projection_plan", "load_adjudication_jsonl", "project_gold_from_files",
    "segment_sentences", "tokenize_with_offsets", "write_adjudication_jsonl",
    "write_adjudication_review_packet",
]
