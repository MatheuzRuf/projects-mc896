"""Sparse categorical segment features fitted on training sentences only.

Every feature is a string; a segment's representation is the multiset of
indices of its active features (one-hot equivalent, never materialized
densely).  Most families depend on one boundary token, and ``in.w`` sums the
one-hot vectors of the tokens inside the segment, so a repeated token counts
once per occurrence.  This decomposition lets segment scores be assembled from
per-token scores and prefix sums.  Lexical
families (token identity and affixes) require a training frequency of at least
``MIN_LEXICAL_FREQUENCY``; unknown features are dropped at encoding time, so an
OOV token keeps only its shape, case, context and length signals.  No feature
looks at the dictionary arm.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from project2_from_project1.src.entity_recognition.corpus import normalize_token
from project2_from_project1.src.gold_projection import OffsetSentence

FEATURE_TEMPLATE_VERSION = "semicrf-sparse-v1"
MIN_LEXICAL_FREQUENCY = 2
AFFIX_LENGTH = 3
BOS, EOS = "<BOS>", "<EOS>"
_LENGTH_BUCKETS = ((1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5"), (10, "6-10"), (20, "11-20"), (40, "21-40"))


def word_shape(text: str) -> str:
    """Map upper/lower/digit to X/x/d, keep punctuation, collapse repeats."""
    symbols = []
    for character in text:
        symbol = "X" if character.isupper() else "x" if character.islower() else "d" if character.isdigit() else character
        if not symbols or symbols[-1] != symbol:
            symbols.append(symbol)
    return "".join(symbols)


def case_class(text: str) -> str:
    if text.isdigit():
        return "digit"
    if not any(character.isalnum() for character in text):
        return "punct"
    if text.islower():
        return "lower"
    if text.isupper():
        return "upper"
    if text[:1].isupper() and text[1:].islower():
        return "title"
    return "mixed"


def length_bucket(size: int) -> str:
    for limit, name in _LENGTH_BUCKETS:
        if size <= limit:
            return name
    return f">{_LENGTH_BUCKETS[-1][0]}"


@dataclass(frozen=True)
class _Token:
    raw: str
    norm: str
    shape: str
    case: str
    is_digit: bool
    is_punct: bool


def _tokens(text: str, sentence: OffsetSentence) -> list[_Token]:
    result = []
    for token in sentence.tokens:
        raw = text[token.span.start:token.span.end]
        case = case_class(raw)
        result.append(_Token(raw, normalize_token(raw), word_shape(raw), case, case == "digit", case == "punct"))
    return result


class FeatureVocabulary:
    """Feature-string index fitted on training sentences only."""

    def __init__(self, features: Sequence[str], lexical_tokens: Iterable[str], prefixes: Iterable[str], suffixes: Iterable[str]):
        self.features = tuple(features)
        self.index = {name: position for position, name in enumerate(self.features)}
        if len(self.index) != len(self.features):
            raise ValueError("feature names must be unique")
        self.lexical_tokens = frozenset(lexical_tokens)
        self.prefixes = frozenset(prefixes)
        self.suffixes = frozenset(suffixes)

    def __len__(self) -> int:
        return len(self.features)

    # -- feature strings -------------------------------------------------
    def _lexical(self, role: str, token: _Token) -> list[str]:
        names = []
        if token.norm in self.lexical_tokens:
            names.append(f"{role}.w={token.norm}")
        if role in ("first", "last"):
            prefix, suffix = token.norm[:AFFIX_LENGTH], token.norm[-AFFIX_LENGTH:]
            if role == "first" and prefix in self.prefixes:
                names.append(f"first.p{AFFIX_LENGTH}={prefix}")
            if role == "last" and suffix in self.suffixes:
                names.append(f"last.s{AFFIX_LENGTH}={suffix}")
        return names

    def _role(self, role: str, token: _Token | None, boundary: str) -> list[str]:
        if token is None:
            return [f"{role}.w={boundary}"]
        names = self._lexical(role, token)
        if role != "in":
            names += [f"{role}.shape={token.shape}", f"{role}.case={token.case}"]
        return names

    def _ids(self, names: Iterable[str]) -> list[int]:
        return [self.index[name] for name in names if name in self.index]

    # -- encoding --------------------------------------------------------
    def token_features(self, text: str, sentence: OffsetSentence) -> dict[str, Any]:
        """Per-token feature ids for each role, plus digit and punctuation flags.

        A segment ``[i, j)`` activates ``first[i]``, ``last[j-1]``, ``prev[i]``
        (token ``i-1`` or BOS), ``next[j-1]`` (token ``j`` or EOS), ``in[t]`` for
        each ``i <= t < j``, its length bucket, ``bias``, and ``has_digit`` or
        ``all_punct`` when applicable.
        """
        tokens = _tokens(text, sentence)
        size = len(tokens)
        return {
            "first": [self._ids(self._role("first", token, BOS)) for token in tokens],
            "last": [self._ids(self._role("last", token, EOS)) for token in tokens],
            "in": [self._ids(self._role("in", token, BOS)) for token in tokens],
            "prev": [self._ids(self._role("prev", tokens[i - 1] if i else None, BOS)) for i in range(size)],
            "next": [self._ids(self._role("next", tokens[i + 1] if i + 1 < size else None, EOS)) for i in range(size)],
            "digit": [token.is_digit for token in tokens],
            "punct": [token.is_punct for token in tokens],
        }

    def segment_constant_ids(self, max_segment: int) -> dict[str, Any]:
        lookup = lambda name: self.index.get(name, -1)  # noqa: E731
        return {
            "all_punct": lookup("all_punct"),
            "bias": lookup("bias"),
            "has_digit": lookup("has_digit"),
            "length": [lookup(f"len={length_bucket(size)}") for size in range(1, max_segment + 1)],
        }

    @classmethod
    def fit(cls, texts_and_sentences: Iterable[tuple[str, OffsetSentence]], max_segment: int) -> "FeatureVocabulary":
        pairs = list(texts_and_sentences)
        counts: Counter[str] = Counter()
        prefixes: Counter[str] = Counter()
        suffixes: Counter[str] = Counter()
        for text, sentence in pairs:
            for token in _tokens(text, sentence):
                counts[token.norm] += 1
                prefixes[token.norm[:AFFIX_LENGTH]] += 1
                suffixes[token.norm[-AFFIX_LENGTH:]] += 1
        lexical = {name for name, count in counts.items() if count >= MIN_LEXICAL_FREQUENCY}
        kept_prefixes = {name for name, count in prefixes.items() if count >= MIN_LEXICAL_FREQUENCY}
        kept_suffixes = {name for name, count in suffixes.items() if count >= MIN_LEXICAL_FREQUENCY}
        probe = cls((), lexical, kept_prefixes, kept_suffixes)
        names = {"bias", "has_digit", "all_punct", f"prev.w={BOS}", f"next.w={EOS}"}
        names.update(f"len={length_bucket(size)}" for size in range(1, max_segment + 1))
        for text, sentence in pairs:
            for token in _tokens(text, sentence):
                for role in ("first", "last", "in", "prev", "next"):
                    names.update(probe._role(role, token, BOS))
        return cls(sorted(names), lexical, kept_prefixes, kept_suffixes)

    def summary(self) -> dict[str, object]:
        families: Counter[str] = Counter(name.split("=", 1)[0] for name in self.features)
        return {
            "affix_length": AFFIX_LENGTH,
            "families": dict(sorted(families.items())),
            "features": len(self.features),
            "lexical_tokens": len(self.lexical_tokens),
            "minimum_lexical_frequency": MIN_LEXICAL_FREQUENCY,
            "template_version": FEATURE_TEMPLATE_VERSION,
        }


ROLES = ("first", "last", "in", "prev", "next")


@dataclass
class EncodedBatch:
    """Sentences of similar length padded to ``max_tokens`` with per-token bags.

    ``bags[role]`` is an ``(ids, offsets)`` pair over the ``B * T`` padded token
    slots in row-major order; padding slots are empty bags.
    """

    lengths: torch.Tensor
    max_segment: int
    bags: dict[str, tuple[torch.Tensor, torch.Tensor]]
    digit: torch.Tensor
    punct: torch.Tensor
    keys: list[tuple[str, int]]

    @property
    def max_tokens(self) -> int:
        return int(self.lengths.max())


def encode_batches(
    vocabulary: FeatureVocabulary,
    sentences: Sequence[tuple[str, str, OffsetSentence]],
    max_segment: int,
    cell_budget: int = 250_000,
) -> list[EncodedBatch]:
    """Group ``(document_id, text, sentence)`` by length into padded batches.

    ``cell_budget`` bounds ``B * T * D`` of the dense segment-score tensor.
    """
    order = sorted(
        (item for item in sentences if item[2].tokens),
        key=lambda item: (len(item[2].tokens), item[0], item[2].sentence_index),
    )
    batches: list[EncodedBatch] = []
    group: list[tuple[str, str, OffsetSentence]] = []

    def flush() -> None:
        if not group:
            return
        width = max(len(item[2].tokens) for item in group)
        ids = {role: [] for role in ROLES}
        offsets = {role: [] for role in ROLES}
        digit = torch.zeros((len(group), width), dtype=torch.bool)
        punct = torch.zeros((len(group), width), dtype=torch.bool)
        for b, (_, text, sentence) in enumerate(group):
            features = vocabulary.token_features(text, sentence)
            size = len(sentence.tokens)
            digit[b, :size] = torch.tensor(features["digit"], dtype=torch.bool)
            punct[b, :size] = torch.tensor(features["punct"], dtype=torch.bool)
            for role in ROLES:
                for position in range(width):
                    offsets[role].append(len(ids[role]))
                    if position < size:
                        ids[role].extend(features[role][position])
        batches.append(EncodedBatch(
            torch.tensor([len(item[2].tokens) for item in group]),
            min(max_segment, width),
            {role: (torch.tensor(ids[role], dtype=torch.long), torch.tensor(offsets[role], dtype=torch.long))
             for role in ROLES},
            digit, punct,
            [(item[0], item[2].sentence_index) for item in group],
        ))
        group.clear()

    for item in order:
        length = len(item[2].tokens)
        cells = (len(group) + 1) * length * min(max_segment, length)
        if group and cells > cell_budget:
            flush()
        group.append(item)
    flush()
    return batches


__all__ = [
    "BOS", "EOS", "EncodedBatch", "FEATURE_TEMPLATE_VERSION", "FeatureVocabulary", "MIN_LEXICAL_FREQUENCY",
    "case_class", "encode_batches", "length_bucket", "word_shape",
]
