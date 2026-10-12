"""Audit external corpora without retaining protected MultiCaRe text.

The protected collection is deliberately a collection of identifiers and
one-way fingerprints.  It is safe to keep its JSON representation with the
source code, while the complete MultiCaRe documents remain outside Git.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from project2_from_project1.src.clinical_documents import ClinicalDocument


class LeakageAuditError(ValueError):
    """Raised when a protected collection or audit report is malformed."""


class TrainingEligibilityError(ValueError):
    """Raised when excluded or quarantined data is sent to training."""


class AuditDecision(str, Enum):
    """The only states an external document can have after an audit."""

    RELEASED = "released"
    EXCLUDED = "excluded"
    QUARANTINED = "quarantined"


_IDENTIFIER_FIELDS = ("pmcid", "pmid", "doi")
_METADATA_ALIASES = {
    "pmcid": ("pmcid", "pmc_id", "pmc"),
    "pmid": ("pmid", "pubmed_id", "pubmedid"),
    "doi": ("doi", "article_doi"),
    "article_id": ("article_id", "articleid", "source_article_id"),
}
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_PMCID = re.compile(r"(?:^|[^A-Z0-9])(PMC\d+)(?:$|[^A-Z0-9])", re.IGNORECASE)
_PMID = re.compile(r"^\d+$")
_DOI_PREFIX = re.compile(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", re.IGNORECASE)


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LeakageAuditError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalise_text(text: str) -> str:
    """Return a stable, case-insensitive textual form for fingerprinting."""

    if not isinstance(text, str):
        raise LeakageAuditError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(normalise_text(text)))


def _fingerprint(tokens: Sequence[str]) -> str:
    return _sha256(" ".join(tokens))


def _shingles(tokens: Sequence[str], width: int = 5) -> frozenset[str]:
    if not tokens:
        return frozenset()
    if len(tokens) < width:
        return frozenset({_fingerprint(tokens)})
    return frozenset(_fingerprint(tokens[index : index + width]) for index in range(len(tokens) - width + 1))


def _chunks(tokens: Sequence[str], size: int, stride: int) -> tuple[tuple[str, ...], ...]:
    if not tokens:
        return ()
    if len(tokens) <= size:
        return (tuple(tokens),)
    starts = list(range(0, len(tokens) - size + 1, stride))
    if starts[-1] != len(tokens) - size:
        starts.append(len(tokens) - size)
    return tuple(tuple(tokens[start : start + size]) for start in starts)


def _normalise_identifier(kind: str, value: object) -> str | None:
    if not isinstance(value, str) and not isinstance(value, int):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if kind == "pmcid":
        match = _PMCID.search(raw.upper())
        return match.group(1).upper() if match else None
    if kind == "pmid":
        return raw if _PMID.fullmatch(raw) else None
    if kind == "doi":
        result = _DOI_PREFIX.sub("", raw).strip().rstrip("/.,;")
        return result.casefold() or None
    raise LeakageAuditError(f"unsupported identifier kind {kind!r}")


def _metadata_value(metadata: Mapping[str, Any], aliases: Sequence[str]) -> object | None:
    lower_metadata = {str(key).casefold(): value for key, value in metadata.items()}
    for alias in aliases:
        if alias in lower_metadata:
            return lower_metadata[alias]
    return None


def document_identifiers(document: ClinicalDocument) -> dict[str, tuple[str, ...]]:
    """Extract canonical PMCID, PMID and DOI values from document provenance."""

    values: dict[str, set[str]] = {kind: set() for kind in _IDENTIFIER_FIELDS}
    for kind in _IDENTIFIER_FIELDS:
        candidates = [_metadata_value(document.metadata, _METADATA_ALIASES[kind])]
        candidates.extend(case.source_id for case in document.cases)
        candidates.append(document.document_id)
        for candidate in candidates:
            normalized = _normalise_identifier(kind, candidate)
            if normalized:
                values[kind].add(normalized)
    return {kind: tuple(sorted(values[kind])) for kind in _IDENTIFIER_FIELDS}


def _article_id(document: ClinicalDocument) -> str:
    value = _metadata_value(document.metadata, _METADATA_ALIASES["article_id"])
    if isinstance(value, (str, int)) and str(value).strip():
        return str(value).strip()
    identifiers = document_identifiers(document)
    for kind in ("pmcid", "pmid", "doi"):
        if identifiers[kind]:
            return f"{kind}:{identifiers[kind][0]}"
    return document.document_id


def _ratio(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    # The audit must detect a case report copied into a larger article.  A
    # Jaccard denominator would dilute that signal with the larger article's
    # unrelated prose, so measure coverage of the smaller fingerprint.
    return len(left & right) / min(len(left), len(right))


@dataclass(frozen=True)
class ProtectedDocumentFingerprint:
    """One-way document and excerpt fingerprints; never stores source text."""

    document_id: str
    content_sha256: str
    token_count: int
    shingles: frozenset[str]
    chunks: tuple[frozenset[str], ...]

    def __post_init__(self) -> None:
        _required_string(self.document_id, "document_id")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_sha256):
            raise LeakageAuditError("content_sha256 must be a SHA-256 hex digest")
        if not isinstance(self.token_count, int) or self.token_count < 0:
            raise LeakageAuditError("token_count must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": [sorted(chunk) for chunk in self.chunks],
            "content_sha256": self.content_sha256,
            "document_id": self.document_id,
            "shingles": sorted(self.shingles),
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtectedDocumentFingerprint":
        try:
            chunks = tuple(frozenset(chunk) for chunk in value["chunks"])
            return cls(
                document_id=value["document_id"],
                content_sha256=value["content_sha256"],
                token_count=value["token_count"],
                shingles=frozenset(value["shingles"]),
                chunks=chunks,
            )
        except (KeyError, TypeError) as error:
            raise LeakageAuditError("invalid protected document fingerprint") from error


@dataclass(frozen=True)
class ProtectedArticle:
    """Protected article metadata and fingerprints, without complete text."""

    article_id: str
    identifiers: Mapping[str, tuple[str, ...]]
    documents: tuple[ProtectedDocumentFingerprint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "article_id", _required_string(self.article_id, "article_id"))
        normalized = {}
        for kind in _IDENTIFIER_FIELDS:
            raw_values = self.identifiers.get(kind, ())
            if isinstance(raw_values, str):
                raw_values = (raw_values,)
            normalized[kind] = tuple(sorted({value for value in (_normalise_identifier(kind, item) for item in raw_values) if value}))
        object.__setattr__(self, "identifiers", normalized)
        object.__setattr__(self, "documents", tuple(self.documents))
        if not self.documents:
            raise LeakageAuditError("a protected article must contain at least one document fingerprint")
        document_ids = [document.document_id for document in self.documents]
        if len(set(document_ids)) != len(document_ids):
            raise LeakageAuditError(f"protected article {self.article_id!r} has duplicate document ids")

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "documents": [document.to_dict() for document in self.documents],
            "identifiers": {kind: list(self.identifiers[kind]) for kind in _IDENTIFIER_FIELDS},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtectedArticle":
        try:
            return cls(
                article_id=value["article_id"],
                identifiers=value.get("identifiers", {}),
                documents=tuple(ProtectedDocumentFingerprint.from_dict(item) for item in value["documents"]),
            )
        except (KeyError, TypeError, AttributeError) as error:
            raise LeakageAuditError("invalid protected article") from error


@dataclass(frozen=True)
class ProtectedCollection:
    """Serializable, text-free fingerprint collection for the protected corpus."""

    articles: tuple[ProtectedArticle, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "articles", tuple(self.articles))
        if not self.articles:
            raise LeakageAuditError("the protected collection cannot be empty")
        article_ids = [article.article_id for article in self.articles]
        if len(set(article_ids)) != len(article_ids):
            raise LeakageAuditError("protected collection has duplicate article ids")
        if self.schema_version != 1:
            raise LeakageAuditError("unsupported protected collection schema_version")

    @property
    def article_count(self) -> int:
        return len(self.articles)

    def to_dict(self) -> dict[str, Any]:
        return {"articles": [article.to_dict() for article in self.articles], "schema_version": self.schema_version}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtectedCollection":
        try:
            return cls(
                articles=tuple(ProtectedArticle.from_dict(item) for item in value["articles"]),
                schema_version=value.get("schema_version", 1),
            )
        except (KeyError, TypeError) as error:
            raise LeakageAuditError("invalid protected collection") from error


def build_protected_collection(documents: Iterable[ClinicalDocument]) -> ProtectedCollection:
    """Fingerprint protected documents grouped by article, discarding all text."""

    grouped: dict[str, list[ClinicalDocument]] = {}
    for document in documents:
        if not isinstance(document, ClinicalDocument):
            raise TypeError("build_protected_collection accepts ClinicalDocument instances")
        grouped.setdefault(_article_id(document), []).append(document)
    articles = []
    for article_id in sorted(grouped):
        article_documents = grouped[article_id]
        identifiers = {kind: set() for kind in _IDENTIFIER_FIELDS}
        fingerprints = []
        for document in sorted(article_documents, key=lambda item: item.document_id):
            for kind, values in document_identifiers(document).items():
                identifiers[kind].update(values)
            tokens = _tokens(document.text)
            fingerprints.append(
                ProtectedDocumentFingerprint(
                    document_id=document.document_id,
                    content_sha256=_sha256(normalise_text(document.text)),
                    token_count=len(tokens),
                    shingles=_shingles(tokens),
                    chunks=tuple(_shingles(chunk) for chunk in _chunks(tokens, size=120, stride=60)),
                )
            )
        articles.append(
            ProtectedArticle(
                article_id=article_id,
                identifiers={kind: tuple(sorted(values)) for kind, values in identifiers.items()},
                documents=tuple(fingerprints),
            )
        )
    return ProtectedCollection(tuple(articles))


def write_protected_collection(collection: ProtectedCollection, path: str | Path) -> None:
    """Write a deterministic fingerprint manifest that contains no source text."""

    if not isinstance(collection, ProtectedCollection):
        raise TypeError("collection must be a ProtectedCollection")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collection.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def load_protected_collection(path: str | Path) -> ProtectedCollection:
    """Load a fingerprint manifest without needing access to protected text."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LeakageAuditError(f"cannot read protected collection: {error}") from error
    if not isinstance(payload, Mapping):
        raise LeakageAuditError("protected collection must be a JSON object")
    return ProtectedCollection.from_dict(payload)


@dataclass(frozen=True)
class AuditRecord:
    """An auditable per-document decision that deliberately omits both texts."""

    document_id: str
    decision: AuditDecision
    rule: str
    evidence: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "document_id": self.document_id,
            "evidence": [dict(item) for item in self.evidence],
            "rule": self.rule,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AuditRecord":
        try:
            return cls(
                document_id=_required_string(value["document_id"], "document_id"),
                decision=AuditDecision(value["decision"]),
                rule=_required_string(value["rule"], "rule"),
                evidence=tuple(dict(item) for item in value.get("evidence", ())),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LeakageAuditError("invalid audit record") from error


@dataclass(frozen=True)
class LeakageAuditReport:
    """Deterministic result of one corpus-vs-protected-collection audit."""

    records: tuple[AuditRecord, ...]
    similarity_threshold: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        ids = [record.document_id for record in self.records]
        if len(set(ids)) != len(ids):
            raise LeakageAuditError("audit report has duplicate document ids")
        if not 0 < self.similarity_threshold <= 1:
            raise LeakageAuditError("similarity_threshold must be in (0, 1]")
        if self.schema_version != 1:
            raise LeakageAuditError("unsupported audit report schema_version")

    @property
    def released_document_ids(self) -> tuple[str, ...]:
        return tuple(record.document_id for record in self.records if record.decision is AuditDecision.RELEASED)

    @property
    def blocked_document_ids(self) -> tuple[str, ...]:
        return tuple(record.document_id for record in self.records if record.decision is not AuditDecision.RELEASED)

    def decision_for(self, document_id: str) -> AuditDecision:
        for record in self.records:
            if record.document_id == document_id:
                return record.decision
        raise TrainingEligibilityError(f"document {document_id!r} is absent from the leakage audit report")

    def to_dict(self) -> dict[str, Any]:
        counts = {decision.value: sum(record.decision is decision for record in self.records) for decision in AuditDecision}
        return {
            "counts": counts,
            "records": [record.to_dict() for record in self.records],
            "schema_version": self.schema_version,
            "similarity_threshold": self.similarity_threshold,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LeakageAuditReport":
        try:
            return cls(
                records=tuple(AuditRecord.from_dict(item) for item in value["records"]),
                similarity_threshold=float(value["similarity_threshold"]),
                schema_version=value.get("schema_version", 1),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LeakageAuditError("invalid leakage audit report") from error


def _identifier_evidence(external: Mapping[str, tuple[str, ...]], protected: ProtectedArticle) -> list[dict[str, Any]]:
    evidence = []
    for kind in _IDENTIFIER_FIELDS:
        shared = sorted(set(external[kind]) & set(protected.identifiers[kind]))
        if shared:
            evidence.append({"identifier": kind, "protected_article_id": protected.article_id, "values": shared})
    return evidence


def _similarity_evidence(tokens: tuple[str, ...], protected: ProtectedArticle) -> list[dict[str, Any]]:
    external_shingles = _shingles(tokens)
    external_chunks = tuple(_shingles(chunk) for chunk in _chunks(tokens, size=120, stride=60))
    evidence = []
    for fingerprint in protected.documents:
        document_similarity = _ratio(external_shingles, fingerprint.shingles)
        document_shared = len(external_shingles & fingerprint.shingles)
        chunk_similarity, chunk_shared = max(
            (
                (_ratio(external_chunk, protected_chunk), len(external_chunk & protected_chunk))
                for external_chunk in external_chunks
                for protected_chunk in fingerprint.chunks
            ),
            default=(0.0, 0),
        )
        score = max(document_similarity, chunk_similarity)
        if score:
            is_chunk = chunk_similarity > document_similarity
            evidence.append(
                {
                    "comparison": "chunk" if is_chunk else "document",
                    "document_similarity": round(document_similarity, 6),
                    "protected_article_id": protected.article_id,
                    "protected_document_id": fingerprint.document_id,
                    "score": round(score, 6),
                    "shared_shingles": chunk_shared if is_chunk else document_shared,
                    "text_similarity": round(chunk_similarity, 6),
                }
            )
    return evidence


def audit_external_corpus(
    documents: Iterable[ClinicalDocument],
    protected_collection: ProtectedCollection,
    *,
    similarity_threshold: float = 0.85,
) -> LeakageAuditReport:
    """Classify every external document as released, excluded, or quarantined.

    Exact identifiers and normalised full-text SHA-256 hashes are decisive and
    therefore exclude the external document.  High similarity from a full
    document or an excerpt is intentionally only a quarantine signal.
    """

    if not isinstance(protected_collection, ProtectedCollection):
        raise TypeError("protected_collection must be a ProtectedCollection")
    if not 0 < similarity_threshold <= 1:
        raise LeakageAuditError("similarity_threshold must be in (0, 1]")
    records = []
    seen_ids = set()
    for document in documents:
        if not isinstance(document, ClinicalDocument):
            raise TypeError("audit_external_corpus accepts ClinicalDocument instances")
        if document.document_id in seen_ids:
            raise LeakageAuditError(f"external corpus has duplicate document id {document.document_id!r}")
        seen_ids.add(document.document_id)
        identifiers = document_identifiers(document)
        identifier_matches = [evidence for article in protected_collection.articles for evidence in _identifier_evidence(identifiers, article)]
        if identifier_matches:
            records.append(AuditRecord(document.document_id, AuditDecision.EXCLUDED, "identifier_match", tuple(identifier_matches)))
            continue

        content_hash = _sha256(normalise_text(document.text))
        hash_matches = [
            {"content_sha256": content_hash, "protected_article_id": article.article_id, "protected_document_id": fingerprint.document_id}
            for article in protected_collection.articles
            for fingerprint in article.documents
            if fingerprint.content_sha256 == content_hash
        ]
        if hash_matches:
            records.append(AuditRecord(document.document_id, AuditDecision.EXCLUDED, "content_hash_match", tuple(hash_matches)))
            continue

        tokens = _tokens(document.text)
        similarities = [item for article in protected_collection.articles for item in _similarity_evidence(tokens, article)]
        # A single shared five-word phrase is weak evidence in clinical prose.
        # Three shingles correspond to at least seven adjacent tokens and keep
        # short generic phrases from being quarantined automatically.
        high_similarity = [
            item
            for item in similarities
            if item["score"] >= similarity_threshold and item["shared_shingles"] >= 3
        ]
        if high_similarity:
            records.append(AuditRecord(document.document_id, AuditDecision.QUARANTINED, "text_similarity_threshold", tuple(high_similarity)))
        elif similarities:
            records.append(AuditRecord(document.document_id, AuditDecision.RELEASED, "similarity_below_threshold", tuple(similarities)))
        else:
            records.append(AuditRecord(document.document_id, AuditDecision.RELEASED, "no_overlap", ()))
    return LeakageAuditReport(tuple(records), similarity_threshold)


def write_leakage_report(report: LeakageAuditReport, path: str | Path) -> None:
    """Write a deterministic report containing evidence, rules and decisions."""

    if not isinstance(report, LeakageAuditReport):
        raise TypeError("report must be a LeakageAuditReport")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def load_leakage_report(path: str | Path) -> LeakageAuditReport:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LeakageAuditError(f"cannot read leakage audit report: {error}") from error
    if not isinstance(payload, Mapping):
        raise LeakageAuditError("leakage audit report must be a JSON object")
    return LeakageAuditReport.from_dict(payload)


def release_for_training(documents: Iterable[ClinicalDocument], report: LeakageAuditReport) -> list[ClinicalDocument]:
    """Return only explicitly released documents; never silently include blocked data."""

    if not isinstance(report, LeakageAuditReport):
        raise TypeError("report must be a LeakageAuditReport")
    released = []
    for document in documents:
        if not isinstance(document, ClinicalDocument):
            raise TypeError("release_for_training accepts ClinicalDocument instances")
        decision = report.decision_for(document.document_id)
        if decision is AuditDecision.RELEASED:
            released.append(document)
    return released


def require_training_eligible(documents: Iterable[ClinicalDocument], report: LeakageAuditReport) -> None:
    """Fail closed when a caller attempts to train on blocked or unaudited data."""

    blocked = []
    for document in documents:
        decision = report.decision_for(document.document_id)
        if decision is not AuditDecision.RELEASED:
            blocked.append(f"{document.document_id} ({decision.value})")
    if blocked:
        raise TrainingEligibilityError("documents are not eligible for training: " + ", ".join(blocked))


__all__ = [
    "AuditDecision",
    "AuditRecord",
    "LeakageAuditError",
    "LeakageAuditReport",
    "ProtectedArticle",
    "ProtectedCollection",
    "ProtectedDocumentFingerprint",
    "TrainingEligibilityError",
    "audit_external_corpus",
    "build_protected_collection",
    "document_identifiers",
    "load_leakage_report",
    "load_protected_collection",
    "normalise_text",
    "release_for_training",
    "require_training_eligible",
    "write_leakage_report",
    "write_protected_collection",
]
