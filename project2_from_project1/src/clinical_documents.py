"""Contracts for lossless clinical-relation corpus documents.

The contracts in this module deliberately describe annotations, rather than
knowledge-graph nodes.  In particular, a :class:`Mention` is an occurrence in
the source text and may be discontinuous, nested with another mention, and
carry more than one source type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ClinicalDocumentValidationError(ValueError):
    """Raised when a clinical-document contract is not internally coherent."""


class CanonicalRelation(str, Enum):
    """The initial supervised relation vocabulary plus the case structure link."""

    DESCRIBES = "DESCRIBES"
    HAS_CONDITION = "HAS_CONDITION"
    HAS_FINDING = "HAS_FINDING"
    HAS_FACTOR = "HAS_FACTOR"
    NEGATES = "NEGATES"
    CAUSES = "CAUSES"
    NO_RELATION = "NO_RELATION"


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClinicalDocumentValidationError(f"{field_name} must be a non-empty string")
    return value


def _as_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence) or not values:
        raise ClinicalDocumentValidationError(f"{field_name} must contain at least one value")
    result = tuple(_non_empty_string(value, field_name) for value in values)
    if len(set(result)) != len(result):
        raise ClinicalDocumentValidationError(f"{field_name} must not contain duplicates")
    return result


@dataclass(frozen=True, order=True)
class Span:
    """A half-open character interval in the document text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise ClinicalDocumentValidationError("span start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise ClinicalDocumentValidationError("span end must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise ClinicalDocumentValidationError(
                f"invalid span [{self.start}, {self.end}): expected 0 <= start < end"
            )

    def to_dict(self) -> list[int]:
        return [self.start, self.end]

    @classmethod
    def from_dict(cls, value: object) -> "Span":
        if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
            raise ClinicalDocumentValidationError("a span must be a [start, end) pair")
        return cls(value[0], value[1])


@dataclass(frozen=True)
class ClinicalCase:
    """The documentary unit, distinct from the person described by it."""

    case_id: str
    source_id: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _non_empty_string(self.case_id, "case_id"))
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _non_empty_string(self.source_id, "source_id"))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "source_id": self.source_id, "provenance": dict(self.provenance)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClinicalCase":
        return cls(
            case_id=value.get("case_id"),
            source_id=value.get("source_id"),
            provenance=value.get("provenance", {}),
        )


@dataclass(frozen=True)
class Patient:
    """The clinical person described by a :class:`ClinicalCase`."""

    patient_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "patient_id", _non_empty_string(self.patient_id, "patient_id"))
        object.__setattr__(self, "attributes", dict(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        return {"patient_id": self.patient_id, "attributes": dict(self.attributes)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Patient":
        return cls(patient_id=value.get("patient_id"), attributes=value.get("attributes", {}))


@dataclass(frozen=True)
class Mention:
    """An annotated text occurrence with lossless source and canonical typing."""

    mention_id: str
    spans: Sequence[Span | Sequence[int]]
    native_types: Sequence[str]
    canonical_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "mention_id", _non_empty_string(self.mention_id, "mention_id"))
        parsed_spans = tuple(span if isinstance(span, Span) else Span.from_dict(span) for span in self.spans)
        if not parsed_spans:
            raise ClinicalDocumentValidationError(f"mention {self.mention_id} must contain at least one span")
        if tuple(sorted(parsed_spans)) != parsed_spans:
            raise ClinicalDocumentValidationError(
                f"mention {self.mention_id} spans must be ordered by start position"
            )
        for previous, current in zip(parsed_spans, parsed_spans[1:]):
            if current.start < previous.end:
                raise ClinicalDocumentValidationError(
                    f"mention {self.mention_id} has overlapping spans; use separate mentions for nesting"
                )
        object.__setattr__(self, "spans", parsed_spans)
        object.__setattr__(self, "native_types", _as_tuple(self.native_types, "native_types"))
        object.__setattr__(self, "canonical_type", _non_empty_string(self.canonical_type, "canonical_type"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "spans": [span.to_dict() for span in self.spans],
            "native_types": list(self.native_types),
            "canonical_type": self.canonical_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Mention":
        return cls(
            mention_id=value.get("mention_id"),
            spans=value.get("spans", ()),
            native_types=value.get("native_types", ()),
            canonical_type=value.get("canonical_type"),
        )


@dataclass(frozen=True)
class AnnotatedRelation:
    """A directed relation preserving its source and canonical labels."""

    relation_id: str
    source_id: str
    target_id: str
    native_label: str
    canonical_label: CanonicalRelation | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_id", _non_empty_string(self.relation_id, "relation_id"))
        object.__setattr__(self, "source_id", _non_empty_string(self.source_id, "source_id"))
        object.__setattr__(self, "target_id", _non_empty_string(self.target_id, "target_id"))
        object.__setattr__(self, "native_label", _non_empty_string(self.native_label, "native_label"))
        try:
            canonical = CanonicalRelation(self.canonical_label)
        except (TypeError, ValueError) as error:
            allowed = ", ".join(relation.value for relation in CanonicalRelation)
            raise ClinicalDocumentValidationError(
                f"relation {self.relation_id} has unsupported canonical_label {self.canonical_label!r}; expected one of {allowed}"
            ) from error
        object.__setattr__(self, "canonical_label", canonical)
        if self.source_id == self.target_id:
            raise ClinicalDocumentValidationError(f"relation {self.relation_id} cannot point to itself")

    def to_dict(self) -> dict[str, str]:
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "native_label": self.native_label,
            "canonical_label": self.canonical_label.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnnotatedRelation":
        return cls(
            relation_id=value.get("relation_id"),
            source_id=value.get("source_id"),
            target_id=value.get("target_id"),
            native_label=value.get("native_label"),
            canonical_label=value.get("canonical_label"),
        )


@dataclass(frozen=True)
class ClinicalDocument:
    """A complete, validated clinical document ready for JSONL interchange."""

    document_id: str
    text: str
    cases: Sequence[ClinicalCase]
    patients: Sequence[Patient]
    mentions: Sequence[Mention]
    relations: Sequence[AnnotatedRelation]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _non_empty_string(self.document_id, "document_id"))
        if not isinstance(self.text, str):
            raise ClinicalDocumentValidationError("text must be a string")
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "patients", tuple(self.patients))
        object.__setattr__(self, "mentions", tuple(self.mentions))
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "metadata", dict(self.metadata))
        self.validate()

    def validate(self) -> None:
        self._validate_unique_identifiers()
        text_length = len(self.text)
        for mention in self.mentions:
            for span in mention.spans:
                if span.end > text_length:
                    raise ClinicalDocumentValidationError(
                        f"mention {mention.mention_id} span [{span.start}, {span.end}) exceeds text length {text_length}"
                    )
        identifiers = {
            *(case.case_id for case in self.cases),
            *(patient.patient_id for patient in self.patients),
            *(mention.mention_id for mention in self.mentions),
        }
        relation_ids = set()
        for relation in self.relations:
            if relation.relation_id in relation_ids:
                raise ClinicalDocumentValidationError(f"duplicate relation_id {relation.relation_id!r}")
            relation_ids.add(relation.relation_id)
            if relation.source_id not in identifiers or relation.target_id not in identifiers:
                raise ClinicalDocumentValidationError(
                    f"relation {relation.relation_id} references unknown endpoint "
                    f"{relation.source_id!r} -> {relation.target_id!r}"
                )
        self._validate_relation_semantics()

    def _validate_unique_identifiers(self) -> None:
        all_ids = [
            *(case.case_id for case in self.cases),
            *(patient.patient_id for patient in self.patients),
            *(mention.mention_id for mention in self.mentions),
        ]
        duplicates = sorted({identifier for identifier in all_ids if all_ids.count(identifier) > 1})
        if duplicates:
            raise ClinicalDocumentValidationError(
                "identifiers must be unique across cases, patients, and mentions: " + ", ".join(duplicates)
            )

    def _validate_relation_semantics(self) -> None:
        case_ids = {case.case_id for case in self.cases}
        patient_ids = {patient.patient_id for patient in self.patients}
        mentions = {mention.mention_id: mention for mention in self.mentions}
        negated_findings: set[str] = set()

        for relation in self.relations:
            label = relation.canonical_label
            if label is CanonicalRelation.DESCRIBES:
                if relation.source_id not in case_ids or relation.target_id not in patient_ids:
                    raise ClinicalDocumentValidationError(
                        f"DESCRIBES relation {relation.relation_id} must point ClinicalCase -> Patient"
                    )
            elif label is CanonicalRelation.NEGATES:
                source = mentions.get(relation.source_id)
                target = mentions.get(relation.target_id)
                if source is None or target is None or source.canonical_type != "Modifier" or target.canonical_type != "Finding":
                    raise ClinicalDocumentValidationError(
                        f"NEGATES relation {relation.relation_id} must point Modifier mention -> Finding mention"
                    )
                negated_findings.add(target.mention_id)
            elif label is CanonicalRelation.HAS_FINDING:
                target = mentions.get(relation.target_id)
                if relation.source_id not in patient_ids or target is None or target.canonical_type != "Finding":
                    raise ClinicalDocumentValidationError(
                        f"HAS_FINDING relation {relation.relation_id} must point Patient -> Finding mention"
                    )
            elif label is CanonicalRelation.HAS_CONDITION:
                target = mentions.get(relation.target_id)
                if relation.source_id not in patient_ids or target is None or target.canonical_type != "Condition":
                    raise ClinicalDocumentValidationError(
                        f"HAS_CONDITION relation {relation.relation_id} must point Patient -> Condition mention"
                    )
            elif label is CanonicalRelation.HAS_FACTOR:
                target = mentions.get(relation.target_id)
                if relation.source_id not in patient_ids or target is None or target.canonical_type != "Factor":
                    raise ClinicalDocumentValidationError(
                        f"HAS_FACTOR relation {relation.relation_id} must point Patient -> Factor mention"
                    )
            elif label is CanonicalRelation.CAUSES:
                source = mentions.get(relation.source_id)
                target = mentions.get(relation.target_id)
                if source is None or target is None or source.canonical_type != "Condition" or target.canonical_type != "Finding":
                    raise ClinicalDocumentValidationError(
                        f"CAUSES relation {relation.relation_id} must point Condition mention -> Finding mention"
                    )

        positive_negations = [
            relation.relation_id
            for relation in self.relations
            if relation.canonical_label is CanonicalRelation.HAS_FINDING
            and relation.target_id in negated_findings
        ]
        if positive_negations:
            raise ClinicalDocumentValidationError(
                "negated Finding mentions cannot also receive HAS_FINDING: " + ", ".join(positive_negations)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "cases": [case.to_dict() for case in self.cases],
            "patients": [patient.to_dict() for patient in self.patients],
            "mentions": [mention.to_dict() for mention in self.mentions],
            "relations": [relation.to_dict() for relation in self.relations],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClinicalDocument":
        if not isinstance(value, Mapping):
            raise ClinicalDocumentValidationError("each JSONL line must contain an object")
        try:
            return cls(
                document_id=value.get("document_id"),
                text=value.get("text"),
                cases=tuple(ClinicalCase.from_dict(item) for item in value.get("cases", ())),
                patients=tuple(Patient.from_dict(item) for item in value.get("patients", ())),
                mentions=tuple(Mention.from_dict(item) for item in value.get("mentions", ())),
                relations=tuple(AnnotatedRelation.from_dict(item) for item in value.get("relations", ())),
                metadata=value.get("metadata", {}),
            )
        except (TypeError, AttributeError) as error:
            raise ClinicalDocumentValidationError("document fields must have their documented JSON types") from error


def write_jsonl(documents: Iterable[ClinicalDocument], path: str | Path) -> None:
    """Write validated documents as deterministic UTF-8 JSON Lines."""

    output_path = Path(path)
    serialized = []
    document_ids = set()
    for document in documents:
        if not isinstance(document, ClinicalDocument):
            raise TypeError("write_jsonl accepts ClinicalDocument instances")
        if document.document_id in document_ids:
            raise ClinicalDocumentValidationError(f"duplicate document_id {document.document_id!r} in JSONL output")
        document_ids.add(document.document_id)
        serialized.append(json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(serialized) + ("\n" if serialized else ""), encoding="utf-8")


def load_jsonl(path: str | Path) -> list[ClinicalDocument]:
    """Load JSONL and report the line that violates the document contract."""

    documents = []
    document_ids = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ClinicalDocumentValidationError(f"JSONL line {line_number} is empty")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ClinicalDocumentValidationError(f"invalid JSON on line {line_number}: {error.msg}") from error
        try:
            document = ClinicalDocument.from_dict(payload)
        except ClinicalDocumentValidationError as error:
            raise ClinicalDocumentValidationError(f"invalid document on JSONL line {line_number}: {error}") from error
        if document.document_id in document_ids:
            raise ClinicalDocumentValidationError(
                f"duplicate document_id {document.document_id!r} on JSONL line {line_number}"
            )
        document_ids.add(document.document_id)
        documents.append(document)
    return documents


__all__ = [
    "AnnotatedRelation",
    "CanonicalRelation",
    "ClinicalCase",
    "ClinicalDocument",
    "ClinicalDocumentValidationError",
    "Mention",
    "Patient",
    "Span",
    "load_jsonl",
    "write_jsonl",
]
