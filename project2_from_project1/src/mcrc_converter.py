"""Controlled conversion of a verified MCRC pickle distribution.

The MCRC release is a pickled BioC collection.  Pickles are executable input,
so the normal project pipeline must never use this module's deserialisation
entry point.  It is intentionally an opt-in, one-time ingestion boundary:
the artifact is hashed before it is opened and the caller must explicitly
acknowledge the unsafe operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import pickle
import re
from typing import Any, Mapping, Sequence
import uuid

from project2_from_project1.src.clinical_documents import (
    AnnotatedRelation,
    CanonicalRelation,
    ClinicalCase,
    ClinicalDocument,
    ClinicalDocumentValidationError,
    Mention,
    Patient,
    Span,
    write_jsonl,
)


_IMMUTABLE_REVISION = re.compile(
    r"(?:[0-9a-f]{7,64}|v?\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.-]+)?)\Z", re.IGNORECASE
)


class MCRCConversionError(ValueError):
    """Raised when a MCRC source cannot safely or faithfully be converted."""


@dataclass(frozen=True)
class MCRCSource:
    """Pinned provenance required before a release artifact can be opened."""

    repository: str
    revision: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("repository", "revision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise MCRCConversionError(f"{field_name} must be a non-empty string")
        if not _IMMUTABLE_REVISION.fullmatch(self.revision):
            raise MCRCConversionError(
                "revision must be an immutable commit SHA or a version tag such as v1.2.3"
            )
        digest = self.artifact_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise MCRCConversionError("artifact_sha256 must be a 64-character hexadecimal SHA-256 digest")
        object.__setattr__(self, "artifact_sha256", digest)

    def to_manifest_dict(self) -> dict[str, str]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "repository": self.repository,
            "revision": self.revision,
        }


_ENTITY_CANONICAL_TYPES = {
    "case": "Patient",
    "condition": "Condition",
    "finding": "Finding",
    "factor": "Factor",
    "modifier": "Modifier",
}

_RELATION_CANONICAL_LABELS = {
    "case_has_condition": CanonicalRelation.HAS_CONDITION,
    "case_has_finding": CanonicalRelation.HAS_FINDING,
    "case_has_factor": CanonicalRelation.HAS_FACTOR,
    "modifier_modifies_finding": CanonicalRelation.NEGATES,
    "condition_causes_finding": CanonicalRelation.CAUSES,
}


class _BioCProxy:
    """Compatibility shell for the BioC classes embedded in the release pickle."""

    def __setstate__(self, state: object) -> None:
        if isinstance(state, Mapping):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and state and isinstance(state[0], Mapping):
            self.__dict__.update(state[0])
        else:
            self.__dict__["_pickle_state"] = state


class _MCRCPickleUnpickler(pickle.Unpickler):
    """Read only the known MCRC BioC classes, without importing their old package."""

    _BIOC_MODULE_PREFIX = "src.util.bioc."
    _ALLOWED_GLOBALS = {
        ("uuid", "UUID"): uuid.UUID,
        ("collections", "OrderedDict"): __import__("collections").OrderedDict,
    }

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith(self._BIOC_MODULE_PREFIX) and name.startswith("BioC"):
            return _BioCProxy
        try:
            return self._ALLOWED_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(f"MCRC pickle requested unsupported global {module}.{name}") from error


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest without loading the artifact into memory."""

    with Path(path).open("rb") as artifact:
        return _sha256_open_file(artifact)


def _sha256_open_file(artifact: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: artifact.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def verify_mcrc_artifact(path: str | Path, source: MCRCSource) -> str:
    """Verify the pinned artifact digest before any deserialisation is attempted."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise MCRCConversionError(f"MCRC artifact does not exist or is not a file: {artifact_path}")
    with artifact_path.open("rb") as artifact:
        actual_digest = _sha256_open_file(artifact)
    if not hmac.compare_digest(actual_digest, source.artifact_sha256):
        raise MCRCConversionError(
            "MCRC artifact SHA-256 mismatch; refusing to deserialize it "
            f"(expected {source.artifact_sha256}, got {actual_digest})"
        )
    return actual_digest


def _load_verified_mcrc_pickle(
    path: str | Path,
    source: MCRCSource,
    *,
    allow_unsafe_deserialization: bool,
) -> tuple[Any, str]:
    if not allow_unsafe_deserialization:
        raise MCRCConversionError(
            "refusing to deserialize MCRC pickle: pass allow_unsafe_deserialization=True "
            "only in a controlled conversion environment"
        )
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise MCRCConversionError(f"MCRC artifact does not exist or is not a file: {artifact_path}")
    with artifact_path.open("rb") as artifact:
        actual_digest = _sha256_open_file(artifact)
        if not hmac.compare_digest(actual_digest, source.artifact_sha256):
            raise MCRCConversionError(
                "MCRC artifact SHA-256 mismatch; refusing to deserialize it "
                f"(expected {source.artifact_sha256}, got {actual_digest})"
            )
        artifact.seek(0)
        return _MCRCPickleUnpickler(artifact).load(), actual_digest


def load_verified_mcrc_pickle(
    path: str | Path,
    source: MCRCSource,
    *,
    allow_unsafe_deserialization: bool = False,
) -> Any:
    """Open a verified pickle only after an explicit controlled-environment opt-in."""

    collection, _ = _load_verified_mcrc_pickle(
        path,
        source,
        allow_unsafe_deserialization=allow_unsafe_deserialization,
    )
    return collection


def _get(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _required_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MCRCConversionError(f"{description} must be a non-empty string")
    return value


def _external_identifier(value: object, description: str) -> str:
    """Normalise the numeric annotation identifiers used by the BioC release."""

    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return _required_string(value, description)


def _as_items(value: object, description: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MCRCConversionError(f"{description} must be a sequence")
    return list(value)


def _infons(value: object) -> Mapping[str, Any]:
    result = _get(value, "infons", {})
    if not isinstance(result, Mapping):
        raise MCRCConversionError("BioC infons must be an object")
    return result


def _native_types(annotation: object, annotation_id: str) -> tuple[str, ...]:
    infons = _infons(annotation)
    raw_types = infons.get("entityType", infons.get("entity_type", infons.get("type")))
    if isinstance(raw_types, str):
        types = (raw_types,)
    elif isinstance(raw_types, Sequence) and not isinstance(raw_types, (str, bytes)):
        types = tuple(_required_string(item, f"annotation {annotation_id} native type") for item in raw_types)
    else:
        raise MCRCConversionError(f"annotation {annotation_id} has no entityType")
    if not types or len(set(types)) != len(types):
        raise MCRCConversionError(f"annotation {annotation_id} has invalid native entity types")
    return types


def _canonical_type(native_types: tuple[str, ...], annotation_id: str) -> str:
    for native_type in native_types:
        canonical = _ENTITY_CANONICAL_TYPES.get(native_type.lower())
        if canonical:
            return canonical
    raise MCRCConversionError(
        f"annotation {annotation_id} has no approved canonical mapping for {list(native_types)!r}"
    )


def _spans(annotation: object, annotation_id: str, passage_offset: int) -> tuple[Span, ...]:
    raw_locations = _get(annotation, "locations")
    if raw_locations is None:
        raise MCRCConversionError(f"annotation {annotation_id} has no locations")
    if isinstance(raw_locations, Sequence) and not isinstance(raw_locations, (str, bytes, Mapping)):
        locations = list(raw_locations)
    else:
        locations = [raw_locations]
    if not locations:
        raise MCRCConversionError(f"annotation {annotation_id} has no locations")
    spans = []
    for location in locations:
        start = _get(location, "offset")
        length = _get(location, "length")
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(length, bool) or not isinstance(length, int):
            raise MCRCConversionError(f"annotation {annotation_id} location must have integer offset and length")
        spans.append(Span(start - passage_offset, start - passage_offset + length))
    return tuple(sorted(spans))


def _normalise_label(label: str) -> str:
    return "_".join(part for part in label.lower().replace("-", "_").replace(" ", "_").split("_") if part)


def _native_relation_label(relation: object, relation_id: str) -> str:
    infons = _infons(relation)
    for key in ("relationType", "relation_type", "type", "label"):
        value = infons.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise MCRCConversionError(f"relation {relation_id} has no relationType")


def _canonical_relation_label(
    native_label: str,
    source_types: tuple[str, ...],
    target_types: tuple[str, ...],
    relation_id: str,
) -> CanonicalRelation:
    """Project a native MCRC relation only when its endpoints support the meaning."""

    normalized = _normalise_label(native_label)
    explicit = _RELATION_CANONICAL_LABELS.get(normalized)
    if explicit is not None:
        return explicit
    source = {item.lower() for item in source_types}
    target = {item.lower() for item in target_types}
    if normalized == "has" and "case" in source:
        if "condition" in target:
            return CanonicalRelation.HAS_CONDITION
        if "finding" in target:
            return CanonicalRelation.HAS_FINDING
        if "factor" in target:
            return CanonicalRelation.HAS_FACTOR
    elif normalized == "modifies" and "modifier" in source and "finding" in target:
        return CanonicalRelation.NEGATES
    elif normalized == "causes" and "condition" in source and "finding" in target:
        return CanonicalRelation.CAUSES
    raise MCRCConversionError(
        f"relation {relation_id} has no approved canonical mapping for {native_label!r} "
        f"with endpoints {list(source_types)!r} -> {list(target_types)!r}"
    )


def _relation_endpoints(relation: object, relation_id: str) -> tuple[str, str]:
    nodes = _as_items(_get(relation, "nodes", ()), f"relation {relation_id} nodes")
    if len(nodes) != 2:
        raise MCRCConversionError(f"relation {relation_id} must have exactly two directed endpoints")
    refs = [
        _external_identifier(
            _get(node, "refid", _get(node, "ref_id")),
            f"relation {relation_id} node refid",
        )
        for node in nodes
    ]
    roles = [str(_get(node, "role", "")).lower() for node in nodes]
    source_roles = {"source", "from", "arg1", "subject", "governor"}
    target_roles = {"target", "to", "arg2", "object", "dependent"}
    source_indexes = [index for index, role in enumerate(roles) if role in source_roles]
    target_indexes = [index for index, role in enumerate(roles) if role in target_roles]
    if len(source_indexes) == len(target_indexes) == 1:
        return refs[source_indexes[0]], refs[target_indexes[0]]
    raise MCRCConversionError(
        f"relation {relation_id} must declare one source and one target role; got {roles!r}"
    )


def _safe_metadata(infons: Mapping[str, Any]) -> dict[str, str]:
    """Keep small document provenance values JSON-safe without copying corpus text."""

    return {str(key): str(value) for key, value in sorted(infons.items(), key=lambda item: str(item[0]))}


def _convert_document(raw_document: object) -> ClinicalDocument:
    raw_document_id = _required_string(_get(raw_document, "id"), "MCRC document id")
    passages = _as_items(_get(raw_document, "passages", ()), f"document {raw_document_id} passages")
    if len(passages) != 1:
        raise MCRCConversionError(f"document {raw_document_id} must contain exactly one passage")
    passage = passages[0]
    text = _get(passage, "text")
    if not isinstance(text, str):
        raise MCRCConversionError(f"document {raw_document_id} passage text must be a string")
    passage_offset = _get(passage, "offset", 0)
    if isinstance(passage_offset, bool) or not isinstance(passage_offset, int):
        raise MCRCConversionError(f"document {raw_document_id} passage offset must be an integer")

    document_id = f"mcrc:{raw_document_id}"
    annotations = _as_items(_get(passage, "annotations", ()), f"document {raw_document_id} annotations")
    mentions: list[Mention] = []
    native_to_mention: dict[str, str] = {}
    native_annotation_types: dict[str, tuple[str, ...]] = {}
    case_annotation_ids: list[str] = []
    for annotation in annotations:
        native_id = _external_identifier(_get(annotation, "id"), f"document {raw_document_id} annotation id")
        if native_id in native_to_mention:
            raise MCRCConversionError(f"document {raw_document_id} has duplicate annotation id {native_id!r}")
        native_types = _native_types(annotation, native_id)
        mention_id = f"{document_id}:mention:{native_id}"
        mentions.append(Mention(mention_id, _spans(annotation, native_id, passage_offset), native_types, _canonical_type(native_types, native_id)))
        native_to_mention[native_id] = mention_id
        native_annotation_types[native_id] = native_types
        if any(native_type.lower() == "case" for native_type in native_types):
            case_annotation_ids.append(native_id)

    patient_id = f"{document_id}:patient"
    patient = Patient(patient_id, {"native_case_mention_ids": [native_to_mention[item] for item in case_annotation_ids]})
    clinical_case = ClinicalCase(
        f"{document_id}:case",
        source_id=raw_document_id,
        provenance={"corpus": "MCRC", "document_infons": _safe_metadata(_infons(raw_document))},
    )
    relations: list[AnnotatedRelation] = [
        AnnotatedRelation(
            relation_id=f"{document_id}:describes",
            source_id=clinical_case.case_id,
            target_id=patient_id,
            native_label="document_describes_case",
            canonical_label=CanonicalRelation.DESCRIBES,
        )
    ]
    for raw_relation in _as_items(_get(passage, "relations", ()), f"document {raw_document_id} relations"):
        raw_relation_id = _external_identifier(_get(raw_relation, "id"), f"document {raw_document_id} relation id")
        native_label = _native_relation_label(raw_relation, raw_relation_id)
        source_native, target_native = _relation_endpoints(raw_relation, raw_relation_id)
        if source_native not in native_to_mention or target_native not in native_to_mention:
            raise MCRCConversionError(f"relation {raw_relation_id} references an unknown annotation")
        canonical_label = _canonical_relation_label(
            native_label,
            native_annotation_types[source_native],
            native_annotation_types[target_native],
            raw_relation_id,
        )
        source_id = patient_id if source_native in case_annotation_ids else native_to_mention[source_native]
        target_id = patient_id if target_native in case_annotation_ids else native_to_mention[target_native]
        relations.append(
            AnnotatedRelation(
                relation_id=f"{document_id}:relation:{raw_relation_id}",
                source_id=source_id,
                target_id=target_id,
                native_label=native_label,
                canonical_label=canonical_label,
            )
        )

    try:
        return ClinicalDocument(
            document_id=document_id,
            text=text,
            cases=[clinical_case],
            patients=[patient],
            mentions=mentions,
            relations=relations,
            metadata={"corpus": "MCRC", "native_document_id": raw_document_id},
        )
    except ClinicalDocumentValidationError as error:
        raise MCRCConversionError(f"document {raw_document_id} violates the clinical document contract: {error}") from error


def convert_mcrc_collection(collection: object) -> list[ClinicalDocument]:
    """Convert an already-opened MCRC BioC collection to validated documents."""

    documents = _as_items(_get(collection, "documents", ()), "MCRC collection documents")
    converted = [_convert_document(document) for document in documents]
    document_ids = [document.document_id for document in converted]
    if len(set(document_ids)) != len(document_ids):
        raise MCRCConversionError("MCRC collection contains duplicate document ids")
    return sorted(converted, key=lambda document: document.document_id)


def _counts(documents: Sequence[ClinicalDocument]) -> dict[str, int]:
    return {
        "documents": len(documents),
        "mentions": sum(len(document.mentions) for document in documents),
        "relations": sum(len(document.relations) for document in documents),
    }


def convert_mcrc_distribution(
    artifact_path: str | Path,
    output_jsonl_path: str | Path,
    manifest_path: str | Path,
    source: MCRCSource,
    *,
    allow_unsafe_deserialization: bool = False,
) -> dict[str, Any]:
    """Convert a pinned MCRC pickle and write deterministic JSONL and manifest files."""

    collection, raw_sha256 = _load_verified_mcrc_pickle(
        artifact_path,
        source,
        allow_unsafe_deserialization=allow_unsafe_deserialization,
    )
    documents = convert_mcrc_collection(collection)
    output_jsonl = Path(output_jsonl_path)
    write_jsonl(documents, output_jsonl)
    manifest = {
        "conversion_result": "success",
        "counts": _counts(documents),
        "jsonl_sha256": sha256_file(output_jsonl),
        "raw_artifact_sha256": raw_sha256,
        "schema_version": 1,
        "source": source.to_manifest_dict(),
    }
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "MCRCConversionError",
    "MCRCSource",
    "convert_mcrc_collection",
    "convert_mcrc_distribution",
    "load_verified_mcrc_pickle",
    "sha256_file",
    "verify_mcrc_artifact",
]
