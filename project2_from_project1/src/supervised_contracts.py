"""Versioned, text-free supervised predictions and canonical graph records.

Offsets are half-open character offsets in the original document. Identifiers
are stable across runs and are scoped by document, not by model output order.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Any

from project2_from_project1.src.clinical_documents import CanonicalRelation, Span


class ContractError(ValueError):
    """A record cannot be safely interpreted according to its schema."""


ENTITY_TYPES = frozenset({"Condition", "Finding", "Factor", "Modifier"})
RELATIONS = frozenset({"HAS_CONDITION", "HAS_FINDING", "HAS_FACTOR", "NEGATES", "CAUSES"})


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field} must be a non-negative integer")
    return value


def _id(prefix: str, *parts: object) -> str:
    data = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return prefix + ":" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _fields(value: object, required: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("record must be a JSON object")
    missing = required - value.keys()
    extra = value.keys() - required
    if missing:
        raise ContractError(f"missing field {sorted(missing)[0]}")
    if extra:
        raise ContractError(f"unexpected field {sorted(extra)[0]}")
    return value


def _version(value: object, expected: int) -> None:
    if type(value) is not int or value != expected:
        raise ContractError(f"schema_version must be {expected}")


@dataclass(frozen=True)
class EntityPrediction:
    document_id: str
    span: Span
    canonical_type: str
    sentence_index: int
    method: str
    model_version: str
    marginal_probability: float | None = None

    def __post_init__(self) -> None:
        for name in ("document_id", "method", "model_version"):
            _string(getattr(self, name), name)
        if not isinstance(self.span, Span):
            raise ContractError("span must be a Span")
        if not isinstance(self.canonical_type, str) or self.canonical_type not in ENTITY_TYPES:
            raise ContractError("canonical_type must be Condition, Finding, Factor or Modifier")
        _integer(self.sentence_index, "sentence_index")
        value = self.marginal_probability
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1 or not math.isfinite(value)):
            raise ContractError("marginal_probability must be null or a finite number in [0, 1]")

    @property
    def prediction_id(self) -> str:
        return _id("entity", self.document_id, self.span.start, self.span.end, self.canonical_type)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "prediction_id": self.prediction_id, "document_id": self.document_id,
                "span": self.span.to_dict(), "canonical_type": self.canonical_type, "sentence_index": self.sentence_index,
                "method": self.method, "model_version": self.model_version, "marginal_probability": self.marginal_probability}

    @classmethod
    def from_dict(cls, value: object) -> EntityPrediction:
        record = _fields(value, {"schema_version", "prediction_id", "document_id", "span", "canonical_type", "sentence_index", "method", "model_version", "marginal_probability"})
        _version(record["schema_version"], 1)
        try:
            span = Span.from_dict(record["span"])
        except ValueError as error:
            raise ContractError(f"span: {error}") from error
        result = cls(record["document_id"], span, record["canonical_type"], record["sentence_index"], record["method"], record["model_version"], record["marginal_probability"])
        if record["prediction_id"] != result.prediction_id:
            raise ContractError("prediction_id does not match document_id, span and canonical_type")
        return result


@dataclass(frozen=True)
class RelationCandidate:
    document_id: str
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    candidate_relation: CanonicalRelation | str
    sentence_distance: int | None

    def __post_init__(self) -> None:
        for name in ("document_id", "source_id", "target_id"):
            _string(getattr(self, name), name)
        if self.source_id == self.target_id:
            raise ContractError("target_id must differ from source_id")
        for name in ("source_type", "target_type"):
            if not isinstance(getattr(self, name), str) or getattr(self, name) not in ENTITY_TYPES | {"Patient"}:
                raise ContractError(f"{name} must be a canonical entity type or Patient")
        try:
            relation = CanonicalRelation(self.candidate_relation)
        except (ValueError, TypeError) as error:
            raise ContractError("candidate_relation must be a supervised positive relation") from error
        if relation.value not in RELATIONS:
            raise ContractError("candidate_relation must be a supervised positive relation")
        object.__setattr__(self, "candidate_relation", relation)
        if self.sentence_distance is not None:
            _integer(self.sentence_distance, "sentence_distance")

    @property
    def candidate_id(self) -> str:
        return _id("candidate", self.document_id, self.source_id, self.target_id, self.source_type, self.target_type, self.candidate_relation.value)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "candidate_id": self.candidate_id, "document_id": self.document_id,
                "source_id": self.source_id, "target_id": self.target_id, "source_type": self.source_type,
                "target_type": self.target_type, "candidate_relation": self.candidate_relation.value, "sentence_distance": self.sentence_distance}

    @classmethod
    def from_dict(cls, value: object) -> RelationCandidate:
        record = _fields(value, {"schema_version", "candidate_id", "document_id", "source_id", "target_id", "source_type", "target_type", "candidate_relation", "sentence_distance"})
        _version(record["schema_version"], 1)
        result = cls(*(record[name] for name in ("document_id", "source_id", "target_id", "source_type", "target_type", "candidate_relation", "sentence_distance")))
        if record["candidate_id"] != result.candidate_id:
            raise ContractError("candidate_id does not match candidate identity fields")
        return result


def _provenance(value: object) -> dict[str, Any]:
    record = _fields(value, {"method", "model_version", "source_ids"})
    _string(record["method"], "provenance.method")
    _string(record["model_version"], "provenance.model_version")
    ids = record["source_ids"]
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ContractError("provenance.source_ids must be a non-empty list")
    for item in ids:
        _string(item, "provenance.source_ids")
    if len(set(ids)) != len(ids):
        raise ContractError("provenance.source_ids must not contain duplicates")
    return {"method": record["method"], "model_version": record["model_version"], "source_ids": list(ids)}


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    document_id: str
    layer: str
    canonical_type: str
    provenance: Mapping[str, Any]
    span: Span | None = None
    sentence_index: int | None = None
    normalized_surface: str | None = None

    def __post_init__(self) -> None:
        for name in ("node_id", "document_id"):
            _string(getattr(self, name), name)
        if self.layer not in ("mention", "concept"):
            raise ContractError("layer must be mention or concept")
        if not isinstance(self.canonical_type, str) or self.canonical_type not in ENTITY_TYPES | {"Patient"}:
            raise ContractError("canonical_type must be a canonical entity type or Patient")
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        if self.layer == "mention" and self.canonical_type != "Patient":
            if not isinstance(self.span, Span):
                raise ContractError("span is required for mention nodes")
            _integer(self.sentence_index, "sentence_index")
        elif self.span is not None or self.sentence_index is not None:
            raise ContractError("span and sentence_index are only allowed on entity mention nodes")
        if self.layer == "concept" and self.canonical_type != "Patient":
            _string(self.normalized_surface, "normalized_surface")
        elif self.normalized_surface is not None:
            raise ContractError("normalized_surface is only allowed on entity concept nodes")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 2, "record_type": "node", "node_id": self.node_id, "document_id": self.document_id,
                "layer": self.layer, "canonical_type": self.canonical_type, "span": self.span.to_dict() if self.span else None,
                "sentence_index": self.sentence_index, "normalized_surface": self.normalized_surface, "provenance": _provenance(self.provenance)}

    @classmethod
    def from_dict(cls, value: object) -> GraphNode:
        record = _fields(value, {"schema_version", "record_type", "node_id", "document_id", "layer", "canonical_type", "span", "sentence_index", "normalized_surface", "provenance"})
        _version(record["schema_version"], 2)
        if record["record_type"] != "node":
            raise ContractError("record_type must be node")
        try:
            span = Span.from_dict(record["span"]) if record["span"] is not None else None
        except ValueError as error:
            raise ContractError(f"span: {error}") from error
        return cls(record["node_id"], record["document_id"], record["layer"], record["canonical_type"], record["provenance"], span, record["sentence_index"], record["normalized_surface"])


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    document_id: str
    layer: str
    source_id: str
    target_id: str
    canonical_relation: CanonicalRelation | str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("edge_id", "document_id", "source_id", "target_id"):
            _string(getattr(self, name), name)
        if self.layer not in ("mention", "concept"):
            raise ContractError("layer must be mention or concept")
        if self.source_id == self.target_id:
            raise ContractError("target_id must differ from source_id")
        try:
            relation = CanonicalRelation(self.canonical_relation)
        except (ValueError, TypeError) as error:
            raise ContractError("canonical_relation must be a supervised positive relation") from error
        if relation.value not in RELATIONS:
            raise ContractError("canonical_relation must be a supervised positive relation")
        object.__setattr__(self, "canonical_relation", relation)
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 2, "record_type": "edge", "edge_id": self.edge_id, "document_id": self.document_id,
                "layer": self.layer, "source_id": self.source_id, "target_id": self.target_id,
                "canonical_relation": self.canonical_relation.value, "provenance": _provenance(self.provenance)}

    @classmethod
    def from_dict(cls, value: object) -> GraphEdge:
        record = _fields(value, {"schema_version", "record_type", "edge_id", "document_id", "layer", "source_id", "target_id", "canonical_relation", "provenance"})
        _version(record["schema_version"], 2)
        if record["record_type"] != "edge":
            raise ContractError("record_type must be edge")
        return cls(*(record[name] for name in ("edge_id", "document_id", "layer", "source_id", "target_id", "canonical_relation", "provenance")))


def _validate_graph(records: list[GraphNode | GraphEdge]) -> None:
    nodes: dict[str, GraphNode] = {}
    edges: set[str] = set()
    for index, record in enumerate(records, 1):
        if isinstance(record, GraphNode):
            if record.node_id in nodes:
                raise ContractError(f"record {index}: duplicate node_id {record.node_id!r}")
            nodes[record.node_id] = record
        elif record.edge_id in edges:
            raise ContractError(f"record {index}: duplicate edge_id {record.edge_id!r}")
        else:
            edges.add(record.edge_id)
    for index, record in enumerate(records, 1):
        if isinstance(record, GraphEdge):
            for field in ("source_id", "target_id"):
                node = nodes.get(getattr(record, field))
                if node is None or node.layer != record.layer or node.document_id != record.document_id:
                    raise ContractError(f"record {index} edge {record.edge_id!r}: {field} must reference a node in the same document and layer")


def write_contract_jsonl(records: Iterable[EntityPrediction | RelationCandidate | GraphNode | GraphEdge], path: str | Path) -> None:
    """Write one homogeneous contract or a mixed v2 graph; preserve record order."""
    items = list(records)
    for index, item in enumerate(items, 1):
        if not isinstance(item, (EntityPrediction, RelationCandidate, GraphNode, GraphEdge)):
            raise ContractError(f"record {index}: record type is an unsupported contract")
    kinds = {"graph" if isinstance(item, (GraphNode, GraphEdge)) else type(item).__name__ for item in items}
    if len(kinds) > 1:
        raise ContractError("records must contain one contract kind")
    if kinds == {"graph"}:
        _validate_graph(items)
    seen: set[str] = set()
    serialized = []
    for index, item in enumerate(items, 1):
        try:
            payload = item.to_dict()
            type(item).from_dict(payload)
            if isinstance(item, (EntityPrediction, RelationCandidate)):
                field = "prediction_id" if isinstance(item, EntityPrediction) else "candidate_id"
                identifier = getattr(item, field)
                if identifier in seen:
                    raise ContractError(f"duplicate {field} {identifier!r}")
                seen.add(identifier)
            serialized.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
        except (ValueError, TypeError) as error:
            raise ContractError(f"record {index}: {error}") from error
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(serialized) + ("\n" if serialized else ""), encoding="utf-8")


def load_contract_jsonl(path: str | Path, kind: str) -> list[EntityPrediction | RelationCandidate | GraphNode | GraphEdge]:
    """Read records with strict schema and line-specific errors."""
    if kind not in ("entity", "candidate", "graph"):
        raise ContractError("kind must be entity, candidate or graph")
    records = []
    seen: set[str] = set()
    contents = Path(path).read_text(encoding="utf-8")
    lines = contents.split("\n")
    if lines[-1] == "":
        lines.pop()
    for line_number, line in enumerate(lines, 1):
        try:
            payload = json.loads(line)
            if kind == "entity":
                record = EntityPrediction.from_dict(payload)
            elif kind == "candidate":
                record = RelationCandidate.from_dict(payload)
            elif isinstance(payload, Mapping) and payload.get("record_type") == "node":
                record = GraphNode.from_dict(payload)
            elif isinstance(payload, Mapping) and payload.get("record_type") == "edge":
                record = GraphEdge.from_dict(payload)
            else:
                raise ContractError("record_type must be node or edge")
            if isinstance(record, (EntityPrediction, RelationCandidate)):
                field = "prediction_id" if isinstance(record, EntityPrediction) else "candidate_id"
                identifier = getattr(record, field)
                if identifier in seen:
                    raise ContractError(f"duplicate {field} {identifier!r}")
                seen.add(identifier)
            records.append(record)
        except (ValueError, TypeError) as error:
            raise ContractError(f"JSONL line {line_number}: {error}") from error
    if kind == "graph":
        _validate_graph(records)
    return records
