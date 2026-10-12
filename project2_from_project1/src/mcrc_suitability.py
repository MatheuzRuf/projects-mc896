"""Reproducible suitability decisions for a converted MCRC corpus.

This module is deliberately downstream of conversion and leakage auditing.  It
never opens the original MCRC pickle and it never treats an audit result as an
implicit permission to train.  A successful decision is the only path that
exposes document ids to the future partitioning step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from project2_from_project1.src.clinical_documents import CanonicalRelation, ClinicalDocument, load_jsonl
from project2_from_project1.src.leakage_audit import AuditDecision, LeakageAuditReport, load_leakage_report


class MCRCSuitabilityError(ValueError):
    """Raised when inputs cannot support an auditable suitability decision."""


class SuitabilityDecision(str, Enum):
    """The explicit, terminal decision for this candidate corpus."""

    APPROVED = "approved"
    REJECTED = "rejected"


_SUPERVISED_RELATIONS = (
    CanonicalRelation.HAS_CONDITION,
    CanonicalRelation.HAS_FINDING,
    CanonicalRelation.HAS_FACTOR,
    CanonicalRelation.NEGATES,
    CanonicalRelation.CAUSES,
)


@dataclass(frozen=True)
class SuitabilityCriteria:
    """Documented thresholds used to decide whether the released corpus is usable.

    A single observed instance is the conservative default for canonical
    coverage.  It proves the class is represented without turning ordinary
    MCRC class imbalance into an automatic rejection.  Callers may set higher
    thresholds for an experiment that needs them.
    """

    minimum_released_documents: int = 1
    minimum_released_supervised_relations: int = 1
    minimum_relations_per_required_canonical: int = 1
    required_canonical_relations: Sequence[CanonicalRelation | str] = _SUPERVISED_RELATIONS
    maximum_blocked_document_fraction: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_released_documents",
            "minimum_released_supervised_relations",
            "minimum_relations_per_required_canonical",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MCRCSuitabilityError(f"{field_name} must be a non-negative integer")
        try:
            required = tuple(CanonicalRelation(label) for label in self.required_canonical_relations)
        except (TypeError, ValueError) as error:
            raise MCRCSuitabilityError("required_canonical_relations contains an unsupported relation") from error
        if len(set(required)) != len(required):
            raise MCRCSuitabilityError("required_canonical_relations must not contain duplicates")
        unsupported = set(required) - set(_SUPERVISED_RELATIONS)
        if unsupported:
            names = ", ".join(sorted(label.value for label in unsupported))
            raise MCRCSuitabilityError(f"required_canonical_relations must be supervised positive classes, not {names}")
        object.__setattr__(self, "required_canonical_relations", required)
        if self.maximum_blocked_document_fraction is not None:
            fraction = self.maximum_blocked_document_fraction
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
                raise MCRCSuitabilityError("maximum_blocked_document_fraction must be in [0, 1] or None")
            object.__setattr__(self, "maximum_blocked_document_fraction", float(fraction))

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_blocked_document_fraction": self.maximum_blocked_document_fraction,
            "minimum_released_documents": self.minimum_released_documents,
            "minimum_released_supervised_relations": self.minimum_released_supervised_relations,
            "minimum_relations_per_required_canonical": self.minimum_relations_per_required_canonical,
            "required_canonical_relations": [label.value for label in self.required_canonical_relations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SuitabilityCriteria":
        if not isinstance(value, Mapping):
            raise MCRCSuitabilityError("criteria must be an object")
        try:
            return cls(
                minimum_released_documents=value["minimum_released_documents"],
                minimum_released_supervised_relations=value["minimum_released_supervised_relations"],
                minimum_relations_per_required_canonical=value["minimum_relations_per_required_canonical"],
                required_canonical_relations=value["required_canonical_relations"],
                maximum_blocked_document_fraction=value.get("maximum_blocked_document_fraction"),
            )
        except KeyError as error:
            raise MCRCSuitabilityError(f"criteria is missing {error.args[0]}") from error


def _relation_support(documents: Sequence[ClinicalDocument]) -> dict[str, Any]:
    native: dict[str, int] = {}
    canonical = {label.value: 0 for label in CanonicalRelation}
    for document in documents:
        for relation in document.relations:
            native[relation.native_label] = native.get(relation.native_label, 0) + 1
            canonical[relation.canonical_label.value] += 1
    return {
        "canonical_relation_support": dict(sorted(canonical.items())),
        "native_relation_support": dict(sorted(native.items())),
    }


def _population_summary(documents: Sequence[ClinicalDocument]) -> dict[str, Any]:
    support = _relation_support(documents)
    return {
        "documents": len(documents),
        "mentions": sum(len(document.mentions) for document in documents),
        "relations": sum(len(document.relations) for document in documents),
        "supervised_relations": sum(
            1
            for document in documents
            for relation in document.relations
            if relation.canonical_label in _SUPERVISED_RELATIONS
        ),
        **support,
    }


def _validate_documents(documents: Iterable[ClinicalDocument]) -> tuple[ClinicalDocument, ...]:
    result = tuple(documents)
    document_ids: set[str] = set()
    for document in result:
        if not isinstance(document, ClinicalDocument):
            raise TypeError("assess_mcrc_suitability accepts ClinicalDocument instances")
        # Reconstructing from the public representation reruns every common
        # contract validator, including relation endpoint and negation rules.
        ClinicalDocument.from_dict(document.to_dict())
        if document.document_id in document_ids:
            raise MCRCSuitabilityError(f"candidate corpus has duplicate document id {document.document_id!r}")
        document_ids.add(document.document_id)
    return result


def _validate_audit_coverage(documents: Sequence[ClinicalDocument], audit_report: LeakageAuditReport) -> None:
    if not isinstance(audit_report, LeakageAuditReport):
        raise TypeError("audit_report must be a LeakageAuditReport")
    corpus_ids = {document.document_id for document in documents}
    audit_ids = {record.document_id for record in audit_report.records}
    if corpus_ids != audit_ids:
        missing = sorted(corpus_ids - audit_ids)
        unexpected = sorted(audit_ids - corpus_ids)
        details = []
        if missing:
            details.append("missing audit records: " + ", ".join(missing))
        if unexpected:
            details.append("audit records outside candidate corpus: " + ", ".join(unexpected))
        raise MCRCSuitabilityError("audit report must cover exactly the converted corpus (" + "; ".join(details) + ")")


@dataclass(frozen=True)
class MCRCSuitabilityReport:
    """Serializable evidence and decision; it contains no clinical source text."""

    decision: SuitabilityDecision | str
    criteria: SuitabilityCriteria
    populations: Mapping[str, Mapping[str, Any]]
    audit: Mapping[str, Any]
    reasons: Sequence[str]
    eligible_document_ids: Sequence[str]
    next_action: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        try:
            decision = SuitabilityDecision(self.decision)
        except (TypeError, ValueError) as error:
            raise MCRCSuitabilityError("decision must be approved or rejected") from error
        if not isinstance(self.criteria, SuitabilityCriteria):
            raise TypeError("criteria must be a SuitabilityCriteria")
        if self.schema_version != 1:
            raise MCRCSuitabilityError("unsupported MCRC suitability report schema_version")
        reasons = tuple(self.reasons)
        if not reasons or any(not isinstance(reason, str) or not reason for reason in reasons):
            raise MCRCSuitabilityError("reasons must contain at least one non-empty string")
        ids = tuple(self.eligible_document_ids)
        if len(set(ids)) != len(ids) or any(not isinstance(item, str) or not item for item in ids):
            raise MCRCSuitabilityError("eligible_document_ids must be unique non-empty strings")
        if decision is SuitabilityDecision.REJECTED and ids:
            raise MCRCSuitabilityError("a rejected corpus cannot expose documents for partitioning")
        if not isinstance(self.next_action, str) or not self.next_action:
            raise MCRCSuitabilityError("next_action must be a non-empty string")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "populations", dict(self.populations))
        object.__setattr__(self, "audit", dict(self.audit))
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "eligible_document_ids", ids)

    @property
    def approved(self) -> bool:
        return self.decision is SuitabilityDecision.APPROVED

    @property
    def documents_for_partitioning(self) -> tuple[str, ...]:
        """Only the approved, explicitly released documents may be partitioned."""

        return self.eligible_document_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit": dict(self.audit),
            "criteria": self.criteria.to_dict(),
            "decision": self.decision.value,
            "eligible_document_ids": list(self.eligible_document_ids),
            "next_action": self.next_action,
            "populations": {name: dict(summary) for name, summary in sorted(self.populations.items())},
            "reasons": list(self.reasons),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MCRCSuitabilityReport":
        if not isinstance(value, Mapping):
            raise MCRCSuitabilityError("MCRC suitability report must be an object")
        try:
            return cls(
                decision=value["decision"],
                criteria=SuitabilityCriteria.from_dict(value["criteria"]),
                populations=value["populations"],
                audit=value["audit"],
                reasons=value["reasons"],
                eligible_document_ids=value["eligible_document_ids"],
                next_action=value["next_action"],
                schema_version=value.get("schema_version", 1),
            )
        except KeyError as error:
            raise MCRCSuitabilityError(f"MCRC suitability report is missing {error.args[0]}") from error


def assess_mcrc_suitability(
    documents: Iterable[ClinicalDocument],
    audit_report: LeakageAuditReport,
    *,
    criteria: SuitabilityCriteria | None = None,
) -> MCRCSuitabilityReport:
    """Make a deterministic decision from validated documents and a complete audit.

    Excluded and quarantined documents are counted as evidence but never used
    for coverage or returned for partitioning.  Class imbalance is visible in
    the support tables; only an explicit configured minimum can reject it.
    """

    candidate = _validate_documents(documents)
    _validate_audit_coverage(candidate, audit_report)
    active_criteria = criteria if criteria is not None else SuitabilityCriteria()
    if not isinstance(active_criteria, SuitabilityCriteria):
        raise TypeError("criteria must be a SuitabilityCriteria")

    by_decision = {
        decision: tuple(
            document
            for document in candidate
            if audit_report.decision_for(document.document_id) is decision
        )
        for decision in AuditDecision
    }
    populations = {
        "before_audit": _population_summary(candidate),
        "released": _population_summary(by_decision[AuditDecision.RELEASED]),
        "excluded": _population_summary(by_decision[AuditDecision.EXCLUDED]),
        "quarantined": _population_summary(by_decision[AuditDecision.QUARANTINED]),
    }
    released_summary = populations["released"]
    blocked_count = len(candidate) - len(by_decision[AuditDecision.RELEASED])
    blocked_fraction = blocked_count / len(candidate) if candidate else 0.0
    audit = {
        "blocked_document_fraction": blocked_fraction,
        "counts": {
            decision.value: len(by_decision[decision])
            for decision in AuditDecision
        },
        "excluded_document_ids": sorted(document.document_id for document in by_decision[AuditDecision.EXCLUDED]),
        "quarantined_document_ids": sorted(document.document_id for document in by_decision[AuditDecision.QUARANTINED]),
        "similarity_threshold": audit_report.similarity_threshold,
    }

    failures: list[str] = []
    if released_summary["documents"] < active_criteria.minimum_released_documents:
        failures.append(
            "released documents below minimum: "
            f"{released_summary['documents']} < {active_criteria.minimum_released_documents}"
        )
    if released_summary["supervised_relations"] < active_criteria.minimum_released_supervised_relations:
        failures.append(
            "released supervised relations below minimum: "
            f"{released_summary['supervised_relations']} < {active_criteria.minimum_released_supervised_relations}"
        )
    canonical_support = released_summary["canonical_relation_support"]
    for label in active_criteria.required_canonical_relations:
        support = canonical_support[label.value]
        if support < active_criteria.minimum_relations_per_required_canonical:
            failures.append(
                f"released {label.value} support below minimum: "
                f"{support} < {active_criteria.minimum_relations_per_required_canonical}"
            )
    if (
        active_criteria.maximum_blocked_document_fraction is not None
        and blocked_fraction > active_criteria.maximum_blocked_document_fraction
    ):
        failures.append(
            "blocked document fraction above maximum: "
            f"{blocked_fraction:.6f} > {active_criteria.maximum_blocked_document_fraction:.6f}"
        )

    validation_reason = f"common clinical-document contract validated for {len(candidate)} converted documents"
    leakage_reason = (
        "leakage audit removed "
        f"{len(by_decision[AuditDecision.EXCLUDED])} excluded and "
        f"{len(by_decision[AuditDecision.QUARANTINED])} quarantined documents; "
        f"{len(by_decision[AuditDecision.RELEASED])} remain released"
    )
    if failures:
        return MCRCSuitabilityReport(
            decision=SuitabilityDecision.REJECTED,
            criteria=active_criteria,
            populations=populations,
            audit=audit,
            reasons=(validation_reason, leakage_reason, *failures),
            eligible_document_ids=(),
            next_action=(
                "Do not produce a training split. Obtain authorised i2b2/n2c2 access "
                "or another corpus, then run the same conversion and leakage audit."
            ),
        )

    released_ids = tuple(sorted(document.document_id for document in by_decision[AuditDecision.RELEASED]))
    return MCRCSuitabilityReport(
        decision=SuitabilityDecision.APPROVED,
        criteria=active_criteria,
        populations=populations,
        audit=audit,
        reasons=(
            validation_reason,
            leakage_reason,
            "released corpus meets configured document, relation, and canonical coverage criteria; "
            "class imbalance is reported in support tables and is not itself a rejection reason",
        ),
        eligible_document_ids=released_ids,
        next_action="Freeze partitions using only eligible_document_ids from this report.",
    )


def assess_mcrc_suitability_from_files(
    converted_jsonl_path: str | Path,
    leakage_report_path: str | Path,
    *,
    criteria: SuitabilityCriteria | None = None,
) -> MCRCSuitabilityReport:
    """Load the safe JSONL and audit report, then make the decision."""

    return assess_mcrc_suitability(
        load_jsonl(converted_jsonl_path),
        load_leakage_report(leakage_report_path),
        criteria=criteria,
    )


def write_mcrc_suitability_report(report: MCRCSuitabilityReport, path: str | Path) -> None:
    """Write deterministic, text-free decision evidence."""

    if not isinstance(report, MCRCSuitabilityReport):
        raise TypeError("report must be an MCRCSuitabilityReport")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_mcrc_suitability_report(path: str | Path) -> MCRCSuitabilityReport:
    """Load a previously written suitability report."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MCRCSuitabilityError(f"cannot read MCRC suitability report: {error}") from error
    return MCRCSuitabilityReport.from_dict(payload)


__all__ = [
    "MCRCSuitabilityError",
    "MCRCSuitabilityReport",
    "SuitabilityCriteria",
    "SuitabilityDecision",
    "assess_mcrc_suitability",
    "assess_mcrc_suitability_from_files",
    "load_mcrc_suitability_report",
    "write_mcrc_suitability_report",
]
