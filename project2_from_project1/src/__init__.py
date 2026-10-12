"""Independent components for the clinical-relation extraction project."""

from project2_from_project1.src.clinical_documents import (
    AnnotatedRelation,
    CanonicalRelation,
    ClinicalCase,
    ClinicalDocument,
    ClinicalDocumentValidationError,
    Mention,
    Patient,
    Span,
    load_jsonl,
    write_jsonl,
)

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
