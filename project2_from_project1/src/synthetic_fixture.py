"""Synthetic clinical-like workspace that exercises the full public study.

The generated reports contain no real clinical text.  Every mention is
token-aligned, contiguous and non-overlapping, so the flat gold projection
needs no adjudication.  The workspace mirrors the real inputs: a lossless
corpus JSONL, an (empty) adjudication file and a frozen supervised dataset
with document folds and a held-out test partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from project2_from_project1.src.clinical_documents import (
    AnnotatedRelation,
    ClinicalCase,
    ClinicalDocument,
    Mention,
    Patient,
    Span,
    write_jsonl,
)
from project2_from_project1.src.mcrc_suitability import MCRCSuitabilityReport, SuitabilityCriteria, SuitabilityDecision
from project2_from_project1.src.run_manifest import sha256_file
from project2_from_project1.src.supervised_dataset import (
    PartitionPolicy,
    build_frozen_supervised_dataset,
    write_frozen_supervised_dataset,
)

FINDINGS = ("fever", "cough", "chest pain", "headache", "rash", "fatigue", "nausea", "dyspnea", "edema", "back pain")
CONDITIONS = ("asthma", "pneumonia", "migraine", "anemia", "gastritis")
FACTORS = ("smoker", "diabetic", "pregnant", "obese")
MODIFIERS = ("no", "denied")


@dataclass(frozen=True)
class SyntheticWorkspace:
    corpus_jsonl: Path
    adjudication_jsonl: Path
    frozen_dataset: Path


def _document(index: int, rng: random.Random) -> ClinicalDocument:
    document_id = f"synthetic:{index:03d}"
    findings = rng.sample(FINDINGS, 4)
    condition = rng.choice(CONDITIONS)
    factor = rng.choice(FACTORS)
    modifier = rng.choice(MODIFIERS)
    pieces: list[tuple[str, str | None, str | None]] = [
        ("The patient is a ", None, None), (factor, "Factor", "factor"), (" adult. ", None, None),
        ("She reported ", None, None), (findings[0], "Finding", "has"), (" and ", None, None),
        (findings[1], "Finding", "has"), (". ", None, None),
        (modifier.capitalize(), "Modifier", "modifier"), (" ", None, None), (findings[2], "Finding", "negated"),
        (" was observed. ", None, None),
        (condition.capitalize(), "Condition", "condition"), (" caused ", None, None),
        (findings[3], "Finding", "caused"), (". ", None, None),
        ("The weather was mild today.", None, None),
    ]
    text, mentions, roles = "", [], {}
    for surface, kind, role in pieces:
        if kind is not None:
            mention_id = f"{document_id}:mention:{len(mentions)}"
            mentions.append(Mention(mention_id, (Span(len(text), len(text) + len(surface)),), (kind.lower(),), kind))
            roles.setdefault(role, []).append(mention_id)
        text += surface
    case_id, patient_id = f"{document_id}:case", f"{document_id}:patient"
    relations = [AnnotatedRelation(f"{document_id}:describes", case_id, patient_id, "describes", "DESCRIBES")]

    def add(source: str, target: str, label: str) -> None:
        relations.append(AnnotatedRelation(f"{document_id}:relation:{len(relations)}", source, target, label.lower(), label))

    add(patient_id, roles["factor"][0], "HAS_FACTOR")
    for target in roles["has"] + roles["caused"]:
        add(patient_id, target, "HAS_FINDING")
    add(patient_id, roles["condition"][0], "HAS_CONDITION")
    add(roles["modifier"][0], roles["negated"][0], "NEGATES")
    add(roles["condition"][0], roles["caused"][0], "CAUSES")
    return ClinicalDocument(document_id, text, (ClinicalCase(case_id),), (Patient(patient_id),), tuple(mentions), tuple(relations))


def make_synthetic_workspace(directory: str | Path, *, documents: int = 12, seed: int = 7, folds: int = 2) -> SyntheticWorkspace:
    """Write a deterministic synthetic corpus, empty adjudication and frozen folds."""
    root = Path(directory)
    rng = random.Random(seed)
    corpus = [_document(index, rng) for index in range(documents)]
    corpus_path = root / "corpus.jsonl"
    write_jsonl(corpus, corpus_path)
    adjudication = root / "adjudication.jsonl"
    adjudication.write_text("", encoding="utf-8")
    ids = tuple(sorted(document.document_id for document in corpus))
    report = MCRCSuitabilityReport(SuitabilityDecision.APPROVED, SuitabilityCriteria(), {}, {},
                                   ["synthetic fixture"], ids, "freeze")
    dataset = build_frozen_supervised_dataset(corpus, report, partition_policy=PartitionPolicy(seed=seed, folds=folds),
                                              source_jsonl_sha256=sha256_file(corpus_path))
    frozen = root / "frozen-dataset"
    write_frozen_supervised_dataset(dataset, frozen)
    return SyntheticWorkspace(corpus_path, adjudication, frozen)


__all__ = ["SyntheticWorkspace", "make_synthetic_workspace"]
