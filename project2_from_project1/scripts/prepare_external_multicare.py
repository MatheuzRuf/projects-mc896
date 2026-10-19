"""Prepare an auditable MultiCaRe candidate snapshot without reading model outputs.

This command does not create gold labels or freeze an external test. It checks
the fixed local source against MCRC and against other local candidates, then
writes a text-free selection report and a private annotation packet.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from project2_from_project1.src.clinical_documents import ClinicalCase, ClinicalDocument, Patient, load_jsonl
from project2_from_project1.src.leakage_audit import (
    AuditDecision,
    audit_external_corpus,
    build_protected_collection,
    normalise_text,
    write_leakage_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _shingles(text: str) -> frozenset[str]:
    tokens = re.findall(r"[^\W_]+", normalise_text(text))
    if len(tokens) < 5:
        return frozenset({" ".join(tokens)})
    return frozenset(" ".join(tokens[index:index + 5]) for index in range(len(tokens) - 4))


def prepare(cases_path: Path, metadata_path: Path, mcrc_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    input_hashes = {name: _sha256(path) for name, path in (
        ("cases_csv", cases_path), ("metadata_csv", metadata_path), ("mcrc_jsonl", mcrc_path)
    )}
    cases = _read_csv(cases_path)
    metadata_rows = _read_csv(metadata_path)
    if not cases or not metadata_rows:
        raise ValueError("cases and metadata must be nonempty")
    metadata = {row["article_id"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError("duplicate article_id in metadata")
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("duplicate case_id in cases")
    documents = []
    for row in sorted(cases, key=lambda item: item["case_id"]):
        case_id, article_id, case_text = row["case_id"], row["article_id"], row["case_text"]
        if not case_id or not article_id or not case_text.strip() or article_id not in metadata:
            raise ValueError(f"case has missing ID, text or article metadata: {case_id!r}")
        article = metadata[article_id]
        documents.append(ClinicalDocument(
            document_id=case_id,
            text=case_text,
            cases=[ClinicalCase(case_id=f"{case_id}:case", source_id=article_id)],
            patients=[Patient(patient_id=f"{case_id}:patient")],
            mentions=[],
            relations=[],
            metadata={"article_id": article_id, "pmcid": article["pmcid"],
                      "pmid": article["pmid"], "doi": article["doi"],
                      "license": article["license"]},
        ))

    mcrc_report = audit_external_corpus(documents, build_protected_collection(load_jsonl(mcrc_path)))
    shingles = [_shingles(document.text) for document in documents]
    normalized_hashes = [hashlib.sha256(normalise_text(document.text).encode("utf-8")).hexdigest()
                         for document in documents]
    overlaps: dict[str, list[dict[str, object]]] = {document.document_id: [] for document in documents}
    for index, left in enumerate(documents):
        for later in range(index + 1, len(documents)):
            right = documents[later]
            if normalized_hashes[index] == normalized_hashes[later]:
                overlaps[right.document_id].append({"with_document_id": left.document_id, "reason": "exact_text"})
                continue
            shared = len(shingles[index] & shingles[later])
            if shared >= 3:
                containment = shared / min(len(shingles[index]), len(shingles[later]))
                if containment >= 0.85:
                    overlaps[right.document_id].append({
                        "with_document_id": left.document_id,
                        "reason": "text_similarity_quarantine", "containment": round(containment, 6),
                    })

    decisions = {record.document_id: record.decision for record in mcrc_report.records}
    selection = []
    for document in documents:
        decision = decisions[document.document_id]
        if decision is AuditDecision.RELEASED and overlaps[document.document_id]:
            decision = AuditDecision.EXCLUDED if any(
                item["reason"] == "exact_text" for item in overlaps[document.document_id]
            ) else AuditDecision.QUARANTINED
        selection.append({
            "document_id": document.document_id,
            "article_id": document.metadata["article_id"],
            "text_sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
            "decision": decision.value,
            "against_mcrc": decisions[document.document_id].value,
            "candidate_overlaps": overlaps[document.document_id],
            "license": document.metadata["license"],
        })

    if input_hashes != {name: _sha256(path) for name, path in (
        ("cases_csv", cases_path), ("metadata_csv", metadata_path), ("mcrc_jsonl", mcrc_path)
    )}:
        raise RuntimeError("input changed while preparing candidate snapshot")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_leakage_report(mcrc_report, output_dir / "mcrc-leakage.json")
    _write_json(output_dir / "selection.json", selection)
    eligible = {item["document_id"] for item in selection if item["decision"] == "released"}
    packet_path = output_dir / "annotation-packet.jsonl"
    with packet_path.open("w", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            if document.document_id in eligible:
                stream.write(json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "annotation_packet_not_gold_not_frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": input_hashes,
        "rules": {"order": "case_id lexical", "document_unit": "one cases.csv row",
                  "mcrc_similarity_threshold": 0.85, "local_containment_threshold": 0.85,
                  "local_shingle_width": 5},
        "counts": dict(Counter(item["decision"] for item in selection)),
        "output_hashes": {name: _sha256(output_dir / name) for name in
                          ("mcrc-leakage.json", "selection.json", "annotation-packet.jsonl")},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mcrc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.cases, args.metadata, args.mcrc, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
