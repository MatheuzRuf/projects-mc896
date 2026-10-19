"""Snapshot the next MultiCaRe v1 candidates after the fixed local sample."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from project2_from_project1.scripts.prepare_external_multicare import _shingles
from project2_from_project1.src.clinical_documents import ClinicalCase, ClinicalDocument, Patient, load_jsonl
from project2_from_project1.src.leakage_audit import (
    AuditDecision, audit_external_corpus, build_protected_collection, normalise_text,
    write_leakage_report,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def prepare(cases_parquet: Path, metadata_parquet: Path, local_selection: Path,
            local_packet: Path, mcrc_path: Path, output_dir: Path,
            *, released_limit: int = 100) -> dict[str, object]:
    if released_limit < 1 or output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("released_limit must be positive and output directory empty")
    import duckdb  # optional dependency required only for the parquet source

    sources = {name: _digest(path) for name, path in (
        ("cases_parquet", cases_parquet), ("metadata_parquet", metadata_parquet),
        ("local_selection", local_selection), ("local_packet", local_packet),
        ("mcrc_jsonl", mcrc_path),
    )}
    local_articles = {row["article_id"] for row in json.loads(local_selection.read_text(encoding="utf-8"))}
    metadata_rows = duckdb.sql(
        "SELECT article_id, article_metadata FROM read_parquet(?)", params=[str(metadata_parquet)]
    ).fetchall()
    metadata = {article_id: value for article_id, value in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError("duplicate article_id in MultiCaRe metadata")
    protected = build_protected_collection(load_jsonl(mcrc_path))
    local_documents = load_jsonl(local_packet)
    local_texts = [document.text for document in local_documents]
    seen_shingles = [(document.document_id, _shingles(document.text)) for document in local_documents]
    seen_hashes = {
        hashlib.sha256(normalise_text(document.text).encode()).hexdigest(): document.document_id
        for document in local_documents
    }
    records: list[dict[str, object]] = []
    selected: list[ClinicalDocument] = []
    audited: list[ClinicalDocument] = []
    query = "SELECT c.case_id,c.case_text,t.article_id FROM read_parquet(?) t, UNNEST(t.cases) AS u(c) ORDER BY c.case_id"
    cursor = duckdb.sql(query, params=[str(cases_parquet)])
    while row := cursor.fetchone():
        case_id, text, article_id = row
        article = metadata.get(article_id)
        if article_id in local_articles or not case_id.endswith("_01") or not article:
            continue
        if article["license"] not in {"CC BY", "CC0"} or not text or not 300 <= len(text) <= 5000:
            continue
        document = ClinicalDocument(
            document_id=case_id, text=text,
            cases=[ClinicalCase(case_id=f"{case_id}:case", source_id=article_id)],
            patients=[Patient(patient_id=f"{case_id}:patient")], mentions=[], relations=[],
            metadata={"article_id": article_id, "pmcid": article.get("pmcid"),
                      "pmid": article.get("pmid"), "doi": article.get("doi"),
                      "license": article["license"], "title": article.get("title")},
        )
        audit = audit_external_corpus([document], protected)
        audited.append(document)
        decision = audit.records[0].decision
        overlaps = []
        if decision is AuditDecision.RELEASED:
            fingerprint = hashlib.sha256(normalise_text(text).encode()).hexdigest()
            shingles = _shingles(text)
            if fingerprint in seen_hashes:
                decision = AuditDecision.EXCLUDED
                overlaps.append({"reason": "exact_text",
                                 "earlier_document_id": seen_hashes[fingerprint]})
            else:
                for earlier_document_id, earlier in seen_shingles:
                    shared = len(shingles & earlier)
                    if shared >= 3 and shared / min(len(shingles), len(earlier)) >= 0.85:
                        overlaps.append({"reason": "text_similarity_quarantine",
                                         "earlier_document_id": earlier_document_id})
                        decision = AuditDecision.QUARANTINED
                        break
            if decision is AuditDecision.RELEASED:
                selected.append(document)
                seen_hashes[fingerprint] = case_id
                seen_shingles.append((case_id, shingles))
        records.append({"document_id": case_id, "article_id": article_id,
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "license": article["license"], "decision": decision.value,
                        "against_mcrc": audit.records[0].decision.value,
                        "candidate_overlaps": overlaps})
        if len(selected) >= released_limit:
            break
    if len(selected) < released_limit:
        raise RuntimeError(f"only {len(selected)} released candidates found")
    if sources != {name: _digest(path) for name, path in (
        ("cases_parquet", cases_parquet), ("metadata_parquet", metadata_parquet),
        ("local_selection", local_selection), ("local_packet", local_packet),
        ("mcrc_jsonl", mcrc_path),
    )}:
        raise RuntimeError("source changed during preparation")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "selection.json", records)
    write_leakage_report(audit_external_corpus(audited, protected), output_dir / "mcrc-leakage.json")
    with (output_dir / "annotation-packet.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for document in selected:
            stream.write(json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"status": "continuation_packet_not_gold_not_frozen",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "preparation_code_sha256": _code_digest(Path(__file__)),
                "source_hashes": sources, "counts": dict(Counter(row["decision"] for row in records)),
                "rules": {"order": "case_id lexical after local 56", "license": ["CC BY", "CC0"],
                          "case_suffix": "_01", "text_length": [300, 5000],
                          "similarity_threshold": 0.85, "released_limit": released_limit},
                "output_hashes": {name: _digest(output_dir / name) for name in (
                    "selection.json", "mcrc-leakage.json", "annotation-packet.jsonl")}}
    _write(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases-parquet", "metadata-parquet", "local-selection", "local-packet", "mcrc", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--released-limit", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(prepare(args.cases_parquet, args.metadata_parquet, args.local_selection,
                             args.local_packet, args.mcrc, args.output_dir,
                             released_limit=args.released_limit), sort_keys=True))


if __name__ == "__main__":
    main()
