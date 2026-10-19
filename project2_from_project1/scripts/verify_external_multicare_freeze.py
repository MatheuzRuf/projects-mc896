"""Verify the frozen external test immediately before any model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from project2_from_project1.scripts.freeze_external_multicare import _code_digest, choose_prefix
from project2_from_project1.scripts.freeze_external_multicare import validate_preparation_manifest
from project2_from_project1.src.clinical_documents import load_jsonl


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(freeze_dir: Path) -> dict[str, object]:
    manifest = json.loads((freeze_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_first_external_model_run" or manifest.get("schema") != "flat-gold-v1":
        raise ValueError("external test is not frozen under flat-gold-v1")
    repository_root = Path(__file__).resolve().parents[2]
    for name, expected in manifest["input_sha256"].items():
        path = (repository_root / manifest["input_paths"][name]).resolve()
        if not path.is_relative_to(repository_root) or _digest(path) != expected:
            raise ValueError(f"frozen input changed: {name}")
    assembly_manifest_path = (repository_root / manifest["input_paths"]["assembly_manifest"]).resolve()
    assembly_manifest = json.loads(assembly_manifest_path.read_text(encoding="utf-8"))
    for name, expected in assembly_manifest["output_hashes"].items():
        if _digest(assembly_manifest_path.parent / name) != expected:
            raise ValueError(f"adjudication output changed: {name}")
    if assembly_manifest["counts"]["candidate_documents"] != manifest["adjudicated_candidate_prefix_documents"]:
        raise ValueError("adjudicated candidate prefix count changed")
    validate_preparation_manifest(
        (repository_root / manifest["input_paths"]["preparation_manifest"]).resolve(),
        (repository_root / manifest["input_paths"]["selection"]).resolve(),
        (repository_root / manifest["input_paths"]["leakage_report"]).resolve(),
        (repository_root / manifest["input_paths"]["packet"]).resolve(),
    )
    project_root = Path(__file__).resolve().parents[1]
    code_paths = {
        "preparation": "scripts/prepare_external_multicare_continuation.py",
        "leakage_audit": "src/leakage_audit.py",
        "review_validation": "scripts/validate_external_review.py",
        "adjudication": "scripts/assemble_external_multicare.py",
        "freeze": "scripts/freeze_external_multicare.py",
        "freeze_verification": "scripts/verify_external_multicare_freeze.py",
    }
    for name, relative in code_paths.items():
        if _code_digest(project_root / relative) != manifest["code_sha256"][name]:
            raise ValueError(f"frozen code changed: {name}")
    for name, expected in manifest["output_sha256"].items():
        if _digest(freeze_dir / name) != expected:
            raise ValueError(f"frozen output changed: {name}")
    documents = load_jsonl(freeze_dir / "external-gold.jsonl")
    chosen, support = choose_prefix(documents)
    if len(documents) != manifest["documents"] or len(chosen) != len(documents) or support != manifest["support"]:
        raise ValueError("frozen support or stop decision changed")
    ids = json.loads((freeze_dir / "selected-ids.json").read_text(encoding="utf-8"))
    expected_ids = [{"document_id": doc.document_id,
                     "text_sha256": hashlib.sha256(doc.text.encode()).hexdigest()} for doc in documents]
    if ids != expected_ids or len({doc.document_id for doc in documents}) != len(documents):
        raise ValueError("frozen IDs or text hashes changed")
    selection = json.loads((repository_root / manifest["input_paths"]["selection"]).read_text(encoding="utf-8"))
    excluded = {row["document_id"] for row in manifest["source_exclusions"]}
    expected_order = [row["document_id"] for row in selection
                      if row["decision"] == "released" and row["document_id"] not in excluded]
    if [doc.document_id for doc in documents] != expected_order[:len(documents)]:
        raise ValueError("frozen gold differs from eligible source order")
    if (any(not doc.metadata.get("article_id") for doc in documents)
            or len({doc.metadata["article_id"] for doc in documents}) != len(documents)):
        raise ValueError("frozen test contains repeated source articles")
    return {"status": "verified_frozen_external_test", "documents": len(documents), "support": support}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.freeze_dir), sort_keys=True))


if __name__ == "__main__":
    main()
