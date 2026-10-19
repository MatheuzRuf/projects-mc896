"""Create a text-only view of an adjudicated candidate prefix without reading gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project2_from_project1.src.clinical_documents import ClinicalDocument, load_jsonl, write_jsonl
from project2_from_project1.src.run_manifest import sha256_file


def prepare(packet: Path, eligibility: Path, assembly_manifest: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("output directory must not exist")
    assembly = json.loads(assembly_manifest.read_text(encoding="utf-8"))
    if assembly.get("status") != "provisional_not_frozen":
        raise ValueError("expected a provisional adjudication manifest")
    if sha256_file(packet) != assembly["input_hashes"]["packet"]:
        raise ValueError("assembly packet hash differs")
    screening = json.loads(eligibility.read_text(encoding="utf-8"))
    prefix = assembly["counts"]["candidate_documents"]
    if screening["screened_prefix_document_count"] < prefix:
        raise ValueError("eligibility screening does not cover the assembled prefix")
    excluded = {item["document_id"] for item in screening["exclusions"]}
    packet_docs = load_jsonl(packet)
    if len(packet_docs) < prefix or any(item.mentions or item.relations for item in packet_docs):
        raise ValueError("packet is incomplete or contains annotations")
    documents = [ClinicalDocument(item.document_id, item.text, (), (), (), (), {})
                 for item in packet_docs[:prefix] if item.document_id not in excluded]
    if len(documents) != assembly["counts"]["documents"]:
        raise ValueError("eligible document count differs from adjudication manifest")
    output.mkdir(parents=True)
    write_jsonl(documents, output / "raw.jsonl")
    manifest = {"status": "provisional_snapshot_user_requested", "documents": len(documents),
                "candidate_prefix": prefix, "raw_sha256": sha256_file(output / "raw.jsonl"),
                "input_sha256": {"packet": sha256_file(packet), "eligibility": sha256_file(eligibility),
                                 "assembly_manifest": sha256_file(assembly_manifest)},
                "assembly_input_sha256": {"packet": assembly["input_hashes"]["packet"],
                                          "eligibility": assembly["input_hashes"]["eligibility"]},
                "eligibility_note": (
                    "Eligibility register differs from the one hashed by the provisional assembly; "
                    "its screened prefix covers this snapshot and reproduces the assembly's eligible count."
                    if sha256_file(eligibility) != assembly["input_hashes"]["eligibility"] else None
                ),
                "code_sha256": sha256_file(__file__)}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "eligibility", "assembly-manifest", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.packet, args.eligibility, args.assembly_manifest, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
