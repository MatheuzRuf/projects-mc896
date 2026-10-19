"""Compare two sealed reviews without exposing either review to the other."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from project2_from_project1.scripts.validate_external_review import validate
from project2_from_project1.src.clinical_documents import load_jsonl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(document: object) -> tuple[set[tuple[object, ...]], set[tuple[object, ...]]]:
    mentions = {mention.mention_id: (mention.spans[0].start, mention.spans[0].end, mention.canonical_type)
                for mention in document.mentions}
    patients = {patient.patient_id for patient in document.patients}
    entities = set(mentions.values())
    relations = set()
    for relation in document.relations:
        source = ("Patient",) if relation.source_id in patients else mentions[relation.source_id]
        target = ("Patient",) if relation.target_id in patients else mentions[relation.target_id]
        relations.add((source, target, relation.canonical_label.value))
    return entities, relations


def compare(packet: Path, guide: Path, review_a: Path, provenance_a: Path,
            review_b: Path, provenance_b: Path) -> dict[str, object]:
    summary_a = validate(packet, review_a, guide, provenance_a)
    summary_b = validate(packet, review_b, guide, provenance_b)
    if summary_a["reviewer_id"] == summary_b["reviewer_id"]:
        raise ValueError("independent reviews need distinct reviewer IDs")
    documents_a = {item.document_id: item for item in load_jsonl(review_a)}
    documents_b = {item.document_id: item for item in load_jsonl(review_b)}
    if documents_a.keys() != documents_b.keys():
        raise ValueError("review document coverage differs")
    comparisons = []
    for document_id in sorted(documents_a):
        entities_a, relations_a = _decision(documents_a[document_id])
        entities_b, relations_b = _decision(documents_b[document_id])
        comparisons.append({
            "document_id": document_id,
            "entity_agreement": entities_a == entities_b,
            "relation_agreement": relations_a == relations_b,
            "entities_only_a": sorted(entities_a - entities_b),
            "entities_only_b": sorted(entities_b - entities_a),
            "relations_only_a": sorted(relations_a - relations_b, key=str),
            "relations_only_b": sorted(relations_b - relations_a, key=str),
        })
    return {
        "status": "comparison_only_not_adjudicated",
        "input_hashes": {"packet": _sha256(packet), "guide": _sha256(guide),
                         "review_a": _sha256(review_a), "provenance_a": _sha256(provenance_a),
                         "review_b": _sha256(review_b), "provenance_b": _sha256(provenance_b)},
        "reviewer_ids": [summary_a["reviewer_id"], summary_b["reviewer_id"]],
        "documents": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "guide", "review-a", "provenance-a", "review-b", "provenance-b", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.packet, args.guide, args.review_a, args.provenance_a,
                     args.review_b, args.provenance_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"documents": len(result["documents"]),
                      "unanimous": sum(item["entity_agreement"] and item["relation_agreement"]
                                       for item in result["documents"]),
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
