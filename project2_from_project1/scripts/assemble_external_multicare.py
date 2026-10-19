"""Assemble provisional external gold from three separately sealed reviews."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from project2_from_project1.scripts.compare_external_reviews import _decision
from project2_from_project1.scripts.validate_external_review import ENTITY_TYPES, RELATION_TYPES, validate
from project2_from_project1.src.clinical_documents import AnnotatedRelation, ClinicalDocument, Mention, Span, load_jsonl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _adjudicate(votes: tuple[ClinicalDocument, ClinicalDocument, ClinicalDocument]) -> tuple[ClinicalDocument, dict[str, object]]:
    """Preserve A/B agreements and use blind C only on disputed individual items."""
    decisions = [_decision(document) for document in votes]
    accepted: list[set[tuple[object, ...]]] = []
    disagreements: list[list[dict[str, object]]] = []
    for part in (0, 1):
        a, b, c = (decision[part] for decision in decisions)
        universe = a | b | c
        accepted.append({key for key in universe if sum(key in vote for vote in (a, b, c)) >= 2})
        disagreements.append([
            {"key": key, "a": key in a, "b": key in b,
             "third_vote": key in c, "accepted": key in accepted[-1]}
            for key in sorted(universe, key=str) if (key in a) != (key in b)
        ])
    original = votes[0]
    entity_keys = sorted(accepted[0], key=lambda key: (key[0], key[1], key[2]))
    mentions = []
    mention_ids = {}
    for number, (start, end, canonical_type) in enumerate(entity_keys, 1):
        mention_id = f"{original.document_id}:gold:m{number:04d}"
        mention_ids[(start, end, canonical_type)] = mention_id
        mentions.append(Mention(mention_id, [Span(start, end)], [canonical_type], canonical_type))
    patients = {patient.patient_id for patient in original.patients}
    if len(patients) != 1:
        raise ValueError(f"expected one logical patient in {original.document_id}")
    patient_id = next(iter(patients))
    relations = []
    for number, (source, target, label) in enumerate(sorted(accepted[1], key=str), 1):
        if source != ("Patient",) and source not in mention_ids:
            raise ValueError(f"relation source lacks majority mention in {original.document_id}")
        if target != ("Patient",) and target not in mention_ids:
            raise ValueError(f"relation target lacks majority mention in {original.document_id}")
        relations.append(AnnotatedRelation(
            f"{original.document_id}:gold:r{number:04d}",
            patient_id if source == ("Patient",) else mention_ids[source],
            patient_id if target == ("Patient",) else mention_ids[target], label, label,
        ))
    adjudicated = ClinicalDocument(original.document_id, original.text, original.cases,
                                    original.patients, mentions, relations, original.metadata)
    if not adjudicated.mentions:
        raise ValueError(f"adjudication produced no mentions in {original.document_id}")
    return adjudicated, {"document_id": original.document_id, "rule": "itemwise_ab_agreement_else_blind_c",
                         "disputed_entities": disagreements[0], "disputed_relations": disagreements[1],
                         "entities": len(mentions), "relations": len(relations)}


def _eligible_support(documents: list[ClinicalDocument], excluded_ids: set[str]) -> tuple[int, Counter, Counter]:
    eligible = [document for document in documents if document.document_id not in excluded_ids]
    entities = Counter(mention.canonical_type for document in eligible for mention in document.mentions)
    relations = Counter(relation.canonical_label.value for document in eligible for relation in document.relations)
    return len(eligible), entities, relations


def assemble(packet: Path, guide: Path, selection: Path, eligibility: Path,
             reviews: tuple[Path, Path, Path],
             provenances: tuple[Path, Path, Path], output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    summaries = [validate(packet, review, guide, provenance)
                 for review, provenance in zip(reviews, provenances)]
    reviewer_ids = [item["reviewer_id"] for item in summaries]
    if len(set(reviewer_ids)) != 3:
        raise ValueError("all three reviewers must have distinct IDs")
    review_sets = [load_jsonl(path) for path in reviews]
    ids = [[item.document_id for item in items] for items in review_sets]
    common_length = min(map(len, review_sets))
    if not common_length or any(item[:common_length] != ids[0][:common_length] for item in ids[1:]):
        raise ValueError("reviews must share a nonempty ordered prefix")
    selected = json.loads(selection.read_text(encoding="utf-8"))
    eligible_ids = [item["document_id"] for item in selected if item["decision"] == "released"]
    if ids[0][:common_length] != eligible_ids[:common_length]:
        raise ValueError("reviewed documents differ from released selection order")
    decisions = []
    adjudicated = []
    for index, document_id in enumerate(ids[0][:common_length]):
        vote_a, vote_b, vote_c = (items[index] for items in review_sets)
        chosen, decision = _adjudicate((vote_a, vote_b, vote_c))
        adjudicated.append(chosen)
        decisions.append(decision)
    eligibility_record = json.loads(eligibility.read_text(encoding="utf-8"))
    if eligibility_record.get("status") != "complete":
        raise ValueError("source eligibility review is not complete")
    excluded_ids = {row["document_id"] for row in eligibility_record["exclusions"]}
    eligible_count, entities, relations = _eligible_support(adjudicated, excluded_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = output_dir / "provisional-gold.jsonl"
    with gold_path.open("w", encoding="utf-8", newline="\n") as stream:
        for document in adjudicated:
            stream.write(json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    decisions_path = output_dir / "decisions.json"
    decisions_path.write_text(json.dumps(decisions, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    counts = {"candidate_documents": len(adjudicated), "documents": eligible_count,
              "excluded_candidate_documents": len(adjudicated) - eligible_count,
              "entities": {label: entities[label] for label in sorted(ENTITY_TYPES)},
              "relations": {label: relations[label] for label in sorted(RELATION_TYPES)},
              "third_vote_documents": sum(bool(item["disputed_entities"] or item["disputed_relations"])
                                          for item in decisions)}
    thresholds = {"documents_at_least_50": eligible_count >= 50,
                  "entities_at_least_100_each": all(entities[label] >= 100 for label in ENTITY_TYPES),
                  "relations_at_least_20_each": all(relations[label] >= 20 for label in RELATION_TYPES)}
    manifest = {
        "status": "provisional_not_frozen",
        "adjudication_rule": "itemwise_ab_agreement_else_blind_c",
        "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
        "adjudication_code_sha256": _code_sha256(Path(__file__)),
        "reviewer_ids": reviewer_ids,
        "reviewed_prefix_lengths": {key: len(items) for key, items in zip("abc", review_sets)},
        "counts": counts,
        "thresholds": thresholds,
        "input_hashes": {"packet": _sha256(packet), "guide": _sha256(guide),
                         "selection": _sha256(selection), "eligibility": _sha256(eligibility),
                         **{f"review_{key}": _sha256(path) for key, path in zip("abc", reviews)},
                         **{f"provenance_{key}": _sha256(path) for key, path in zip("abc", provenances)}},
        "output_hashes": {"provisional-gold.jsonl": _sha256(gold_path),
                          "decisions.json": _sha256(decisions_path)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "guide", "selection", "eligibility", "review-a", "review-b", "review-c",
                 "provenance-a", "provenance-b", "provenance-c", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    result = assemble(args.packet, args.guide, args.selection, args.eligibility,
                      (args.review_a, args.review_b, args.review_c),
                      (args.provenance_a, args.provenance_b, args.provenance_c), args.output_dir)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "thresholds": result["thresholds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
