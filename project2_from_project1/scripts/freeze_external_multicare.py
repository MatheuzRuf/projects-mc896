"""Freeze the first rule-compliant prefix of fresh MultiCaRe cases."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from project2_from_project1.scripts.validate_external_review import ENTITY_TYPES, RELATION_TYPES
from project2_from_project1.src.clinical_documents import load_jsonl


EXPECTED_PREPARATION_SOURCE_HASHES = {
    "cases_parquet": "1f988382b57379d6e39c285895510b4d210bb704a4b8b9c7535349170eb16da1",
    "metadata_parquet": "71a0f2d56bbac4721bc99f25b488848ff9d665c3a9be361e3fbd02b96107ba80",
    "local_selection": "36faaeb4449b00b3dc4de67d2a48640a98c9d823f70ecf4f22d79459220ce386",
    "local_packet": "997d490dd5742dd61f14d7d29c2b05fc87eff7138a2bfaa39023b3e11fc9f420",
    "mcrc_jsonl": "5104d36fa42b6a532b465403b870d89dbfd289ae301ed4b1964ef6b701ebda9b",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _counts(documents: list[object]) -> tuple[dict[str, int], dict[str, int]]:
    entities = Counter(mention.canonical_type for document in documents for mention in document.mentions)
    relations = Counter(relation.canonical_label.value for document in documents for relation in document.relations)
    return ({name: entities[name] for name in sorted(ENTITY_TYPES)},
            {name: relations[name] for name in sorted(RELATION_TYPES)})


def choose_prefix(documents: list[object], *, cap: int = 100) -> tuple[list[object], dict[str, object]]:
    """Apply the registered stop rule; an unfinished prefix is never the cap."""
    for size in range(50, min(len(documents), cap) + 1):
        entities, relations = _counts(documents[:size])
        if all(value >= 100 for value in entities.values()) and all(value >= 20 for value in relations.values()):
            return documents[:size], {"stop_reason": "all_support_thresholds_met", "entities": entities,
                                     "relations": relations, "inconclusive_relations": []}
    if len(documents) < cap:
        raise ValueError(f"reviewed eligible prefix has {len(documents)} documents; 100-document cap not reached")
    entities, relations = _counts(documents[:cap])
    if not all(value >= 100 for value in entities.values()):
        raise ValueError(f"entity support below 100 at cap: {entities}")
    return documents[:cap], {"stop_reason": "document_cap", "entities": entities,
                            "relations": relations,
                            "inconclusive_relations": sorted(name for name, value in relations.items() if value < 20)}


def validate_eligibility_coverage(released: list[str], excluded_ids: list[str],
                                  screened_count: object, chosen_ids: list[str]) -> None:
    """Require the source review to cover every exclusion and frozen document."""
    if (not isinstance(screened_count, int) or isinstance(screened_count, bool)
            or not 0 <= screened_count <= len(released)):
        raise ValueError("source eligibility screened prefix is invalid")
    positions = {document_id: index for index, document_id in enumerate(released)}
    if any(positions[document_id] >= screened_count for document_id in excluded_ids):
        raise ValueError("source eligibility exclusion is outside screened prefix")
    if chosen_ids and positions[chosen_ids[-1]] >= screened_count:
        raise ValueError("source eligibility review does not cover frozen prefix")


def validate_preparation_manifest(preparation_manifest: Path, selection: Path,
                                  audit: Path, packet: Path) -> dict[str, object]:
    """Bind prepared outputs to the manifest that records their raw sources."""
    manifest = json.loads(preparation_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "continuation_packet_not_gold_not_frozen":
        raise ValueError("unexpected external preparation status")
    expected_paths = {
        "selection.json": selection,
        "mcrc-leakage.json": audit,
        "annotation-packet.jsonl": packet,
    }
    if set(manifest.get("output_hashes", {})) != set(expected_paths):
        raise ValueError("external preparation outputs are incomplete")
    for name, path in expected_paths.items():
        if manifest["output_hashes"][name] != _digest(path):
            raise ValueError(f"external preparation output changed: {name}")
    if manifest.get("source_hashes") != EXPECTED_PREPARATION_SOURCE_HASHES or not manifest.get("created_at_utc"):
        raise ValueError("external preparation source provenance is incomplete")
    preparation_script = Path(__file__).with_name("prepare_external_multicare_continuation.py")
    if manifest.get("preparation_code_sha256") != _code_digest(preparation_script):
        raise ValueError("external preparation code changed after snapshot")
    return manifest


def freeze(assembly: Path, reviews: tuple[Path, Path, Path], provenances: tuple[Path, Path, Path],
           preparation_manifest: Path, selection: Path, eligibility: Path, audit: Path, packet: Path,
           prior_packet: Path, pilot_packet: Path, guide: Path, selection_rule: Path, pool_amendment: Path,
           eligibility_completion: Path, scope_decision: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty")
    if not packet.read_bytes().startswith(prior_packet.read_bytes()):
        raise ValueError("expanded candidate packet changed the sealed prefix")
    if _digest(pilot_packet) != EXPECTED_PREPARATION_SOURCE_HASHES["local_packet"]:
        raise ValueError("pilot packet differs from preparation input")
    validate_preparation_manifest(preparation_manifest, selection, audit, packet)
    assembly_manifest = json.loads((assembly / "manifest.json").read_text(encoding="utf-8"))
    if (assembly_manifest.get("status") != "provisional_not_frozen"
            or assembly_manifest.get("adjudication_rule") != "itemwise_ab_agreement_else_blind_c"):
        raise ValueError("assembly is not itemwise adjudicated")
    for name, expected in assembly_manifest["output_hashes"].items():
        if _digest(assembly / name) != expected:
            raise ValueError(f"assembly output changed: {name}")
    for name, path in (("packet", packet), ("guide", guide), ("selection", selection),
                       ("eligibility", eligibility)):
        if assembly_manifest["input_hashes"][name] != _digest(path):
            raise ValueError(f"assembly input changed: {name}")
    for key, path in zip("abc", reviews):
        if assembly_manifest["input_hashes"][f"review_{key}"] != _digest(path):
            raise ValueError(f"review {key} changed after adjudication")
    for key, path in zip("abc", provenances):
        if assembly_manifest["input_hashes"][f"provenance_{key}"] != _digest(path):
            raise ValueError(f"review {key} provenance changed after adjudication")
    project_root = Path(__file__).resolve().parents[1]
    if assembly_manifest.get("adjudication_code_sha256") != _code_digest(
            project_root / "scripts/assemble_external_multicare.py"):
        raise ValueError("adjudication code changed after assembly")
    originals = load_jsonl(packet)
    adjudicated = load_jsonl(assembly / "provisional-gold.jsonl")
    if [(doc.document_id, doc.text, doc.metadata) for doc in adjudicated] != [
        (doc.document_id, doc.text, doc.metadata) for doc in originals[:len(adjudicated)]
    ]:
        raise ValueError("adjudicated documents differ from packet prefix")
    decisions = json.loads((assembly / "decisions.json").read_text(encoding="utf-8"))
    if [row["document_id"] for row in decisions] != [doc.document_id for doc in adjudicated]:
        raise ValueError("adjudication decisions and gold IDs differ")
    selection_rows = json.loads(selection.read_text(encoding="utf-8"))
    audit_records = {row["document_id"]: row for row in json.loads(audit.read_text(encoding="utf-8"))["records"]}
    if (len(audit_records) != len(selection_rows)
            or len({row["document_id"] for row in selection_rows}) != len(selection_rows)):
        raise ValueError("selection or leakage audit has missing or duplicate IDs")
    for row in selection_rows:
        audit_row = audit_records.get(row["document_id"])
        if audit_row is None or audit_row["decision"] != row["against_mcrc"]:
            raise ValueError(f"leakage decision mismatch: {row['document_id']}")
        if row["decision"] == "released" and (audit_row["decision"] != "released"
                                              or row.get("candidate_overlaps")):
            raise ValueError(f"blocked candidate selected: {row['document_id']}")
    released = [row["document_id"] for row in selection_rows if row["decision"] == "released"]
    eligibility_record = json.loads(eligibility.read_text(encoding="utf-8"))
    if eligibility_record.get("status") != "complete":
        raise ValueError("source eligibility review is not complete")
    exclusions = eligibility_record["exclusions"]
    excluded_ids = [row["document_id"] for row in exclusions]
    if len(excluded_ids) != len(set(excluded_ids)) or not set(excluded_ids) <= set(released):
        raise ValueError("invalid source eligibility exclusions")
    eligible = [doc for doc in adjudicated if doc.document_id not in excluded_ids]
    if [doc.document_id for doc in eligible] != [case_id for case_id in released
                                                 if case_id not in excluded_ids][:len(eligible)]:
        raise ValueError("selected documents are not in registered order")
    chosen, support = choose_prefix(eligible)
    validate_eligibility_coverage(released, excluded_ids,
                                  eligibility_record.get("screened_prefix_document_count"),
                                  [doc.document_id for doc in chosen])
    if (any(not doc.metadata.get("article_id") for doc in chosen)
            or len({doc.metadata["article_id"] for doc in chosen}) != len(chosen)):
        raise ValueError("frozen test contains multiple documents from one article")
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = output_dir / "external-gold.jsonl"
    with gold_path.open("w", encoding="utf-8", newline="\n") as stream:
        for doc in chosen:
            stream.write(json.dumps(doc.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    ids_path = output_dir / "selected-ids.json"
    ids_path.write_text(json.dumps([
        {"document_id": doc.document_id, "text_sha256": hashlib.sha256(doc.text.encode()).hexdigest()}
        for doc in chosen
    ], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    input_files = dict((name, path) for name, path in (
        ("assembly_manifest", assembly / "manifest.json"),
        ("preparation_manifest", preparation_manifest), ("selection", selection),
        ("source_eligibility", eligibility), ("leakage_report", audit), ("packet", packet),
        ("prior_100_candidate_packet", prior_packet), ("pilot_local_packet", pilot_packet),
        ("guide", guide), ("selection_rule", selection_rule),
        ("pool_amendment", pool_amendment), ("scope_decision", scope_decision),
        ("source_eligibility_completion", eligibility_completion),
    ))
    input_files.update({f"review_{key}": path for key, path in zip("abc", reviews)})
    input_files.update({f"provenance_{key}": path for key, path in zip("abc", provenances)})
    inputs = {name: _digest(path) for name, path in input_files.items()}
    repository_root = Path(__file__).resolve().parents[2]
    input_paths = {name: str(path.resolve().relative_to(repository_root)).replace("\\", "/")
                   for name, path in input_files.items()}
    code_hashes = {name: _code_digest(project_root / path) for name, path in (
        ("preparation", "scripts/prepare_external_multicare_continuation.py"),
        ("leakage_audit", "src/leakage_audit.py"),
        ("review_validation", "scripts/validate_external_review.py"),
        ("adjudication", "scripts/assemble_external_multicare.py"),
        ("freeze", "scripts/freeze_external_multicare.py"),
        ("freeze_verification", "scripts/verify_external_multicare_freeze.py"),
    )}
    manifest = {"status": "frozen_before_first_external_model_run",
                "schema": "flat-gold-v1",
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "documents": len(chosen), "support": support,
                "adjudicated_candidate_prefix_documents": len(adjudicated),
                "source_eligibility_screened_prefix_documents": eligibility_record[
                    "screened_prefix_document_count"
                ],
                "review_type": "independent_model_reviews_with_blind_third_itemwise_vote",
                "pilot_local_documents_excluded": True,
                "selection_rule_observed_pilot_annotation_support": True,
                "source_exclusions": exclusions, "input_sha256": inputs,
                "input_paths": input_paths,
                "code_sha256": code_hashes,
                "output_sha256": {"external-gold.jsonl": _digest(gold_path),
                                  "selected-ids.json": _digest(ids_path)}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assembly", "review-a", "review-b", "review-c", "provenance-a",
                 "provenance-b", "provenance-c", "selection", "eligibility", "audit", "packet", "prior-packet",
                 "pilot-packet", "guide", "preparation-manifest",
                 "selection-rule", "pool-amendment", "eligibility-completion",
                 "scope-decision", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.assembly, (args.review_a, args.review_b, args.review_c),
                    (args.provenance_a, args.provenance_b, args.provenance_c),
                    args.preparation_manifest, args.selection, args.eligibility, args.audit,
                    args.packet, args.prior_packet, args.pilot_packet, args.guide,
                    args.selection_rule, args.pool_amendment,
                    args.eligibility_completion, args.scope_decision, args.output_dir)
    print(json.dumps({"status": result["status"], "documents": result["documents"],
                      "support": result["support"]}, sort_keys=True))


if __name__ == "__main__":
    main()
