"""Prepare reviewer-readable adjudication cases and collect independent model reviews.

``prepare`` renders every case of the bound projection plan with sentence
context, inline candidate markers, short aliases (``c1``, ``c2``...) and the gold
relations in which each candidate's source mentions participate.  The output
contains clinical text and must stay in the access-controlled artifacts tree.

``check`` and ``collect`` validate one reviewer's alias-based JSONL
(``{"case_id", "selected", "justification"}`` per line) and convert it to the
review format consumed by ``assemble_gold_adjudication``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.clinical_documents import load_jsonl
from project2_from_project1.src.gold_projection import (
    AdjudicationError,
    build_adjudication_binding,
    build_projection_plan,
)
from project2_from_project1.src.run_manifest import sha256_file

_FORMAT_VERSION = 1


def _relations(document) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    for relation in document.relations:
        label = relation.canonical_label.value
        roles.setdefault(relation.source_id, []).append(f"source of {label}")
        roles.setdefault(relation.target_id, []).append(f"target of {label}")
    return roles


def prepare(source_jsonl: Path, output_json: Path, case_ids: set[str] | None = None) -> dict[str, Any]:
    documents = load_jsonl(source_jsonl)
    plan = build_projection_plan(documents)
    binding = build_adjudication_binding(plan, sha256_file(source_jsonl))
    by_document = {document.document_id: document for document in documents}
    candidates = {candidate.candidate_id: candidate for candidate in plan.candidates}
    cases = []
    for case in plan.cases:
        if case_ids is not None and case.case_id not in case_ids:
            continue
        document = by_document[case.document_id]
        sentences = plan.sentences[case.document_id]
        members = [candidates[candidate_id] for candidate_id in case.candidate_ids]
        members.sort(key=lambda item: (item.span.start, -item.span.end, item.canonical_type))
        indexes = [item.sentence_index for item in members if item.sentence_index is not None]
        if indexes:
            first, last = max(0, min(indexes) - 1), min(len(sentences) - 1, max(indexes) + 1)
            window = (sentences[first].span.start, sentences[last].span.end)
        else:
            window = (max(0, min(item.span.start for item in members) - 300),
                      min(len(document.text), max(item.span.end for item in members) + 300))
        window = (min(window[0], *(item.span.start for item in members)),
                  max(window[1], *(item.span.end for item in members)))
        roles = _relations(document)
        aliases = {}
        rendered = []
        for number, item in enumerate(members, 1):
            alias = f"c{number}"
            aliases[alias] = item.candidate_id
            relation_roles = sorted({role for mention in item.source_mention_ids for role in roles.get(mention, ())})
            rendered.append({
                "alias": alias,
                "duplicates_collapsed": len(item.source_mention_ids),
                "gold_relations": relation_roles,
                "issues": list(item.issue_reasons),
                "surface": document.text[item.span.start:item.span.end],
                "original_surface": " | ".join(document.text[span.start:span.end] for span in item.original_spans),
                "transformations": list(item.transformations),
                "type": item.canonical_type,
            })
        # Insertions at one offset end up right-to-left, so insert openings
        # (innermost first) before closings (outermost first) to nest properly.
        markers = []
        for index, (alias, item) in enumerate(zip(aliases, members)):
            markers.append((item.span.start, 0, -index, f"[{alias}:"))
            markers.append((item.span.end, 1, index, f":{alias}]"))
        text = document.text[window[0]:window[1]]
        for position, _, _, marker in sorted(markers, key=lambda item: (-item[0], item[1], item[2])):
            offset = position - window[0]
            text = text[:offset] + marker + text[offset:]
        cases.append({
            "aliases": aliases,
            "candidates": rendered,
            "case_id": case.case_id,
            "context": re.sub(r"\s+", " ", text).strip(),
            "reasons": list(case.reasons),
        })
    payload = {"binding": binding.to_dict(), "cases": cases, "format_version": _FORMAT_VERSION,
               "sensitive_clinical_text": True}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return payload


def _load_alias_review(packet: dict[str, Any], review_jsonl: Path) -> list[dict[str, Any]]:
    by_case = {case["case_id"]: case for case in packet["cases"]}
    seen = {}
    errors = []
    for number, line in enumerate(review_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"line {number}: invalid JSON")
            continue
        if not isinstance(value, dict) or set(value) != {"case_id", "selected", "justification"}:
            errors.append(f"line {number}: fields must be exactly case_id, selected, justification")
            continue
        case = by_case.get(value["case_id"])
        if case is None:
            errors.append(f"line {number}: unknown case_id")
            continue
        if value["case_id"] in seen:
            errors.append(f"line {number}: duplicate review for {value['case_id']}")
            continue
        selected = value["selected"]
        if not isinstance(selected, list) or len(set(selected)) != len(selected) or any(alias not in case["aliases"] for alias in selected):
            errors.append(f"line {number}: selected must list distinct aliases of this case")
            continue
        candidates = {item["alias"]: item for item in case["candidates"]}
        if any(set(candidates[alias]["issues"]) & {"cross_sentence", "discontinuous", "over_max_tokens", "token_misalignment"} for alias in selected):
            errors.append(f"line {number}: a selected candidate is not token aligned")
        justification = value["justification"]
        if not isinstance(justification, str) or len(justification.strip()) < 20:
            errors.append(f"line {number}: justification must be a specific sentence")
            continue
        lowered = justification.casefold()
        leaked = [item["surface"] for item in case["candidates"]
                  if len(item["surface"]) >= 4 and item["surface"].casefold() in lowered]
        if leaked:
            errors.append(f"line {number}: justification quotes clinical text; describe candidates abstractly")
        seen[value["case_id"]] = value
    return [errors, seen]  # type: ignore[list-item]


def check(packet_json: Path, review_jsonl: Path, expected: set[str] | None = None) -> list[str]:
    packet = json.loads(packet_json.read_text(encoding="utf-8"))
    errors, seen = _load_alias_review(packet, review_jsonl)
    wanted = expected if expected is not None else {case["case_id"] for case in packet["cases"]}
    missing = sorted(wanted - set(seen))
    if missing:
        errors.append(f"{len(missing)} cases are missing, first {missing[0]}")
    extra = sorted(set(seen) - wanted)
    if extra:
        errors.append(f"{len(extra)} reviews are not expected, first {extra[0]}")
    return errors


def collect(packet_json: Path, review_jsonl: Path, reviewer_id: str, output_jsonl: Path) -> int:
    errors = check(packet_json, review_jsonl)
    if errors:
        raise AdjudicationError("; ".join(errors[:10]))
    packet = json.loads(packet_json.read_text(encoding="utf-8"))
    by_case = {case["case_id"]: case for case in packet["cases"]}
    _, seen = _load_alias_review(packet, review_jsonl)
    lines = []
    for case_id in sorted(seen):
        value = seen[case_id]
        aliases = by_case[case_id]["aliases"]
        lines.append(json.dumps({
            "case_id": case_id,
            "justification": value["justification"].strip(),
            "reviewer_id": reviewer_id,
            "selected_candidate_ids": sorted(aliases[alias] for alias in value["selected"]),
        }, ensure_ascii=False, sort_keys=True))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-jsonl", required=True)
    prepare_parser.add_argument("--output-json", required=True)
    prepare_parser.add_argument("--case-ids", help="Optional file with one case_id per line")
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--packet", required=True)
    check_parser.add_argument("--review", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--packet", required=True)
    collect_parser.add_argument("--review", required=True)
    collect_parser.add_argument("--reviewer-id", required=True)
    collect_parser.add_argument("--output-jsonl", required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        ids = None
        if arguments.case_ids:
            ids = {line.strip() for line in Path(arguments.case_ids).read_text(encoding="utf-8").splitlines() if line.strip()}
        payload = prepare(Path(arguments.source_jsonl), Path(arguments.output_json), ids)
        print(json.dumps({"cases": len(payload["cases"])}))
    elif arguments.command == "check":
        errors = check(Path(arguments.packet), Path(arguments.review))
        print("\n".join(errors) if errors else "OK")
        raise SystemExit(1 if errors else 0)
    else:
        print(json.dumps({"reviews": collect(Path(arguments.packet), Path(arguments.review),
                                             arguments.reviewer_id, Path(arguments.output_jsonl))}))


if __name__ == "__main__":
    main()
