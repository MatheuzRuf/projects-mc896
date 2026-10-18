"""Assemble independent model-review files into the bound adjudication JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from project2_from_project1.src.clinical_documents import load_jsonl
from project2_from_project1.src.gold_projection import (
    AdjudicationError,
    AdjudicationRecord,
    ReviewerVote,
    build_adjudication_binding,
    build_projection_plan,
    write_adjudication_jsonl,
)


_REVIEW_FIELDS = {"case_id", "reviewer_id", "selected_candidate_ids", "justification"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AdjudicationError(f"{path} line {line_number}: invalid JSON") from error
        if not isinstance(value, dict) or set(value) != _REVIEW_FIELDS:
            raise AdjudicationError(f"{path} line {line_number}: invalid review fields")
        case_id = value["case_id"]
        reviewer_id = value["reviewer_id"]
        selected = value["selected_candidate_ids"]
        justification = value["justification"]
        if not isinstance(case_id, str) or not case_id:
            raise AdjudicationError(f"{path} line {line_number}: invalid case_id")
        if case_id in reviews:
            raise AdjudicationError(f"{path}: duplicate review for {case_id}")
        if not isinstance(reviewer_id, str) or not reviewer_id:
            raise AdjudicationError(f"{path} line {line_number}: invalid reviewer_id")
        if (
            not isinstance(selected, list)
            or any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in selected)
            or len(selected) != len(set(selected))
        ):
            raise AdjudicationError(f"{path} line {line_number}: invalid selected_candidate_ids")
        if not isinstance(justification, str) or not justification.strip():
            raise AdjudicationError(f"{path} line {line_number}: invalid justification")
        reviews[case_id] = value
    return reviews


def assemble_adjudication(
    source_jsonl: str | Path,
    review_a_jsonl: str | Path,
    review_b_jsonl: str | Path,
    third_review_jsonl: str | Path,
    output_jsonl: str | Path,
) -> tuple[AdjudicationRecord, ...]:
    source = Path(source_jsonl).resolve()
    review_paths = tuple(Path(path).resolve() for path in (review_a_jsonl, review_b_jsonl, third_review_jsonl))
    output = Path(output_jsonl).resolve()
    if len({source, *review_paths, output}) != 5:
        raise AdjudicationError("source, reviews, and adjudication output paths must all be different")
    source_hash = _sha256(source)
    plan = build_projection_plan(load_jsonl(source))
    binding = build_adjudication_binding(plan, source_hash)
    first_reviews, second_reviews, third_reviews = (
        _load_reviews(path) for path in review_paths
    )
    expected = {case.case_id for case in plan.cases}
    for label, reviews in (("first", first_reviews), ("second", second_reviews)):
        if set(reviews) != expected:
            missing = sorted(expected - set(reviews))
            extra = sorted(set(reviews) - expected)
            detail = f"missing {missing[0]}" if missing else f"unknown {extra[0]}"
            raise AdjudicationError(f"{label} review set is incomplete: {detail}")

    records = []
    disagreements = set()
    for case in plan.cases:
        first = first_reviews[case.case_id]
        second = second_reviews[case.case_id]
        first_vote = ReviewerVote(
            first["reviewer_id"], tuple(first["selected_candidate_ids"]), first["justification"], "model"
        )
        second_vote = ReviewerVote(
            second["reviewer_id"], tuple(second["selected_candidate_ids"]), second["justification"], "model"
        )
        if first_vote.reviewer_id == second_vote.reviewer_id:
            raise AdjudicationError(f"case {case.case_id} requires distinct model reviewers")
        first_selection = tuple(sorted(first_vote.selected_candidate_ids))
        second_selection = tuple(sorted(second_vote.selected_candidate_ids))
        if first_selection == second_selection:
            records.append(AdjudicationRecord(
                case.case_id,
                (first_vote, second_vote),
                first_selection,
                "consensus",
                "Two independent model reviews agreed on the same flat selection; "
                f"review_a={first_vote.justification} review_b={second_vote.justification}",
                (first_vote.reviewer_id, second_vote.reviewer_id),
                binding,
            ))
            continue
        disagreements.add(case.case_id)
        if case.case_id not in third_reviews:
            raise AdjudicationError(f"missing independent third review for {case.case_id}")
        third = third_reviews[case.case_id]
        third_vote = ReviewerVote(
            third["reviewer_id"], tuple(third["selected_candidate_ids"]), third["justification"], "model"
        )
        if third_vote.reviewer_id in {first_vote.reviewer_id, second_vote.reviewer_id}:
            raise AdjudicationError(f"case {case.case_id} requires a distinct third model reviewer")
        outcome = tuple(sorted(third_vote.selected_candidate_ids))
        records.append(AdjudicationRecord(
            case.case_id,
            (first_vote, second_vote, third_vote),
            outcome,
            "third_vote",
            "Independent initial model reviews differed; the recorded outcome follows the independent third vote. "
            f"third_review={third_vote.justification}",
            (),
            binding,
        ))
    extra_third = sorted(set(third_reviews) - disagreements)
    if extra_third:
        raise AdjudicationError(f"unnecessary third review for {extra_third[0]}")
    plan.project(records)
    if _sha256(source) != source_hash:
        raise AdjudicationError("lossless source changed while assembling adjudication")
    write_adjudication_jsonl(records, output)
    return tuple(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--review-a-jsonl", required=True)
    parser.add_argument("--review-b-jsonl", required=True)
    parser.add_argument("--third-review-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()
    records = assemble_adjudication(
        args.source_jsonl,
        args.review_a_jsonl,
        args.review_b_jsonl,
        args.third_review_jsonl,
        args.output_jsonl,
    )
    print(json.dumps({
        "cases": len(records),
        "consensus": sum(record.resolution == "consensus" for record in records),
        "third_vote": sum(record.resolution == "third_vote" for record in records),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
