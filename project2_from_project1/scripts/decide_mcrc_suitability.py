"""Decide whether a converted and audited MCRC corpus may be partitioned."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.clinical_documents import CanonicalRelation
from project2_from_project1.src.mcrc_suitability import (
    SuitabilityCriteria,
    assess_mcrc_suitability_from_files,
    write_mcrc_suitability_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-jsonl", required=True, help="Safe MCRC JSONL produced by convert_mcrc.py")
    parser.add_argument("--leakage-report", required=True, help="Audit report produced against protected MultiCaRe")
    parser.add_argument("--output-report", required=True, help="Destination for the suitability decision JSON")
    parser.add_argument("--minimum-released-documents", type=int, default=1)
    parser.add_argument("--minimum-released-supervised-relations", type=int, default=1)
    parser.add_argument("--minimum-relations-per-required-canonical", type=int, default=1)
    parser.add_argument(
        "--required-canonical-relation",
        action="append",
        choices=[label.value for label in (
            CanonicalRelation.HAS_CONDITION,
            CanonicalRelation.HAS_FINDING,
            CanonicalRelation.HAS_FACTOR,
            CanonicalRelation.NEGATES,
            CanonicalRelation.CAUSES,
        )],
        help="Repeat to replace the default required canonical coverage set",
    )
    parser.add_argument(
        "--maximum-blocked-document-fraction",
        type=float,
        help="Optional maximum fraction of excluded plus quarantined documents",
    )
    args = parser.parse_args()
    criteria_kwargs = {
        "minimum_released_documents": args.minimum_released_documents,
        "minimum_released_supervised_relations": args.minimum_released_supervised_relations,
        "minimum_relations_per_required_canonical": args.minimum_relations_per_required_canonical,
        "maximum_blocked_document_fraction": args.maximum_blocked_document_fraction,
    }
    if args.required_canonical_relation is not None:
        criteria_kwargs["required_canonical_relations"] = args.required_canonical_relation
    report = assess_mcrc_suitability_from_files(
        args.converted_jsonl,
        args.leakage_report,
        criteria=SuitabilityCriteria(**criteria_kwargs),
    )
    write_mcrc_suitability_report(report, args.output_report)
    print(report.decision.value)


if __name__ == "__main__":
    main()
