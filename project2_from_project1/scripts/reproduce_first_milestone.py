"""Prepare or evaluate the first clinical-relation extraction milestone."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.clinical_documents import CanonicalRelation
from project2_from_project1.src.first_milestone import evaluate_first_milestone, prepare_first_milestone
from project2_from_project1.src.mcrc_suitability import SuitabilityCriteria
from project2_from_project1.src.supervised_dataset import CandidatePolicy, PartitionPolicy


def _prepare(arguments: argparse.Namespace) -> None:
    criteria_values = {
        "minimum_released_documents": arguments.minimum_released_documents,
        "minimum_released_supervised_relations": arguments.minimum_released_supervised_relations,
        "minimum_relations_per_required_canonical": arguments.minimum_relations_per_required_canonical,
        "maximum_blocked_document_fraction": arguments.maximum_blocked_document_fraction,
    }
    if arguments.required_canonical_relation is not None:
        criteria_values["required_canonical_relations"] = arguments.required_canonical_relation
    result = prepare_first_milestone(
        arguments.safe_jsonl,
        arguments.protected_collection,
        arguments.output_directory,
        candidate_policy=CandidatePolicy(None if arguments.maximum_sentence_distance == -1 else arguments.maximum_sentence_distance),
        partition_policy=PartitionPolicy(
            seed=arguments.seed,
            test_fraction=arguments.test_fraction,
            folds=arguments.folds,
            maximum_train_negatives_per_positive=arguments.maximum_train_negatives_per_positive,
        ),
        suitability_criteria=SuitabilityCriteria(**criteria_values),
        similarity_threshold=arguments.similarity_threshold,
    )
    print(result.status)
    print(result.manifest_path)


def _evaluate(arguments: argparse.Namespace) -> None:
    print(
        evaluate_first_milestone(
            arguments.prepared_directory,
            arguments.predictions,
            arguments.output,
            partition=arguments.partition,
            seed=arguments.seed,
            minimum_positive_support=arguments.minimum_positive_support,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Audit safe corpus and freeze an approved dataset")
    prepare.add_argument("--safe-jsonl", required=True, help="Validated safe clinical-document JSONL; never a raw pickle")
    prepare.add_argument("--protected-collection", required=True, help="Text-free protected collection fingerprint JSON")
    prepare.add_argument("--output-directory", required=True, help="Ignored directory for generated reports and frozen candidates")
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--test-fraction", type=float, default=0.2)
    prepare.add_argument("--folds", type=int, default=5)
    prepare.add_argument("--maximum-sentence-distance", type=int, default=1, help="Inclusive window; use -1 for no limit")
    prepare.add_argument("--maximum-train-negatives-per-positive", type=int, default=2)
    prepare.add_argument("--similarity-threshold", type=float, default=0.85)
    prepare.add_argument("--minimum-released-documents", type=int, default=1)
    prepare.add_argument("--minimum-released-supervised-relations", type=int, default=1)
    prepare.add_argument("--minimum-relations-per-required-canonical", type=int, default=1)
    prepare.add_argument("--maximum-blocked-document-fraction", type=float)
    prepare.add_argument(
        "--required-canonical-relation",
        action="append",
        choices=[label.value for label in (
            CanonicalRelation.HAS_CONDITION,
            CanonicalRelation.HAS_FINDING,
            CanonicalRelation.HAS_FACTOR,
            CanonicalRelation.NEGATES,
            CanonicalRelation.CAUSES,
        )],
        help="Repeat to replace the default required coverage set",
    )
    prepare.set_defaults(handler=_prepare)

    evaluate = commands.add_parser("evaluate", help="Evaluate compatible predictions against prepared gold candidates")
    evaluate.add_argument("--prepared-directory", required=True, help="Output directory from the prepare command")
    evaluate.add_argument("--predictions", required=True, help="Complete common-contract prediction JSONL")
    evaluate.add_argument("--output", required=True, help="Ignored destination for evaluation JSON")
    evaluate.add_argument("--partition", default="test", help="test or fold-XX-validation")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--minimum-positive-support", type=int, default=5)
    evaluate.set_defaults(handler=_evaluate)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
