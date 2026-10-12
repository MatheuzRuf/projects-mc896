"""Freeze supervised relation candidates from an approved safe MCRC corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.supervised_dataset import (
    CandidatePolicy,
    PartitionPolicy,
    build_frozen_supervised_dataset_from_files,
    write_frozen_supervised_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-jsonl", required=True, help="Safe MCRC JSONL produced by convert_mcrc.py")
    parser.add_argument("--suitability-report", required=True, help="Approved report produced by decide_mcrc_suitability.py")
    parser.add_argument("--output-directory", required=True, help="Directory for frozen candidate JSONL files and manifest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--maximum-sentence-distance", type=int, default=1, help="Inclusive sentence window; use -1 for no limit")
    parser.add_argument("--maximum-train-negatives-per-positive", type=int, default=2)
    args = parser.parse_args()
    candidate_policy = CandidatePolicy(None if args.maximum_sentence_distance == -1 else args.maximum_sentence_distance)
    partition_policy = PartitionPolicy(args.seed, args.test_fraction, args.folds, args.maximum_train_negatives_per_positive)
    dataset = build_frozen_supervised_dataset_from_files(
        args.converted_jsonl,
        args.suitability_report,
        candidate_policy=candidate_policy,
        partition_policy=partition_policy,
    )
    write_frozen_supervised_dataset(dataset, args.output_directory)


if __name__ == "__main__":
    main()
