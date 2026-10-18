"""Select n-gram order and k using only frozen cross-validation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.ngrams.model_selection import (
    select_ngram_knn_from_folds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Safe clinical-document JSONL")
    parser.add_argument(
        "--frozen-dataset",
        required=True,
        help="Directory containing fold JSONL files",
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--n-neighbors", type=int, action="append", dest="neighbor_values"
    )
    parser.add_argument(
        "--maximum-ngram-order",
        type=int,
        choices=(1, 2, 3),
        action="append",
        dest="maximum_ngram_orders",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-positive-support", type=int, default=5)
    parser.add_argument("--model-version", default="ngram-knn-v1")
    arguments = parser.parse_args()

    summary = select_ngram_knn_from_folds(
        arguments.corpus,
        arguments.frozen_dataset,
        arguments.output_directory,
        neighbor_values=arguments.neighbor_values or (1, 3, 5),
        maximum_ngram_orders=arguments.maximum_ngram_orders or (1, 2, 3),
        seed=arguments.seed,
        minimum_positive_support=arguments.minimum_positive_support,
        model_version=arguments.model_version,
    )
    print(json.dumps(summary["best_configuration"], sort_keys=True))
    print(Path(arguments.output_directory) / "model-selection-summary.json")


if __name__ == "__main__":
    main()
