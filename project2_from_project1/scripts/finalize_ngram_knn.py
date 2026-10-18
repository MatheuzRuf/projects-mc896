"""Fit the selected n-gram k-NN and evaluate the fixed test partition once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.ngrams.model_selection import (
    finalize_selected_ngram_knn,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--frozen-dataset", required=True)
    parser.add_argument("--selection-summary", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--maximum-negatives-per-positive", type=int, default=2)
    parser.add_argument("--minimum-positive-support", type=int, default=5)
    arguments = parser.parse_args()
    summary = finalize_selected_ngram_knn(
        arguments.corpus,
        arguments.frozen_dataset,
        arguments.selection_summary,
        arguments.output_directory,
        maximum_negatives_per_positive=arguments.maximum_negatives_per_positive,
        minimum_positive_support=arguments.minimum_positive_support,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
