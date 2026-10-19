"""Evaluate {dictionary, CRF} x {rules, k-NN} end to end, beside the gold-entity ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.relation_factorial import run_factorial_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--gold-projection", required=True)
    parser.add_argument("--frozen-dataset", required=True, help="Directory with manifest.json and fold candidate files")
    parser.add_argument("--entity-study", required=True, help="Output directory of run_entity_nested_cv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--minimum-positive-support", type=int, default=5)
    arguments = parser.parse_args()
    summary = run_factorial_study(arguments.corpus, arguments.gold_projection, arguments.frozen_dataset,
                                  arguments.entity_study, arguments.output_dir, folds=arguments.folds,
                                  minimum_positive_support=arguments.minimum_positive_support)
    print(json.dumps({arm: round(data["pooled"]["macro_f1"], 4) if data["pooled"] else None
                      for arm, data in summary["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
