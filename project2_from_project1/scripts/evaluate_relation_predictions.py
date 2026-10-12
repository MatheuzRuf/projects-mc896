"""Evaluate common-contract relation predictions against a frozen gold partition."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.relation_evaluation import (
    EvaluationConfig,
    evaluate_predictions,
    load_relation_predictions,
    load_supervised_candidates,
    write_evaluation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-candidates", required=True, help="Frozen validation or test candidate JSONL")
    parser.add_argument("--predictions", required=True, help="Common-contract prediction JSONL")
    parser.add_argument("--output", required=True, help="Output evaluation JSON report")
    parser.add_argument("--seed", type=int, default=0, help="Recorded reproducibility seed")
    parser.add_argument("--minimum-positive-support", type=int, default=5, help="Support below this is inconclusive")
    args = parser.parse_args()
    report = evaluate_predictions(
        load_supervised_candidates(args.gold_candidates),
        load_relation_predictions(args.predictions),
        config=EvaluationConfig(seed=args.seed, minimum_positive_support=args.minimum_positive_support),
        metadata={"gold_candidates_path": Path(args.gold_candidates).name, "predictions_path": Path(args.predictions).name},
    )
    write_evaluation_report(report, args.output)


if __name__ == "__main__":
    main()
