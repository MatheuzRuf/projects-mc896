"""Train n-gram k-NN, predict one frozen partition, and optionally evaluate it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.ngrams.pipeline import run_ngram_knn_from_files
from project2_from_project1.src.relation_evaluation import (
    EvaluationConfig,
    evaluate_predictions,
    load_supervised_candidates,
    write_evaluation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Safe clinical-document JSONL")
    parser.add_argument("--train-candidates", required=True, help="Frozen training candidate JSONL")
    parser.add_argument("--predict-candidates", required=True, help="Frozen validation or test candidate JSONL")
    parser.add_argument("--predictions", required=True, help="Output common-contract prediction JSONL")
    parser.add_argument("--manifest", help="Output run manifest; defaults beside predictions")
    parser.add_argument("--evaluation", help="Optional output evaluation JSON")
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument(
        "--maximum-ngram-order", type=int, choices=(1, 2, 3), default=3
    )
    parser.add_argument("--model-version", default="ngram-knn-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-positive-support", type=int, default=5)
    arguments = parser.parse_args()

    run = run_ngram_knn_from_files(
        arguments.corpus,
        arguments.train_candidates,
        arguments.predict_candidates,
        arguments.predictions,
        n_neighbors=arguments.n_neighbors,
        maximum_ngram_order=arguments.maximum_ngram_order,
        model_version=arguments.model_version,
        manifest_path=arguments.manifest,
    )
    print(f"predictions={len(run.predictions)} vocabulary={len(run.vocabulary)}")

    if arguments.evaluation:
        report = evaluate_predictions(
            load_supervised_candidates(arguments.predict_candidates),
            run.predictions,
            config=EvaluationConfig(
                seed=arguments.seed,
                minimum_positive_support=arguments.minimum_positive_support,
            ),
            metadata={
                "feature_family": f"word-count-ngrams-1-{arguments.maximum_ngram_order}",
                "maximum_ngram_order": arguments.maximum_ngram_order,
                "model_version": arguments.model_version,
                "n_neighbors": arguments.n_neighbors,
                "partition": Path(arguments.predict_candidates).name,
            },
        )
        write_evaluation_report(report, arguments.evaluation)
        print(Path(arguments.evaluation))


if __name__ == "__main__":
    main()
