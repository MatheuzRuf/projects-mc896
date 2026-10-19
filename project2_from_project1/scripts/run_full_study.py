"""Run the whole study from raw clinical text to the Knowledge Graph for every arm.

Use ``--synthetic DIR`` to generate and run the synthetic fixture instead of
real inputs; it needs no clinical data and finishes in seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.study_pipeline import run_full_study
from project2_from_project1.src.synthetic_fixture import make_synthetic_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", help="Lossless ClinicalDocument JSONL")
    parser.add_argument("--adjudication", help="Bound adjudication JSONL")
    parser.add_argument("--frozen-dataset", help="Directory with the frozen partitions and fold candidates")
    parser.add_argument("--synthetic", help="Generate the synthetic fixture in this directory and run it")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.synthetic:
        workspace = make_synthetic_workspace(arguments.synthetic)
        corpus, adjudication, frozen = workspace.corpus_jsonl, workspace.adjudication_jsonl, workspace.frozen_dataset
    elif arguments.corpus and arguments.adjudication and arguments.frozen_dataset:
        corpus, adjudication, frozen = arguments.corpus, arguments.adjudication, arguments.frozen_dataset
    else:
        parser.error("give --synthetic or all of --corpus, --adjudication and --frozen-dataset")
    manifest = run_full_study(corpus, adjudication, frozen, arguments.output_dir, folds=arguments.folds,
                              max_iterations=arguments.max_iterations, workers=arguments.workers,
                              threads_per_worker=arguments.threads_per_worker)
    print(json.dumps({"arms": {arm: {"relation_macro_f1": data["relation_macro_f1"], "failed_folds": data["failed_folds"]}
                               for arm, data in manifest["arms"].items()},
                      "duration_seconds": manifest["duration_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
