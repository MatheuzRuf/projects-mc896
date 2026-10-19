"""Run the nested document cross-validation comparing dictionary and semi-Markov CRF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.entity_recognition.nested_cv import StudyInputs, run_nested_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--gold-projection", required=True)
    parser.add_argument("--gold-projection-manifest", required=True)
    parser.add_argument("--partitions", required=True, help="Frozen supervised-dataset manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="*", help="Outer folds to run; default all")
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--workers", type=int, default=1, help="Parallel CRF fits")
    parser.add_argument("--threads-per-worker", type=int, default=1)
    arguments = parser.parse_args()
    summary = run_nested_study(
        StudyInputs(arguments.corpus, arguments.gold_projection, arguments.gold_projection_manifest, arguments.partitions),
        arguments.output_dir, folds=arguments.folds, max_iterations=arguments.max_iterations,
        workers=arguments.workers, threads_per_worker=arguments.threads_per_worker,
    )
    print(json.dumps({arm: data["aggregate"].get("macro_f1") for arm, data in summary["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
