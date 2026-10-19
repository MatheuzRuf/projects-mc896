"""Train the semi-Markov CRF on one fold's training gold and evaluate its validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from project2_from_project1.src.entity_recognition.crf_model import TrainingConfig
from project2_from_project1.src.entity_recognition.run import run_crf_fold_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Lossless ClinicalDocument JSONL")
    parser.add_argument("--gold-projection", required=True, help="flat-gold-v1 gold-projection.jsonl")
    parser.add_argument("--gold-projection-manifest", required=True, help="Manifest written beside the projection")
    parser.add_argument("--partitions", required=True, help="Frozen supervised-dataset manifest with document folds")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--l2", type=float, required=True, help="L2 strength; selected by nested CV in ticket 05")
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--threads", type=int, help="torch CPU threads; recorded in the manifest")
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    if arguments.threads:
        torch.set_num_threads(arguments.threads)

    manifest = run_crf_fold_from_files(
        arguments.corpus, arguments.gold_projection, arguments.gold_projection_manifest,
        arguments.partitions, arguments.fold, arguments.output_dir,
        config=TrainingConfig(l2=arguments.l2, max_iterations=arguments.max_iterations),
    )
    print(json.dumps({
        "converged": manifest["converged"],
        "fold": manifest["fold"],
        "iterations": manifest["training"]["iterations"],
        "primary_metric": manifest["primary_metric"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
