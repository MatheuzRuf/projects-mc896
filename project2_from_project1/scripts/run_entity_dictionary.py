"""Fit the canonical dictionary on one fold's training gold and evaluate its validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.entity_recognition.run import run_dictionary_fold_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Lossless ClinicalDocument JSONL")
    parser.add_argument("--gold-projection", required=True, help="flat-gold-v1 gold-projection.jsonl")
    parser.add_argument("--gold-projection-manifest", required=True, help="Manifest written beside the projection")
    parser.add_argument("--partitions", required=True, help="Frozen supervised-dataset manifest with document folds")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    manifest = run_dictionary_fold_from_files(
        arguments.corpus, arguments.gold_projection, arguments.gold_projection_manifest,
        arguments.partitions, arguments.fold, arguments.output_dir,
    )
    print(json.dumps({
        "dictionary_entries": manifest["dictionary"]["entries"],
        "fold": manifest["fold"],
        "primary_metric": manifest["primary_metric"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
