"""Fit and run canonical relation rules for one frozen validation fold."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.canonical_relation_rules import run_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-candidates", required=True)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = run_from_files(args.training_candidates, args.partition_manifest, args.fold,
                              args.candidates, args.candidate_manifest, args.output_dir)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
