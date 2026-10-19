"""Run the same fold-trained n-gram k-NN on dictionary and CRF candidates."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.ngrams.predicted_pipeline import run_from_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--training-candidates", required=True)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--fold", type=int, required=True)
    for arm in ("dictionary", "crf"):
        parser.add_argument(f"--{arm}-candidates", required=True)
        parser.add_argument(f"--{arm}-candidate-manifest", required=True)
        parser.add_argument(f"--{arm}-entities", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-neighbors", type=int, required=True)
    parser.add_argument("--maximum-ngram-order", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--model-version", default="ngram-knn-v1")
    args = parser.parse_args()
    arms = {name: (getattr(args, f"{name}_candidates"), getattr(args, f"{name}_candidate_manifest"),
                   getattr(args, f"{name}_entities")) for name in ("dictionary", "crf")}
    manifests = run_from_files(args.corpus, args.training_candidates, args.partition_manifest, args.fold,
                               arms, args.output_dir, n_neighbors=args.n_neighbors,
                               maximum_ngram_order=args.maximum_ngram_order, model_version=args.model_version)
    print(json.dumps({name: manifest["counts"] for name, manifest in manifests.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
