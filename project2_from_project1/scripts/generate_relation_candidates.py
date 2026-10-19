"""Generate gold-free relation candidates from an entity prediction JSONL."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project2_from_project1.src.relation_candidates import run_from_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, help='JSONL with document_id and text')
    parser.add_argument('--predictions', required=True, help='EntityPrediction JSONL')
    parser.add_argument('--prediction-manifest', help='Optional recognizer manifest for provenance')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    manifest = run_from_files(args.corpus, args.predictions, args.output_dir, args.prediction_manifest)
    print(json.dumps(manifest['counts'], sort_keys=True))


if __name__ == '__main__':
    main()
