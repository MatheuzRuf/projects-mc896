"""Build the adjudicated, text-free flat entity gold projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project2_from_project1.src.gold_projection import project_gold_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, help="Lossless ClinicalDocument JSONL")
    parser.add_argument("--adjudication-jsonl", required=True, help="Versioned independent-review decisions")
    parser.add_argument("--output-dir", required=True, help="Directory for derived projection artifacts")
    args = parser.parse_args()

    output = Path(args.output_dir)
    manifest = project_gold_from_files(
        args.source_jsonl,
        args.adjudication_jsonl,
        output / "gold-projection.jsonl",
        output / "adjudication-audit.jsonl",
        output / "manifest.json",
    )
    print(json.dumps({
        "counts": manifest["counts"],
        "input_sha256": manifest["input_sha256"],
        "output_sha256": manifest["output_sha256"],
        "reviews": manifest["reviews"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
