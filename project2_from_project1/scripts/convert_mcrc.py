"""One-time, explicit converter for a verified MCRC pickle distribution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.mcrc_converter import MCRCSource, convert_mcrc_distribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="Verified CaseReportCorpus.pickle path")
    parser.add_argument("--output-jsonl", required=True, help="Destination for the safe JSONL corpus")
    parser.add_argument("--manifest", required=True, help="Destination for the conversion manifest")
    parser.add_argument("--repository", required=True, help="Source repository URL")
    parser.add_argument("--revision", required=True, help="Immutable source commit or version")
    parser.add_argument("--sha256", required=True, help="Expected SHA-256 of --artifact")
    parser.add_argument(
        "--allow-unsafe-deserialization",
        action="store_true",
        help="Required acknowledgement: only run this once in a controlled environment",
    )
    args = parser.parse_args()
    convert_mcrc_distribution(
        args.artifact,
        args.output_jsonl,
        args.manifest,
        MCRCSource(args.repository, args.revision, args.sha256),
        allow_unsafe_deserialization=args.allow_unsafe_deserialization,
    )


if __name__ == "__main__":
    main()
