"""Export a supervised JSONL v2 graph to the legacy visualizer inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from project2_from_project1.src.supervised_contracts import load_contract_jsonl
from project2_from_project1.src.supervised_graph import export_legacy_csv


def _write_cases_csv(raw_jsonl: Path, output: Path) -> int:
    rows: list[dict[str, str]] = []
    with raw_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            document_id = payload.get("document_id")
            text = payload.get("text")
            if not isinstance(document_id, str) or not isinstance(text, str):
                raise ValueError(
                    f"raw JSONL line {line_number}: document_id and text must be strings"
                )
            rows.append({"case_id": document_id, "case_text": text})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "case_text"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export graph JSONL v2 to nodes.csv/edges.csv for the Project 1 visualizer."
    )
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layer", choices=("concept", "mention"), default="concept")
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        help="Optional raw documents used to create cases.csv for the interactive interface.",
    )
    arguments = parser.parse_args()

    records = load_contract_jsonl(arguments.graph, "graph")
    counts = export_legacy_csv(
        records,
        {},
        arguments.output_dir / "nodes.csv",
        arguments.output_dir / "edges.csv",
        layer=arguments.layer,
    )
    documents = None
    if arguments.raw_jsonl is not None:
        documents = _write_cases_csv(arguments.raw_jsonl, arguments.output_dir / "cases.csv")

    print(
        json.dumps(
            {
                "documents": documents,
                "edges": counts["edges"],
                "layer": arguments.layer,
                "nodes": counts["nodes"],
                "output_dir": str(arguments.output_dir),
                "sensitive_outputs": ["nodes.csv", "cases.csv"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
