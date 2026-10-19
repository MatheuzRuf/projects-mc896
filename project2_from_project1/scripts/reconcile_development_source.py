"""Copy the development corpus with the exact bytes bound to the adjudication.

This only reconciles line endings. It does not rebind or regenerate partitions,
train models, or open external annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reconcile(corpus: Path, adjudication: Path, partitions: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("output directory already exists")
    original = corpus.read_bytes()
    adjudication_bytes = adjudication.read_bytes()
    partition_bytes = partitions.read_bytes()
    partition_data = json.loads(partition_bytes)
    if partition_data.get("source_jsonl_sha256") != _digest(original):
        raise ValueError("original corpus does not match the frozen partition source")
    bindings = set()
    for line in adjudication_bytes.splitlines():
        if not line.strip():
            raise ValueError("adjudication has an empty record")
        bindings.add(json.loads(line)["binding"]["source_sha256"])
    if len(bindings) != 1:
        raise ValueError("adjudication must have exactly one source binding")
    canonical = original.replace(b"\r\n", b"\n")
    expected = next(iter(bindings))
    if _digest(canonical) != expected:
        raise ValueError("line-ending reconciliation does not match the adjudication")
    manifest = {
        "status": "source_reconciled_partitions_not_regenerated",
        "transformation": "replace CRLF line endings with LF",
        "adjudication_source_sha256": expected,
        "input_sha256": {"corpus": _digest(original), "adjudication": _digest(adjudication_bytes),
                         "partitions": _digest(partition_bytes)},
        "output_sha256": {"corpus.jsonl": _digest(canonical)},
        "code_sha256": _digest(Path(__file__).read_bytes()),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "corpus.jsonl").write_bytes(canonical)
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "adjudication", "partitions", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.corpus, args.adjudication, args.partitions, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
