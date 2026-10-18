"""Export the sensitive local packet used by independent human reviewers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from project2_from_project1.src.clinical_documents import load_jsonl
from project2_from_project1.src.gold_projection import (
    AdjudicationError,
    MAX_MENTION_TOKENS,
    build_adjudication_binding,
    build_projection_plan,
    write_adjudication_review_packet,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_review_packet(
    source_jsonl: str | Path,
    output_dir: str | Path,
    *,
    max_mention_tokens: int = MAX_MENTION_TOKENS,
) -> dict[str, object]:
    source = Path(source_jsonl).resolve()
    output = Path(output_dir).resolve()
    packet = output / "review-cases.jsonl"
    manifest_path = output / "manifest.json"
    if len({source, packet, manifest_path}) != 3:
        raise AdjudicationError("source, review packet, and manifest paths must all be different")
    source_hash = _sha256(source)
    plan = build_projection_plan(
        load_jsonl(source),
        max_mention_tokens=max_mention_tokens,
    )
    binding = build_adjudication_binding(
        plan,
        source_hash,
        max_mention_tokens=max_mention_tokens,
    )
    output.mkdir(parents=True, exist_ok=True)
    staged_packet: Path | None = None
    staged_manifest: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as handle:
            staged_packet = Path(handle.name)
        write_adjudication_review_packet(plan, binding, staged_packet)
        manifest: dict[str, object] = {
            "binding": binding.to_dict(),
            "cases": len(plan.cases),
            "packet_sha256": _sha256(staged_packet),
            "schema_version": 1,
            "sensitive_clinical_text": True,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output, delete=False,
        ) as handle:
            handle.write(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            staged_manifest = Path(handle.name)
        if _sha256(source) != source_hash:
            raise AdjudicationError("lossless source changed while exporting the review packet")
        staged_packet.replace(packet)
        staged_packet = None
        if _sha256(source) != source_hash:
            raise AdjudicationError("lossless source changed while publishing the review packet")
        staged_manifest.replace(manifest_path)
        staged_manifest = None
    finally:
        for staged in (staged_packet, staged_manifest):
            if staged is not None and staged.exists():
                staged.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, help="Lossless ClinicalDocument JSONL")
    parser.add_argument("--output-dir", required=True, help="Access-controlled output directory")
    parser.add_argument("--max-mention-tokens", type=int, default=MAX_MENTION_TOKENS)
    args = parser.parse_args()

    manifest = export_review_packet(
        args.source_jsonl,
        args.output_dir,
        max_mention_tokens=args.max_mention_tokens,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
