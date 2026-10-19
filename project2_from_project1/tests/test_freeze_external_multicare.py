from types import SimpleNamespace

import pytest

from project2_from_project1.scripts.freeze_external_multicare import (
    EXPECTED_PREPARATION_SOURCE_HASHES, choose_prefix, validate_eligibility_coverage,
    validate_preparation_manifest,
)


def _documents(causes_per_document: int, count: int):
    entity_types = ("Condition", "Factor", "Finding", "Modifier")
    relation_types = ("HAS_CONDITION", "HAS_FACTOR", "HAS_FINDING", "NEGATES")
    return [SimpleNamespace(
        mentions=[SimpleNamespace(canonical_type=kind) for kind in entity_types for _ in range(2)],
        relations=[SimpleNamespace(canonical_label=SimpleNamespace(value=kind))
                   for kind in relation_types for _ in range(1)] +
                  [SimpleNamespace(canonical_label=SimpleNamespace(value="CAUSES"))
                   for _ in range(causes_per_document)],
    ) for _ in range(count)]


def test_stops_at_first_supported_document_after_fifty():
    chosen, support = choose_prefix(_documents(1, 70))
    assert len(chosen) == 50
    assert support["stop_reason"] == "all_support_thresholds_met"
    assert support["inconclusive_relations"] == []


def test_only_cap_allows_inconclusive_relation():
    with pytest.raises(ValueError, match="cap not reached"):
        choose_prefix(_documents(0, 99))
    chosen, support = choose_prefix(_documents(0, 100))
    assert len(chosen) == 100
    assert support["inconclusive_relations"] == ["CAUSES"]


def test_frozen_prefix_must_be_covered_by_source_screen():
    released = [f"doc-{number}" for number in range(1, 6)]

    validate_eligibility_coverage(released, ["doc-2"], 4, ["doc-1", "doc-3", "doc-4"])

    with pytest.raises(ValueError, match="does not cover frozen prefix"):
        validate_eligibility_coverage(released, ["doc-2"], 4,
                                      ["doc-1", "doc-3", "doc-4", "doc-5"])


def test_source_exclusion_must_be_inside_screened_prefix():
    with pytest.raises(ValueError, match="exclusion is outside screened prefix"):
        validate_eligibility_coverage(["doc-1", "doc-2", "doc-3"], ["doc-3"], 2, ["doc-1"])


def test_preparation_manifest_binds_outputs_and_source_provenance(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    from project2_from_project1.scripts import prepare_external_multicare_continuation

    outputs = {}
    paths = {}
    for name in ("selection.json", "mcrc-leakage.json", "annotation-packet.jsonl"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
        outputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "status": "continuation_packet_not_gold_not_frozen",
        "created_at_utc": "2026-10-19T00:00:00+00:00",
        "source_hashes": EXPECTED_PREPARATION_SOURCE_HASHES,
        "preparation_code_sha256": hashlib.sha256(
            Path(prepare_external_multicare_continuation.__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "output_hashes": outputs,
    }), encoding="utf-8")

    validate_preparation_manifest(manifest_path, paths["selection.json"],
                                  paths["mcrc-leakage.json"], paths["annotation-packet.jsonl"])
    paths["selection.json"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="preparation output changed"):
        validate_preparation_manifest(manifest_path, paths["selection.json"],
                                      paths["mcrc-leakage.json"], paths["annotation-packet.jsonl"])
