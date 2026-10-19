"""Public seam: raw synthetic text to the Knowledge Graph for every arm."""

import json

import pytest

from project2_from_project1.src.entity_recognition.crf_model import NonFiniteTrainingError
from project2_from_project1.src.run_manifest import sha256_file
from project2_from_project1.src.study_pipeline import run_full_study
from project2_from_project1.src.supervised_contracts import load_contract_jsonl
from project2_from_project1.src.synthetic_fixture import make_synthetic_workspace

END_TO_END = ("dictionary+rules", "dictionary+knn", "crf+rules", "crf+knn")


def _exploding_fit(documents, config):
    raise NonFiniteTrainingError("semi-Markov CRF objective or gradient is not finite",
                                 {"function_evaluations": 1, "objective_finite": False, "gradient_finite": False})


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    workspace = make_synthetic_workspace(root / "workspace")
    manifest = run_full_study(workspace.corpus_jsonl, workspace.adjudication_jsonl, workspace.frozen_dataset,
                              root / "output", max_iterations=20)
    return workspace, root / "output", manifest


def test_every_arm_runs_from_raw_text_to_graph_with_valid_contracts(study):
    workspace, output, manifest = study
    assert set(manifest["arms"]) == {*END_TO_END, "gold+rules", "gold+knn"}
    for arm in END_TO_END:
        assert manifest["arms"][arm]["role"] == "end_to_end"
        assert manifest["arms"][arm]["completed_folds"] == [1, 2] and manifest["arms"][arm]["failed_folds"] == []
    assert manifest["arms"]["gold+knn"]["role"] == "diagnostic_ceiling"
    for fold in ("fold-01", "fold-02"):
        load_contract_jsonl(output / "entities" / "dictionary" / fold / "predictions.jsonl", "entity")
        load_contract_jsonl(output / "entities" / "crf" / fold / "outer" / "predictions.jsonl", "entity")
        for recognizer in ("dictionary", "crf"):
            candidates = load_contract_jsonl(output / "relations" / "candidates" / recognizer / fold / "candidates.jsonl", "candidate")
            assert candidates and all("label" not in item.to_dict() for item in candidates)
        for arm in ("dictionary-rules", "dictionary-knn", "crf-rules", "crf-knn"):
            graph = load_contract_jsonl(output / "graph" / arm / fold / "graph.jsonl", "graph")
            assert {item.layer for item in graph} == {"mention", "concept"}


def test_root_manifest_references_every_child_artifact_and_never_the_test_partition(study):
    workspace, output, manifest = study
    files = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} - {"manifest.json"}
    assert set(manifest["artifact_hashes"]) == files
    for name, digest in manifest["artifact_hashes"].items():
        assert sha256_file(output / name) == digest
    assert manifest["git"]["commit"] and manifest["dependencies"]["packages"]["torch"]
    assert set(manifest["stage_seconds"]) == {"gold_projection", "entity_recognition", "relations", "graph"}
    assert [item["fold"] for item in manifest["partitions"]] == [1, 2]
    assert manifest["test_partition_read"] is False
    test_ids = json.loads((workspace.frozen_dataset / "manifest.json").read_text(encoding="utf-8"))["partitions"]["test_document_ids"]
    for path in (output / "entities").rglob("predictions.jsonl"):
        assert not any(json.loads(line)["document_id"] in test_ids for line in path.read_text(encoding="utf-8").splitlines())


def test_rerun_is_deterministic(study, tmp_path):
    workspace, output, manifest = study
    again = run_full_study(workspace.corpus_jsonl, workspace.adjudication_jsonl, workspace.frozen_dataset,
                           tmp_path / "again", max_iterations=20)
    stable = [name for name in manifest["artifact_hashes"]
              if name.endswith(("predictions.jsonl", "candidates.jsonl", "graph.jsonl", "gold-projection.jsonl"))]
    assert stable
    for name in stable:
        assert again["artifact_hashes"][name] == manifest["artifact_hashes"][name], name


def test_non_finite_crf_fails_only_its_arms_without_fallback(tmp_path):
    workspace = make_synthetic_workspace(tmp_path / "workspace")
    manifest = run_full_study(workspace.corpus_jsonl, workspace.adjudication_jsonl, workspace.frozen_dataset,
                              tmp_path / "output", max_iterations=20, crf_fit=_exploding_fit)
    for arm in ("crf+rules", "crf+knn"):
        assert manifest["arms"][arm]["failed_folds"] == [1, 2] and manifest["arms"][arm]["relation_macro_f1"] is None
    for arm in ("dictionary+rules", "dictionary+knn", "gold+rules", "gold+knn"):
        assert manifest["arms"][arm]["completed_folds"] == [1, 2]
    outer = tmp_path / "output" / "entities" / "crf" / "fold-01" / "outer"
    assert json.loads((outer / "manifest.json").read_text(encoding="utf-8"))["status"] == "failed_non_finite"
    assert not (outer / "predictions.jsonl").exists()
    assert not (tmp_path / "output" / "relations" / "candidates" / "crf").exists()
    relations = json.loads((tmp_path / "output" / "relations" / "summary.json").read_text(encoding="utf-8"))
    assert relations["arms"]["crf+knn"]["folds"][0]["status"] == "skipped_failed_recognizer"
    assert relations["arms"]["crf+knn"]["folds"][0]["recognizer_status"] == "failed_non_finite"
