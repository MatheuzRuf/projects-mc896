"""Public preparation seam: models are fixed using development documents only."""

import json
from pathlib import Path
import subprocess
import sys

from project2_from_project1.src.gold_projection import project_gold_from_files
from project2_from_project1.src.run_manifest import sha256_file
from project2_from_project1.src.synthetic_fixture import make_synthetic_workspace


def test_prepare_external_models_freezes_development_only(tmp_path):
    workspace = make_synthetic_workspace(tmp_path / "input")
    projection = tmp_path / "projection"
    projection.mkdir()
    project_gold_from_files(workspace.corpus_jsonl, workspace.adjudication_jsonl,
                            projection / "gold.jsonl", projection / "audit.jsonl", projection / "manifest.json")
    output = tmp_path / "models"
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "project2_from_project1.scripts.prepare_external_models",
               "--corpus", str(workspace.corpus_jsonl), "--projection", str(projection / "gold.jsonl"),
               "--projection-manifest", str(projection / "manifest.json"),
               "--partitions", str(workspace.frozen_dataset / "manifest.json"),
               "--output-dir", str(output), "--max-iterations", "1"]
    run = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    partitions = json.loads((workspace.frozen_dataset / "manifest.json").read_text(encoding="utf-8"))["partitions"]
    assert not set(manifest["training_document_ids"]) & set(partitions["test_document_ids"])
    assert manifest["crf"]["status"] == "completed"
    assert manifest["crf"]["converged"] is False
    assert manifest["configuration"]["l2"] == 0.1
    assert set(manifest["artifact_sha256"]) >= {"dictionary.jsonl", "crf.json", "rules.json", "knn.json", "knn-matrix.npz"}
    for name, expected in manifest["artifact_sha256"].items():
        assert sha256_file(output / name) == expected
    assert subprocess.run(command, cwd=root, capture_output=True, text=True).returncode != 0
