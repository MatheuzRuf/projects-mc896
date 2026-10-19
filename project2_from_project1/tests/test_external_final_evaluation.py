"""Exercise external inference and reporting through the public command."""

import json
from pathlib import Path
import random
import subprocess
import sys

from project2_from_project1.src.clinical_documents import ClinicalDocument, write_jsonl
from project2_from_project1.src.gold_projection import project_gold_from_files
from project2_from_project1.src.run_manifest import sha256_file
from project2_from_project1.src.synthetic_fixture import _document, make_synthetic_workspace


def test_synthetic_external_run_preserves_all_arms_and_inconclusive_labels(tmp_path):
    root = Path(__file__).resolve().parents[2]
    workspace = make_synthetic_workspace(tmp_path / "training")
    projection = tmp_path / "projection"
    projection.mkdir()
    project_gold_from_files(workspace.corpus_jsonl, workspace.adjudication_jsonl,
                            projection / "gold.jsonl", projection / "audit.jsonl", projection / "manifest.json")
    models = tmp_path / "models"
    prepare = [sys.executable, "-m", "project2_from_project1.scripts.prepare_external_models",
               "--corpus", str(workspace.corpus_jsonl), "--projection", str(projection / "gold.jsonl"),
               "--projection-manifest", str(projection / "manifest.json"),
               "--partitions", str(workspace.frozen_dataset / "manifest.json"),
               "--output-dir", str(models), "--max-iterations", "1"]
    run = subprocess.run(prepare, cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    gold = [_document(index, random.Random(index)) for index in range(100, 103)]
    raw = [ClinicalDocument(item.document_id, item.text, (), (), (), (), item.metadata) for item in gold]
    # An assembly can retain source-screened records; they must not be scored
    # when absent from the raw, eligible snapshot.
    gold = [*gold, _document(103, random.Random(103))]
    raw_path, gold_path = tmp_path / "raw.jsonl", tmp_path / "gold.jsonl"
    write_jsonl(raw, raw_path)
    write_jsonl(gold, gold_path)
    output = tmp_path / "results"
    command = [sys.executable, "-m", "project2_from_project1.scripts.run_external_evaluation",
               "--models", str(models), "--raw", str(raw_path), "--gold", str(gold_path),
               "--training-corpus", str(workspace.corpus_jsonl), "--training-projection", str(projection / "gold.jsonl"),
               "--output-dir", str(output)]
    run = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["documents"] == 3
    assert report["gold_records_excluded_from_snapshot"] == 1
    assert set(report["arms"]) == {"dictionary+rules", "dictionary+knn", "crf+rules", "crf+knn", "gold+rules", "gold+knn"}
    assert report["arms"]["gold+knn"]["role"] == "diagnostic_ceiling"
    assert report["arms"]["crf+knn"]["crf_converged"] is False
    assert report["arms"]["dictionary+rules"]["relations"]["per_label"]["CAUSES"]["conclusion"] == "inconclusive_insufficient_support"
    assert report["arms"]["dictionary+knn"]["graph"] is not None
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert all(sha256_file(output / name) == value for name, value in manifest["artifact_sha256"].items())
    assert subprocess.run(command, cwd=root, capture_output=True, text=True).returncode != 0
