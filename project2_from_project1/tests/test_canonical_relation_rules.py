import json

import pytest

from project2_from_project1.src.canonical_relation_rules import fit_rules, predict_candidates, run_from_files
from project2_from_project1.src.clinical_documents import CanonicalRelation, Span
from project2_from_project1.src.entity_recognition.corpus import RawDocument
from project2_from_project1.src.gold_projection import SENTENCE_SEGMENTER_VERSION, TOKENIZER_VERSION
from project2_from_project1.src.relation_candidates import generate_candidates, run_from_files as generate_from_files
from project2_from_project1.src.relation_evaluation import load_relation_predictions, write_relation_predictions
from project2_from_project1.src.run_manifest import sha256_file
from project2_from_project1.src.supervised_contracts import EntityPrediction, RelationCandidate, write_contract_jsonl
from project2_from_project1.src.supervised_dataset import SupervisedCandidate


def training(relation, source_type, target_type, distance, label, suffix):
    return SupervisedCandidate("train", "source" + suffix, "target" + suffix, CanonicalRelation(relation),
                               CanonicalRelation(label), source_type, target_type, distance)


def test_empty_prediction_contract_round_trip(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_relation_predictions([], path)
    assert path.read_bytes() == b""
    assert load_relation_predictions(path) == []


def test_public_rules_preserve_hypothesis_direction_and_reject_leakage():
    examples = [training("NEGATES", "Modifier", "Finding", 0, "NEGATES", "1"),
                training("HAS_FINDING", "Patient", "Finding", None, "HAS_FINDING", "2"),
                training("CAUSES", "Condition", "Finding", 0, "CAUSES", "3"),
                training("CAUSES", "Condition", "Finding", 0, "NO_RELATION", "4")]
    model = fit_rules(examples, ["train"])
    doc = RawDocument("validation", "Asthma no cough.")
    entities = [EntityPrediction(doc.document_id, Span(start, end), kind, 0, "dictionary", "1")
                for start, end, kind in [(0, 6, "Condition"), (7, 9, "Modifier"), (10, 15, "Finding")]]
    candidates = generate_candidates([doc], entities)
    predictions = predict_candidates(candidates, model)
    by_pair = {prediction.pair_key: prediction for prediction in predictions}
    assert len(predictions) == len(candidates)
    for candidate in candidates:
        result = by_pair[(candidate.document_id, candidate.source_id, candidate.target_id)]
        assert result.canonical_label.value in (candidate.candidate_relation.value, "NO_RELATION")
        assert result.evidence["candidate_id"] == candidate.candidate_id
        assert result.evidence["rule_id"]
    assert next(item for item in predictions if item.evidence["rule_id"] == "negates-distance-0").canonical_label.value == "NEGATES"
    assert next(item for item in predictions if item.evidence["rule_id"] == "causes-distance-0").canonical_label.value == "NO_RELATION"
    with pytest.raises(ValueError, match="training partition"):
        fit_rules(examples, ["validation"])
    bad = RelationCandidate("validation", "source", "target", "Finding", "Condition", "CAUSES", 0)
    with pytest.raises(ValueError, match="direction"):
        predict_candidates([bad], model)
    with pytest.raises(ValueError, match="duplicate"):
        predict_candidates([candidates[0], candidates[0]], model)


def test_file_pipeline_uses_same_rules_for_recognizers_and_freezes_artifacts(tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps(training("HAS_FINDING", "Patient", "Finding", None, "HAS_FINDING", "1").to_dict()) + "\n", encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"document_id": "validation", "text": "Cough.", "relations": ["gold poison"]}) + "\n", encoding="utf-8")
    partitions = tmp_path / "partitions.json"
    partitions.write_text(json.dumps({"source_jsonl_sha256": sha256_file(corpus),
        "artifact_hashes": {"fold-01-train.jsonl": sha256_file(train_file)},
        "partitions": {"test_document_ids": [], "folds": [{"fold": 1, "train_document_ids": ["train"], "validation_document_ids": ["validation"]}]}}), encoding="utf-8")
    results = []
    for method in ("dictionary", "crf"):
        entities = tmp_path / (method + ".jsonl")
        source = tmp_path / (method + "-manifest.json")
        write_contract_jsonl([EntityPrediction("validation", Span(0, 5), "Finding", 0, method, "1")], entities)
        source.write_text(json.dumps({"artifact_hashes": {"corpus_jsonl_sha256": sha256_file(corpus), "predictions_jsonl_sha256": sha256_file(entities)},
            "configuration": {"sentence_segmenter_version": SENTENCE_SEGMENTER_VERSION, "tokenizer_version": TOKENIZER_VERSION},
            "partitions": {"train_document_ids": ["train"], "validation_document_ids": ["validation"]}}), encoding="utf-8")
        candidate_dir = tmp_path / (method + "-candidates")
        generate_from_files(corpus, entities, candidate_dir, source)
        output = tmp_path / (method + "-output")
        manifest = run_from_files(train_file, partitions, 1, candidate_dir / "candidates.jsonl", candidate_dir / "manifest.json", output)
        predictions = load_relation_predictions(output / "predictions.jsonl")
        assert len(predictions) == 1 and predictions[0].canonical_label.value == "HAS_FINDING"
        assert manifest["output_hashes"]["predictions_sha256"] == sha256_file(output / "predictions.jsonl")
        assert manifest["code_hashes"]["rules"] == manifest["code_sha256"]
        assert manifest["git"]["dirty_scope"] == "tracked_files"
        assert "Cough" not in (output / "manifest.json").read_text(encoding="utf-8")
        assert "gold poison" not in (output / "predictions.jsonl").read_text(encoding="utf-8")
        results.append(((output / "rules.json").read_bytes(), (output / "predictions.jsonl").read_bytes()))
    assert results[0] == results[1]
    write_contract_jsonl([], entities)
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_payload["artifact_hashes"]["predictions_jsonl_sha256"] = sha256_file(entities)
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    generate_from_files(corpus, entities, candidate_dir, source)
    empty_output = tmp_path / "empty"
    empty_manifest = run_from_files(train_file, partitions, 1, candidate_dir / "candidates.jsonl", candidate_dir / "manifest.json", empty_output)
    assert empty_manifest["counts"]["predictions"] == 0
    assert load_relation_predictions(empty_output / "predictions.jsonl") == []
    train_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen fold"):
        run_from_files(train_file, partitions, 1, candidate_dir / "candidates.jsonl", candidate_dir / "manifest.json", tmp_path / "bad")
