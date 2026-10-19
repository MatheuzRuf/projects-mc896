import json

import pytest

from project2_from_project1.src.clinical_documents import CanonicalRelation, Span, write_jsonl
from project2_from_project1.src.entity_recognition.corpus import RawDocument
from project2_from_project1.src.gold_projection import SENTENCE_SEGMENTER_VERSION, TOKENIZER_VERSION
from project2_from_project1.src.ngrams.pipeline import build_candidate_examples, train_and_predict_ngram_knn
from project2_from_project1.src.ngrams.predicted_pipeline import run_from_files
from project2_from_project1.src.relation_candidates import generate_candidates, run_from_files as generate_from_files
from project2_from_project1.src.relation_evaluation import load_relation_predictions
from project2_from_project1.src.run_manifest import sha256_file
from project2_from_project1.src.supervised_contracts import EntityPrediction, write_contract_jsonl
from project2_from_project1.tests.test_ngram_pipeline import _candidate, _document


def test_predicted_features_use_spans_without_labels_or_gold_mentions():
    train = _document(1, "severe pain today")
    validation = _document(2, "rare pain today")
    entity = EntityPrediction(validation.document_id, Span(0, 4), "Finding", 0, "dictionary", "1")
    candidates = generate_candidates([RawDocument(validation.document_id, validation.text)], [entity])
    examples = build_candidate_examples([validation], candidates, include_labels=False, entity_predictions=[entity])
    assert examples[0].label is None
    assert examples[0].ngram_counts[0]["rare"] == 1
    result = train_and_predict_ngram_knn([train, validation], [_candidate(train, CanonicalRelation.HAS_FINDING)],
                                         candidates, n_neighbors=1, entity_predictions=[entity])
    assert (1, ("rare",)) not in result.vocabulary
    assert result.predictions[0].evidence["candidate_id"] == candidates[0].candidate_id
    assert result.predictions[0].neighbor_vote_fraction == 1.0
    assert result.predictions[0].confidence is None
    with pytest.raises(ValueError, match="predicted entity"):
        build_candidate_examples([validation], candidates, include_labels=False)


def test_file_run_applies_one_model_to_both_arms_and_checks_artifacts(tmp_path):
    train = _document(1, "severe pain today")
    validation = _document(2, "severe pain today")
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl([train, validation], corpus)
    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps(_candidate(train, CanonicalRelation.HAS_FINDING).to_dict()) + "\n", encoding="utf-8")
    partition = tmp_path / "partitions.json"
    partition.write_text(json.dumps({"source_jsonl_sha256": sha256_file(corpus),
        "artifact_hashes": {"fold-01-train.jsonl": sha256_file(training)},
        "partitions": {"test_document_ids": [], "folds": [{"fold": 1, "train_document_ids": [train.document_id],
                        "validation_document_ids": [validation.document_id]}]}}), encoding="utf-8")
    arms = {}
    for method in ("dictionary", "crf"):
        recognizer_method = {"dictionary": "canonical-dictionary", "crf": "semi-markov-crf"}[method]
        entity_path = tmp_path / f"{method}.jsonl"
        write_contract_jsonl([EntityPrediction(validation.document_id, Span(7, 11), "Finding", 0, recognizer_method, "1")], entity_path)
        entity_manifest = tmp_path / f"{method}-manifest.json"
        entity_manifest.write_text(json.dumps({"method": recognizer_method, "model_version": "1", "artifact_hashes": {"corpus_jsonl_sha256": sha256_file(corpus),
            "predictions_jsonl_sha256": sha256_file(entity_path)},
            "configuration": {"sentence_segmenter_version": SENTENCE_SEGMENTER_VERSION, "tokenizer_version": TOKENIZER_VERSION},
            "partitions": {"train_document_ids": [train.document_id], "validation_document_ids": [validation.document_id]}}), encoding="utf-8")
        candidate_dir = tmp_path / f"{method}-candidates"
        generate_from_files(corpus, entity_path, candidate_dir, entity_manifest)
        arms[method] = (candidate_dir / "candidates.jsonl", candidate_dir / "manifest.json", entity_path)
    output = tmp_path / "output"
    manifests = run_from_files(corpus, training, partition, 1, arms, output, n_neighbors=1, maximum_ngram_order=3)
    dictionary = load_relation_predictions(output / "dictionary" / "predictions.jsonl")
    crf = load_relation_predictions(output / "crf" / "predictions.jsonl")
    assert [item.to_dict() for item in dictionary] == [item.to_dict() for item in crf]
    assert len(dictionary) == 1 and dictionary[0].canonical_label == CanonicalRelation.HAS_FINDING
    assert manifests["dictionary"]["counts"]["candidates"] == 1
    assert manifests["dictionary"]["output_hashes"]["predictions_sha256"] == sha256_file(output / "dictionary" / "predictions.jsonl")
    with pytest.raises(ValueError, match="different recognizer"):
        run_from_files(corpus, training, partition, 1, {"dictionary": arms["dictionary"], "crf": arms["dictionary"]},
                       tmp_path / "wrong-arm", n_neighbors=1, maximum_ngram_order=3)
    arms["crf"][2].write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="inputs do not match"):
        run_from_files(corpus, training, partition, 1, arms, tmp_path / "bad", n_neighbors=1, maximum_ngram_order=3)
    entity_manifest = tmp_path / "crf-manifest.json"
    source = json.loads(entity_manifest.read_text(encoding="utf-8"))
    source["artifact_hashes"]["predictions_jsonl_sha256"] = sha256_file(arms["crf"][2])
    entity_manifest.write_text(json.dumps(source), encoding="utf-8")
    generate_from_files(corpus, arms["crf"][2], arms["crf"][0].parent, entity_manifest)
    empty_output = tmp_path / "empty-output"
    empty = run_from_files(corpus, training, partition, 1, arms, empty_output, n_neighbors=1, maximum_ngram_order=3)
    assert empty["crf"]["counts"]["predictions"] == 0
    assert load_relation_predictions(empty_output / "crf" / "predictions.jsonl") == []
    assert len(load_relation_predictions(empty_output / "dictionary" / "predictions.jsonl")) == 1
