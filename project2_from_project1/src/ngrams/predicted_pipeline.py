"""Apply one fold's gold-trained k-NN to two predicted-entity arms."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

from project2_from_project1.src.clinical_documents import ClinicalDocument
from project2_from_project1.src.entity_recognition.corpus import RawDocument, load_document_folds
from project2_from_project1.src.entity_recognition import dictionary, crf_model
from project2_from_project1.src.ngrams.pipeline import train_and_predict_ngram_knn
from project2_from_project1.src.relation_candidates import generate_candidates
from project2_from_project1.src.relation_evaluation import load_supervised_candidates, write_relation_predictions
from project2_from_project1.src.run_manifest import dependency_versions, git_provenance, sha256_file
from project2_from_project1.src.supervised_contracts import load_contract_jsonl


def run_from_files(corpus_path, training_path, partition_path, fold_index, arms, output_dir, *,
                   n_neighbors, maximum_ngram_order, model_version="ngram-knn-v1"):
    """Fit once and write aligned predictions and provenance for each named arm.

    Each arm maps to (candidate JSONL, candidate manifest, entity JSONL).
    """
    started = time.perf_counter()
    corpus_path, training_path, partition_path, output_dir = map(Path, (corpus_path, training_path, partition_path, output_dir))
    if not arms or not set(arms) <= {"dictionary", "crf"}:
        raise ValueError("arms must be a non-empty subset of dictionary and crf")
    arms = {name: tuple(map(Path, paths)) for name, paths in arms.items()}
    if any(len(paths) != 3 for paths in arms.values()):
        raise ValueError("each arm requires candidates, manifest and entity predictions")
    inputs = {"corpus": corpus_path, "training": training_path, "partitions": partition_path}
    for name, paths in arms.items():
        inputs.update({f"{name}_{key}": path for key, path in zip(("candidates", "candidate_manifest", "entities"), paths)})
    outputs = [output_dir / name / filename for name in arms for filename in ("predictions.jsonl", "manifest.json")]
    if {path.resolve() for path in inputs.values()} & {path.resolve() for path in outputs}:
        raise ValueError("output paths must not overwrite inputs")
    hashes = {name: sha256_file(path) for name, path in inputs.items()}
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    folds = {fold.index: fold for fold in load_document_folds(partition)}
    if fold_index not in folds:
        raise ValueError("unknown fold")
    fold = folds[fold_index]
    if partition.get("source_jsonl_sha256") != hashes["corpus"]:
        raise ValueError("corpus differs from frozen partition")
    if partition.get("artifact_hashes", {}).get(f"fold-{fold_index:02d}-train.jsonl") != hashes["training"]:
        raise ValueError("training candidates do not match frozen fold")
    train_ids, validation_ids = set(fold.train_document_ids), set(fold.validation_document_ids)
    training = load_supervised_candidates(training_path)
    if any(item.document_id not in train_ids for item in training):
        raise ValueError("training candidate outside training partition")
    documents, raw_documents, seen = [], [], set()
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        document_id, text = record["document_id"], record["text"]
        if document_id in seen:
            raise ValueError("duplicate corpus document")
        seen.add(document_id)
        if document_id in train_ids:
            documents.append(ClinicalDocument.from_dict(record))
        elif document_id in validation_ids:
            documents.append(ClinicalDocument(document_id, text, [], [], [], []))
            raw_documents.append(RawDocument(document_id, text))
    if not (train_ids | validation_ids) <= seen:
        raise ValueError("fold references a missing corpus document")
    candidates_by_arm, entities = {}, []
    for name, (candidate_path, manifest_path, entity_path) in arms.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output_hashes", {}).get("candidates_sha256") != hashes[f"{name}_candidates"]:
            raise ValueError(f"{name} candidates do not match their manifest")
        source_hashes = manifest.get("input_hashes", {})
        if source_hashes.get("corpus_sha256") != hashes["corpus"] or source_hashes.get("entity_predictions_sha256") != hashes[f"{name}_entities"]:
            raise ValueError(f"{name} inputs do not match candidate manifest")
        if set(manifest.get("prediction_document_ids") or ()) != validation_ids:
            raise ValueError(f"{name} manifest must identify the validation partition")
        if set((manifest.get("source_partitions") or {}).get("train_document_ids") or ()) != train_ids:
            raise ValueError(f"{name} recognizer training partition differs")
        arm_entities = load_contract_jsonl(entity_path, "entity")
        expected_method = {"dictionary": dictionary.METHOD, "crf": crf_model.METHOD}[name]
        recognizer = manifest.get("source_recognizer") or {}
        if recognizer.get("method") != expected_method or not recognizer.get("model_version"):
            raise ValueError(f"{name} manifest identifies a different recognizer")
        if any((item.method, item.model_version) != (expected_method, recognizer["model_version"]) for item in arm_entities):
            raise ValueError(f"{name} entity predictions identify a different recognizer")
        arm_candidates = load_contract_jsonl(candidate_path, "candidate")
        if tuple(arm_candidates) != generate_candidates(raw_documents, arm_entities):
            raise ValueError(f"{name} candidates do not match predicted entities")
        candidates_by_arm[name] = arm_candidates
        entities.extend(arm_entities)
    combined = [candidate for name in arms for candidate in candidates_by_arm[name]]
    run = train_and_predict_ngram_knn(documents, training, combined, n_neighbors=n_neighbors,
                                      maximum_ngram_order=maximum_ngram_order, model_version=model_version,
                                      entity_predictions=entities)
    if hashes != {name: sha256_file(path) for name, path in inputs.items()}:
        raise ValueError("input changed during the run")
    offset, results = 0, {}
    code_dir = Path(__file__).resolve().parent
    code_paths = {"predicted_pipeline": Path(__file__), "pipeline": code_dir / "pipeline.py",
                  "features": code_dir / "feature_matrix.py", "extractor": code_dir / "ngrams_extractor.py",
                  "classifier": code_dir / "knn_classifier.py", "preprocessing": code_dir.parent / "preprocessing" / "text.py",
                  "candidates": code_dir.parent / "relation_candidates.py", "contracts": code_dir.parent / "supervised_contracts.py",
                  "prediction_contract": code_dir.parent / "relation_evaluation.py", "folds": code_dir.parent / "entity_recognition" / "corpus.py",
                  "cli": code_dir.parent.parent / "scripts" / "run_predicted_ngram_knn.py"}
    for name in arms:
        arm_candidates = candidates_by_arm[name]
        predictions = run.predictions[offset:offset + len(arm_candidates)]
        offset += len(arm_candidates)
        arm_dir = output_dir / name
        prediction_path = arm_dir / "predictions.jsonl"
        write_relation_predictions(predictions, prediction_path)
        counts = Counter(item.canonical_label.value for item in predictions)
        manifest = {**run.manifest_dict(), "schema_version": 1, "fold": fold_index, "arm": name,
                    "model_version": model_version, "prediction_examples": len(predictions),
                    "input_hashes": hashes, "output_hashes": {"predictions_sha256": sha256_file(prediction_path)},
                    "counts": {"candidates": len(arm_candidates), "predictions": len(predictions), "by_label": dict(sorted(counts.items()))},
                    "partitions": {"train_document_ids": list(fold.train_document_ids), "validation_document_ids": list(fold.validation_document_ids)},
                    "label_access": {"fit": "training candidates only", "predict": "unlabeled candidates and predicted entity spans"},
                    "code_hashes": {key: sha256_file(path) for key, path in code_paths.items()},
                    "git": {**git_provenance(), "dirty_scope": "tracked_files"}, "environment": dependency_versions(),
                    "duration_seconds": round(time.perf_counter() - started, 6)}
        (arm_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        results[name] = manifest
    return results
