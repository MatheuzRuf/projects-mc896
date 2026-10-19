"""Train-only, typed relation rules over canonical unlabeled candidates."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Iterable

from project2_from_project1.src.clinical_documents import CanonicalRelation
from project2_from_project1.src.entity_recognition.corpus import load_document_folds
from project2_from_project1.src.relation_candidates import PAIR_RELATIONS, PATIENT_RELATIONS, logical_patient_id
from project2_from_project1.src.relation_evaluation import RelationPrediction, load_supervised_candidates, write_relation_predictions
from project2_from_project1.src.run_manifest import dependency_versions, git_provenance, sha256_file
from project2_from_project1.src.supervised_contracts import RelationCandidate, load_contract_jsonl
from project2_from_project1.src.supervised_dataset import SupervisedCandidate

METHOD = "canonical-relation-rules"
MODEL_VERSION = "canonical-relation-rules-v1"
TYPES = {**{relation: ("Patient", kind) for kind, relation in PATIENT_RELATIONS.items()},
         **{relation: (kind, "Finding") for kind, relation in PAIR_RELATIONS.items()}}


def _rule_key(candidate: RelationCandidate | SupervisedCandidate) -> tuple[str, int | None]:
    relation = candidate.candidate_relation.value
    if (candidate.source_type, candidate.target_type) != TYPES[relation]:
        raise ValueError("candidate types or direction are incompatible with its relation")
    distance = candidate.sentence_distance
    if (candidate.source_type == "Patient" and distance is not None) or (
        candidate.source_type != "Patient" and (type(distance) is not int or distance not in (0, 1))
    ):
        raise ValueError("candidate sentence distance is incompatible with its endpoints")
    return relation, distance


def fit_rules(candidates: Iterable[SupervisedCandidate], training_document_ids: Iterable[str]) -> dict:
    """Fit one majority decision per typed hypothesis and distance; ties are negative."""
    ids = tuple(training_document_ids)
    if not ids or any(not isinstance(item, str) or not item.strip() for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("training document IDs must be nonempty and unique")
    training_ids = set(ids)
    counts: dict[tuple[str, int | None], Counter] = {}
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, SupervisedCandidate) or candidate.document_id not in training_ids:
            raise ValueError("training candidates must belong to the training partition")
        key = _rule_key(candidate)
        pair = (candidate.document_id, candidate.source_id, candidate.target_id)
        if pair in seen:
            raise ValueError("duplicate training candidate pair")
        seen.add(pair)
        counts.setdefault(key, Counter())[candidate.label.value] += 1
    rules = []
    for relation in sorted(TYPES):
        for distance in ((None,) if TYPES[relation][0] == "Patient" else (0, 1)):
            counter = counts.get((relation, distance), Counter())
            positive, negative = counter[relation], counter[CanonicalRelation.NO_RELATION.value]
            rules.append({"rule_id": f"{relation.lower()}-{'patient' if distance is None else 'distance-' + str(distance)}",
                          "candidate_relation": relation, "sentence_distance": distance,
                          "positive_count": positive, "negative_count": negative,
                          "decision": relation if positive > negative else CanonicalRelation.NO_RELATION.value})
    return {"schema_version": 1, "method": METHOD, "model_version": MODEL_VERSION,
            "training_document_ids": sorted(ids), "rules": rules}


def predict_candidates(candidates: Iterable[RelationCandidate], model: dict) -> tuple[RelationPrediction, ...]:
    """Predict exactly once per compatible candidate, without access to gold."""
    if model.get("method") != METHOD or model.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported rule model")
    rules = {(rule["candidate_relation"], rule["sentence_distance"]): rule for rule in model["rules"]}
    expected = {(relation, distance) for relation, types in TYPES.items()
                for distance in ((None,) if types[0] == "Patient" else (0, 1))}
    if len(model["rules"]) != len(expected) or set(rules) != expected:
        raise ValueError("rule model must cover each hypothesis and distance exactly once")
    output, seen = [], set()
    for candidate in candidates:
        if not isinstance(candidate, RelationCandidate):
            raise ValueError("expected canonical relation candidates")
        key = _rule_key(candidate)
        pair = (candidate.document_id, candidate.source_id, candidate.target_id)
        if pair in seen:
            raise ValueError("duplicate candidate pair")
        seen.add(pair)
        if candidate.source_type == "Patient" and candidate.source_id != logical_patient_id(candidate.document_id):
            raise ValueError("invalid logical Patient ID")
        rule = rules[key]
        label = rule["decision"]
        if label not in (key[0], CanonicalRelation.NO_RELATION.value) or not isinstance(rule.get("rule_id"), str) or not rule["rule_id"]:
            raise ValueError("invalid rule decision or ID")
        output.append(RelationPrediction(candidate.document_id, candidate.source_id, candidate.target_id,
                                         label, label, METHOD, {"rule_id": rule["rule_id"], "candidate_id": candidate.candidate_id},
                                         model_version=MODEL_VERSION))
    return tuple(output)


def run_from_files(training_path: str | Path, partition_manifest_path: str | Path, fold_index: int,
                   candidates_path: str | Path, candidate_manifest_path: str | Path, output_dir: str | Path) -> dict:
    """Fit on the frozen training partition and run on its validation candidates."""
    started = time.perf_counter()
    training_path, partition_manifest_path, candidates_path, candidate_manifest_path, output_dir = map(
        Path, (training_path, partition_manifest_path, candidates_path, candidate_manifest_path, output_dir))
    inputs = (training_path, partition_manifest_path, candidates_path, candidate_manifest_path)
    outputs = tuple(output_dir / name for name in ("rules.json", "predictions.jsonl", "manifest.json"))
    if {path.resolve() for path in inputs} & {path.resolve() for path in outputs}:
        raise ValueError("output paths must not overwrite inputs")
    hashes = {name: sha256_file(path) for name, path in zip(("training", "partitions", "candidates", "candidate_manifest"), inputs)}
    partition_manifest = json.loads(partition_manifest_path.read_text(encoding="utf-8"))
    folds = {fold.index: fold for fold in load_document_folds(partition_manifest)}
    if fold_index not in folds:
        raise ValueError("unknown fold")
    fold = folds[fold_index]
    expected_name = f"fold-{fold_index:02d}-train.jsonl"
    if partition_manifest.get("artifact_hashes", {}).get(expected_name) != hashes["training"]:
        raise ValueError("training candidates do not match the frozen fold")
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("output_hashes", {}).get("candidates_sha256") != hashes["candidates"]:
        raise ValueError("candidates do not match their manifest")
    if candidate_manifest.get("input_hashes", {}).get("corpus_sha256") != partition_manifest.get("source_jsonl_sha256"):
        raise ValueError("candidate corpus differs from the frozen partition corpus")
    if set(candidate_manifest.get("prediction_document_ids") or ()) != set(fold.validation_document_ids):
        raise ValueError("candidate manifest must identify the complete validation partition")
    source_partitions = candidate_manifest.get("source_partitions") or {}
    if set(source_partitions.get("train_document_ids") or ()) != set(fold.train_document_ids):
        raise ValueError("recognizer and relation training partitions differ")
    model = fit_rules(load_supervised_candidates(training_path), fold.train_document_ids)
    fit_seconds = time.perf_counter() - started
    candidates = load_contract_jsonl(candidates_path, "candidate")
    validation_ids = set(fold.validation_document_ids)
    if any(candidate.document_id not in validation_ids for candidate in candidates):
        raise ValueError("candidate outside validation partition")
    predict_started = time.perf_counter()
    predictions = predict_candidates(candidates, model)
    predict_seconds = time.perf_counter() - predict_started
    if hashes != {name: sha256_file(path) for name, path in zip(("training", "partitions", "candidates", "candidate_manifest"), inputs)}:
        raise ValueError("input changed during the run")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(model, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    write_relation_predictions(predictions, outputs[1])
    counts = Counter(item.canonical_label.value for item in predictions)
    source_dir = Path(__file__).resolve().parent
    code_paths = {"rules": Path(__file__), "candidates": source_dir / "relation_candidates.py",
                  "prediction_contract": source_dir / "relation_evaluation.py", "contracts": source_dir / "supervised_contracts.py",
                  "dataset": source_dir / "supervised_dataset.py", "folds": source_dir / "entity_recognition" / "corpus.py",
                  "cli": source_dir.parent / "scripts" / "run_canonical_relation_rules.py"}
    manifest = {"schema_version": 1, "method": METHOD, "model_version": MODEL_VERSION, "fold": fold_index,
                "input_hashes": hashes, "output_hashes": {"rules_sha256": sha256_file(outputs[0]), "predictions_sha256": sha256_file(outputs[1])},
                "code_sha256": sha256_file(__file__), "code_hashes": {name: sha256_file(path) for name, path in code_paths.items()},
                "counts": {"candidates": len(candidates), "predictions": len(predictions), "by_label": dict(sorted(counts.items()))},
                "configuration": {"features": ["candidate_relation", "sentence_distance"], "positive_decision": "strict_majority", "tie_or_unseen": "NO_RELATION"},
                "determinism": {"randomness": "none", "prediction_order": "document_source_target"},
                "partitions": {"train_document_ids": list(fold.train_document_ids), "validation_document_ids": list(fold.validation_document_ids)},
                "label_access": {"fit": "training candidates only", "predict": "unlabeled candidates only"},
                "timing_seconds": {"fit_and_input_validation": round(fit_seconds, 6), "predict": round(predict_seconds, 6), "total": round(time.perf_counter() - started, 6)},
                "git": {**git_provenance(), "dirty_scope": "tracked_files"}, "environment": dependency_versions()}
    outputs[2].write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest
