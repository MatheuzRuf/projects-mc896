"""Factorial end-to-end relation study: {dictionary, CRF} x {rules, k-NN}.

Per outer fold, entity predictions from the nested entity study are turned into
unlabeled candidates, classified by the fold's train-only rules and by one
fold-trained n-gram k-NN, and evaluated strictly against the flat gold.  The
same classifiers applied to gold candidates form the gold-entity ceiling,
reported as a diagnostic and never as an end-to-end pipeline.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from project2_from_project1.src import canonical_relation_rules as rules
from project2_from_project1.src import relation_candidates
from project2_from_project1.src.clinical_documents import load_jsonl
from project2_from_project1.src.entity_recognition.corpus import load_document_folds
from project2_from_project1.src.gold_projection import load_flat_gold_jsonl
from project2_from_project1.src.ngrams import predicted_pipeline
from project2_from_project1.src.ngrams.pipeline import train_and_predict_ngram_knn
from project2_from_project1.src.relation_end_to_end import (
    FALSE_NEGATIVE_REASONS,
    FALSE_POSITIVE_REASONS,
    LABELS,
    PATIENT,
    build_gold,
    evaluate_end_to_end,
)
from project2_from_project1.src.relation_evaluation import (
    RelationPrediction,
    load_relation_predictions,
    load_supervised_candidates,
    write_relation_predictions,
)
from project2_from_project1.src.run_manifest import dependency_versions, git_provenance, sha256_file
from project2_from_project1.src.supervised_contracts import load_contract_jsonl

RECOGNIZERS = ("dictionary", "crf")
CLASSIFIERS = ("rules", "knn")
KNN_N_NEIGHBORS = 1
KNN_MAXIMUM_NGRAM_ORDER = 1


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _entity_paths(entity_study: Path, recognizer: str, fold: int) -> Path:
    if recognizer == "dictionary":
        return entity_study / "dictionary" / f"fold-{fold:02d}"
    return entity_study / "crf" / f"fold-{fold:02d}" / "outer"


def _ceiling(corpus: Path, frozen: Path, fold_index: int, output: Path, documents, fold) -> dict[str, Path]:
    """Apply the fold's rules and k-NN to gold validation candidates."""
    manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    training_path = frozen / f"fold-{fold_index:02d}-train.jsonl"
    validation_path = frozen / f"fold-{fold_index:02d}-validation.jsonl"
    for path in (training_path, validation_path):
        if manifest["artifact_hashes"].get(path.name) != sha256_file(path):
            raise ValueError(f"{path.name} does not match the frozen dataset manifest")
    training = load_supervised_candidates(training_path)
    validation = load_supervised_candidates(validation_path)
    if any(item.document_id not in set(fold.validation_document_ids) for item in validation):
        raise ValueError("gold validation candidate outside the validation partition")

    model = rules.fit_rules(training, fold.train_document_ids)
    table = {(rule["candidate_relation"], rule["sentence_distance"]): rule for rule in model["rules"]}
    rule_predictions = []
    for candidate in validation:
        rule = table[rules._rule_key(candidate)]
        rule_predictions.append(RelationPrediction(
            candidate.document_id, candidate.source_id, candidate.target_id, rule["decision"], rule["decision"],
            rules.METHOD, {"rule_id": rule["rule_id"], "entity_source": "gold"}, model_version=rules.MODEL_VERSION))
    knn = train_and_predict_ngram_knn(
        [document for document in documents if document.document_id in set(fold.train_document_ids) | set(fold.validation_document_ids)],
        training, validation, n_neighbors=KNN_N_NEIGHBORS, maximum_ngram_order=KNN_MAXIMUM_NGRAM_ORDER)
    paths = {}
    for name, predictions, extra in (("rules", rule_predictions, {"rules": model["rules"]}),
                                     ("knn", knn.predictions, knn.manifest_dict())):
        directory = output / "ceiling" / name / f"fold-{fold_index:02d}"
        write_relation_predictions(predictions, directory / "predictions.jsonl")
        _write_json(directory / "manifest.json", {
            **({"configuration": extra} if name == "knn" else {"rules": extra["rules"]}),
            "entity_source": "gold (diagnostic ceiling, not an end-to-end pipeline)",
            "fold": fold_index,
            "input_hashes": {"corpus_sha256": sha256_file(corpus), "training_sha256": sha256_file(training_path),
                             "validation_candidates_sha256": sha256_file(validation_path)},
            "method": rules.METHOD if name == "rules" else "ngram-knn-cosine-uniform",
            "output_hashes": {"predictions_sha256": sha256_file(directory / "predictions.jsonl")},
            "schema_version": 1,
        })
        paths[name] = directory
    return paths


def run_fold(corpus: Path, projection: Path, frozen: Path, entity_study: Path, fold_index: int, output: Path,
             *, minimum_positive_support: int = 5) -> dict[str, Any]:
    partitions = frozen / "manifest.json"
    folds = {fold.index: fold for fold in load_document_folds(partitions)}
    fold = folds[fold_index]
    documents = load_jsonl(corpus)
    validation_documents = [document for document in documents if document.document_id in set(fold.validation_document_ids)]
    gold = build_gold(validation_documents, [item for item in load_flat_gold_jsonl(projection)
                                             if item.document_id in set(fold.validation_document_ids)])
    training_path = frozen / f"fold-{fold_index:02d}-train.jsonl"
    tag = f"fold-{fold_index:02d}"

    arm_inputs = {}
    failed = {}
    for recognizer in RECOGNIZERS:
        entity_dir = _entity_paths(entity_study, recognizer, fold_index)
        entity_manifest = json.loads((entity_dir / "manifest.json").read_text(encoding="utf-8"))
        if entity_manifest.get("status") != "completed":
            # A failed recognizer removes only its own arms for this fold; nothing replaces it.
            failed[recognizer] = {"recognizer_status": entity_manifest.get("status"), "failure": entity_manifest.get("failure"),
                                  "entity_manifest_sha256": sha256_file(entity_dir / "manifest.json")}
            continue
        candidate_dir = output / "candidates" / recognizer / tag
        relation_candidates.run_from_files(corpus, entity_dir / "predictions.jsonl", candidate_dir, entity_dir / "manifest.json")
        rules.run_from_files(training_path, partitions, fold_index, candidate_dir / "candidates.jsonl",
                             candidate_dir / "manifest.json", output / "rules" / recognizer / tag)
        arm_inputs[recognizer] = (candidate_dir / "candidates.jsonl", candidate_dir / "manifest.json",
                                  entity_dir / "predictions.jsonl")
    if arm_inputs:
        predicted_pipeline.run_from_files(corpus, training_path, partitions, fold_index, arm_inputs, output / "knn" / tag,
                                          n_neighbors=KNN_N_NEIGHBORS, maximum_ngram_order=KNN_MAXIMUM_NGRAM_ORDER)
    ceiling = _ceiling(corpus, frozen, fold_index, output, documents, fold)

    results = {}
    for recognizer in arm_inputs:
        entity_dir = _entity_paths(entity_study, recognizer, fold_index)
        entities = load_contract_jsonl(entity_dir / "predictions.jsonl", "entity")
        resolve = {(item.document_id, item.prediction_id): (item.span.start, item.span.end, item.canonical_type)
                   for item in entities}
        for document_id in fold.validation_document_ids:
            resolve[(document_id, relation_candidates.logical_patient_id(document_id))] = PATIENT
        candidates = load_contract_jsonl(arm_inputs[recognizer][0], "candidate")
        candidate_keys = {(item.document_id, resolve[(item.document_id, item.source_id)],
                           resolve[(item.document_id, item.target_id)], item.candidate_relation.value) for item in candidates}
        recognized = {(item.document_id, resolve[(item.document_id, item.prediction_id)]) for item in entities}
        for classifier in CLASSIFIERS:
            directory = (output / "rules" / recognizer / tag) if classifier == "rules" else (output / "knn" / tag / recognizer)
            predictions = load_relation_predictions(directory / "predictions.jsonl")
            if len(predictions) != len(candidates):
                raise ValueError(f"{recognizer}/{classifier} predictions are not one-to-one with candidates")
            results[f"{recognizer}+{classifier}"] = (evaluate_end_to_end(
                gold, fold.validation_document_ids, predictions, resolve, recognized_entities=recognized,
                candidates=candidate_keys, minimum_positive_support=minimum_positive_support,
                metadata={"classifier": classifier, "entity_source": recognizer, "fold": fold_index}),
                {"entities": entity_dir / "manifest.json", "candidates": arm_inputs[recognizer][1],
                 "classifier": directory / "manifest.json"})

    resolve = dict(gold.mention_keys)
    resolve.update({key: PATIENT for key in gold.patient_ids})
    gold_candidates = load_supervised_candidates(frozen / f"fold-{fold_index:02d}-validation.jsonl")
    candidate_keys = {(item.document_id, resolve[(item.document_id, item.source_id)],
                       resolve[(item.document_id, item.target_id)], item.candidate_relation.value) for item in gold_candidates}
    recognized = {(document_id, key) for (document_id, _), key in gold.mention_keys.items()}
    for classifier in CLASSIFIERS:
        predictions = load_relation_predictions(ceiling[classifier] / "predictions.jsonl")
        results[f"gold+{classifier}"] = (evaluate_end_to_end(
            gold, fold.validation_document_ids, predictions, resolve, recognized_entities=recognized,
            candidates=candidate_keys, minimum_positive_support=minimum_positive_support,
            metadata={"classifier": classifier, "entity_source": "gold", "fold": fold_index,
                      "role": "diagnostic_ceiling"}),
            {"candidates": frozen / "manifest.json", "classifier": ceiling[classifier] / "manifest.json"})

    rows = {}
    for arm, (evaluation, manifests) in results.items():
        path = output / "evaluations" / arm.replace("+", "-") / f"{tag}.json"
        _write_json(path, evaluation)
        rows[arm] = {"evaluation": evaluation, "evaluation_sha256": sha256_file(path), "status": "completed",
                     "manifests": {name: {"path": str(item), "sha256": sha256_file(item)} for name, item in manifests.items()}}
    for recognizer, failure in failed.items():
        for classifier in CLASSIFIERS:
            rows[f"{recognizer}+{classifier}"] = {**failure, "evaluation": None, "status": "skipped_failed_recognizer"}
    return rows


def _spread(values: Sequence[float]) -> dict[str, float | None]:
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else None}


def _pooled(evaluations: Sequence[Mapping[str, Any]], minimum_positive_support: int) -> dict[str, Any]:
    per_label = {}
    for label in LABELS:
        tp = sum(item["per_label"][label]["true_positive"] for item in evaluations)
        predicted = sum(item["per_label"][label]["predicted"] for item in evaluations)
        support = sum(item["per_label"][label]["support"] for item in evaluations)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_label[label] = {
            "conclusion": "conclusive" if support >= minimum_positive_support else "inconclusive_insufficient_support",
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "precision": precision, "predicted": predicted, "recall": recall, "support": support, "true_positive": tp,
        }
    tp = sum(row["true_positive"] for row in per_label.values())
    predicted = sum(row["predicted"] for row in per_label.values())
    support = sum(row["support"] for row in per_label.values())
    precision, recall = (tp / predicted if predicted else 0.0), (tp / support if support else 0.0)
    errors = {side: Counter() for side in ("false_negative", "false_positive")}
    for item in evaluations:
        for side in errors:
            errors[side].update(item["errors"][side])
    return {
        "errors": {"false_negative": {reason: errors["false_negative"][reason] for reason in FALSE_NEGATIVE_REASONS},
                   "false_positive": {reason: errors["false_positive"][reason] for reason in FALSE_POSITIVE_REASONS}},
        "macro_f1": sum(row["f1"] for row in per_label.values()) / len(LABELS),
        "micro": {"f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                  "precision": precision, "predicted": predicted, "recall": recall, "support": support, "true_positive": tp},
        "per_label": per_label,
    }


def run_factorial_study(corpus: str | Path, projection: str | Path, frozen_dataset: str | Path, entity_study: str | Path,
                        output_directory: str | Path, *, folds: Sequence[int] | None = None,
                        minimum_positive_support: int = 5) -> dict[str, Any]:
    started = time.perf_counter()
    corpus, projection, frozen, entity_study, output = map(Path, (corpus, projection, frozen_dataset, entity_study, output_directory))
    indexes = sorted(folds) if folds else sorted(fold.index for fold in load_document_folds(frozen / "manifest.json"))
    per_fold = {index: run_fold(corpus, projection, frozen, entity_study, index, output,
                                minimum_positive_support=minimum_positive_support) for index in indexes}
    arms = {}
    names = [f"{recognizer}+{classifier}" for recognizer in (*RECOGNIZERS, "gold") for classifier in CLASSIFIERS]
    for arm in names:
        completed = [index for index in indexes if per_fold[index][arm]["status"] == "completed"]
        evaluations = [per_fold[index][arm]["evaluation"] for index in completed]
        folds_rows = []
        for index in indexes:
            item = per_fold[index][arm]
            if item["status"] != "completed":
                folds_rows.append({"fold": index, "status": item["status"], "failure": item.get("failure"),
                                   "recognizer_status": item.get("recognizer_status"),
                                   "entity_manifest_sha256": item.get("entity_manifest_sha256")})
                continue
            folds_rows.append({"fold": index, "status": "completed", "evaluation_sha256": item["evaluation_sha256"],
                               "inconclusive_labels": item["evaluation"]["inconclusive_labels"],
                               "macro_f1": item["evaluation"]["macro_f1"], "manifests": item["manifests"],
                               "micro_f1": item["evaluation"]["micro"]["f1"]})
        arms[arm] = {
            "across_folds": {
                "macro_f1": _spread([item["macro_f1"] for item in evaluations]),
                "micro_f1": _spread([item["micro"]["f1"] for item in evaluations]),
                "per_label_f1": {label: _spread([item["per_label"][label]["f1"] for item in evaluations]) for label in LABELS},
            } if evaluations else None,
            "completed_folds": completed,
            "failed_folds": [index for index in indexes if index not in completed],
            "folds": folds_rows,
            "pooled": _pooled(evaluations, minimum_positive_support) if evaluations else None,
            "role": "diagnostic_ceiling" if arm.startswith("gold+") else "end_to_end",
        }
    summary = {
        "arms": arms,
        "configuration": {"classifiers": list(CLASSIFIERS), "folds": indexes,
                          "knn": {"maximum_ngram_order": KNN_MAXIMUM_NGRAM_ORDER, "n_neighbors": KNN_N_NEIGHBORS},
                          "matching": "strict: document, source span+type, target span+type, label",
                          "minimum_positive_support": minimum_positive_support, "recognizers": list(RECOGNIZERS)},
        "duration_seconds": round(time.perf_counter() - started, 3),
        "environment": dependency_versions(),
        "git": git_provenance(),
        "input_hashes": {"corpus_sha256": sha256_file(corpus), "gold_projection_sha256": sha256_file(projection),
                         "partition_manifest_sha256": sha256_file(frozen / "manifest.json"),
                         "entity_study_summary_sha256": sha256_file(entity_study / "summary.json")},
        "interpretation": "exploratory development folds; gold+* rows are diagnostic ceilings, not pipelines",
        "schema_version": 1,
        "test_partition_read": False,
    }
    _write_json(output / "summary.json", summary)
    return summary


__all__ = ["CLASSIFIERS", "RECOGNIZERS", "run_factorial_study", "run_fold"]
