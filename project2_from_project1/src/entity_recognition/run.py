"""File-driven execution of one recognizer fold with strict, ordered phases.

1. verify that corpus, flat gold projection and partitions share one source;
2. fit on gold training documents;
3. predict from raw validation text only;
4. freeze predictions on disk and reload them;
5. evaluate the frozen predictions; publish the manifest last.

Both recognizer arms share these phases, the evaluator and the manifest layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from project2_from_project1.src.clinical_documents import ClinicalDocument, load_jsonl
from project2_from_project1.src.entity_recognition import crf_model, dictionary as dictionary_arm
from project2_from_project1.src.entity_recognition.corpus import (
    DocumentFold,
    EntityRecognitionError,
    GoldEntityDocument,
    RawDocument,
    join_gold_projection,
    load_document_folds,
    select_documents,
)
from project2_from_project1.src.entity_recognition.crf_features import FEATURE_TEMPLATE_VERSION
from project2_from_project1.src.entity_recognition.dictionary import CanonicalDictionary, predict_documents, write_dictionary
from project2_from_project1.src.entity_recognition.evaluation import evaluate_entities, training_vocabulary
from project2_from_project1.src.gold_projection import (
    MAX_MENTION_TOKENS,
    PROJECTION_VERSION,
    SENTENCE_SEGMENTER_VERSION,
    TOKENIZER_VERSION,
    load_flat_gold_jsonl,
)
from project2_from_project1.src.run_manifest import dependency_versions, git_provenance, sha256_file, sha256_text
from project2_from_project1.src.supervised_contracts import (
    EntityPrediction,
    load_contract_jsonl,
    write_contract_jsonl,
)

SEGMENTATION = {
    "max_mention_tokens": MAX_MENTION_TOKENS,
    "projection_version": PROJECTION_VERSION,
    "sentence_segmenter_version": SENTENCE_SEGMENTER_VERSION,
    "tokenizer_version": TOKENIZER_VERSION,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def verify_inputs(
    corpus_jsonl: Path, projection_jsonl: Path, projection_manifest_json: Path, partition_manifest_json: Path,
) -> dict[str, str]:
    """Fail unless every input derives from the same lossless corpus file."""
    hashes = {
        "corpus_jsonl_sha256": sha256_file(corpus_jsonl),
        "gold_projection_jsonl_sha256": sha256_file(projection_jsonl),
        "gold_projection_manifest_sha256": sha256_file(projection_manifest_json),
        "partition_manifest_sha256": sha256_file(partition_manifest_json),
    }
    projection_manifest = json.loads(projection_manifest_json.read_text(encoding="utf-8"))
    partitions = json.loads(partition_manifest_json.read_text(encoding="utf-8"))
    if projection_manifest.get("output_sha256") != hashes["gold_projection_jsonl_sha256"]:
        raise EntityRecognitionError("gold projection file does not match its manifest output_sha256")
    if projection_manifest.get("input_sha256") != hashes["corpus_jsonl_sha256"]:
        raise EntityRecognitionError("gold projection was derived from a different corpus file")
    if projection_manifest.get("configuration", {}).get("projection_version") != PROJECTION_VERSION:
        raise EntityRecognitionError(f"gold projection must be {PROJECTION_VERSION}")
    if partitions.get("source_jsonl_sha256") != hashes["corpus_jsonl_sha256"]:
        raise EntityRecognitionError("document partitions were frozen from a different corpus file")
    return hashes


@dataclass(frozen=True)
class FoldInputs:
    input_hashes: dict[str, str]
    fold: DocumentFold
    sources: dict[str, ClinicalDocument]
    gold: dict[str, GoldEntityDocument]

    @property
    def training(self) -> tuple[GoldEntityDocument, ...]:
        return select_documents(self.gold, self.fold.train_document_ids, "train")

    def raw_validation(self) -> list[RawDocument]:
        """Validation text only; gold validation mentions are deliberately not reachable here."""
        missing = sorted(set(self.fold.validation_document_ids) - self.sources.keys())
        if missing:
            raise EntityRecognitionError(f"validation document absent from corpus: {missing[0]!r}")
        return [RawDocument(document_id, self.sources[document_id].text) for document_id in self.fold.validation_document_ids]


def prepare_fold(
    corpus_jsonl: str | Path, projection_jsonl: str | Path, projection_manifest_json: str | Path,
    partition_manifest_json: str | Path, fold_index: int,
) -> FoldInputs:
    corpus, projection = Path(corpus_jsonl), Path(projection_jsonl)
    partitions = Path(partition_manifest_json)
    hashes = verify_inputs(corpus, projection, Path(projection_manifest_json), partitions)
    folds = {fold.index: fold for fold in load_document_folds(partitions)}
    if fold_index not in folds:
        raise EntityRecognitionError(f"fold {fold_index} is not defined; expected one of {sorted(folds)}")
    sources = {document.document_id: document for document in load_jsonl(corpus)}
    gold = join_gold_projection(sources.values(), load_flat_gold_jsonl(projection))
    return FoldInputs(hashes, folds[fold_index], sources, gold)


def _check_flat(predictions: Sequence[EntityPrediction]) -> None:
    by_document: dict[str, list[EntityPrediction]] = {}
    for prediction in predictions:
        by_document.setdefault(prediction.document_id, []).append(prediction)
    for document_id, items in by_document.items():
        items.sort(key=lambda item: (item.span.start, item.span.end))
        for left, right in zip(items, items[1:]):
            if right.span.start < left.span.end:
                raise EntityRecognitionError(f"recognizer produced overlapping predictions in {document_id!r}")


def _freeze_and_evaluate(
    inputs: FoldInputs, predictions: Sequence[EntityPrediction], output: Path, method: str, model_version: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    _check_flat(predictions)
    predictions_path = output / "predictions.jsonl"
    write_contract_jsonl(predictions, predictions_path)
    frozen = tuple(load_contract_jsonl(predictions_path, "entity"))
    training = inputs.training
    surfaces, tokens = training_vocabulary(training)
    evaluation = evaluate_entities(
        select_documents(inputs.gold, inputs.fold.validation_document_ids, "validation"), frozen,
        training_surfaces=surfaces, training_tokens=tokens,
        metadata={"fold": inputs.fold.index, "method": method, "model_version": model_version, "partition": "validation"},
    )
    evaluation_path = output / "evaluation.json"
    _write_json(evaluation_path, evaluation)
    return evaluation, {
        "evaluation_json_sha256": sha256_file(evaluation_path),
        "predictions_jsonl_sha256": sha256_file(predictions_path),
    }


def _manifest(
    inputs: FoldInputs, *, method: str, model_version: str, started: float, status: str,
    artifact_hashes: Mapping[str, str], configuration: Mapping[str, Any], determinism: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None, extra: Mapping[str, Any], sensitive: Sequence[str],
) -> dict[str, Any]:
    return {
        "artifact_hashes": {**inputs.input_hashes, **artifact_hashes},
        "configuration": {**SEGMENTATION, **configuration},
        "determinism": dict(determinism),
        "duration_seconds": round(time.perf_counter() - started, 6),
        "environment": dependency_versions(),
        "fold": inputs.fold.index,
        "git": git_provenance(),
        "label_access": {
            "evaluate": "validation flat gold, after predictions were frozen",
            "fit": "train flat gold mentions",
            "predict": "validation raw text only",
        },
        "method": method,
        "model_version": model_version,
        "partitions": {
            "train_document_ids": list(inputs.fold.train_document_ids),
            "validation_document_ids": list(inputs.fold.validation_document_ids),
        },
        "primary_metric": (
            {"name": evaluation["primary_metric"], "value": evaluation["strict"]["macro_f1"]} if evaluation else None
        ),
        "schema_version": 1,
        "sensitive_artifacts": list(sensitive),
        "status": status,
        **extra,
    }


def run_dictionary_fold_from_files(
    corpus_jsonl: str | Path,
    projection_jsonl: str | Path,
    projection_manifest_json: str | Path,
    partition_manifest_json: str | Path,
    fold_index: int,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Run the canonical-dictionary arm on one validation fold and write artifacts."""
    started = time.perf_counter()
    output = Path(output_directory)
    inputs = prepare_fold(corpus_jsonl, projection_jsonl, projection_manifest_json, partition_manifest_json, fold_index)
    dictionary = CanonicalDictionary.fit(inputs.training)
    predictions, match_stats = predict_documents(dictionary, inputs.raw_validation())
    evaluation, hashes = _freeze_and_evaluate(
        inputs, predictions, output, dictionary_arm.METHOD, dictionary_arm.MODEL_VERSION,
    )
    hashes["dictionary_jsonl_sha256"] = write_dictionary(dictionary, output / "dictionary.jsonl")
    manifest = _manifest(
        inputs, method=dictionary_arm.METHOD, model_version=dictionary_arm.MODEL_VERSION, started=started,
        status="completed", artifact_hashes=hashes,
        configuration={
            "ambiguity_policy": "strict_training_majority_else_abstain",
            "match_scope": "whole_tokens_within_one_sentence",
            "normalization": dictionary_arm.NORMALIZATION,
            "overlap_policy": "longest_first_then_leftmost",
        },
        determinism={"deterministic": True, "random_seed": None},
        evaluation=evaluation,
        extra={"dictionary": dictionary.summary(), "matching": match_stats.to_dict()},
        sensitive=["dictionary.jsonl"],
    )
    _write_json(output / "manifest.json", manifest)
    return manifest


def run_crf_fold_from_files(
    corpus_jsonl: str | Path,
    projection_jsonl: str | Path,
    projection_manifest_json: str | Path,
    partition_manifest_json: str | Path,
    fold_index: int,
    output_directory: str | Path,
    *,
    config: crf_model.TrainingConfig,
    fit: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Train the semi-Markov CRF on one fold's training gold and evaluate its validation.

    A finite run that does not converge is still evaluated and marked
    ``converged=false``.  A non-finite objective writes a ``failed_non_finite``
    manifest with no predictions and re-raises; nothing replaces the model.
    """
    started = time.perf_counter()
    output = Path(output_directory)
    inputs = prepare_fold(corpus_jsonl, projection_jsonl, projection_manifest_json, partition_manifest_json, fold_index)
    determinism = {
        "deterministic": True,
        "dtype": "float64",
        "initialization": "zeros",
        "random_seed": None,
        "torch_deterministic_algorithms": True,
        "torch_threads": torch.get_num_threads(),
    }
    configuration = {**config.to_dict(), "feature_template_version": FEATURE_TEMPLATE_VERSION}
    common = {"method": crf_model.METHOD, "model_version": crf_model.MODEL_VERSION, "started": started,
              "configuration": configuration, "determinism": determinism, "sensitive": ["model.json"]}
    try:
        model, training = (fit or crf_model.SemiMarkovCRF.fit)(inputs.training, config)
    except crf_model.NonFiniteTrainingError as error:
        manifest = _manifest(
            inputs, status="failed_non_finite", artifact_hashes={}, evaluation=None,
            extra={"converged": None, "failure": {"message": str(error), **error.diagnostic}},
            **{**common, "sensitive": []},
        )
        _write_json(output / "manifest.json", manifest)
        raise
    predictions = model.predict(inputs.raw_validation())
    evaluation, hashes = _freeze_and_evaluate(inputs, predictions, output, crf_model.METHOD, crf_model.MODEL_VERSION)
    model_text = json.dumps(model.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    (output / "model.json").write_text(model_text, encoding="utf-8")
    hashes["model_json_sha256"] = sha256_text(model_text)
    manifest = _manifest(
        inputs, status="completed", artifact_hashes=hashes, evaluation=evaluation,
        extra={"converged": training.converged, "features": model.vocabulary.summary(), "training": training.to_dict()},
        **common,
    )
    _write_json(output / "manifest.json", manifest)
    return manifest


__all__ = ["FoldInputs", "prepare_fold", "run_crf_fold_from_files", "run_dictionary_fold_from_files", "verify_inputs"]
