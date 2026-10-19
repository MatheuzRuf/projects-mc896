"""Run frozen models on raw external documents, then score against external gold."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scipy.sparse import load_npz
import torch

from project2_from_project1.src import canonical_relation_rules
from project2_from_project1.src.clinical_documents import load_jsonl
from project2_from_project1.src.entity_recognition.corpus import RawDocument, join_gold_projection, select_documents
from project2_from_project1.src.entity_recognition.crf_model import SemiMarkovCRF
from project2_from_project1.src.entity_recognition.dictionary import CanonicalDictionary, DictionaryEntry, predict_documents
from project2_from_project1.src.entity_recognition.evaluation import evaluate_entities, training_vocabulary
from project2_from_project1.src.gold_projection import (
    FlatGoldDocument, FlatGoldMention, MAX_MENTION_TOKENS, load_flat_gold_jsonl,
    segment_sentences, tokenize_with_offsets,
)
from project2_from_project1.src.ngrams.feature_matrix import NgramFeatureVectorizer
from project2_from_project1.src.ngrams.knn_classifier import NgramKNNClassifier
from project2_from_project1.src.ngrams.pipeline import build_candidate_examples
from project2_from_project1.src.relation_candidates import generate_candidates, logical_patient_id
from project2_from_project1.src.relation_end_to_end import PATIENT, build_gold, evaluate_end_to_end
from project2_from_project1.src.relation_evaluation import (
    RelationPrediction, load_relation_predictions, write_relation_predictions,
)
from project2_from_project1.src.run_manifest import dependency_versions, git_provenance, sha256_file
from project2_from_project1.src.supervised_contracts import EntityPrediction, load_contract_jsonl, write_contract_jsonl
from project2_from_project1.src.supervised_graph import build_gold_graph, build_predicted_graph, evaluate_graph


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_models(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "models_frozen_before_external_annotation_access":
        raise ValueError("models are not frozen")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"model artifact changed: {name}")
    project = Path(__file__).resolve().parents[1]
    for name, digest in manifest["code_sha256"].items():
        if sha256_file(project / name) != digest:
            raise ValueError(f"model code changed: {name}")
    return manifest


def _knn(documents, entities, candidates, directory: Path):
    model = json.loads((directory / "knn.json").read_text(encoding="utf-8"))
    vocabulary = tuple((entry[0], tuple(entry[1])) for entry in model["vocabulary"])
    examples = build_candidate_examples(documents, candidates, include_labels=False, entity_predictions=entities)
    prediction_matrix = NgramFeatureVectorizer._transform(tuple(examples), vocabulary).X
    training_matrix = load_npz(directory / "knn-matrix.npz")
    predictions = [None] * len(candidates)
    for relation in sorted({candidate.candidate_relation.value for candidate in candidates}):
        train_rows = [index for index, value in enumerate(model["relations"]) if value == relation]
        rows = [index for index, candidate in enumerate(candidates) if candidate.candidate_relation.value == relation]
        if not train_rows:
            raise ValueError(f"no training examples for {relation}")
        classifier = NgramKNNClassifier(n_neighbors=model["n_neighbors"])
        classifier.fit(training_matrix[train_rows], [model["labels"][index] for index in train_rows])
        labels, fractions = classifier.predict_with_vote_fraction(prediction_matrix[rows])
        for index, label, fraction in zip(rows, labels, fractions, strict=True):
            candidate = candidates[index]
            if str(label) not in (relation, "NO_RELATION"):
                raise ValueError("k-NN returned an incompatible relation")
            predictions[index] = RelationPrediction(candidate.document_id, candidate.source_id, candidate.target_id,
                                                    str(label), str(label), "ngram-knn-cosine-uniform",
                                                    {"candidate_id": candidate.candidate_id, "candidate_relation": relation},
                                                    model_version="ngram-knn-v1", neighbor_vote_fraction=float(fraction))
    return tuple(predictions)


def _predict_relations(raw_docs, raw, entities, name: str, models: Path, output: Path):
    candidates = generate_candidates(raw, entities)
    directory = output / name
    directory.mkdir(parents=True)
    write_contract_jsonl(entities, directory / "entities.jsonl")
    write_contract_jsonl(candidates, directory / "candidates.jsonl")
    rules = json.loads((models / "rules.json").read_text(encoding="utf-8"))
    predictions = {"rules": canonical_relation_rules.predict_candidates(candidates, rules),
                   "knn": _knn(raw_docs, entities, candidates, models)}
    for classifier, values in predictions.items():
        write_relation_predictions(values, directory / f"{classifier}.jsonl")
    return candidates, predictions


def _flat_from_adjudicated_documents(documents) -> tuple[tuple[FlatGoldDocument, ...], dict[str, int]]:
    """Project already-adjudicated external mentions without development records.

    The external assembly's ``ClinicalDocument`` records are the adjudicated
    authority, unlike the development corpus whose selection decisions live in
    a separate adjudication JSONL.  This conversion checks the same sentence
    and token alignment constraints as the flat-gold format.
    """
    result = []
    losses: Counter[str] = Counter()
    for document in documents:
        sentences = segment_sentences(document.text, tokenize_with_offsets(document.text))
        flat_mentions = []
        for mention in document.mentions:
            if len(mention.spans) != 1:
                losses["discontinuous"] += 1
                continue
            span = mention.spans[0]
            sentence = next((item for item in sentences if item.span.start <= span.start and span.end <= item.span.end), None)
            if sentence is None:
                losses["cross_sentence"] += 1
                continue
            starts = {token.span.start: index for index, token in enumerate(sentence.tokens)}
            ends = {token.span.end: index + 1 for index, token in enumerate(sentence.tokens)}
            if span.start not in starts or span.end not in ends:
                losses["token_misalignment"] += 1
                continue
            token_start, token_end = starts[span.start], ends[span.end]
            if token_end - token_start > MAX_MENTION_TOKENS:
                losses["over_max_tokens"] += 1
                continue
            flat_mentions.append(FlatGoldMention(
                document.document_id, span, mention.canonical_type, sentence.sentence_index,
                token_start, token_end, (mention.mention_id,), None, (),
            ))
        flat_mentions.sort(key=lambda item: (item.span.start, item.span.end, item.canonical_type))
        non_overlapping = []
        for mention in flat_mentions:
            if non_overlapping and mention.span.start < non_overlapping[-1].span.end:
                losses["overlap"] += 1
                continue
            non_overlapping.append(mention)
        result.append(FlatGoldDocument(document.document_id, tuple(sentences), tuple(non_overlapping)))
    return tuple(result), dict(sorted(losses.items()))


def _load_frozen_predictions(output: Path, models: Path, raw_path: Path, recognizers: tuple[str, ...]) -> dict:
    """Reload predictions frozen before gold access after an interrupted scorer."""
    manifest_path = output / "predictions-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("cannot resume without a frozen predictions manifest")
    prediction_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (prediction_manifest.get("models_manifest_sha256") != sha256_file(models / "manifest.json")
            or prediction_manifest.get("raw_sha256") != sha256_file(raw_path)):
        raise ValueError("frozen predictions do not match the supplied models or raw snapshot")
    expected = prediction_manifest.get("prediction_sha256", {})
    for name, digest in expected.items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"frozen prediction changed: {name}")
    frozen = {}
    for recognizer in recognizers:
        directory = output / recognizer
        entities = tuple(load_contract_jsonl(directory / "entities.jsonl", "entity"))
        candidates = tuple(load_contract_jsonl(directory / "candidates.jsonl", "candidate"))
        frozen[recognizer] = (entities, candidates, {
            classifier: tuple(load_relation_predictions(directory / f"{classifier}.jsonl"))
            for classifier in ("rules", "knn")
        })
    return frozen


def run(models: Path, raw_path: Path, gold_path: Path, training_corpus: Path, training_projection: Path, output: Path,
        *, resume_frozen_predictions: bool = False) -> dict:
    if output.exists() and not resume_frozen_predictions:
        raise ValueError("output directory must not exist")
    if resume_frozen_predictions and not output.is_dir():
        raise ValueError("resume output directory does not exist")
    manifest = _read_models(models)
    if sha256_file(training_corpus) != manifest["input_sha256"]["corpus_jsonl_sha256"] or \
       sha256_file(training_projection) != manifest["input_sha256"]["gold_projection_jsonl_sha256"]:
        raise ValueError("training data differs from the frozen models")
    raw_docs = load_jsonl(raw_path)
    if not raw_docs or any(item.mentions or item.relations for item in raw_docs):
        raise ValueError("raw external documents must contain no annotations")
    raw = [RawDocument(item.document_id, item.text) for item in raw_docs]
    ids = sorted(item.document_id for item in raw)
    if set(ids) & set(manifest["training_document_ids"]):
        raise ValueError("external documents overlap training")
    completed_recognizers = ("dictionary",) + (("crf",) if manifest["crf"]["status"] == "completed" else ())
    if resume_frozen_predictions:
        frozen = _load_frozen_predictions(output, models, raw_path, completed_recognizers)
    else:
        output.mkdir(parents=True)
        entries = [json.loads(line) for line in (models / "dictionary.jsonl").read_text(encoding="utf-8").splitlines()]
        dictionary = CanonicalDictionary([DictionaryEntry(tuple(row["surface_tokens"]), row["type_counts"]) for row in entries],
                                         manifest["training_document_ids"])
        dictionary_entities, _ = predict_documents(dictionary, raw)
        frozen = {"dictionary": (dictionary_entities, *_predict_relations(raw_docs, raw, dictionary_entities, "dictionary", models, output))}
        if manifest["crf"]["status"] == "completed":
            print("Predicting CRF entities", flush=True)
            torch.set_num_threads(1)
            model = SemiMarkovCRF.from_dict(json.loads((models / "crf.json").read_text(encoding="utf-8")))
            crf_entities = model.predict(raw)
            frozen["crf"] = (crf_entities, *_predict_relations(raw_docs, raw, crf_entities, "crf", models, output))
        prediction_hashes = {str(path.relative_to(output)): sha256_file(path) for path in output.rglob("*.jsonl")}
        _json(output / "predictions-manifest.json", {"models_manifest_sha256": sha256_file(models / "manifest.json"),
                                                     "raw_sha256": sha256_file(raw_path), "prediction_sha256": prediction_hashes})
    # Gold is first opened only after all end-to-end predictions have been written.
    all_gold_docs = load_jsonl(gold_path)
    raw_by_id = {item.document_id: item.text for item in raw_docs}
    gold_by_id = {item.document_id: item for item in all_gold_docs}
    if len(gold_by_id) != len(all_gold_docs):
        raise ValueError("external gold has duplicate document IDs")
    if set(raw_by_id) - set(gold_by_id) or any(gold_by_id[document_id].text != text
                                                for document_id, text in raw_by_id.items()):
        raise ValueError("external gold IDs or texts differ from raw inputs")
    # The supplied provisional assembly retains records screened out of this
    # user-authorized snapshot.  They are never evaluated or silently treated
    # as predictions; only raw snapshot IDs are opened for scoring.
    gold_docs = [gold_by_id[document_id] for document_id in ids]
    flat, entity_projection_losses = _flat_from_adjudicated_documents(gold_docs)
    gold_entities = tuple(EntityPrediction(doc.document_id, mention.span, mention.canonical_type, mention.sentence_index,
                                           "gold-entity-ceiling", "flat-gold-v1")
                          for doc in flat for mention in doc.mentions)
    frozen["gold"] = (gold_entities, *_predict_relations(raw_docs, raw, gold_entities, "gold", models, output))
    gold_entity_docs = join_gold_projection(gold_docs, flat)
    train_gold = join_gold_projection(load_jsonl(training_corpus), load_flat_gold_jsonl(training_projection))
    surfaces, tokens = training_vocabulary(select_documents(train_gold, manifest["training_document_ids"], "training"))
    gold_relations = build_gold(gold_docs, flat)
    gold_graph, graph_losses = build_gold_graph(gold_docs, flat)
    texts = {item.document_id: item.text for item in raw_docs}
    arms = {}
    for recognizer in ("dictionary", "crf", "gold"):
        if recognizer not in frozen:
            for classifier in ("rules", "knn"):
                arms[f"{recognizer}+{classifier}"] = {"status": "skipped_failed_recognizer", "failure": manifest["crf"].get("failure")}
            continue
        entities, candidates, predictions = frozen[recognizer]
        recognized = {(item.document_id, (item.span.start, item.span.end, item.canonical_type)) for item in entities}
        resolve = {(item.document_id, item.prediction_id): (item.span.start, item.span.end, item.canonical_type) for item in entities}
        resolve.update({(document_id, logical_patient_id(document_id)): PATIENT for document_id in ids})
        candidate_keys = {(item.document_id, resolve[(item.document_id, item.source_id)],
                           resolve[(item.document_id, item.target_id)], item.candidate_relation.value) for item in candidates}
        entity_eval = None
        if recognizer != "gold":
            entity_eval = evaluate_entities(gold_entity_docs.values(), entities, training_surfaces=surfaces, training_tokens=tokens)
            for row in entity_eval["strict"]["per_type"].values():
                row["conclusion"] = "conclusive" if row["support"] >= 100 else "inconclusive_insufficient_support"
        for classifier in ("rules", "knn"):
            arm = f"{recognizer}+{classifier}"
            relation_eval = evaluate_end_to_end(gold_relations, ids, predictions[classifier], resolve,
                                                recognized_entities=recognized, candidates=candidate_keys,
                                                minimum_positive_support=20)
            graph_eval = None
            if recognizer != "gold":
                graph = build_predicted_graph(texts, ids, entities, predictions[classifier])
                graph_eval = evaluate_graph(gold_graph, graph)
                write_contract_jsonl(graph, output / recognizer / f"graph-{classifier}.jsonl")
            arms[arm] = {"status": "completed", "role": "diagnostic_ceiling" if recognizer == "gold" else "end_to_end",
                         "crf_converged": manifest["crf"].get("converged") if recognizer == "crf" else None,
                         "entities": entity_eval, "relations": relation_eval, "graph": graph_eval}
    report = {"status": "external_evaluation_of_supplied_snapshot", "documents": len(ids),
              "gold_records_excluded_from_snapshot": len(all_gold_docs) - len(gold_docs), "arms": arms,
              "entity_projection_losses": entity_projection_losses, "graph_projection_losses": graph_losses,
              "minimum_relation_support": 20, "minimum_entity_support": 100,
              "resumed_frozen_predictions": resume_frozen_predictions,
              "interpretation": "External snapshot supplied by caller; conclusive claims depend on its documented freeze and selection protocol."}
    _json(output / "report.json", report)
    artifacts = {str(path.relative_to(output)).replace("\\", "/"): sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file()}
    _json(output / "manifest.json", {"status": "completed", "artifact_sha256": artifacts,
                                     "input_sha256": {"raw": sha256_file(raw_path), "gold": sha256_file(gold_path),
                                                      "models_manifest": sha256_file(models / "manifest.json")},
                                     "code_sha256": sha256_file(__file__), "git": git_provenance(),
                                     "environment": dependency_versions()})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "raw", "gold", "training-corpus", "training-projection", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-frozen-predictions", action="store_true")
    args = parser.parse_args()
    result = run(args.models, args.raw, args.gold, args.training_corpus, args.training_projection, args.output_dir,
                 resume_frozen_predictions=args.resume_frozen_predictions)
    print(json.dumps({"status": result["status"], "documents": result["documents"]}))


if __name__ == "__main__":
    main()
