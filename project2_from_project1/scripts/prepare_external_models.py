"""Fit and freeze the four development-only model components for external evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.sparse import save_npz
import torch

from project2_from_project1.src import canonical_relation_rules
from project2_from_project1.src.clinical_documents import load_jsonl
from project2_from_project1.src.entity_recognition.corpus import join_gold_projection, load_document_folds, select_documents
from project2_from_project1.src.entity_recognition.crf_model import NonFiniteTrainingError, SemiMarkovCRF, TrainingConfig
from project2_from_project1.src.entity_recognition.dictionary import CanonicalDictionary, write_dictionary
from project2_from_project1.src.entity_recognition.run import verify_inputs
from project2_from_project1.src.gold_projection import load_flat_gold_jsonl
from project2_from_project1.src.ngrams.feature_matrix import NgramFeatureVectorizer
from project2_from_project1.src.ngrams.pipeline import build_candidate_examples
from project2_from_project1.src.run_manifest import dependency_versions, git_provenance, sha256_file
from project2_from_project1.src.supervised_dataset import PartitionPolicy, _sample_train_negatives, generate_candidates


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def prepare(corpus: Path, projection: Path, projection_manifest: Path, partitions: Path, output: Path,
            *, max_iterations: int = 250) -> dict:
    if output.exists():
        raise ValueError("output directory must not exist")
    input_hashes = verify_inputs(corpus, projection, projection_manifest, partitions)
    partition_data = json.loads(partitions.read_text(encoding="utf-8"))
    folds = load_document_folds(partition_data)
    train_ids = sorted({document_id for fold in folds for document_id in fold.validation_document_ids})
    if not train_ids or set(train_ids) & set(partition_data["partitions"]["test_document_ids"]):
        raise ValueError("development IDs are empty or overlap the held-out partition")
    if any(set(fold.train_document_ids) | set(fold.validation_document_ids) != set(train_ids) for fold in folds):
        raise ValueError("folds do not share a complete development partition")
    sources = load_jsonl(corpus)
    by_id = {document.document_id: document for document in sources}
    gold = join_gold_projection(sources, load_flat_gold_jsonl(projection))
    training = select_documents(gold, train_ids, "development")
    documents = [by_id[document_id] for document_id in train_ids]
    policy = partition_data["partition_policy"]
    sample_policy = PartitionPolicy(seed=policy["seed"], test_fraction=policy["test_fraction"], folds=policy["folds_requested"],
                                    maximum_train_negatives_per_positive=policy["maximum_train_negatives_per_positive"])
    all_candidates = tuple(candidate for document in documents for candidate in generate_candidates(document))
    candidates = _sample_train_negatives(all_candidates, sample_policy, 0)
    output.mkdir(parents=True)
    write_dictionary(CanonicalDictionary.fit(training), output / "dictionary.jsonl")
    write_json(output / "rules.json", canonical_relation_rules.fit_rules(candidates, train_ids))
    examples = build_candidate_examples(documents, candidates, include_labels=True)
    vectorizer = NgramFeatureVectorizer(maximum_ngram_order=1)
    matrix = vectorizer.fit_transform(examples)
    if not matrix.X.shape[1]:
        raise ValueError("no training features for k-NN")
    save_npz(output / "knn-matrix.npz", matrix.X)
    write_json(output / "knn.json", {"vocabulary": vectorizer.vocabulary, "labels": list(matrix.y),
                                     "relations": [candidate.candidate_relation.value for candidate in candidates],
                                     "n_neighbors": 1, "maximum_ngram_order": 1})
    print("Training the final CRF on development documents", flush=True)
    torch.set_num_threads(1)
    config = TrainingConfig(l2=0.1, max_iterations=max_iterations)
    try:
        model, diagnostics = SemiMarkovCRF.fit(training, config)
    except NonFiniteTrainingError as error:
        crf = {"status": "failed_non_finite", "converged": None, "failure": {"message": str(error), **error.diagnostic}}
    else:
        write_json(output / "crf.json", model.to_dict())
        crf = {"status": "completed", "converged": diagnostics.converged, "training": diagnostics.to_dict()}
    paths = sorted(path for path in output.iterdir() if path.is_file())
    project_root = Path(__file__).resolve().parents[1]
    code = sorted((project_root / "src").rglob("*.py")) + [Path(__file__)]
    selection_evidence = project_root / "docs/entity-recognizer-comparison.md"
    manifest = {"status": "models_frozen_before_external_annotation_access", "schema_version": 1,
                "training_document_ids": train_ids, "input_sha256": input_hashes,
                "configuration": {"l2": 0.1, "l2_selection": "highest rounded mean inner validation macro-F1 across the five documented development folds",
                                  "max_iterations": max_iterations, "negative_sampling_fold": 0,
                                  "knn_neighbors": 1, "maximum_ngram_order": 1},
                "selection_evidence_sha256": sha256_file(selection_evidence),
                "artifact_sha256": {path.name: sha256_file(path) for path in paths},
                "code_sha256": {str(path.relative_to(project_root)).replace("\\", "/"): sha256_file(path) for path in code},
                "crf": crf, "git": git_provenance(), "environment": dependency_versions()}
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "projection", "projection-manifest", "partitions", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=250)
    args = parser.parse_args()
    manifest = prepare(args.corpus, args.projection, args.projection_manifest, args.partitions, args.output_dir,
                       max_iterations=args.max_iterations)
    print(json.dumps({"status": manifest["status"], "crf": manifest["crf"]["status"]}))


if __name__ == "__main__":
    main()
