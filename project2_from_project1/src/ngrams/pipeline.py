"""End-to-end n-gram k-NN relation-classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from project2_from_project1.src.clinical_documents import (
    CanonicalRelation,
    ClinicalDocument,
    load_jsonl,
)
from project2_from_project1.src.ngrams.feature_matrix import (
    CandidateMetadata,
    NgramCandidateExample,
    NgramFeatureVectorizer,
)
from project2_from_project1.src.ngrams.knn_classifier import NgramKNNClassifier
from project2_from_project1.src.ngrams.ngrams_extractor import (
    NgramFeature,
    extract_candidate_tokens,
    generate_ngrams,
)
from project2_from_project1.src.relation_evaluation import (
    RelationPrediction,
    load_supervised_candidates,
    write_relation_predictions,
)
from project2_from_project1.src.supervised_dataset import SupervisedCandidate


class NgramKNNPipelineError(ValueError):
    """Raised when frozen candidates cannot be classified safely."""


@dataclass(frozen=True)
class NgramKNNRun:
    """Predictions and deterministic metadata from one fitted k-NN run."""

    predictions: tuple[RelationPrediction, ...]
    vocabulary: tuple[NgramFeature, ...]
    training_examples: int
    prediction_examples: int
    n_neighbors: int

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "feature_family": "word-count-ngrams-1-3",
            "method": "ngram-knn-cosine-uniform",
            "n_neighbors": self.n_neighbors,
            "prediction_examples": self.prediction_examples,
            "training_examples": self.training_examples,
            "vocabulary_size": len(self.vocabulary),
        }


def _documents_by_id(documents: Iterable[ClinicalDocument]) -> dict[str, ClinicalDocument]:
    by_id: dict[str, ClinicalDocument] = {}
    for document in documents:
        if not isinstance(document, ClinicalDocument):
            raise TypeError("documents must contain ClinicalDocument instances")
        if document.document_id in by_id:
            raise NgramKNNPipelineError(
                f"duplicate clinical document id {document.document_id!r}"
            )
        by_id[document.document_id] = document
    return by_id


def build_candidate_examples(
    documents: Iterable[ClinicalDocument],
    candidates: Iterable[SupervisedCandidate],
    *,
    include_labels: bool,
) -> tuple[NgramCandidateExample, ...]:
    """Create exactly one n-gram example for each frozen candidate."""

    by_id = _documents_by_id(documents)
    examples: list[NgramCandidateExample] = []
    for candidate in candidates:
        if not isinstance(candidate, SupervisedCandidate):
            raise TypeError("candidates must contain SupervisedCandidate instances")
        try:
            document = by_id[candidate.document_id]
        except KeyError as error:
            raise NgramKNNPipelineError(
                f"candidate references missing document {candidate.document_id!r}"
            ) from error

        document_payload = document.to_dict()
        mentions_by_id = {
            mention["mention_id"]: mention
            for mention in document_payload["mentions"]
        }
        tokens = extract_candidate_tokens(
            candidate.to_dict(), document.text, mentions_by_id
        )
        metadata = CandidateMetadata(
            document_id=candidate.document_id,
            source_id=candidate.source_id,
            target_id=candidate.target_id,
            additional_ids=(
                ("candidate_relation", candidate.candidate_relation.value),
                ("gold_relation_id", candidate.gold_relation_id or ""),
            ),
        )
        examples.append(
            NgramCandidateExample(
                ngram_counts=generate_ngrams(tokens),
                metadata=metadata,
                label=candidate.label.value if include_labels else None,
            )
        )
    return tuple(examples)


def train_and_predict_ngram_knn(
    documents: Iterable[ClinicalDocument],
    training_candidates: Sequence[SupervisedCandidate],
    prediction_candidates: Sequence[SupervisedCandidate],
    *,
    n_neighbors: int = 3,
    model_version: str = "ngram-knn-v1",
) -> NgramKNNRun:
    """Fit on one frozen training partition and classify another partition."""

    documents = tuple(documents)
    training_candidates = tuple(training_candidates)
    prediction_candidates = tuple(prediction_candidates)
    if not training_candidates:
        raise NgramKNNPipelineError("training candidates cannot be empty")
    if not prediction_candidates:
        raise NgramKNNPipelineError("prediction candidates cannot be empty")

    training_examples = build_candidate_examples(
        documents, training_candidates, include_labels=True
    )
    prediction_examples = build_candidate_examples(
        documents, prediction_candidates, include_labels=False
    )

    vectorizer = NgramFeatureVectorizer()
    training_matrix = vectorizer.fit_transform(training_examples)
    if training_matrix.X.shape[1] == 0:
        raise NgramKNNPipelineError(
            "training candidates produced no n-gram features"
        )
    prediction_matrix = vectorizer.transform(prediction_examples)

    labels: list[str | None] = [None] * len(prediction_candidates)
    vote_fractions: list[float | None] = [None] * len(prediction_candidates)
    candidate_relations = sorted(
        {candidate.candidate_relation for candidate in prediction_candidates},
        key=lambda relation: relation.value,
    )
    for candidate_relation in candidate_relations:
        training_rows = [
            index
            for index, candidate in enumerate(training_candidates)
            if candidate.candidate_relation is candidate_relation
        ]
        prediction_rows = [
            index
            for index, candidate in enumerate(prediction_candidates)
            if candidate.candidate_relation is candidate_relation
        ]
        if not training_rows:
            raise NgramKNNPipelineError(
                f"training partition has no candidates for {candidate_relation.value}"
            )
        if n_neighbors > len(training_rows):
            raise NgramKNNPipelineError(
                f"n_neighbors={n_neighbors} exceeds the {len(training_rows)} "
                f"training candidates for {candidate_relation.value}"
            )

        classifier = NgramKNNClassifier(n_neighbors=n_neighbors).fit(
            training_matrix.X[training_rows],
            [training_candidates[index].label.value for index in training_rows],
        )
        relation_labels, relation_fractions = classifier.predict_with_vote_fraction(
            prediction_matrix.X[prediction_rows]
        )
        for row, raw_label, vote_fraction in zip(
            prediction_rows, relation_labels, relation_fractions, strict=True
        ):
            labels[row] = str(raw_label)
            vote_fractions[row] = float(vote_fraction)

    if any(label is None for label in labels) or any(
        fraction is None for fraction in vote_fractions
    ):
        raise NgramKNNPipelineError("not every prediction row was classified")

    predictions: list[RelationPrediction] = []
    for candidate, metadata, raw_label, vote_fraction in zip(
        prediction_candidates,
        prediction_matrix.metadata,
        labels,
        vote_fractions,
        strict=True,
    ):
        if (
            metadata.document_id,
            metadata.source_id,
            metadata.target_id,
        ) != (
            candidate.document_id,
            candidate.source_id,
            candidate.target_id,
        ):
            raise NgramKNNPipelineError("prediction rows lost candidate alignment")
        label = CanonicalRelation(raw_label)
        if label not in (candidate.candidate_relation, CanonicalRelation.NO_RELATION):
            raise NgramKNNPipelineError(
                f"classifier produced incompatible {label.value} for "
                f"{candidate.candidate_relation.value} candidate"
            )
        predictions.append(
            RelationPrediction(
                document_id=candidate.document_id,
                source_id=candidate.source_id,
                target_id=candidate.target_id,
                native_label=label.value,
                canonical_label=label,
                method="ngram-knn-cosine-uniform",
                evidence={
                    "candidate_relation": candidate.candidate_relation.value,
                    "feature_family": "word-count-ngrams-1-3",
                    "n_neighbors": n_neighbors,
                    "sentence_distance": candidate.sentence_distance,
                },
                sentences=(),
                model_version=model_version,
                neighbor_vote_fraction=vote_fraction,
            )
        )

    return NgramKNNRun(
        predictions=tuple(predictions),
        vocabulary=vectorizer.vocabulary,
        training_examples=len(training_examples),
        prediction_examples=len(prediction_examples),
        n_neighbors=n_neighbors,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_ngram_knn_from_files(
    corpus_jsonl: str | Path,
    training_candidates_jsonl: str | Path,
    prediction_candidates_jsonl: str | Path,
    output_predictions_jsonl: str | Path,
    *,
    n_neighbors: int = 3,
    model_version: str = "ngram-knn-v1",
    manifest_path: str | Path | None = None,
) -> NgramKNNRun:
    """Run the classifier from frozen files and write auditable predictions."""

    corpus_path = Path(corpus_jsonl)
    training_path = Path(training_candidates_jsonl)
    prediction_path = Path(prediction_candidates_jsonl)
    output_path = Path(output_predictions_jsonl)
    run = train_and_predict_ngram_knn(
        load_jsonl(corpus_path),
        load_supervised_candidates(training_path),
        load_supervised_candidates(prediction_path),
        n_neighbors=n_neighbors,
        model_version=model_version,
    )
    write_relation_predictions(run.predictions, output_path)

    manifest_output = (
        Path(manifest_path)
        if manifest_path is not None
        else output_path.with_suffix(".manifest.json")
    )
    manifest: Mapping[str, Any] = {
        **run.manifest_dict(),
        "artifact_hashes": {
            "corpus_jsonl_sha256": sha256_file(corpus_path),
            "prediction_candidates_jsonl_sha256": sha256_file(prediction_path),
            "predictions_jsonl_sha256": sha256_file(output_path),
            "training_candidates_jsonl_sha256": sha256_file(training_path),
        },
        "model_version": model_version,
        "schema_version": 1,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return run


__all__ = [
    "NgramKNNPipelineError",
    "NgramKNNRun",
    "build_candidate_examples",
    "run_ngram_knn_from_files",
    "sha256_file",
    "train_and_predict_ngram_knn",
]
