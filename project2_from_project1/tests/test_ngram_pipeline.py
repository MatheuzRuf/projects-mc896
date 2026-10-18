import json

from project2_from_project1.src.clinical_documents import (
    AnnotatedRelation,
    CanonicalRelation,
    ClinicalCase,
    ClinicalDocument,
    Mention,
    Patient,
    Span,
    write_jsonl,
)
from project2_from_project1.src.ngrams.pipeline import (
    build_candidate_examples,
    run_ngram_knn_from_files,
    train_and_predict_ngram_knn,
)
from project2_from_project1.src.ngrams.model_selection import (
    finalize_selected_ngram_knn,
    select_ngram_knn_from_folds,
)
from project2_from_project1.src.relation_evaluation import load_relation_predictions
from project2_from_project1.src.supervised_dataset import SupervisedCandidate


def _document(index: int, text: str) -> ClinicalDocument:
    document_id = f"document:{index}"
    case_id = f"{document_id}:case"
    patient_id = f"{document_id}:patient"
    finding_start = text.index("pain")
    finding = Mention(
        f"{document_id}:finding",
        [Span(finding_start, finding_start + 4)],
        ["finding"],
        "Finding",
    )
    return ClinicalDocument(
        document_id=document_id,
        text=text,
        cases=[ClinicalCase(case_id)],
        patients=[Patient(patient_id)],
        mentions=[finding],
        relations=[
            AnnotatedRelation(
                f"{document_id}:describes",
                case_id,
                patient_id,
                "document_describes_case",
                "DESCRIBES",
            )
        ],
    )


def _candidate(document: ClinicalDocument, label: CanonicalRelation) -> SupervisedCandidate:
    return SupervisedCandidate(
        document_id=document.document_id,
        source_id=document.patients[0].patient_id,
        target_id=document.mentions[0].mention_id,
        candidate_relation=CanonicalRelation.HAS_FINDING,
        label=label,
        source_type="Patient",
        target_type="Finding",
        sentence_distance=None,
    )


def _causal_document(index: int, text: str) -> ClinicalDocument:
    document_id = f"causal:{index}"
    condition_start = text.index("Condition")
    finding_start = text.index("pain")
    condition = Mention(
        f"{document_id}:condition",
        [Span(condition_start, condition_start + 9)],
        ["condition"],
        "Condition",
    )
    finding = Mention(
        f"{document_id}:finding",
        [Span(finding_start, finding_start + 4)],
        ["finding"],
        "Finding",
    )
    return ClinicalDocument(
        document_id=document_id,
        text=text,
        cases=[],
        patients=[],
        mentions=[condition, finding],
        relations=[],
    )


def _cause_candidate(document: ClinicalDocument, label: CanonicalRelation) -> SupervisedCandidate:
    return SupervisedCandidate(
        document_id=document.document_id,
        source_id=document.mentions[0].mention_id,
        target_id=document.mentions[1].mention_id,
        candidate_relation=CanonicalRelation.CAUSES,
        label=label,
        source_type="Condition",
        target_type="Finding",
        sentence_distance=0,
    )


def _write_candidates(path, candidates) -> None:
    path.write_text(
        "".join(
            json.dumps(candidate.to_dict(), sort_keys=True) + "\n"
            for candidate in candidates
        ),
        encoding="utf-8",
    )


def test_build_candidate_examples_preserves_one_row_per_candidate():
    document = _document(1, "Patient reports pain clearly.")
    candidate = _candidate(document, CanonicalRelation.HAS_FINDING)

    examples = build_candidate_examples([document], [candidate], include_labels=True)

    assert len(examples) == 1
    assert examples[0].metadata.document_id == document.document_id
    assert examples[0].label == "HAS_FINDING"
    assert examples[0].ngram_counts[0]["pain"] == 1


def test_in_memory_pipeline_predicts_every_candidate_with_vote_fraction():
    positive = _document(1, "Patient reports pain clearly.")
    negative = _document(2, "Routine observation mentions pain incidentally.")
    held_out = _document(3, "Patient reports pain clearly.")

    run = train_and_predict_ngram_knn(
        [positive, negative, held_out],
        [
            _candidate(positive, CanonicalRelation.HAS_FINDING),
            _candidate(negative, CanonicalRelation.NO_RELATION),
        ],
        [_candidate(held_out, CanonicalRelation.HAS_FINDING)],
        n_neighbors=1,
    )

    assert len(run.predictions) == 1
    assert run.predictions[0].canonical_label is CanonicalRelation.HAS_FINDING
    assert run.predictions[0].neighbor_vote_fraction == 1.0
    assert run.training_examples == 2
    assert run.prediction_examples == 1


def test_pipeline_never_predicts_a_relation_incompatible_with_candidate_type():
    finding_positive = _document(1, "Patient reports pain clearly.")
    finding_negative = _document(2, "Routine observation mentions pain incidentally.")
    finding_held_out = _document(3, "Patient reports pain clearly.")
    cause_positive = _causal_document(1, "Condition strongly causes pain.")
    cause_negative = _causal_document(2, "Condition appears unrelated to pain.")
    cause_held_out = _causal_document(3, "Condition strongly causes pain.")
    prediction_candidates = [
        _candidate(finding_held_out, CanonicalRelation.HAS_FINDING),
        _cause_candidate(cause_held_out, CanonicalRelation.CAUSES),
    ]

    run = train_and_predict_ngram_knn(
        [
            finding_positive,
            finding_negative,
            finding_held_out,
            cause_positive,
            cause_negative,
            cause_held_out,
        ],
        [
            _candidate(finding_positive, CanonicalRelation.HAS_FINDING),
            _candidate(finding_negative, CanonicalRelation.NO_RELATION),
            _cause_candidate(cause_positive, CanonicalRelation.CAUSES),
            _cause_candidate(cause_negative, CanonicalRelation.NO_RELATION),
        ],
        prediction_candidates,
        n_neighbors=1,
    )

    for candidate, prediction in zip(
        prediction_candidates, run.predictions, strict=True
    ):
        assert prediction.canonical_label in {
            candidate.candidate_relation,
            CanonicalRelation.NO_RELATION,
        }


def test_file_pipeline_writes_predictions_and_auditable_manifest(tmp_path):
    positive = _document(1, "Patient reports pain clearly.")
    negative = _document(2, "Routine observation mentions pain incidentally.")
    held_out = _document(3, "Patient reports pain clearly.")
    corpus = tmp_path / "corpus.jsonl"
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl([positive, negative, held_out], corpus)
    _write_candidates(
        train,
        [
            _candidate(positive, CanonicalRelation.HAS_FINDING),
            _candidate(negative, CanonicalRelation.NO_RELATION),
        ],
    )
    _write_candidates(
        validation,
        [_candidate(held_out, CanonicalRelation.HAS_FINDING)],
    )

    run_ngram_knn_from_files(
        corpus,
        train,
        validation,
        predictions,
        n_neighbors=1,
        model_version="test-model-1",
    )

    loaded = load_relation_predictions(predictions)
    manifest = json.loads(
        predictions.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert len(loaded) == 1
    assert manifest["training_examples"] == 2
    assert manifest["prediction_examples"] == 1
    assert manifest["artifact_hashes"]["predictions_jsonl_sha256"]
    assert manifest["model_version"] == "test-model-1"


def test_model_selection_uses_validation_folds_without_opening_test(tmp_path):
    positive_one = _document(1, "Patient reports pain clearly.")
    negative_one = _document(2, "Routine observation mentions pain incidentally.")
    positive_two = _document(3, "Patient reports pain persistently.")
    negative_two = _document(4, "Routine review mentions pain incidentally.")
    documents = [positive_one, negative_one, positive_two, negative_two]
    corpus = tmp_path / "corpus.jsonl"
    frozen = tmp_path / "frozen"
    output = tmp_path / "selection"
    frozen.mkdir()
    write_jsonl(documents, corpus)
    _write_candidates(
        frozen / "fold-01-train.jsonl",
        [
            _candidate(positive_one, CanonicalRelation.HAS_FINDING),
            _candidate(negative_one, CanonicalRelation.NO_RELATION),
        ],
    )
    _write_candidates(
        frozen / "fold-01-validation.jsonl",
        [
            _candidate(positive_two, CanonicalRelation.HAS_FINDING),
            _candidate(negative_two, CanonicalRelation.NO_RELATION),
        ],
    )
    _write_candidates(
        frozen / "fold-02-train.jsonl",
        [
            _candidate(positive_two, CanonicalRelation.HAS_FINDING),
            _candidate(negative_two, CanonicalRelation.NO_RELATION),
        ],
    )
    _write_candidates(
        frozen / "fold-02-validation.jsonl",
        [
            _candidate(positive_one, CanonicalRelation.HAS_FINDING),
            _candidate(negative_one, CanonicalRelation.NO_RELATION),
        ],
    )
    # A malformed test partition would fail if model selection tried to open it.
    (frozen / "test.jsonl").write_text("not json\n", encoding="utf-8")

    summary = select_ngram_knn_from_folds(
        corpus,
        frozen,
        output,
        neighbor_values=(1,),
        maximum_ngram_orders=(1, 3),
        minimum_positive_support=1,
        model_version="selection-test",
    )

    assert summary["folds"] == ["fold-01", "fold-02"]
    assert summary["test_partition_used"] is False
    assert len(summary["configurations"]) == 2
    assert summary["best_configuration"]["n_neighbors"] == 1
    assert summary["best_configuration"]["maximum_ngram_order"] in (1, 3)
    assert (output / "model-selection-summary.json").is_file()


def test_finalization_builds_full_development_train_and_evaluates_test(tmp_path):
    documents = [
        _document(1, "Patient reports pain clearly."),
        _document(2, "Routine observation mentions pain incidentally."),
        _document(3, "Patient reports pain persistently."),
        _document(4, "Routine review mentions pain incidentally."),
        _document(5, "Patient reports pain acutely."),
        _document(6, "Routine note mentions pain incidentally."),
    ]
    corpus = tmp_path / "corpus.jsonl"
    frozen = tmp_path / "frozen"
    selection_path = tmp_path / "selection.json"
    output = tmp_path / "final"
    frozen.mkdir()
    write_jsonl(documents, corpus)
    for fold, indices in ((1, (0, 1)), (2, (2, 3))):
        candidates = [
            _candidate(documents[indices[0]], CanonicalRelation.HAS_FINDING),
            _candidate(documents[indices[1]], CanonicalRelation.NO_RELATION),
        ]
        _write_candidates(frozen / f"fold-{fold:02d}-validation.jsonl", candidates)
        _write_candidates(frozen / f"fold-{fold:02d}-train.jsonl", candidates)
    _write_candidates(
        frozen / "test.jsonl",
        [
            _candidate(documents[4], CanonicalRelation.HAS_FINDING),
            _candidate(documents[5], CanonicalRelation.NO_RELATION),
        ],
    )
    selection_path.write_text(
        json.dumps(
            {
                "best_configuration": {
                    "maximum_ngram_order": 1,
                    "n_neighbors": 1,
                },
                "model_version": "selection-test",
                "seed": 42,
                "test_partition_used": False,
            }
        ),
        encoding="utf-8",
    )

    summary = finalize_selected_ngram_knn(
        corpus,
        frozen,
        selection_path,
        output,
        minimum_positive_support=1,
    )

    assert summary["test_partition_used_once_after_selection"] is True
    assert summary["training_manifest"]["document_count"] == 4
    assert summary["training_manifest"]["candidate_count"] == 4
    assert (output / "final-test.predictions.jsonl").is_file()
    assert (output / "final-test.evaluation.json").is_file()
