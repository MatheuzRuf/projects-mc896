import json

import pytest

from project2_from_project1.src.clinical_documents import ClinicalDocument, Mention, Span, write_jsonl
from project2_from_project1.src.entity_recognition import (
    CanonicalDictionary,
    EntityRecognitionError,
    RawDocument,
    evaluate_entities,
    join_gold_projection,
    load_document_folds,
    run_dictionary_fold_from_files,
    training_vocabulary,
)
from project2_from_project1.src.gold_projection import (
    build_projection_plan,
    load_flat_gold_jsonl,
    project_gold_from_files,
)
from project2_from_project1.src.run_manifest import sha256_file
from project2_from_project1.src.supervised_contracts import EntityPrediction, load_contract_jsonl

TYPE_TO_NATIVE = {"Condition": "condition", "Finding": "finding", "Factor": "factor", "Modifier": "modifier"}


def _document(document_id, text, mentions):
    items = []
    cursor = {}
    for index, (surface, canonical_type) in enumerate(mentions):
        start = text.index(surface, cursor.get(surface, 0))
        cursor[surface] = start + len(surface)
        items.append(Mention(f"{document_id}:m{index}", (Span(start, start + len(surface)),),
                             (TYPE_TO_NATIVE[canonical_type],), canonical_type))
    return ClinicalDocument(document_id, text, (), (), tuple(items), ())


TRAIN = (
    _document("d1", "Patient denies chest pain. Asthma was noted. Fever and cough.", [
        ("denies", "Modifier"), ("chest pain", "Finding"), ("Asthma", "Condition"),
        ("Fever", "Finding"), ("cough", "Finding"),
    ]),
    _document("d2", "No fever. Edema present. Chest pain worsened.", [
        ("No", "Modifier"), ("fever", "Finding"), ("Edema", "Finding"), ("Chest pain", "Finding"),
    ]),
    _document("d3", "Edema of legs. Pain in asthma.", [
        ("Edema", "Condition"), ("Pain", "Finding"), ("asthma", "Condition"),
    ]),
)
VALIDATION = _document("d4", "She has chest pain and edema. Asthma flared. Rash appeared.", [
    ("chest pain", "Finding"), ("edema", "Finding"), ("Asthma", "Condition"), ("Rash", "Finding"),
])
TEST = _document("d5", "Held-out fever.", [("fever", "Finding")])


def _gold(documents):
    plan = build_projection_plan(documents)
    assert not plan.cases
    return join_gold_projection(documents, plan.project(()))


def test_dictionary_uses_training_majority_abstains_on_ties_and_prefers_longest_match():
    gold = _gold((*TRAIN, VALIDATION))
    dictionary = CanonicalDictionary.fit(gold[name] for name in ("d1", "d2", "d3"))

    entries = {entry.surface: entry for entry in dictionary.entries}
    assert entries[("chest", "pain")].type_counts == {"Finding": 2}
    assert entries[("asthma",)].resolved_type == "Condition"
    assert entries[("edema",)].type_counts == {"Condition": 1, "Finding": 1}
    assert entries[("edema",)].resolved_type is None
    assert ("rash",) not in entries
    assert dictionary.training_document_ids == ("d1", "d2", "d3")

    predictions, stats = dictionary.predict(RawDocument("d4", VALIDATION.text))
    text = VALIDATION.text
    assert [(text[p.span.start:p.span.end], p.canonical_type, p.sentence_index) for p in predictions] == [
        ("chest pain", "Finding", 0), ("Asthma", "Condition", 1),
    ]
    assert stats.abstained_surface_occurrences == 1
    assert stats.discarded_by_longer_overlap == 1  # "pain" lost to "chest pain"
    assert all(p.method == "canonical-dictionary" and p.marginal_probability is None for p in predictions)


def test_dictionary_never_matches_across_sentences_and_rejects_gold_bearing_input():
    gold = _gold(TRAIN)
    dictionary = CanonicalDictionary.fit(gold.values())
    predictions, _ = dictionary.predict(RawDocument("x", "The chest. Pain resolved."))
    assert [(p.span.to_dict(), p.canonical_type) for p in predictions] == [([11, 15], "Finding")]
    with pytest.raises(TypeError, match="RawDocument"):
        dictionary.predict(VALIDATION)


def test_strict_evaluation_decomposes_errors_and_reports_oov_and_length():
    gold = _gold((*TRAIN, VALIDATION))
    training = [gold[name] for name in ("d1", "d2", "d3")]
    surfaces, tokens = training_vocabulary(training)

    def prediction(start, end, canonical_type):
        return EntityPrediction("d4", Span(start, end), canonical_type, 0, "test", "v1")

    text = VALIDATION.text
    chest = text.index("chest pain")
    edema = text.index("edema")
    asthma = text.index("Asthma")
    report = evaluate_entities(
        [gold["d4"]],
        [
            prediction(chest, chest + 10, "Finding"),          # exact
            prediction(edema, edema + 5, "Condition"),          # type mismatch
            prediction(asthma, asthma + 13, "Condition"),       # boundary mismatch ("Asthma flared")
            prediction(0, 3, "Modifier"),                       # spurious ("She")
        ],
        training_surfaces=surfaces, training_tokens=tokens,
    )

    strict = report["strict"]
    assert strict["micro"]["true_positive"] == 1
    assert strict["micro"]["precision"] == pytest.approx(0.25)
    assert strict["micro"]["recall"] == pytest.approx(0.25)
    assert strict["per_type"]["Finding"]["support"] == 3
    assert strict["per_type"]["Finding"]["f1"] == pytest.approx(0.5)
    assert strict["macro_f1"] == pytest.approx(0.5 / 4)
    assert report["errors"]["false_positive"] == {
        "boundary_and_type_mismatch": 0, "boundary_mismatch": 1, "spurious": 1, "type_mismatch": 1,
    }
    assert report["errors"]["false_negative"] == {"missing": 1, "overlapped_by_incorrect_prediction": 2}
    assert report["diagnostics"]["oov_surface"]["unseen_surface"]["support"] == 1  # "Rash"
    assert report["diagnostics"]["oov_token"]["contains_unseen_token"]["support"] == 1
    assert report["diagnostics"]["length_tokens"]["2"]["true_positive"] == 1
    assert report["diagnostics"]["lenient_overlap_same_type"]["matched_gold"] == 2


def _write_inputs(tmp_path, documents, name="run"):
    directory = tmp_path / name
    corpus = directory / "mcrc.jsonl"
    write_jsonl(documents, corpus)
    decisions = directory / "adjudication.jsonl"
    decisions.write_text("", encoding="utf-8")
    projection = directory / "gold" / "gold-projection.jsonl"
    projection_manifest = directory / "gold" / "manifest.json"
    project_gold_from_files(corpus, decisions, projection, directory / "gold" / "audit.jsonl", projection_manifest)
    partitions = directory / "partitions.json"
    partitions.write_text(json.dumps({
        "partitions": {
            "folds": [
                {"fold": 1, "train_document_ids": ["d1", "d2", "d3"], "validation_document_ids": ["d4"]},
                {"fold": 2, "train_document_ids": ["d2", "d3", "d4"], "validation_document_ids": ["d1"]},
            ],
            "test_document_ids": ["d5"],
        },
        "source_jsonl_sha256": sha256_file(corpus),
    }), encoding="utf-8")
    return corpus, projection, projection_manifest, partitions


def test_fold_run_writes_contracts_evaluation_and_text_free_manifest(tmp_path):
    inputs = _write_inputs(tmp_path, (*TRAIN, VALIDATION, TEST))
    output = tmp_path / "out"

    manifest = run_dictionary_fold_from_files(*inputs, 1, output)

    predictions = load_contract_jsonl(output / "predictions.jsonl", "entity")
    assert {prediction.document_id for prediction in predictions} == {"d4"}
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["strict"]["micro"]["true_positive"] == 2
    assert manifest["partitions"] == {"train_document_ids": ["d1", "d2", "d3"], "validation_document_ids": ["d4"]}
    assert manifest["dictionary"]["abstained_tied_surfaces"] == 1
    assert manifest["artifact_hashes"]["predictions_jsonl_sha256"] == sha256_file(output / "predictions.jsonl")
    assert manifest["artifact_hashes"]["corpus_jsonl_sha256"] == sha256_file(inputs[0])
    assert manifest["primary_metric"]["name"] == "strict_macro_f1"
    assert '"d5"' not in json.dumps(manifest)
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8").casefold()
    for surface in ("chest", "asthma", "edema", "fever"):
        assert surface not in manifest_text
        assert surface not in (output / "evaluation.json").read_text(encoding="utf-8").casefold()

    second = tmp_path / "again"
    run_dictionary_fold_from_files(*inputs, 1, second)
    for name in ("predictions.jsonl", "evaluation.json", "dictionary.jsonl"):
        assert (output / name).read_bytes() == (second / name).read_bytes()


def test_predictions_do_not_depend_on_labels_of_the_predicted_partition(tmp_path):
    relabeled = _document("d4", VALIDATION.text, [("She", "Factor"), ("flared", "Modifier")])
    original = _write_inputs(tmp_path, (*TRAIN, VALIDATION, TEST), "original")
    changed = _write_inputs(tmp_path, (*TRAIN, relabeled, TEST), "changed")

    run_dictionary_fold_from_files(*original, 1, tmp_path / "a")
    run_dictionary_fold_from_files(*changed, 1, tmp_path / "b")

    assert (tmp_path / "a" / "predictions.jsonl").read_bytes() == (tmp_path / "b" / "predictions.jsonl").read_bytes()
    assert (tmp_path / "a" / "evaluation.json").read_bytes() != (tmp_path / "b" / "evaluation.json").read_bytes()


def test_dictionary_learned_on_one_fold_ignores_other_fold_surfaces(tmp_path):
    inputs = _write_inputs(tmp_path, (*TRAIN, VALIDATION, TEST))
    run_dictionary_fold_from_files(*inputs, 1, tmp_path / "fold1")
    surfaces = [json.loads(line)["surface_tokens"]
                for line in (tmp_path / "fold1" / "dictionary.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ["rash"] not in surfaces


def test_run_rejects_inputs_derived_from_different_sources(tmp_path):
    corpus, projection, projection_manifest, partitions = _write_inputs(tmp_path, (*TRAIN, VALIDATION, TEST))
    payload = json.loads(partitions.read_text(encoding="utf-8"))
    payload["source_jsonl_sha256"] = "0" * 64
    partitions.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EntityRecognitionError, match="different corpus"):
        run_dictionary_fold_from_files(corpus, projection, projection_manifest, partitions, 1, tmp_path / "out")
    with pytest.raises(EntityRecognitionError, match="not defined"):
        payload["source_jsonl_sha256"] = sha256_file(corpus)
        partitions.write_text(json.dumps(payload), encoding="utf-8")
        run_dictionary_fold_from_files(corpus, projection, projection_manifest, partitions, 9, tmp_path / "out")


def test_folds_touching_test_documents_are_rejected():
    with pytest.raises(EntityRecognitionError, match="held-out test"):
        load_document_folds({"partitions": {
            "folds": [{"fold": 1, "train_document_ids": ["a", "t"], "validation_document_ids": ["b"]}],
            "test_document_ids": ["t"],
        }})


def test_projection_loader_round_trips_and_rejects_tampered_ids(tmp_path):
    corpus, projection, _, _ = _write_inputs(tmp_path, (*TRAIN, VALIDATION, TEST))
    loaded = load_flat_gold_jsonl(projection)
    assert [document.document_id for document in loaded] == ["d1", "d2", "d3", "d4", "d5"]
    lines = projection.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["mentions"][0]["mention_id"] = "gold:tampered"
    projection.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mention_id"):
        load_flat_gold_jsonl(projection)
