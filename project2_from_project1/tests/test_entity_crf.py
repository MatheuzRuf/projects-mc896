import json

import pytest
import torch

from project2_from_project1.src.entity_recognition import (
    EntityRecognitionError,
    NonFiniteTrainingError,
    RawDocument,
    SemiMarkovCRF,
    TrainingConfig,
    run_crf_fold_from_files,
)
from project2_from_project1.src.entity_recognition.crf_features import (
    BOS,
    EOS,
    FeatureVocabulary,
    case_class,
    encode_batches,
    length_bucket,
    word_shape,
)
from project2_from_project1.src.entity_recognition.corpus import normalize_token
from project2_from_project1.src.gold_projection import segment_sentences, tokenize_with_offsets
from project2_from_project1.src.run_manifest import sha256_file
from project2_from_project1.src.supervised_contracts import load_contract_jsonl

from project2_from_project1.tests.test_entity_dictionary import TEST, TRAIN, VALIDATION, _document, _gold, _write_inputs

CONFIG = TrainingConfig(l2=0.1)


def _training():
    gold = _gold((*TRAIN, VALIDATION))
    return gold, [gold[name] for name in ("d1", "d2", "d3")]


def test_features_are_fitted_on_training_with_minimum_frequency_and_keep_oov_shape():
    gold, training = _training()
    vocabulary = FeatureVocabulary.fit(((document.text, sentence) for document in training for sentence in document.sentences), 40)
    names = set(vocabulary.features)

    assert "first.w=pain" in names and "in.w=chest" in names       # each seen twice
    assert "first.w=legs" not in names and "in.w=denies" not in names  # seen once in training
    assert not any("rash" in name or "flared" in name for name in names)  # validation-only tokens
    assert not any(name.startswith("dict") for name in names)
    assert {"bias", "has_digit", "all_punct", "len=1", "len=21-40", f"prev.w={BOS}", f"next.w={EOS}"} <= names

    text = VALIDATION.text
    sentence = segment_sentences(text, tokenize_with_offsets(text))[2]  # "Rash appeared."
    features = vocabulary.token_features(text, sentence)
    rash = [vocabulary.features[index] for index in features["first"][0]]
    assert sorted(rash) == ["first.case=title", "first.shape=Xx"]
    assert word_shape("Fever") == "Xx" and word_shape("12") == "d" and case_class("mRNA") == "mixed"


def _expected_names(vocabulary, text, sentence, start, size):
    tokens = [text[token.span.start:token.span.end] for token in sentence.tokens]

    def role(name, raw):
        if raw is None:
            return [f"{name}.w={BOS if name == 'prev' else EOS}"]
        norm = normalize_token(raw)
        found = []
        if norm in vocabulary.lexical_tokens:
            found.append(f"{name}.w={norm}")
        if name == "first" and norm[:3] in vocabulary.prefixes:
            found.append(f"first.p3={norm[:3]}")
        if name == "last" and norm[-3:] in vocabulary.suffixes:
            found.append(f"last.s3={norm[-3:]}")
        if name != "in":
            found += [f"{name}.shape={word_shape(raw)}", f"{name}.case={case_class(raw)}"]
        return found

    stop = start + size
    names = ["bias", f"len={length_bucket(size)}", *role("first", tokens[start]), *role("last", tokens[stop - 1]),
             *role("prev", tokens[start - 1] if start else None), *role("next", tokens[stop] if stop < len(tokens) else None)]
    for raw in tokens[start:stop]:
        names += role("in", raw)
    if any(raw.isdigit() for raw in tokens[start:stop]):
        names.append("has_digit")
    if all(case_class(raw) == "punct" for raw in tokens[start:stop]):
        names.append("all_punct")
    return [name for name in names if name in vocabulary.index]


def test_segment_scores_equal_the_sum_of_active_feature_weights():
    gold, training = _training()
    vocabulary = FeatureVocabulary.fit(((document.text, sentence) for document in training for sentence in document.sentences), 3)
    model = SemiMarkovCRF(vocabulary, 3)
    model.weights = torch.randn(model.weights.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    document = gold["d4"]
    sentences = [(document.document_id, document.text, sentence) for sentence in document.sentences]
    for batch in encode_batches(vocabulary, sentences, 3):
        scores = model.segment_scores(batch)
        for b, (_, sentence_index) in enumerate(batch.keys):
            sentence = document.sentences[sentence_index]
            for start in range(len(sentence.tokens)):
                for size in range(1, min(3, len(sentence.tokens) - start) + 1):
                    names = _expected_names(vocabulary, document.text, sentence, start, size)
                    expected = sum((model.weights[vocabulary.index[name]] for name in names), torch.zeros(5, dtype=torch.float64))
                    assert torch.allclose(scores[b, start, size - 1], expected)


def test_training_is_deterministic_converges_and_predicts_flat_entities_with_marginals():
    gold, training = _training()
    first, diagnostics = SemiMarkovCRF.fit(training, CONFIG)
    second, _ = SemiMarkovCRF.fit(training, CONFIG)

    assert diagnostics.converged and diagnostics.stop_reason in ("gradient_tolerance", "change_tolerance")
    assert diagnostics.iterations <= 250
    assert torch.equal(first.weights, second.weights) and torch.equal(first.transitions, second.transitions)

    predictions = first.predict([gold["d4"].raw()])
    assert predictions == second.predict([gold["d4"].raw()])
    text = VALIDATION.text
    found = {(text[p.span.start:p.span.end], p.canonical_type) for p in predictions}
    assert ("chest pain", "Finding") in found and ("Asthma", "Condition") in found
    assert all(0.0 <= p.marginal_probability <= 1.0 and p.method == "semi-markov-crf" for p in predictions)
    spans = sorted((p.span.start, p.span.end) for p in predictions)
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))

    restored = SemiMarkovCRF.from_dict(json.loads(json.dumps(first.to_dict())))
    assert restored.predict([gold["d4"].raw()]) == predictions
    with pytest.raises(TypeError, match="RawDocument"):
        first.predict([gold["d4"]])


def test_untrained_model_starts_at_zero_and_never_truncates_long_gold_mentions():
    gold, training = _training()
    model = SemiMarkovCRF(FeatureVocabulary.fit([], 40), 40)
    assert all(not parameter.any() for parameter in model.parameters)
    with pytest.raises(EntityRecognitionError, match="never truncated"):
        SemiMarkovCRF.fit(training, TrainingConfig(l2=0.1, max_segment=1))


def test_finite_non_converged_run_is_evaluated_and_flagged(tmp_path):
    inputs = _write_inputs(tmp_path, (*TRAIN, VALIDATION, TEST))
    manifest = run_crf_fold_from_files(*inputs, 1, tmp_path / "out", config=TrainingConfig(l2=0.1, max_iterations=1))

    assert manifest["status"] == "completed"
    assert manifest["converged"] is False
    assert manifest["training"]["stop_reason"] == "max_iterations"
    assert (tmp_path / "out" / "evaluation.json").exists()
    assert manifest["primary_metric"]["name"] == "strict_macro_f1"


def test_non_finite_training_fails_with_structured_diagnostic_and_no_fallback(tmp_path):
    inputs = _write_inputs(tmp_path, (*TRAIN, VALIDATION, TEST))

    def exploding(documents, config):
        raise NonFiniteTrainingError("semi-Markov CRF objective or gradient is not finite",
                                     {"function_evaluations": 3, "objective_finite": False, "gradient_finite": True})

    with pytest.raises(NonFiniteTrainingError):
        run_crf_fold_from_files(*inputs, 1, tmp_path / "out", config=CONFIG, fit=exploding)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_non_finite"
    assert manifest["failure"]["objective_finite"] is False
    assert manifest["primary_metric"] is None
    assert not (tmp_path / "out" / "predictions.jsonl").exists()


def test_fold_run_writes_predictions_evaluation_and_text_free_manifest(tmp_path):
    inputs = _write_inputs(tmp_path, (*TRAIN, VALIDATION, TEST))
    output = tmp_path / "out"
    manifest = run_crf_fold_from_files(*inputs, 1, output, config=CONFIG)

    predictions = load_contract_jsonl(output / "predictions.jsonl", "entity")
    assert {p.document_id for p in predictions} == {"d4"}
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["diagnostics"]["marginal_probability"]["interpretation"] == "uncalibrated_model_posterior"
    assert manifest["converged"] is True
    assert manifest["configuration"]["l2"] == 0.1
    assert manifest["configuration"]["max_iterations"] == 250
    assert manifest["determinism"]["initialization"] == "zeros"
    assert manifest["environment"]["packages"]["torch"]
    assert manifest["artifact_hashes"]["model_json_sha256"] == sha256_file(output / "model.json")
    assert manifest["artifact_hashes"]["predictions_jsonl_sha256"] == sha256_file(output / "predictions.jsonl")
    text = (output / "manifest.json").read_text(encoding="utf-8").casefold()
    for surface in ("chest", "asthma", "edema", "fever"):
        assert surface not in text

    again = tmp_path / "again"
    run_crf_fold_from_files(*inputs, 1, again, config=CONFIG)
    for name in ("predictions.jsonl", "evaluation.json", "model.json"):
        assert (output / name).read_bytes() == (again / name).read_bytes()


def test_crf_predictions_do_not_depend_on_labels_of_the_predicted_partition(tmp_path):
    relabeled = _document("d4", VALIDATION.text, [("She", "Factor"), ("flared", "Modifier")])
    original = _write_inputs(tmp_path, (*TRAIN, VALIDATION, TEST), "original")
    changed = _write_inputs(tmp_path, (*TRAIN, relabeled, TEST), "changed")
    run_crf_fold_from_files(*original, 1, tmp_path / "a", config=CONFIG)
    run_crf_fold_from_files(*changed, 1, tmp_path / "b", config=CONFIG)
    assert (tmp_path / "a" / "predictions.jsonl").read_bytes() == (tmp_path / "b" / "predictions.jsonl").read_bytes()


def test_unknown_tokens_in_raw_text_are_handled_without_vocabulary_growth():
    _, training = _training()
    model, _ = SemiMarkovCRF.fit(training, CONFIG)
    size = len(model.vocabulary)
    model.predict([RawDocument("new", "Zzyzx 42 !!")])
    assert len(model.vocabulary) == size
