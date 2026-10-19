import json
import pytest
from project2_from_project1.src.clinical_documents import Span
from project2_from_project1.src.entity_recognition.corpus import RawDocument
from project2_from_project1.src.relation_candidates import generate_candidates, logical_patient_id, run_from_files
from project2_from_project1.src.gold_projection import SENTENCE_SEGMENTER_VERSION, TOKENIZER_VERSION
from project2_from_project1.src.run_manifest import sha256_file
from project2_from_project1.src.supervised_contracts import EntityPrediction, write_contract_jsonl


def prediction(doc, term, kind, sentence, method='dictionary'):
    start = doc.text.index(term)
    return EntityPrediction(doc.document_id, Span(start, start + len(term)), kind, sentence, method, '1')


def test_candidates_use_types_direction_and_versioned_sentence_window():
    doc = RawDocument('d', 'Dr. Smith has asthma. No cough. Fever. Later pain.')
    entities = [prediction(doc, 'asthma', 'Condition', 0), prediction(doc, 'No', 'Modifier', 1),
                prediction(doc, 'cough', 'Finding', 1), prediction(doc, 'Fever', 'Finding', 2),
                prediction(doc, 'pain', 'Factor', 3)]
    candidates = generate_candidates([doc], entities)
    triples = {(c.source_type, c.target_type, c.candidate_relation.value, c.sentence_distance) for c in candidates}
    assert triples == {('Patient', 'Condition', 'HAS_CONDITION', None), ('Patient', 'Finding', 'HAS_FINDING', None),
                       ('Patient', 'Factor', 'HAS_FACTOR', None), ('Modifier', 'Finding', 'NEGATES', 0),
                       ('Modifier', 'Finding', 'NEGATES', 1), ('Condition', 'Finding', 'CAUSES', 1)}
    assert all(c.source_id == logical_patient_id('d') for c in candidates if c.source_type == 'Patient')
    assert not any(c.target_id == entities[3].prediction_id and c.candidate_relation.value == 'CAUSES' for c in candidates)
    assert all('label' not in c.to_dict() and 'gold_relation_id' not in c.to_dict() for c in candidates)
    assert generate_candidates([doc], list(reversed(entities))) == candidates
    other = [EntityPrediction(e.document_id, e.span, e.canonical_type, e.sentence_index, 'crf', '2') for e in entities]
    assert generate_candidates([doc], other) == candidates


def test_invalid_predictions_are_rejected():
    doc = RawDocument('d', 'One. Two.')
    wrong = prediction(doc, 'Two', 'Finding', 0)
    with pytest.raises(ValueError, match='sentence'):
        generate_candidates([doc], [wrong])
    with pytest.raises(ValueError, match='unknown document'):
        generate_candidates([], [wrong])
    with pytest.raises(ValueError, match='duplicate'):
        good = prediction(doc, 'Two', 'Finding', 1)
        generate_candidates([doc], [good, good])


def test_file_run_is_deterministic_and_ignores_gold(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    entities = tmp_path / 'entities.jsonl'
    doc = RawDocument('doc', 'Asthma. Cough.')
    corpus.write_text(json.dumps({'document_id': doc.document_id, 'text': doc.text, 'patients': [{'patient_id': 'gold'}], 'relations': 'irrelevant'}) + '\n', encoding='utf8')
    write_contract_jsonl([prediction(doc, 'Asthma', 'Condition', 0), prediction(doc, 'Cough', 'Finding', 1)], entities)
    first = run_from_files(corpus, entities, tmp_path / 'one')
    second = run_from_files(corpus, entities, tmp_path / 'two')
    assert {key: value for key, value in first.items() if key != 'duration_seconds'} == {
        key: value for key, value in second.items() if key != 'duration_seconds'}
    assert (tmp_path / 'one' / 'candidates.jsonl').read_bytes() == (tmp_path / 'two' / 'candidates.jsonl').read_bytes()
    assert first['counts']['candidates'] == 3
    assert first['input_hashes']['entity_predictions_sha256']
    assert len(first['code_sha256']) == 64
    assert first['git']['dirty_scope'] == 'tracked_files'
    corpus.write_text(json.dumps({'document_id': doc.document_id, 'text': doc.text, 'patients': [], 'relations': []}) + '\n', encoding='utf8')
    run_from_files(corpus, entities, tmp_path / 'three')
    assert (tmp_path / 'one' / 'candidates.jsonl').read_bytes() == (tmp_path / 'three' / 'candidates.jsonl').read_bytes()


def test_recognizer_manifest_rejects_changed_corpus(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    entities = tmp_path / 'entities.jsonl'
    manifest = tmp_path / 'recognizer.json'
    doc = RawDocument('doc', 'Asthma.')
    corpus.write_text(json.dumps({'document_id': doc.document_id, 'text': doc.text}) + '\n' +
                      json.dumps({'document_id': 'empty', 'text': 'No entities.'}) + '\n', encoding='utf8')
    write_contract_jsonl([prediction(doc, 'Asthma', 'Condition', 0)], entities)
    manifest.write_text(json.dumps({
        'artifact_hashes': {'corpus_jsonl_sha256': sha256_file(corpus), 'predictions_jsonl_sha256': sha256_file(entities)},
        'configuration': {'sentence_segmenter_version': SENTENCE_SEGMENTER_VERSION, 'tokenizer_version': TOKENIZER_VERSION},
        'partitions': {'validation_document_ids': ['doc', 'empty'], 'train_document_ids': []},
        'method': 'dictionary', 'model_version': '1',
    }), encoding='utf8')
    result = run_from_files(corpus, entities, tmp_path / 'good', manifest)
    assert result['prediction_document_ids'] == ['doc', 'empty']
    assert result['source_recognizer'] == {'method': 'dictionary', 'model_version': '1'}
    corpus.write_text(json.dumps({'document_id': doc.document_id, 'text': 'Allergy.'}) + '\n', encoding='utf8')
    with pytest.raises(ValueError, match='corpus does not match'):
        run_from_files(corpus, entities, tmp_path / 'bad', manifest)


def test_manifest_keeps_empty_fold_and_rejects_outside_prediction(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    entities = tmp_path / 'entities.jsonl'
    manifest = tmp_path / 'recognizer.json'
    corpus.write_text(''.join(json.dumps({'document_id': doc, 'text': 'Cough.'}) + '\n'
                              for doc in ('validation', 'train')), encoding='utf8')
    write_contract_jsonl([], entities)
    source = {'artifact_hashes': {'corpus_jsonl_sha256': sha256_file(corpus),
                                  'predictions_jsonl_sha256': sha256_file(entities)},
              'configuration': {'sentence_segmenter_version': SENTENCE_SEGMENTER_VERSION,
                                'tokenizer_version': TOKENIZER_VERSION},
              'partitions': {'validation_document_ids': ['validation'], 'train_document_ids': ['train']},
              'method': 'semicrf', 'model_version': '1'}
    manifest.write_text(json.dumps(source), encoding='utf8')
    result = run_from_files(corpus, entities, tmp_path / 'empty', manifest)
    assert result['prediction_document_ids'] == ['validation']
    assert result['source_recognizer']['method'] == 'semicrf'
    assert result['counts']['candidates'] == 0
    write_contract_jsonl([prediction(RawDocument('train', 'Cough.'), 'Cough', 'Finding', 0)], entities)
    source['artifact_hashes']['predictions_jsonl_sha256'] = sha256_file(entities)
    manifest.write_text(json.dumps(source), encoding='utf8')
    with pytest.raises(ValueError, match='outside manifest validation partition'):
        run_from_files(corpus, entities, tmp_path / 'outside', manifest)
