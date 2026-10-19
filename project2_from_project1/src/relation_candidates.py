"""Gold-free, deterministic relation candidates from entity predictions."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

from project2_from_project1.src.entity_recognition.corpus import RawDocument
from project2_from_project1.src.gold_projection import SENTENCE_SEGMENTER_VERSION, TOKENIZER_VERSION
from project2_from_project1.src.run_manifest import dependency_versions, git_provenance, sha256_file
from project2_from_project1.src.supervised_contracts import EntityPrediction, RelationCandidate, load_contract_jsonl, write_contract_jsonl

PATIENT_RELATIONS = {'Condition': 'HAS_CONDITION', 'Finding': 'HAS_FINDING', 'Factor': 'HAS_FACTOR'}
PAIR_RELATIONS = {'Condition': 'CAUSES', 'Modifier': 'NEGATES'}


def logical_patient_id(document_id: str) -> str:
    """Identity of the logical patient, derived solely from the document ID."""
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError('document_id must be a non-empty string')
    encoded = json.dumps([document_id], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return 'patient:' + hashlib.sha256(encoded).hexdigest()


def generate_candidates(documents: Iterable[RawDocument], predictions: Iterable[EntityPrediction]) -> tuple[RelationCandidate, ...]:
    """Build compatible pairs within one sentence; Patient has no sentence."""
    document_map = {}
    for document in documents:
        if not isinstance(document, RawDocument) or not isinstance(document.document_id, str) or not document.document_id.strip() or not isinstance(document.text, str):
            raise ValueError('documents must have a non-empty ID and text')
        if document.document_id in document_map:
            raise ValueError(f'duplicate document_id {document.document_id!r}')
        document_map[document.document_id] = document
    by_document = defaultdict(list)
    seen = set()
    sentences = {}
    for prediction in predictions:
        if not isinstance(prediction, EntityPrediction):
            raise ValueError('predictions must be EntityPrediction records')
        if prediction.document_id not in document_map:
            raise ValueError(f'unknown document {prediction.document_id!r}')
        if prediction.prediction_id in seen:
            raise ValueError(f'duplicate prediction {prediction.prediction_id!r}')
        seen.add(prediction.prediction_id)
        if prediction.document_id not in sentences:
            sentences[prediction.document_id] = document_map[prediction.document_id].sentences
        document_sentences = sentences[prediction.document_id]
        if prediction.sentence_index >= len(document_sentences):
            raise ValueError(f'prediction {prediction.prediction_id} has an invalid sentence index')
        sentence = document_sentences[prediction.sentence_index]
        if not (sentence.span.start <= prediction.span.start < prediction.span.end <= sentence.span.end):
            raise ValueError(f'prediction {prediction.prediction_id} does not lie in its sentence')
        by_document[prediction.document_id].append(prediction)
    candidates = []
    for document_id in sorted(by_document):
        entities = sorted(by_document[document_id], key=lambda item: (item.span.start, item.span.end, item.canonical_type))
        findings = [item for item in entities if item.canonical_type == 'Finding']
        for entity in entities:
            relation = PATIENT_RELATIONS.get(entity.canonical_type)
            if relation:
                candidates.append(RelationCandidate(document_id, logical_patient_id(document_id), entity.prediction_id,
                                                    'Patient', entity.canonical_type, relation, None))
            relation = PAIR_RELATIONS.get(entity.canonical_type)
            if relation:
                for finding in findings:
                    distance = abs(entity.sentence_index - finding.sentence_index)
                    if distance <= 1:
                        candidates.append(RelationCandidate(document_id, entity.prediction_id, finding.prediction_id,
                                                            entity.canonical_type, 'Finding', relation, distance))
    return tuple(sorted(candidates, key=lambda item: (item.document_id, item.source_id, item.target_id, item.candidate_relation.value)))


def _load_raw_documents(path: Path) -> list[RawDocument]:
    documents = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            record = json.loads(line)
            documents.append(RawDocument(record['document_id'], record['text']))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'invalid raw document on line {line_number}: {error}') from error
    return documents


def run_from_files(corpus_path: str | Path, predictions_path: str | Path, output_dir: str | Path,
                   prediction_manifest_path: str | Path | None = None) -> dict:
    """Write candidates and a text-free manifest from existing JSONL artifacts."""
    started = time.perf_counter()
    corpus_path, predictions_path, output_dir = Path(corpus_path), Path(predictions_path), Path(output_dir)
    inputs = [corpus_path, predictions_path] + ([Path(prediction_manifest_path)] if prediction_manifest_path else [])
    if any(path.resolve() in {(output_dir / 'candidates.jsonl').resolve(), (output_dir / 'manifest.json').resolve()} for path in inputs):
        raise ValueError('output paths must not overwrite inputs')
    documents = _load_raw_documents(corpus_path)
    predictions = load_contract_jsonl(predictions_path, 'entity')
    source_manifest = None
    predicted_document_ids = None
    if prediction_manifest_path:
        source_manifest = json.loads(Path(prediction_manifest_path).read_text(encoding='utf-8'))
        if source_manifest.get('artifact_hashes', {}).get('predictions_jsonl_sha256') != sha256_file(predictions_path):
            raise ValueError('entity predictions do not match their manifest')
        if source_manifest.get('artifact_hashes', {}).get('corpus_jsonl_sha256') != sha256_file(corpus_path):
            raise ValueError('corpus does not match the entity manifest')
        configuration = source_manifest.get('configuration', {})
        if (configuration.get('sentence_segmenter_version') != SENTENCE_SEGMENTER_VERSION
                or configuration.get('tokenizer_version') != TOKENIZER_VERSION):
            raise ValueError('entity manifest uses a different sentence segmentation')
        partitions = source_manifest.get('partitions', {})
        if 'validation_document_ids' in partitions:
            predicted_document_ids = partitions['validation_document_ids']
            if (not isinstance(predicted_document_ids, list)
                    or any(not isinstance(item, str) or not item.strip() for item in predicted_document_ids)
                    or len(set(predicted_document_ids)) != len(predicted_document_ids)):
                raise ValueError('invalid validation_document_ids in entity manifest')
            predicted_document_ids = sorted(predicted_document_ids)
            if not set(predicted_document_ids) <= {document.document_id for document in documents}:
                raise ValueError('entity manifest references an unknown validation document')
            if not {prediction.document_id for prediction in predictions} <= set(predicted_document_ids):
                raise ValueError('entity prediction outside manifest validation partition')
    candidates = generate_candidates(documents, predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / 'candidates.jsonl'
    write_contract_jsonl(candidates, candidate_path)
    counts = Counter(item.candidate_relation.value for item in candidates)
    manifest = {
        'schema_version': 1,
        'configuration': {'maximum_sentence_distance': 1, 'sentence_window_inclusive': True,
                          'sentence_segmenter_version': SENTENCE_SEGMENTER_VERSION, 'tokenizer_version': TOKENIZER_VERSION,
                          'patient_identity': 'sha256-json-document-id-v1'},
        'input_hashes': {'corpus_sha256': sha256_file(corpus_path), 'entity_predictions_sha256': sha256_file(predictions_path),
                         'entity_manifest_sha256': sha256_file(prediction_manifest_path) if prediction_manifest_path else None},
        'output_hashes': {'candidates_sha256': sha256_file(candidate_path)},
        'code_sha256': sha256_file(__file__),
        'counts': {'documents': len(documents), 'predictions': len(predictions), 'candidates': len(candidates),
                   'by_relation': {relation: counts[relation] for relation in sorted((*PATIENT_RELATIONS.values(), *PAIR_RELATIONS.values()))}},
        'entity_methods': [{'method': method, 'model_version': version} for method, version in sorted({(p.method, p.model_version) for p in predictions})],
        'corpus_document_ids': sorted(document.document_id for document in documents),
        'prediction_document_ids': predicted_document_ids,
        'source_partitions': source_manifest.get('partitions') if source_manifest else None,
        'source_recognizer': ({'method': source_manifest.get('method'), 'model_version': source_manifest.get('model_version')}
                              if source_manifest else None),
        'determinism': {'candidate_order': 'document_source_target_relation', 'randomness': 'none'},
        'git': {**git_provenance(), 'dirty_scope': 'tracked_files'}, 'dependencies': dependency_versions(),
    }
    manifest['duration_seconds'] = round(time.perf_counter() - started, 6)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return manifest
