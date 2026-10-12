import hashlib
import pickle
import sys
from types import ModuleType

import pytest

from project2_from_project1.src.clinical_documents import CanonicalRelation, load_jsonl
from project2_from_project1.src.mcrc_converter import (
    MCRCConversionError,
    MCRCSource,
    convert_mcrc_distribution,
)


def _source_for(path):
    return MCRCSource(
        repository="https://github.com/adahealth/medical_case_report_corpus",
        revision="9b4b06f",
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _collection():
    return {
        "documents": [
            {
                "id": "doc-z",
                "passages": [
                    {
                        "offset": 0,
                        "text": "Patient had fever. No cough.",
                        "annotations": [
                            {"id": "case", "infons": {"entityType": "case"}, "locations": [{"offset": 0, "length": 7}]},
                            {"id": "condition", "infons": {"entityType": "condition"}, "locations": [{"offset": 12, "length": 5}]},
                            {"id": "modifier", "infons": {"entityType": "modifier"}, "locations": [{"offset": 19, "length": 2}]},
                            {"id": "finding", "infons": {"entityType": ["finding", "symptom"]}, "locations": [{"offset": 22, "length": 5}]},
                        ],
                        "relations": [
                            {"id": "has-condition", "infons": {"relationType": "case_has_condition"}, "nodes": [{"refid": "case", "role": "source"}, {"refid": "condition", "role": "target"}]},
                            {"id": "negates", "infons": {"relationType": "modifier_modifies_finding"}, "nodes": [{"refid": "modifier", "role": "source"}, {"refid": "finding", "role": "target"}]},
                        ],
                    }
                ],
            },
            {"id": "doc-a", "passages": [{"offset": 0, "text": "Case.", "annotations": [], "relations": []}]},
        ]
    }


def _pickle_collection(tmp_path):
    artifact = tmp_path / "CaseReportCorpus.pickle"
    artifact.write_bytes(pickle.dumps(_collection()))
    return artifact


def _official_shape_collection():
    text = "Patient had severe fever. No cough. Later fever returned."
    first_fever = text.index("fever")
    second_fever = text.rindex("fever")
    return {
        "documents": [{
            "id": "official-shape",
            "passages": [{
                "offset": 0,
                "text": text,
                "annotations": [
                    {"id": "case", "infons": {"entityType": "case"}, "locations": {"offset": 0, "length": 7}},
                    {"id": "condition", "infons": {"entityType": "condition"}, "locations": {"offset": text.index("severe fever"), "length": len("severe fever")}},
                    {"id": "nested", "infons": {"entityType": "finding"}, "locations": {"offset": first_fever, "length": len("fever")}},
                    {"id": "modifier", "infons": {"entityType": "modifier"}, "locations": {"offset": text.index("No"), "length": len("No")}},
                    {"id": "cough", "infons": {"entityType": "finding"}, "locations": {"offset": text.index("cough"), "length": len("cough")}},
                    {"id": "discontinuous", "infons": {"entityType": ["finding", "symptom"]}, "locations": [{"offset": first_fever, "length": len("fever")}, {"offset": second_fever, "length": len("fever")}]}],
                "relations": [
                    {"id": "has", "infons": {"type": "has"}, "nodes": [{"refid": "condition", "role": "Dependent"}, {"refid": "case", "role": "Governor"}]},
                    {"id": "modifies", "infons": {"type": "modifies"}, "nodes": [{"refid": "cough", "role": "Dependent"}, {"refid": "modifier", "role": "Governor"}]},
                    {"id": "causes", "infons": {"type": "causes"}, "nodes": [{"refid": "discontinuous", "role": "Dependent"}, {"refid": "condition", "role": "Governor"}]}],
            }],
        }]
    }


def test_official_bioc_shape_preserves_discontinuous_nested_and_cross_sentence_annotations(tmp_path):
    collection = _official_shape_collection()
    passage = collection["documents"][0]["passages"][0]
    numeric_ids = {annotation["id"]: index for index, annotation in enumerate(passage["annotations"], start=1)}
    for annotation in passage["annotations"]:
        annotation["id"] = numeric_ids[annotation["id"]]
    for index, relation in enumerate(passage["relations"], start=1):
        relation["id"] = index
        for node in relation["nodes"]:
            node["refid"] = numeric_ids[node["refid"]]
    artifact = tmp_path / "CaseReportCorpus.pickle"
    artifact.write_bytes(pickle.dumps(collection))
    output = tmp_path / "mcrc.jsonl"
    convert_mcrc_distribution(
        artifact,
        output,
        tmp_path / "manifest.json",
        _source_for(artifact),
        allow_unsafe_deserialization=True,
    )

    [document] = load_jsonl(output)
    nested, discontinuous = document.mentions[2], document.mentions[5]
    assert nested.spans[0].to_dict() == discontinuous.spans[0].to_dict()
    assert [span.to_dict() for span in discontinuous.spans] == [[19, 24], [42, 47]]
    assert discontinuous.native_types == ("finding", "symptom")
    causes = next(relation for relation in document.relations if relation.canonical_label is CanonicalRelation.CAUSES)
    assert causes.canonical_label is CanonicalRelation.CAUSES
    assert causes.source_id == document.mentions[1].mention_id
    assert causes.target_id == document.mentions[5].mention_id


def test_ambiguous_relation_roles_are_rejected_instead_of_guessing_direction(tmp_path):
    collection = _official_shape_collection()
    collection["documents"][0]["passages"][0]["relations"][0]["nodes"][0]["role"] = "unknown"
    artifact = tmp_path / "CaseReportCorpus.pickle"
    artifact.write_bytes(pickle.dumps(collection))

    with pytest.raises(MCRCConversionError, match="must declare one source and one target role"):
        convert_mcrc_distribution(
            artifact,
            tmp_path / "mcrc.jsonl",
            tmp_path / "manifest.json",
            _source_for(artifact),
            allow_unsafe_deserialization=True,
        )


def test_source_revision_must_be_immutable():
    with pytest.raises(MCRCConversionError, match="immutable commit SHA"):
        MCRCSource(
            repository="https://github.com/adahealth/medical_case_report_corpus",
            revision="master",
            artifact_sha256="0" * 64,
        )


def test_release_bioc_classes_are_loaded_without_importing_their_original_package(tmp_path, monkeypatch):
    module_name = "src.util.bioc.bioc_collection"
    for package_name in ("src", "src.util", "src.util.bioc"):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    module = ModuleType(module_name)
    collection_type = type("BioCCollection", (), {})
    collection_type.__module__ = module_name
    setattr(module, "BioCCollection", collection_type)
    monkeypatch.setitem(sys.modules, module_name, module)
    collection = collection_type()
    collection.documents = _official_shape_collection()["documents"]
    artifact = tmp_path / "CaseReportCorpus.pickle"
    artifact.write_bytes(pickle.dumps(collection))
    monkeypatch.delitem(sys.modules, module_name)

    output = tmp_path / "mcrc.jsonl"
    convert_mcrc_distribution(
        artifact,
        output,
        tmp_path / "manifest.json",
        _source_for(artifact),
        allow_unsafe_deserialization=True,
    )

    assert load_jsonl(output)[0].document_id == "mcrc:official-shape"


def test_verified_conversion_preserves_complex_annotations_and_writes_audit_manifest(tmp_path):
    artifact = _pickle_collection(tmp_path)
    jsonl = tmp_path / "mcrc.jsonl"
    manifest = tmp_path / "mcrc.manifest.json"

    result = convert_mcrc_distribution(
        artifact,
        jsonl,
        manifest,
        _source_for(artifact),
        allow_unsafe_deserialization=True,
    )

    documents = load_jsonl(jsonl)
    assert [document.document_id for document in documents] == ["mcrc:doc-a", "mcrc:doc-z"]
    document = documents[1]
    assert document.mentions[3].native_types == ("finding", "symptom")
    assert document.mentions[3].spans[0].to_dict() == [22, 27]
    assert {relation.canonical_label for relation in document.relations} == {
        CanonicalRelation.DESCRIBES,
        CanonicalRelation.HAS_CONDITION,
        CanonicalRelation.NEGATES,
    }
    assert result["conversion_result"] == "success"
    assert result["counts"] == {"documents": 2, "mentions": 4, "relations": 4}
    assert "Patient had fever" not in manifest.read_text(encoding="utf-8")


def test_hash_mismatch_stops_before_pickle_is_opened(tmp_path, monkeypatch):
    artifact = _pickle_collection(tmp_path)
    source = MCRCSource(
        repository="https://github.com/adahealth/medical_case_report_corpus",
        revision="9b4b06f",
        artifact_sha256="0" * 64,
    )
    opened = False

    def unexpected_load(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("pickle deserialization must not run after a failed integrity check")

    monkeypatch.setattr("project2_from_project1.src.mcrc_converter._MCRCPickleUnpickler.load", unexpected_load)

    with pytest.raises(MCRCConversionError, match="SHA-256 mismatch"):
        convert_mcrc_distribution(
            artifact,
            tmp_path / "mcrc.jsonl",
            tmp_path / "manifest.json",
            source,
            allow_unsafe_deserialization=True,
        )

    assert not opened


def test_unsafe_deserialization_requires_an_explicit_opt_in(tmp_path):
    artifact = _pickle_collection(tmp_path)

    with pytest.raises(MCRCConversionError, match="allow_unsafe_deserialization=True"):
        convert_mcrc_distribution(
            artifact,
            tmp_path / "mcrc.jsonl",
            tmp_path / "manifest.json",
            _source_for(artifact),
        )


def test_equivalent_conversions_are_byte_for_byte_deterministic(tmp_path):
    artifact = _pickle_collection(tmp_path)
    source = _source_for(artifact)
    paths = []
    for run in ("first", "second"):
        jsonl = tmp_path / f"{run}.jsonl"
        manifest = tmp_path / f"{run}.manifest.json"
        convert_mcrc_distribution(artifact, jsonl, manifest, source, allow_unsafe_deserialization=True)
        paths.append((jsonl, manifest))

    assert paths[0][0].read_bytes() == paths[1][0].read_bytes()
    assert paths[0][1].read_bytes() == paths[1][1].read_bytes()
