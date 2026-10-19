import json

import pytest

from project2_from_project1.scripts.assemble_gold_adjudication import assemble_adjudication
from project2_from_project1.scripts.model_adjudication import check, collect, prepare
from project2_from_project1.src.clinical_documents import ClinicalDocument, Mention, Span, write_jsonl
from project2_from_project1.src.gold_projection import AdjudicationError, load_adjudication_jsonl


def _corpus(tmp_path):
    text = "She reported severe cough. No fever."
    document = ClinicalDocument("doc-1", text, (), (), (
        Mention("long", (Span(13, 25),), ("finding",), "Finding"),
        Mention("short", (Span(20, 25),), ("finding",), "Finding"),
        Mention("fever", (Span(30, 35),), ("finding",), "Finding"),
    ), ())
    path = tmp_path / "mcrc.jsonl"
    write_jsonl([document], path)
    return path


def _review(path, case_id, selected, justification):
    path.write_text(json.dumps({"case_id": case_id, "selected": selected, "justification": justification}) + "\n",
                    encoding="utf-8")


def test_prepare_marks_candidates_and_collect_maps_aliases_to_bound_ids(tmp_path):
    source = _corpus(tmp_path)
    packet_path = tmp_path / "packet.json"
    packet = prepare(source, packet_path)
    [case] = packet["cases"]
    assert case["context"] == "She reported [c1:severe [c2:cough:c2]:c1]. No fever."
    assert [item["type"] for item in case["candidates"]] == ["Finding", "Finding"]

    leaky = tmp_path / "leaky.jsonl"
    _review(leaky, case["case_id"], ["c1"], "keeps severe cough as the complete concept")
    assert any("clinical text" in error for error in check(packet_path, leaky))

    for name, selected in (("a", ["c1"]), ("b", ["c2"]), ("c", ["c1"])):
        alias = tmp_path / f"{name}.alias.jsonl"
        _review(alias, case["case_id"], selected, "the selected candidate is the complete concept per rule four")
        assert check(packet_path, alias) == []
        collect(packet_path, alias, f"model-review-{name}", tmp_path / f"{name}.jsonl")
    collected = json.loads((tmp_path / "a.jsonl").read_text(encoding="utf-8"))
    assert collected["selected_candidate_ids"] == [case["aliases"]["c1"]]

    output = tmp_path / "adjudication.jsonl"
    assemble_adjudication(source, tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl", output)
    [record] = load_adjudication_jsonl(output)
    assert record.resolution == "third_vote"
    assert record.selected_candidate_ids == (case["aliases"]["c1"],)
    assert all(vote.reviewer_kind == "model" for vote in record.votes)


def test_collect_refuses_incomplete_or_unknown_reviews(tmp_path):
    source = _corpus(tmp_path)
    packet_path = tmp_path / "packet.json"
    prepare(source, packet_path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(AdjudicationError, match="missing"):
        collect(packet_path, empty, "model-review-a", tmp_path / "out.jsonl")
    unknown = tmp_path / "unknown.jsonl"
    _review(unknown, "adjudication:other", [], "no candidate is a valid entity in this context")
    assert any("unknown case_id" in error for error in check(packet_path, unknown))
