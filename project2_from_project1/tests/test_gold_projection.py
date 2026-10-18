import hashlib
import json

import pytest

from project2_from_project1.scripts.export_gold_adjudication_packet import export_review_packet
from project2_from_project1.src.clinical_documents import ClinicalDocument, Mention, Span, write_jsonl
from project2_from_project1.src.gold_projection import (
    AdjudicationError,
    AdjudicationRecord,
    ReviewerVote,
    build_adjudication_binding,
    build_projection_plan,
    load_adjudication_jsonl,
    project_gold_from_files,
    segment_sentences,
    tokenize_with_offsets,
    write_adjudication_jsonl,
    write_adjudication_review_packet,
)


def _document(text, mentions):
    return ClinicalDocument("doc-1", text, (), (), mentions, ())


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tokenization_and_sentence_boundaries_preserve_original_offsets():
    text = "H. influenzae was treated with 5 mg. twice daily. Fever resolved."

    tokens = tokenize_with_offsets(text)
    sentences = segment_sentences(text, tokens)

    assert [(token.span.to_dict(), text[token.span.start:token.span.end]) for token in tokens[:4]] == [
        ([0, 1], "H"),
        ([1, 2], "."),
        ([3, 13], "influenzae"),
        ([14, 17], "was"),
    ]
    assert [sentence.span.to_dict() for sentence in sentences] == [[0, 49], [50, 65]]
    assert [token.span.to_dict() for token in sentences[0].tokens[-4:]] == [
        [35, 36], [37, 42], [43, 48], [48, 49],
    ]
    new_sentence = "Dose was 5 mg. Patient improved."
    assert [item.span.to_dict() for item in segment_sentences(new_sentence)] == [[0, 14], [15, 32]]
    compound = "Dose 5 mg/day post-op."
    compound_tokens = tokenize_with_offsets(compound)
    assert [compound[item.span.start:item.span.end] for item in compound_tokens] == [
        "Dose", "5", "mg", "/", "day", "post", "-", "op", ".",
    ]
    mixed = "5.0 B12"
    assert [mixed[item.span.start:item.span.end] for item in tokenize_with_offsets(mixed)] == [
        "5", ".", "0", "B", "12",
    ]
    punctuation_run = "Wait... Next?! Done."
    assert [item.span.to_dict() for item in segment_sentences(punctuation_run)] == [
        [0, 7], [8, 14], [15, 20],
    ]
    initials = "J. R. Smith improved. A. Jones recovered. Grade I. Next."
    assert [initials[item.span.start:item.span.end] for item in segment_sentences(initials)] == [
        "J. R. Smith improved.", "A. Jones recovered.", "Grade I.", "Next.",
    ]
    abbreviations = "Given p.o. twice daily. Administer i.v. antibiotics."
    assert [
        abbreviations[item.span.start:item.span.end]
        for item in segment_sentences(abbreviations)
    ] == [
        "Given p.o. twice daily.", "Administer i.v. antibiotics.",
    ]
    mid_sentence_names = "Patient J. Smith improved. Stable. Seen by A. Jones today. Stable."
    assert [
        mid_sentence_names[item.span.start:item.span.end]
        for item in segment_sentences(mid_sentence_names)
    ] == [
        "Patient J. Smith improved.", "Stable.", "Seen by A. Jones today.", "Stable.",
    ]


def test_projection_collapses_duplicates_and_requires_review_for_trimmed_boundaries():
    document = _document(
        "  Asthma causes cough.",
        (
            Mention("m1", (Span(2, 8),), ("condition",), "Condition"),
            Mention("m2", (Span(2, 8),), ("condition",), "Condition"),
            Mention("m3", (Span(0, 8),), ("condition",), "Condition"),
            Mention("m4", (Span(16, 21),), ("finding",), "Finding"),
        ),
    )
    plan = build_projection_plan((document,))

    assert len(plan.candidates) == 3
    duplicate = next(
        candidate for candidate in plan.candidates
        if candidate.original_spans == (Span(2, 8),)
    )
    assert duplicate.source_mention_ids == ("m1", "m2")
    [case] = plan.cases
    assert set(case.reasons) == {"boundary_trim", "overlap"}

    with pytest.raises(AdjudicationError, match="missing adjudication"):
        plan.project(())

    selected = duplicate.candidate_id
    record = AdjudicationRecord(
        case.case_id,
        (
            ReviewerVote("reviewer-a", (selected,), "prefer exact token boundaries"),
            ReviewerVote("reviewer-b", (selected,), "prefer exact token boundaries"),
        ),
        (selected,),
        "consensus",
        "both independent reviews selected the exact aligned annotation",
        ("reviewer-a", "reviewer-b"),
    )
    [projection] = plan.project((record,))

    assert [(mention.span.to_dict(), mention.canonical_type) for mention in projection.mentions] == [
        ([2, 8], "Condition"),
        ([16, 21], "Finding"),
    ]
    assert projection.mentions[0].source_mention_ids == ("m1", "m2")


def test_disagreement_requires_a_third_vote_matching_the_recorded_outcome():
    document = _document(
        "severe cough",
        (
            Mention("long", (Span(0, 12),), ("finding",), "Finding"),
            Mention("short", (Span(7, 12),), ("finding",), "Finding"),
        ),
    )
    plan = build_projection_plan((document,))
    [case] = plan.cases
    candidates = {candidate.span: candidate.candidate_id for candidate in plan.candidates}
    long_id, short_id = candidates[Span(0, 12)], candidates[Span(7, 12)]

    incomplete = AdjudicationRecord(
        case.case_id,
        (
            ReviewerVote("reviewer-a", (long_id,), "complete clinical phrase"),
            ReviewerVote("reviewer-b", (short_id,), "minimal clinical head"),
        ),
        (long_id,),
        "third_vote",
        "the longer annotation preserves the source concept",
    )
    with pytest.raises(AdjudicationError, match="third vote"):
        plan.project((incomplete,))

    resolved = AdjudicationRecord(
        case.case_id,
        incomplete.votes + (ReviewerVote("reviewer-c", (long_id,), "source concept includes severity"),),
        (long_id,),
        "third_vote",
        incomplete.justification,
    )
    [projection] = plan.project((resolved,))
    assert [mention.span for mention in projection.mentions] == [Span(0, 12)]

    independent_third = AdjudicationRecord(
        case.case_id,
        incomplete.votes + (ReviewerVote("reviewer-c", (), "neither boundary is reliable"),),
        (),
        "third_vote",
        "the independent third review rejected both admissible alternatives",
    )
    [projection] = plan.project((independent_third,))
    assert projection.mentions == ()

    reconciled = AdjudicationRecord(
        case.case_id,
        incomplete.votes,
        (short_id,),
        "consensus",
        "the initial reviewers reconciled their decisions after discussion",
        ("reviewer-a", "reviewer-b"),
    )
    [projection] = plan.project((reconciled,))
    assert [mention.span for mention in projection.mentions] == [Span(7, 12)]

    too_many = AdjudicationRecord(
        case.case_id,
        resolved.votes + (ReviewerVote("reviewer-d", (short_id,), "late dissent"),),
        (long_id,),
        "third_vote",
        resolved.justification,
    )
    with pytest.raises(AdjudicationError, match="exactly three"):
        plan.project((too_many,))

    unnecessary_third = AdjudicationRecord(
        case.case_id,
        (
            ReviewerVote("reviewer-a", (long_id,), "complete phrase"),
            ReviewerVote("reviewer-b", (long_id,), "complete phrase"),
            ReviewerVote("reviewer-c", (short_id,), "late dissent"),
        ),
        (short_id,),
        "third_vote",
        "a third vote is invalid when the initial reviews agree",
    )
    with pytest.raises(AdjudicationError, match="initial reviews to disagree"):
        plan.project((unnecessary_third,))


def test_same_span_with_different_types_is_recorded_as_type_conflict():
    document = _document(
        "volvulus",
        (
            Mention("condition", (Span(0, 8),), ("condition",), "Condition"),
            Mention("finding", (Span(0, 8),), ("finding",), "Finding"),
        ),
    )
    [case] = build_projection_plan((document,)).cases
    assert case.reasons == ("overlap", "type_conflict")


def test_partial_word_boundary_is_adjusted_only_through_recorded_adjudication():
    document = _document(
        "upper respiratory infection",
        (Mention("cut", (Span(7, 27),), ("condition",), "Condition"),),
    )
    plan = build_projection_plan((document,))
    [candidate] = plan.candidates
    [case] = plan.cases

    assert candidate.span == Span(6, 27)
    assert candidate.transformations == ("expand_start_to_token_boundary",)
    assert case.reasons == ("token_boundary_adjustment",)

    record = AdjudicationRecord(
        case.case_id,
        (
            ReviewerVote("reviewer-a", (candidate.candidate_id,), "restore complete token"),
            ReviewerVote("reviewer-b", (candidate.candidate_id,), "restore complete token"),
        ),
        (candidate.candidate_id,),
        "consensus",
        "both reviews restore the complete source token",
        ("reviewer-a", "reviewer-b"),
    )
    [projection] = plan.project((record,))
    assert projection.mentions[0].span == Span(6, 27)


def test_file_projection_is_deterministic_preserves_source_and_keeps_manifest_text_free(tmp_path):
    source = tmp_path / "lossless.jsonl"
    document = _document(
        "Asthma causes cough.",
        (
            Mention("condition", (Span(0, 6),), ("condition",), "Condition"),
            Mention("finding", (Span(14, 19),), ("finding",), "Finding"),
        ),
    )
    write_jsonl((document,), source)
    before = _digest(source)
    decisions = tmp_path / "adjudication.jsonl"
    write_adjudication_jsonl((), decisions)

    outputs = []
    for run in ("first", "second"):
        projection = tmp_path / f"{run}.jsonl"
        audit = tmp_path / f"{run}.audit.jsonl"
        manifest = tmp_path / f"{run}.manifest.json"
        result = project_gold_from_files(source, decisions, projection, audit, manifest)
        outputs.append((projection, audit, manifest))
        assert result["counts"] == {
            "adjudication_cases": 0,
            "documents": 1,
            "input_mentions": 2,
            "output_mentions": 2,
            "rejected_candidates": 0,
            "sentences": 1,
            "tokens": 4,
        }

    assert _digest(source) == before
    assert outputs[0][0].read_bytes() == outputs[1][0].read_bytes()
    assert outputs[0][1].read_bytes() == outputs[1][1].read_bytes()
    assert outputs[0][2].read_bytes() == outputs[1][2].read_bytes()
    manifest_payload = json.loads(outputs[0][2].read_text(encoding="utf-8"))
    assert manifest_payload["input_sha256"] == before
    assert manifest_payload["output_sha256"] == _digest(outputs[0][0])
    assert "Asthma" not in outputs[0][2].read_text(encoding="utf-8")

    paths = [source, decisions, tmp_path / "projection", tmp_path / "audit", tmp_path / "manifest"]
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            colliding = list(paths)
            colliding[right] = colliding[left]
            with pytest.raises(AdjudicationError, match="all be different"):
                project_gold_from_files(*colliding)


@pytest.mark.parametrize(
    "payload, field",
    [
        ({"schema_version": True, "case_id": "case", "binding": {}, "votes": [], "selected_candidate_ids": [],
          "resolution": "consensus", "resolution_reviewer_ids": [], "justification": "reason"}, "schema_version"),
        ({"schema_version": 1, "case_id": "case", "binding": {}, "votes": [], "selected_candidate_ids": [],
          "resolution": "consensus", "resolution_reviewer_ids": [], "justification": "reason", "extra": 1}, "extra"),
        ({"schema_version": 1, "case_id": "case", "binding": {}, "votes": ["invalid"], "selected_candidate_ids": [],
          "resolution": "consensus", "resolution_reviewer_ids": [], "justification": "reason"}, "vote"),
    ],
)
def test_adjudication_jsonl_rejects_non_strict_records(tmp_path, payload, field):
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AdjudicationError, match=field):
        load_adjudication_jsonl(path)


def test_file_projection_rejects_adjudication_bound_to_a_different_source(tmp_path):
    source = tmp_path / "lossless.jsonl"
    document = _document(
        "severe cough",
        (
            Mention("long", (Span(0, 12),), ("finding",), "Finding"),
            Mention("short", (Span(7, 12),), ("finding",), "Finding"),
        ),
    )
    write_jsonl((document,), source)
    plan = build_projection_plan((document,))
    binding = build_adjudication_binding(plan, _digest(source))
    [case] = plan.cases
    selected = next(candidate.candidate_id for candidate in plan.candidates if candidate.span == Span(0, 12))
    record = AdjudicationRecord(
        case.case_id,
        (
            ReviewerVote("reviewer-a", (selected,), "complete phrase"),
            ReviewerVote("reviewer-b", (selected,), "complete phrase"),
        ),
        (selected,),
        "consensus",
        "the reviewers independently selected the complete phrase",
        ("reviewer-a", "reviewer-b"),
        binding,
    )
    decisions = tmp_path / "adjudication.jsonl"
    write_adjudication_jsonl((record,), decisions)

    write_jsonl((ClinicalDocument("doc-1", "severE cough", (), (), document.mentions, ()),), source)
    with pytest.raises(AdjudicationError, match="not bound"):
        project_gold_from_files(
            source,
            decisions,
            tmp_path / "projection",
            tmp_path / "audit",
            tmp_path / "manifest",
        )


def test_manifest_attributes_type_conflict_only_to_affected_rejected_candidates(tmp_path):
    source = tmp_path / "lossless.jsonl"
    document = _document(
        "pain severe cough",
        (
            Mention("condition", (Span(0, 11),), ("condition",), "Condition"),
            Mention("finding", (Span(0, 11),), ("finding",), "Finding"),
            Mention("overlap", (Span(5, 17),), ("finding",), "Finding"),
        ),
    )
    write_jsonl((document,), source)
    plan = build_projection_plan((document,))
    binding = build_adjudication_binding(plan, _digest(source))
    [case] = plan.cases
    selected = next(
        candidate.candidate_id
        for candidate in plan.candidates
        if candidate.span == Span(0, 11) and candidate.canonical_type == "Condition"
    )
    record = AdjudicationRecord(
        case.case_id,
        (
            ReviewerVote("reviewer-a", (selected,), "select the condition annotation"),
            ReviewerVote("reviewer-b", (selected,), "select the condition annotation"),
        ),
        (selected,),
        "consensus",
        "both reviewers selected the same flat annotation",
        ("reviewer-a", "reviewer-b"),
        binding,
    )
    decisions = tmp_path / "adjudication.jsonl"
    write_adjudication_jsonl((record,), decisions)
    result = project_gold_from_files(
        source,
        decisions,
        tmp_path / "projection",
        tmp_path / "audit",
        tmp_path / "manifest",
    )

    assert result["rejections"]["by_reason"] == {"overlap": 2, "type_conflict": 1}
    rejected = {item["candidate_id"]: item["reasons"] for item in result["rejections"]["candidates"]}
    overlap_id = next(
        candidate.candidate_id for candidate in plan.candidates if candidate.span == Span(5, 17)
    )
    assert rejected[overlap_id] == ["overlap"]


def test_review_packet_contains_context_for_human_adjudication_and_exact_binding(tmp_path):
    document = _document(
        "The patient had severe cough overnight.",
        (
            Mention("long", (Span(16, 28),), ("finding",), "Finding"),
            Mention("short", (Span(23, 28),), ("finding",), "Finding"),
        ),
    )
    plan = build_projection_plan((document,))
    binding = build_adjudication_binding(plan, "0" * 64)
    packet = tmp_path / "review-cases.jsonl"
    write_adjudication_review_packet(plan, binding, packet)

    payload = json.loads(packet.read_text(encoding="utf-8"))
    assert payload["binding"] == binding.to_dict()
    assert payload["context"]["text"] == document.text
    assert {candidate["projection_surface"] for candidate in payload["candidates"]} == {
        "severe cough", "cough",
    }


@pytest.mark.parametrize("source_name", ["review-cases.jsonl", "manifest.json"])
def test_review_packet_export_rejects_source_output_collisions(tmp_path, source_name):
    source = tmp_path / source_name
    write_jsonl((_document("Asthma.", ()),), source)
    before = source.read_bytes()

    with pytest.raises(AdjudicationError, match="all be different"):
        export_review_packet(source, tmp_path)

    assert source.read_bytes() == before
