from project2_from_project1.scripts.validate_external_review import _aligned_mention
from project2_from_project1.src.clinical_documents import Span
from project2_from_project1.src.gold_projection import segment_sentences, tokenize_with_offsets


def _check(text: str) -> bool:
    tokens = tokenize_with_offsets(text)
    return _aligned_mention(text, Span(0, len(text)), tokens, segment_sentences(text, tokens))


def test_hgvs_dots_within_one_clinical_expression_are_allowed():
    assert _check("homozygotic for c.712C>T/p.H238Y in MC2R")


def test_real_sentence_boundary_within_mention_is_rejected():
    assert not _check("fever. cough")
