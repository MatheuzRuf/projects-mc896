"""Linear semi-Markov CRF: zero init, full-batch L-BFGS, L2, CPU float64.

Objective (minimized)::

    sum_sentences [log Z(x) - score(x, gold segmentation)] + (l2 / 2) * ||theta||^2

where ``theta`` holds segment-feature weights, label transitions and start/end
parameters.  A finite run that exhausts ``max_iterations`` is returned with
``converged=False``; a non-finite objective or gradient raises
:class:`NonFiniteTrainingError`.  There is no optimizer or dictionary fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from project2_from_project1.src.clinical_documents import Span
from project2_from_project1.src.entity_recognition.corpus import EntityRecognitionError, GoldEntityDocument, RawDocument
from project2_from_project1.src.entity_recognition.crf_features import (
    FEATURE_TEMPLATE_VERSION,
    EncodedBatch,
    FeatureVocabulary,
    encode_batches,
)
from project2_from_project1.src.entity_recognition.semicrf import (
    NEG,
    OUTSIDE,
    Segment,
    log_partition,
    path_score,
    segment_marginals,
    viterbi,
)
from project2_from_project1.src.gold_projection import MAX_MENTION_TOKENS
from project2_from_project1.src.supervised_contracts import ENTITY_TYPES, EntityPrediction

METHOD = "semi-markov-crf"
MODEL_VERSION = "semi-markov-crf-v1"
LABELS = ("O", *sorted(ENTITY_TYPES))
DTYPE = torch.float64


class NonFiniteTrainingError(EntityRecognitionError):
    """The objective or its gradient became NaN or infinite; the run is invalid."""

    def __init__(self, message: str, diagnostic: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostic = dict(diagnostic)


@dataclass(frozen=True)
class TrainingConfig:
    l2: float
    max_iterations: int = 250
    tolerance_grad: float = 1e-5
    tolerance_change: float = 1e-9
    history_size: int = 10
    max_segment: int = MAX_MENTION_TOKENS

    def __post_init__(self) -> None:
        if isinstance(self.l2, bool) or not isinstance(self.l2, (int, float)) or not math.isfinite(self.l2) or self.l2 < 0:
            raise EntityRecognitionError("l2 must be a finite non-negative number")
        for name in ("max_iterations", "history_size", "max_segment"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise EntityRecognitionError(f"{name} must be a positive integer")
        if self.max_segment > MAX_MENTION_TOKENS:
            raise EntityRecognitionError(f"max_segment cannot exceed {MAX_MENTION_TOKENS}")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "initialization": "zeros", "line_search": "strong_wolfe",
                "optimizer": "torch.optim.LBFGS full-batch", "regularized_parameters": "all"}


@dataclass(frozen=True)
class TrainingDiagnostics:
    converged: bool
    stop_reason: str
    iterations: int
    function_evaluations: int
    final_objective: float
    final_gradient_max_abs: float
    duration_seconds: float
    training_sentences: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gold_paths(document: GoldEntityDocument, max_segment: int) -> dict[int, list[Segment]]:
    by_sentence: dict[int, list] = {}
    for mention in document.mentions:
        size = mention.token_end - mention.token_start
        if size > max_segment:
            raise EntityRecognitionError(
                f"gold mention {mention.mention_id} spans {size} tokens; the limit is {max_segment} and "
                "mentions are never truncated"
            )
        by_sentence.setdefault(mention.sentence_index, []).append(mention)
    paths = {}
    for sentence in document.sentences:
        mentions = sorted(by_sentence.get(sentence.sentence_index, ()), key=lambda item: item.token_start)
        path, position = [], 0
        for mention in mentions:
            path.extend(Segment(token, 1, OUTSIDE) for token in range(position, mention.token_start))
            path.append(Segment(mention.token_start, mention.token_end - mention.token_start,
                                LABELS.index(mention.canonical_type)))
            position = mention.token_end
        path.extend(Segment(token, 1, OUTSIDE) for token in range(position, len(sentence.tokens)))
        paths[sentence.sentence_index] = path
    return paths


class SemiMarkovCRF:
    def __init__(self, vocabulary: FeatureVocabulary, max_segment: int, weights: torch.Tensor | None = None,
                 transitions: torch.Tensor | None = None, start: torch.Tensor | None = None,
                 end: torch.Tensor | None = None) -> None:
        self.vocabulary = vocabulary
        self.max_segment = max_segment
        labels = len(LABELS)
        self.weights = weights if weights is not None else torch.zeros((len(vocabulary), labels), dtype=DTYPE)
        self.transitions = transitions if transitions is not None else torch.zeros((labels, labels), dtype=DTYPE)
        self.start = start if start is not None else torch.zeros(labels, dtype=DTYPE)
        self.end = end if end is not None else torch.zeros(labels, dtype=DTYPE)

    @property
    def parameters(self) -> tuple[torch.Tensor, ...]:
        return self.weights, self.transitions, self.start, self.end

    def _row(self, index: int) -> torch.Tensor:
        return self.weights[index] if index >= 0 else torch.zeros(len(LABELS), dtype=DTYPE)

    def segment_scores(self, batch: EncodedBatch) -> torch.Tensor:
        """Dense ``S[b, i, d-1, y]`` assembled from per-token role scores.

        Entries outside a sentence are left for the dynamic program to mask.
        """
        size, tokens, segment, labels = len(batch.lengths), batch.max_tokens, batch.max_segment, len(LABELS)
        role = {
            name: F.embedding_bag(ids, self.weights, offsets, mode="sum").view(size, tokens, labels)
            for name, (ids, offsets) in batch.bags.items()
        }
        starts = torch.arange(tokens).view(tokens, 1)
        stops = (starts + torch.arange(1, segment + 1).view(1, segment)).clamp(max=tokens)
        lasts = stops - 1
        zero = torch.zeros((size, 1, labels), dtype=DTYPE)
        inner = torch.cat((zero, role["in"].cumsum(1)), dim=1)
        constants = self.vocabulary.segment_constant_ids(self.max_segment)
        length = torch.stack([self._row(index) for index in constants["length"][:segment]])
        scores = (
            (role["first"] + role["prev"])[:, starts]
            + (role["last"] + role["next"])[:, lasts]
            + inner[:, stops] - inner[:, starts]
            + length.view(1, 1, segment, labels)
            + self._row(constants["bias"])
        )
        digits = torch.cat((torch.zeros((size, 1)), batch.digit.cumsum(1)), dim=1)
        puncts = torch.cat((torch.zeros((size, 1)), batch.punct.cumsum(1)), dim=1)
        widths = (stops - starts).view(1, tokens, segment)
        has_digit = (digits[:, stops] - digits[:, starts]) > 0
        all_punct = (puncts[:, stops] - puncts[:, starts]) == widths
        scores = scores + has_digit.unsqueeze(-1).to(DTYPE) * self._row(constants["has_digit"])
        return scores + all_punct.unsqueeze(-1).to(DTYPE) * self._row(constants["all_punct"])

    # -- training ----------------------------------------------------------
    @classmethod
    def fit(cls, documents: Sequence[GoldEntityDocument], config: TrainingConfig) -> tuple["SemiMarkovCRF", TrainingDiagnostics]:
        began = time.perf_counter()
        documents = sorted(documents, key=lambda item: item.document_id)
        sentences = [(document.document_id, document.text, sentence) for document in documents for sentence in document.sentences]
        if not sentences:
            raise EntityRecognitionError("training partition contains no sentences")
        paths = {}
        for document in documents:
            for index, path in _gold_paths(document, config.max_segment).items():
                paths[(document.document_id, index)] = path
        vocabulary = FeatureVocabulary.fit(((text, sentence) for _, text, sentence in sentences), config.max_segment)
        model = cls(vocabulary, config.max_segment)
        batches = [(batch, [paths[key] for key in batch.keys]) for batch in encode_batches(vocabulary, sentences, config.max_segment)]
        parameters = [parameter.requires_grad_(True) for parameter in model.parameters]
        optimizer = torch.optim.LBFGS(
            parameters, lr=1.0, max_iter=config.max_iterations, max_eval=config.max_iterations * 25,
            tolerance_grad=config.tolerance_grad, tolerance_change=config.tolerance_change,
            history_size=config.history_size, line_search_fn="strong_wolfe",
        )
        trace = {"evaluations": 0, "objective": math.nan, "gradient": math.nan}

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            total = 0.0
            for batch, gold in batches:
                scores = model.segment_scores(batch)
                arguments = (scores, batch.lengths, model.transitions, model.start, model.end)
                loss = (log_partition(*arguments) - path_score(*arguments, gold)).sum()
                loss.backward()
                total += float(loss.detach())
            penalty = 0.5 * config.l2 * sum((parameter * parameter).sum() for parameter in parameters)
            penalty.backward()
            value = total + float(penalty.detach())
            gradient = max(float(parameter.grad.abs().max()) for parameter in parameters)
            trace.update(evaluations=trace["evaluations"] + 1, objective=value, gradient=gradient)
            if not math.isfinite(value) or not math.isfinite(gradient):
                raise NonFiniteTrainingError("semi-Markov CRF objective or gradient is not finite", {
                    "function_evaluations": trace["evaluations"],
                    "objective_finite": math.isfinite(value),
                    "gradient_finite": math.isfinite(gradient),
                })
            return torch.tensor(value, dtype=DTYPE)

        deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            optimizer.step(closure)
        finally:
            torch.use_deterministic_algorithms(deterministic)
            for parameter in parameters:
                parameter.requires_grad_(False)
                parameter.grad = None
        state = optimizer.state[optimizer.param_groups[0]["params"][0]]
        iterations = int(state.get("n_iter", 0))
        evaluations = int(state.get("func_evals", trace["evaluations"]))
        if trace["gradient"] <= config.tolerance_grad:
            reason = "gradient_tolerance"
        elif evaluations >= config.max_iterations * 25:
            reason = "max_function_evaluations"
        elif iterations >= config.max_iterations:
            reason = "max_iterations"
        else:
            reason = "change_tolerance"
        diagnostics = TrainingDiagnostics(
            reason in ("gradient_tolerance", "change_tolerance"), reason, iterations, evaluations,
            trace["objective"], trace["gradient"], round(time.perf_counter() - began, 6), len(sentences),
        )
        return model, diagnostics

    # -- inference ---------------------------------------------------------
    def predict(self, documents: Iterable[RawDocument]) -> tuple[EntityPrediction, ...]:
        """Viterbi segmentation per sentence; each entity carries its uncalibrated marginal."""
        lookup = {}
        sentences = []
        for document in documents:
            if not isinstance(document, RawDocument):
                raise TypeError("the semi-Markov CRF predicts RawDocument instances only")
            for sentence in document.sentences:
                lookup[(document.document_id, sentence.sentence_index)] = sentence
                sentences.append((document.document_id, document.text, sentence))
        predictions = []
        with torch.no_grad():
            for batch in encode_batches(self.vocabulary, sentences, self.max_segment):
                scores = self.segment_scores(batch)
                arguments = (scores, batch.lengths, self.transitions, self.start, self.end)
                paths, _ = viterbi(*arguments)
                marginals = segment_marginals(*arguments)
                for b, ((document_id, sentence_index), path) in enumerate(zip(batch.keys, paths)):
                    tokens = lookup[(document_id, sentence_index)].tokens
                    for segment in path:
                        if segment.label == OUTSIDE:
                            continue
                        predictions.append(EntityPrediction(
                            document_id,
                            Span(tokens[segment.start].span.start, tokens[segment.start + segment.size - 1].span.end),
                            LABELS[segment.label], sentence_index, METHOD, MODEL_VERSION,
                            float(marginals[b, segment.start, segment.size - 1, segment.label]),
                        ))
        return tuple(sorted(predictions, key=lambda item: (item.document_id, item.span.start, item.span.end)))

    # -- serialization (contains training token strings: keep private) -----
    def to_dict(self) -> dict[str, Any]:
        return {
            "end": self.end.tolist(),
            "features": list(self.vocabulary.features),
            "labels": list(LABELS),
            "lexical_tokens": sorted(self.vocabulary.lexical_tokens),
            "max_segment": self.max_segment,
            "prefixes": sorted(self.vocabulary.prefixes),
            "schema_version": 1,
            "start": self.start.tolist(),
            "suffixes": sorted(self.vocabulary.suffixes),
            "template_version": FEATURE_TEMPLATE_VERSION,
            "transitions": self.transitions.tolist(),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemiMarkovCRF":
        if value.get("schema_version") != 1 or value.get("template_version") != FEATURE_TEMPLATE_VERSION:
            raise EntityRecognitionError("unsupported semi-Markov CRF model version")
        if tuple(value["labels"]) != LABELS:
            raise EntityRecognitionError("model labels do not match the canonical label order")
        vocabulary = FeatureVocabulary(value["features"], value["lexical_tokens"], value["prefixes"], value["suffixes"])
        tensor = lambda name: torch.tensor(value[name], dtype=DTYPE)  # noqa: E731
        weights = tensor("weights").reshape(len(vocabulary), len(LABELS))
        return cls(vocabulary, value["max_segment"], weights, tensor("transitions"), tensor("start"), tensor("end"))


__all__ = ["LABELS", "METHOD", "MODEL_VERSION", "NonFiniteTrainingError", "SemiMarkovCRF", "TrainingConfig", "TrainingDiagnostics"]
