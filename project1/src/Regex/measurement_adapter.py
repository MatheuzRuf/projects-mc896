"""Saída estruturada para as expressões regulares de medidas existentes."""

import re
from dataclasses import dataclass

from src.Regex.regex import padraoFinal, regex_interpretacoes, valor
from src.extraction.entities import Entity


NUMBER_PATTERN = r"\d+(?:[.,]\d+)?"
REFERENCE_RANGE_PATTERN = re.compile(
    rf"^({NUMBER_PATTERN})\s*-\s*({NUMBER_PATTERN})$"
)


@dataclass(frozen=True)
class ReferenceRange:
    low: str
    high: str


@dataclass(frozen=True)
class StructuredMeasurement:
    text: str
    value: str | None
    unit: str | None
    reference_range: ReferenceRange | None
    interpretation: str | None
    entity: Entity | None
    start: int
    end: int


def _sentence_bounds(text: str, position: int) -> tuple[int, int]:
    sentence_start = max(
        text.rfind(".", 0, position),
        text.rfind("!", 0, position),
        text.rfind("?", 0, position),
        text.rfind("\n", 0, position),
    ) + 1

    endings = [
        ending
        for separator in ".!?\n"
        if (ending := text.find(separator, position)) != -1
    ]
    sentence_end = min(endings) if endings else len(text)

    return sentence_start, sentence_end


def _nearest_entity(
    text: str,
    start: int,
    end: int,
    entities: list[Entity],
    max_distance: int,
) -> Entity | None:
    sentence_start, sentence_end = _sentence_bounds(text, start)
    match_center = (start + end) / 2
    nearest: Entity | None = None
    nearest_distance = float("inf")

    for entity in entities:
        if entity.type not in {"Exam", "Finding"}:
            continue

        for entity_start, entity_end in entity.positions:
            entity_center = (entity_start + entity_end) / 2
            if not sentence_start <= entity_center <= sentence_end:
                continue

            distance = abs(match_center - entity_center)
            if distance <= max_distance and distance < nearest_distance:
                nearest = entity
                nearest_distance = distance

    return nearest


def _parse_value(raw_value: str) -> tuple[str | None, ReferenceRange | None]:
    range_match = REFERENCE_RANGE_PATTERN.fullmatch(raw_value)
    if range_match is None:
        return raw_value, None

    return None, ReferenceRange(
        low=range_match.group(1),
        high=range_match.group(2),
    )


def extract_structured_measurements(
    text: str,
    entities: list[Entity],
    max_entity_distance: int = 120,
) -> list[StructuredMeasurement]:
    """
    Adapta os regex existentes para uma saída nomeada e contextualizada.

    A associação é limitada ao Exam ou Finding mais próximo que esteja na
    mesma sentença e dentro de ``max_entity_distance`` caracteres.
    """

    if max_entity_distance < 0:
        raise ValueError("max_entity_distance must be non-negative")

    results: list[StructuredMeasurement] = []

    for match in re.finditer(padraoFinal, text):
        number_match = re.search(valor, match.group())
        if number_match is None:
            continue

        raw_value = number_match.group()
        value, reference_range = _parse_value(raw_value)
        unit = match.group()[number_match.end():].strip() or None

        results.append(
            StructuredMeasurement(
                text=match.group(),
                value=value,
                unit=unit,
                reference_range=reference_range,
                interpretation=None,
                entity=_nearest_entity(
                    text,
                    match.start(),
                    match.end(),
                    entities,
                    max_entity_distance,
                ),
                start=match.start(),
                end=match.end(),
            )
        )

    for match in re.finditer(regex_interpretacoes, text, re.IGNORECASE):
        results.append(
            StructuredMeasurement(
                text=match.group(),
                value=None,
                unit=None,
                reference_range=None,
                interpretation=match.group().lower(),
                entity=_nearest_entity(
                    text,
                    match.start(),
                    match.end(),
                    entities,
                    max_entity_distance,
                ),
                start=match.start(),
                end=match.end(),
            )
        )

    return sorted(results, key=lambda result: (result.start, result.end))


__all__ = [
    "ReferenceRange",
    "StructuredMeasurement",
    "extract_structured_measurements",
]
