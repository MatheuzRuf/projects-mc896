from src.Regex.measurement_adapter import extract_structured_measurements
from src.extraction.entities import Entity


def test_extracts_values_reference_range_and_case_insensitive_interpretation():
    text = "Lipase was Elevated at 850 U/L (reference range 10-140 U/L)."
    lipase = Entity(label="lipase", type="Exam", positions=[(0, 6)])

    results = extract_structured_measurements(text, [lipase])

    value = next(result for result in results if result.text == "850 U/L")
    reference_range = next(
        result
        for result in results
        if result.reference_range is not None
    )
    interpretation = next(
        result
        for result in results
        if result.interpretation == "elevated"
    )

    assert value.value == "850"
    assert value.unit == "U/L"
    assert value.entity == lipase
    assert reference_range.reference_range.low == "10"
    assert reference_range.reference_range.high == "140"
    assert reference_range.unit == "U/L"
    assert interpretation.entity == lipase


def test_extracts_decimal_size_and_duration():
    text = "CRP was 12.5 mg/L. The lesion measured 4 cm. Symptoms lasted 5 days."

    results = extract_structured_measurements(text, [])
    values = {
        (result.value, result.unit)
        for result in results
        if result.value is not None
    }

    assert ("12.5", "mg/L") in values
    assert ("4", "cm") in values
    assert ("5", "days") in values


def test_does_not_associate_entity_from_another_sentence():
    text = "Lipase was requested. The result was 850 U/L."
    lipase = Entity(label="lipase", type="Exam", positions=[(0, 6)])

    results = extract_structured_measurements(text, [lipase])
    value = next(result for result in results if result.text == "850 U/L")

    assert value.entity is None


def test_text_without_measurement_returns_empty_list():
    assert extract_structured_measurements("The patient recovered.", []) == []
