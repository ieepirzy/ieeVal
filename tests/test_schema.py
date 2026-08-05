from pathlib import Path

import pytest

from ieeval.schema import (
    Dataset,
    HardNegativeProvenance,
    QueryRecord,
    SchemaValidationError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_loads_and_validates_the_readme_fixture():
    dataset = Dataset.from_json(FIXTURES_DIR / "toy_dataset.json")

    assert dataset.schema_version == 1
    assert len(dataset.queries) == 1

    query = dataset.queries[0]
    assert query.id == "ocr-lineage-fi"
    assert query.text == "Miksi OCR-tuloksen pitäisi korvata alkuperäinen artefakti?"
    assert query.language == "fi"
    assert query.query_type == "retrieval"
    assert query.namespace == "architecture"
    assert query.positive_artifact_ids == ("artifact-123",)
    assert query.hard_negative_ids == ("artifact-456",)
    assert query.expected_tags == ("ocr", "lineage")
    assert query.provenance == "manually curated"
    assert query.notes == "Tests exact intent against a topical near miss."


def test_query_record_round_trips_through_to_dict():
    dataset = Dataset.from_json(FIXTURES_DIR / "toy_dataset.json")
    query = dataset.queries[0]

    reloaded = QueryRecord.from_dict(query.to_dict())

    assert reloaded == query


def test_malformed_fixture_missing_positive_artifact_ids_is_rejected():
    with pytest.raises(SchemaValidationError, match="positive_artifact_ids"):
        Dataset.from_json(FIXTURES_DIR / "toy_dataset_malformed_missing_positives.json")


def test_missing_required_field_error_names_the_field():
    payload = {
        "text": "some text",
        "language": "en",
        "query_type": "retrieval",
        "namespace": "ns",
        "positive_artifact_ids": ["a-1"],
    }

    with pytest.raises(SchemaValidationError, match=r"missing required field\(s\): id"):
        QueryRecord.from_dict(payload)


def test_requires_at_least_one_positive_artifact_id():
    payload = {
        "id": "q-1",
        "text": "some text",
        "language": "en",
        "query_type": "retrieval",
        "namespace": "ns",
        "positive_artifact_ids": [],
    }

    with pytest.raises(SchemaValidationError, match="at least one positive_artifact_id"):
        QueryRecord.from_dict(payload)


@pytest.mark.parametrize("language", ["", "english", "E", "fi_FI", "123"])
def test_implausible_language_codes_are_rejected(language):
    payload = {
        "id": "q-1",
        "text": "some text",
        "language": language,
        "query_type": "retrieval",
        "namespace": "ns",
        "positive_artifact_ids": ["a-1"],
    }

    with pytest.raises(SchemaValidationError, match="language"):
        QueryRecord.from_dict(payload)


@pytest.mark.parametrize("language", ["fi", "en", "sv", "en-US", "zh-Hans"])
def test_plausible_language_codes_are_accepted(language):
    payload = {
        "id": "q-1",
        "text": "some text",
        "language": language,
        "query_type": "retrieval",
        "namespace": "ns",
        "positive_artifact_ids": ["a-1"],
    }

    record = QueryRecord.from_dict(payload)
    assert record.language == language


def test_unknown_field_is_rejected():
    payload = {
        "id": "q-1",
        "text": "some text",
        "language": "en",
        "query_type": "retrieval",
        "namespace": "ns",
        "positive_artifact_ids": ["a-1"],
        "totally_unexpected_field": "surprise",
    }

    with pytest.raises(SchemaValidationError, match="totally_unexpected_field"):
        QueryRecord.from_dict(payload)


def test_duplicate_query_ids_are_rejected():
    payload = {
        "schema_version": 1,
        "queries": [
            {
                "id": "dup",
                "text": "a",
                "language": "en",
                "query_type": "retrieval",
                "namespace": "ns",
                "positive_artifact_ids": ["a-1"],
            },
            {
                "id": "dup",
                "text": "b",
                "language": "en",
                "query_type": "retrieval",
                "namespace": "ns",
                "positive_artifact_ids": ["a-2"],
            },
        ],
    }

    with pytest.raises(SchemaValidationError, match="duplicate query id"):
        Dataset.from_dict(payload)


def test_hard_negative_provenance_covers_readme_vocabulary():
    labels = {member.value for member in HardNegativeProvenance}
    assert labels == {
        "lexical mining",
        "dense mining",
        "cross-model mining",
        "metadata-constrained sampling",
        "manual/adversarial",
    }


def test_future_schema_version_is_rejected():
    payload = {
        "schema_version": 999,
        "queries": [
            {
                "id": "q-1",
                "text": "a",
                "language": "en",
                "query_type": "retrieval",
                "namespace": "ns",
                "positive_artifact_ids": ["a-1"],
            }
        ],
    }

    with pytest.raises(SchemaValidationError, match="schema_version"):
        Dataset.from_dict(payload)
