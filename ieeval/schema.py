"""Versioned dataset/query schema for ieeVal retrieval fixtures.

This module defines the on-disk contract described in README.md's "Dataset
contract" section: a query record with explicit positives, explicit hard
negatives, and a human-readable provenance note, plus a fixed vocabulary for
*how* a hard negative was sourced.

Fixtures are plain JSON (stdlib ``json``, no third-party parser) shaped as::

    {
      "schema_version": 1,
      "queries": [
        {
          "id": "ocr-lineage-fi",
          "text": "Miksi OCR-tuloksen pitäisi korvata alkuperäinen artefakti?",
          "language": "fi",
          "query_type": "retrieval",
          "namespace": "architecture",
          "positive_artifact_ids": ["artifact-123"],
          "hard_negative_ids": ["artifact-456"],
          "expected_tags": ["ocr", "lineage"],
          "provenance": "manually curated",
          "notes": "Tests exact intent against a topical near miss."
        }
      ]
    }

which is a direct JSON transliteration of the YAML example in README.md (JSON
is used for the fixture on disk because the project intentionally ships with
``dependencies = []`` in pyproject.toml and stdlib has no YAML parser).

Schema version compatibility rule
----------------------------------
``SCHEMA_VERSION`` tracks the shape of :class:`QueryRecord` and the top-level
fixture document, not any particular dataset's content.

- Additive, backward-compatible changes (a new *optional* field with a
  default that older readers can safely ignore or that keeps prior fixtures
  valid) do **not** bump ``SCHEMA_VERSION``.
- Removing a field, renaming a field, changing a field's meaning or type, or
  tightening validation such that a previously-valid fixture becomes invalid
  **does** bump ``SCHEMA_VERSION``.

Consumers should refuse (or explicitly migrate) fixtures whose
``schema_version`` is newer than the version they were built against, rather
than guessing at compatibility.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

#: Current version of the dataset/query schema defined in this module.
#: See "Schema version compatibility rule" above for what bumps this.
SCHEMA_VERSION = 1

#: Fields every query record must supply; a fixture missing any of these is
#: rejected outright rather than silently defaulted.
REQUIRED_QUERY_FIELDS = (
    "id",
    "text",
    "language",
    "query_type",
    "namespace",
    "positive_artifact_ids",
)

#: Fields a query record may supply, with defaults applied when absent.
OPTIONAL_QUERY_FIELDS = (
    "hard_negative_ids",
    "expected_tags",
    "provenance",
    "notes",
)

KNOWN_QUERY_FIELDS = frozenset(REQUIRED_QUERY_FIELDS) | frozenset(OPTIONAL_QUERY_FIELDS)

# A deliberately loose BCP-47-ish check: 2-3 lowercase letters for the
# primary language subtag, optionally followed by hyphen-separated subtags
# (region, script, variant, ...). It accepts "fi", "en", "sv", "en-US",
# "zh-Hans", etc. It is not a full BCP-47 validator.
_LANGUAGE_TAG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")


class SchemaValidationError(ValueError):
    """Raised when a dataset/query fixture does not satisfy the schema.

    Carries a clear, specific message about *what* is wrong (missing field,
    empty positives, malformed language tag, ...) rather than surfacing a
    generic ``KeyError``/``TypeError`` from dataclass construction.
    """


class HardNegativeProvenance(StrEnum):
    """Fixed vocabulary for how a hard-negative artifact was sourced.

    Per README.md: "Hard negatives are explicit and retain their source
    (lexical mining, dense mining, cross-model mining, metadata-constrained
    sampling, or manual/adversarial selection)."

    This is a separate concept from :attr:`QueryRecord.provenance`, which is
    a free-text note about how the *query record as a whole* was authored
    (e.g. "manually curated"). ``HardNegativeProvenance`` is the closed
    vocabulary intended for tracking, per hard-negative id, which mining
    strategy produced it once fixtures grow beyond hand-authored toy data
    (e.g. a future ``{artifact_id: HardNegativeProvenance}`` side table keyed
    by ``hard_negative_ids``). It is defined now so downstream mining
    tooling has a stable set of labels to target.
    """

    LEXICAL_MINING = "lexical mining"
    DENSE_MINING = "dense mining"
    CROSS_MODEL_MINING = "cross-model mining"
    METADATA_CONSTRAINED_SAMPLING = "metadata-constrained sampling"
    MANUAL_ADVERSARIAL = "manual/adversarial"


@dataclass(frozen=True)
class QueryRecord:
    """A single versioned query record, matching README.md's dataset contract.

    Field names and shape mirror the "Dataset contract" YAML example
    exactly: ``id``, ``text``, ``language``, ``query_type``, ``namespace``,
    ``positive_artifact_ids``, ``hard_negative_ids``, ``expected_tags``,
    ``provenance``, ``notes``.
    """

    id: str
    text: str
    language: str
    query_type: str
    namespace: str
    positive_artifact_ids: tuple[str, ...]
    hard_negative_ids: tuple[str, ...] = field(default_factory=tuple)
    expected_tags: tuple[str, ...] = field(default_factory=tuple)
    provenance: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QueryRecord:
        """Build and validate a :class:`QueryRecord` from a decoded JSON object.

        Raises :class:`SchemaValidationError` with a specific, actionable
        message for missing/unknown/malformed fields instead of letting a
        generic ``TypeError``/``KeyError`` leak out of dataclass construction.
        """
        if not isinstance(data, Mapping):
            raise SchemaValidationError(
                f"query record must be a JSON object, got {type(data).__name__}"
            )

        record_id = data.get("id", "<unknown id>")

        missing = [name for name in REQUIRED_QUERY_FIELDS if name not in data]
        if missing:
            raise SchemaValidationError(
                f"query record {record_id!r} is missing required field(s): "
                f"{', '.join(sorted(missing))}"
            )

        unknown = set(data) - KNOWN_QUERY_FIELDS
        if unknown:
            raise SchemaValidationError(
                f"query record {record_id!r} has unknown field(s): "
                f"{', '.join(sorted(unknown))} (known fields: "
                f"{', '.join(sorted(KNOWN_QUERY_FIELDS))})"
            )

        record = cls(
            id=str(data["id"]),
            text=str(data["text"]),
            language=str(data["language"]),
            query_type=str(data["query_type"]),
            namespace=str(data["namespace"]),
            positive_artifact_ids=_as_str_tuple(
                data["positive_artifact_ids"],
                field_name="positive_artifact_ids",
                record_id=record_id,
            ),
            hard_negative_ids=_as_str_tuple(
                data.get("hard_negative_ids", ()),
                field_name="hard_negative_ids",
                record_id=record_id,
            ),
            expected_tags=_as_str_tuple(
                data.get("expected_tags", ()),
                field_name="expected_tags",
                record_id=record_id,
            ),
            provenance=str(data.get("provenance", "")),
            notes=str(data.get("notes", "")),
        )
        record.validate()
        return record

    def validate(self) -> None:
        """Validate field contents, raising :class:`SchemaValidationError` on failure.

        Checks, in order: required text fields are non-empty, at least one
        positive artifact id is present, and the language tag has a
        plausible BCP-47-ish shape.
        """
        for field_name in ("id", "text", "language", "query_type", "namespace"):
            if not getattr(self, field_name).strip():
                raise SchemaValidationError(
                    f"query record {self.id!r} has an empty required field: {field_name!r}"
                )

        if len(self.positive_artifact_ids) == 0:
            raise SchemaValidationError(
                f"query record {self.id!r} must have at least one positive_artifact_id"
            )

        if not _LANGUAGE_TAG_RE.match(self.language):
            raise SchemaValidationError(
                f"query record {self.id!r} has an implausible language tag "
                f"{self.language!r}; expected a BCP-47-ish code such as 'fi', "
                "'en', or 'en-US'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to the plain-dict shape used on disk."""
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "query_type": self.query_type,
            "namespace": self.namespace,
            "positive_artifact_ids": list(self.positive_artifact_ids),
            "hard_negative_ids": list(self.hard_negative_ids),
            "expected_tags": list(self.expected_tags),
            "provenance": self.provenance,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Dataset:
    """A versioned collection of query records loaded from a fixture file."""

    schema_version: int
    queries: tuple[QueryRecord, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Dataset:
        if not isinstance(data, Mapping):
            raise SchemaValidationError(
                f"dataset fixture must be a JSON object, got {type(data).__name__}"
            )
        if "queries" not in data:
            raise SchemaValidationError("dataset fixture is missing required field: 'queries'")

        raw_queries = data["queries"]
        if not isinstance(raw_queries, list):
            raise SchemaValidationError(
                f"dataset fixture field 'queries' must be a list, got {type(raw_queries).__name__}"
            )
        if len(raw_queries) == 0:
            raise SchemaValidationError("dataset fixture must contain at least one query")

        schema_version = data.get("schema_version", SCHEMA_VERSION)
        if not isinstance(schema_version, int):
            raise SchemaValidationError(
                f"dataset fixture field 'schema_version' must be an int, got "
                f"{type(schema_version).__name__}"
            )
        if schema_version > SCHEMA_VERSION:
            raise SchemaValidationError(
                f"dataset fixture declares schema_version={schema_version}, which is newer "
                f"than the schema this code understands ({SCHEMA_VERSION}); refusing to guess "
                "at compatibility"
            )

        queries = tuple(QueryRecord.from_dict(entry) for entry in raw_queries)

        seen_ids: dict[str, int] = {}
        for index, query in enumerate(queries):
            if query.id in seen_ids:
                raise SchemaValidationError(
                    f"duplicate query id {query.id!r} at index {index} "
                    f"(first seen at index {seen_ids[query.id]})"
                )
            seen_ids[query.id] = index

        return cls(schema_version=schema_version, queries=queries)

    @classmethod
    def from_json(cls, path: str | Path) -> Dataset:
        """Load and validate a dataset fixture from a JSON file on disk."""
        text = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"{path}: not valid JSON ({exc})") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "queries": [query.to_dict() for query in self.queries],
        }


def _as_str_tuple(value: Any, *, field_name: str, record_id: Any) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        raise SchemaValidationError(
            f"query record {record_id!r} field {field_name!r} must be a list of strings, "
            f"got {type(value).__name__}"
        )
    items = list(value)
    for item in items:
        if not isinstance(item, str):
            raise SchemaValidationError(
                f"query record {record_id!r} field {field_name!r} must contain only strings, "
                f"found {type(item).__name__}: {item!r}"
            )
    return tuple(items)
