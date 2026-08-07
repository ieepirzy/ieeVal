"""Immutable run-manifest and artifact-directory contract.

This module is the "product boundary" described in README.md's "Project
boundary" section: a downstream tool such as ieeVisualizer "should need no
privileged access to the experiment process—only this bundle." Concretely
that means this module:

- has no import-time dependency on the rest of ``ieeval`` (not even
  :mod:`ieeval.schema` -- a run references a dataset only by the opaque
  ``dataset_revision`` string, never by importing dataset types), and
- reads/writes nothing but plain JSON and JSON Lines files via the stdlib,
  so a separate process can reconstruct a :class:`RunManifest` from disk with
  nothing more than this one module.

Artifact-directory layout
--------------------------
Each run gets its own directory, named after its ``run_id``, under a root of
the caller's choosing (e.g. ``runs/``)::

    <root>/<run_id>/
        manifest.json    # the RunManifest, as formatted JSON
        metrics.json      # {"aggregate": {...}, "slices": {...}}
        rankings.jsonl     # one JSON object per query: ranking output
        errors.jsonl        # one JSON object per query: error/diagnostic record

``manifest.json`` is always written. ``metrics.json``, ``rankings.jsonl``, and
``errors.jsonl`` are written whenever the caller supplies content for them
(metrics computation and per-query rankings/errors are out of scope for this
module -- see ROADMAP.md Phase 1) -- but the *names and format* of those
files are part of the contract every run directory follows, so tooling that
only understands the directory layout can find them.

Immutability and lineage
-------------------------
``write_run`` refuses to write into a directory that already exists --
existing run artifacts are never mutated in place, per ROADMAP.md's "Do not
mutate completed run artifacts; emit a new run with explicit lineage" rule.

``run_id`` is derived deterministically (a SHA-256 digest, truncated) from
the run's own identity fields (model/tokenizer revision, prompt, pooling,
normalization, distance metric, precision, truncation dimension, dataset
revision, seed, retrieval/reranking configuration, backend, hardware,
dependency versions, and ``parent_run_id``). Two runs with byte-identical
configuration and no declared lineage therefore compute the same
``run_id`` and the same directory -- this is intentional: it makes accidental
duplicate runs detectable (the second `write_run` call fails loudly instead
of silently overwriting). To intentionally re-run an identical configuration
(e.g. to check reproducibility), set ``parent_run_id`` to the prior run's id;
that value is itself part of the hash, so the new run gets a distinct id
while still recording explicit lineage back to what it repeats.

Schema version compatibility rule
----------------------------------
Same rule as :mod:`ieeval.schema`: additive optional fields with safe
defaults do not bump ``SCHEMA_VERSION``; removing/renaming a field, changing
a field's meaning, or changing the artifact-directory layout in a
reader-breaking way does.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Current version of the run-manifest schema and artifact-directory layout.
SCHEMA_VERSION = 1

MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.json"
RANKINGS_FILENAME = "rankings.jsonl"
ERRORS_FILENAME = "errors.jsonl"

#: Manifest fields that identify a run's configuration (used to derive
#: run_id) as opposed to purely informational metadata (e.g. created_at).
_IDENTITY_FIELDS = (
    "model_revision",
    "tokenizer_revision",
    "prompt",
    "pooling",
    "normalization",
    "distance_metric",
    "precision",
    "truncation_dimension",
    "dataset_revision",
    "seed",
    "retrieval_config",
    "reranking_config",
    "backend",
    "hardware",
    "dependency_versions",
    "parent_run_id",
)

#: Exact shape produced by :func:`compute_run_id`: ``"run-"`` followed by the
#: first 16 hex digits of a SHA-256 digest. ``RunManifest.validate`` enforces
#: this on every manifest -- including ones reconstructed via
#: :meth:`RunManifest.from_dict`/:func:`read_manifest` -- because ``run_id``
#: is used verbatim to build a filesystem path in :func:`write_run`; anything
#: that doesn't match this shape (e.g. containing ``/`` or ``..``) must never
#: reach that path join.
_RUN_ID_RE = re.compile(r"^run-[0-9a-f]{16}$")


class ManifestValidationError(ValueError):
    """Raised when a run manifest does not satisfy the schema."""


class RunArtifactExistsError(FileExistsError):
    """Raised by :func:`write_run` when a run directory already exists.

    Run artifacts are immutable once written; emit a new run (optionally
    with ``parent_run_id`` set to this one) instead of overwriting it.
    """


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def compute_run_id(identity: Mapping[str, Any]) -> str:
    """Derive a deterministic run id from a run's identity fields.

    Same identity fields (including ``parent_run_id``) always produce the
    same id; this is what makes accidental duplicate runs detectable and
    intentional reruns distinguishable (see module docstring).
    """
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return f"run-{digest[:16]}"


@dataclass(frozen=True)
class RunManifest:
    """Immutable record of everything needed to reproduce and interpret a run.

    Field set is taken directly from README.md's "Run artifacts" section:
    "model and tokenizer revision, prompt/instruction, pooling,
    normalization, distance metric, precision, truncation dimension, dataset
    revision, seed, retrieval and reranking configuration, backend,
    hardware, and dependency versions."
    """

    schema_version: int
    run_id: str
    created_at: str
    parent_run_id: str | None
    model_revision: str
    tokenizer_revision: str
    prompt: str | None
    pooling: str
    normalization: str
    distance_metric: str
    precision: str
    truncation_dimension: int | None
    dataset_revision: str
    seed: int
    retrieval_config: dict[str, Any] = field(default_factory=dict)
    reranking_config: dict[str, Any] | None = None
    backend: str = ""
    hardware: str = ""
    dependency_versions: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate field contents, raising :class:`ManifestValidationError`."""
        required_str_fields = (
            "run_id",
            "created_at",
            "model_revision",
            "tokenizer_revision",
            "pooling",
            "normalization",
            "distance_metric",
            "precision",
            "dataset_revision",
            "backend",
            "hardware",
        )
        for field_name in required_str_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ManifestValidationError(
                    f"run manifest {self.run_id!r} has an empty/invalid required field: "
                    f"{field_name!r}"
                )

        if not _RUN_ID_RE.fullmatch(self.run_id):
            raise ManifestValidationError(
                f"run manifest field 'run_id' must match {_RUN_ID_RE.pattern!r} (the "
                f"shape produced by compute_run_id), got {self.run_id!r}"
            )

        if not isinstance(self.schema_version, int):
            raise ManifestValidationError(
                f"run manifest {self.run_id!r} field 'schema_version' must be an int, "
                f"got {type(self.schema_version).__name__}"
            )
        if self.schema_version > SCHEMA_VERSION:
            raise ManifestValidationError(
                f"run manifest {self.run_id!r} declares schema_version="
                f"{self.schema_version}, which is newer than the schema this code "
                f"understands ({SCHEMA_VERSION}); refusing to guess at compatibility"
            )

        if not isinstance(self.seed, int):
            raise ManifestValidationError(
                f"run manifest {self.run_id!r} field 'seed' must be an int, "
                f"got {type(self.seed).__name__}"
            )

        if self.truncation_dimension is not None and self.truncation_dimension <= 0:
            raise ManifestValidationError(
                f"run manifest {self.run_id!r} field 'truncation_dimension' must be a "
                f"positive int or None, got {self.truncation_dimension!r}"
            )

        if not isinstance(self.retrieval_config, dict):
            raise ManifestValidationError(
                f"run manifest {self.run_id!r} field 'retrieval_config' must be an object, "
                f"got {type(self.retrieval_config).__name__}"
            )

        if self.reranking_config is not None and not isinstance(self.reranking_config, dict):
            raise ManifestValidationError(
                f"run manifest {self.run_id!r} field 'reranking_config' must be an object "
                f"or null, got {type(self.reranking_config).__name__}"
            )

        if not isinstance(self.dependency_versions, dict):
            raise ManifestValidationError(
                f"run manifest {self.run_id!r} field 'dependency_versions' must be an "
                f"object, got {type(self.dependency_versions).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "parent_run_id": self.parent_run_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "prompt": self.prompt,
            "pooling": self.pooling,
            "normalization": self.normalization,
            "distance_metric": self.distance_metric,
            "precision": self.precision,
            "truncation_dimension": self.truncation_dimension,
            "dataset_revision": self.dataset_revision,
            "seed": self.seed,
            "retrieval_config": dict(self.retrieval_config),
            "reranking_config": (
                dict(self.reranking_config) if self.reranking_config is not None else None
            ),
            "backend": self.backend,
            "hardware": self.hardware,
            "dependency_versions": dict(self.dependency_versions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        if not isinstance(data, Mapping):
            raise ManifestValidationError(
                f"run manifest must be a JSON object, got {type(data).__name__}"
            )

        required = (
            "schema_version",
            "run_id",
            "created_at",
            "model_revision",
            "tokenizer_revision",
            "pooling",
            "normalization",
            "distance_metric",
            "precision",
            "dataset_revision",
            "seed",
            "backend",
            "hardware",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ManifestValidationError(
                f"run manifest is missing required field(s): {', '.join(sorted(missing))}"
            )

        manifest = cls(
            schema_version=int(data["schema_version"]),
            run_id=str(data["run_id"]),
            created_at=str(data["created_at"]),
            parent_run_id=(
                str(data["parent_run_id"]) if data.get("parent_run_id") is not None else None
            ),
            model_revision=str(data["model_revision"]),
            tokenizer_revision=str(data["tokenizer_revision"]),
            prompt=(str(data["prompt"]) if data.get("prompt") is not None else None),
            pooling=str(data["pooling"]),
            normalization=str(data["normalization"]),
            distance_metric=str(data["distance_metric"]),
            precision=str(data["precision"]),
            truncation_dimension=(
                int(data["truncation_dimension"])
                if data.get("truncation_dimension") is not None
                else None
            ),
            dataset_revision=str(data["dataset_revision"]),
            seed=int(data["seed"]),
            retrieval_config=dict(data.get("retrieval_config") or {}),
            reranking_config=(
                dict(data["reranking_config"]) if data.get("reranking_config") is not None else None
            ),
            backend=str(data["backend"]),
            hardware=str(data["hardware"]),
            dependency_versions=dict(data.get("dependency_versions") or {}),
        )
        manifest.validate()
        return manifest


def build_manifest(
    *,
    created_at: str,
    model_revision: str,
    tokenizer_revision: str,
    pooling: str,
    normalization: str,
    distance_metric: str,
    precision: str,
    dataset_revision: str,
    seed: int,
    backend: str,
    hardware: str,
    prompt: str | None = None,
    truncation_dimension: int | None = None,
    retrieval_config: dict[str, Any] | None = None,
    reranking_config: dict[str, Any] | None = None,
    dependency_versions: dict[str, str] | None = None,
    parent_run_id: str | None = None,
    schema_version: int = SCHEMA_VERSION,
) -> RunManifest:
    """Construct a validated :class:`RunManifest`, deriving ``run_id`` for you.

    ``created_at`` is caller-supplied (rather than read from the clock here)
    so callers -- including tests -- get deterministic, reproducible
    manifests; it is metadata only and does not affect the derived
    ``run_id``.
    """
    identity = {
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "prompt": prompt,
        "pooling": pooling,
        "normalization": normalization,
        "distance_metric": distance_metric,
        "precision": precision,
        "truncation_dimension": truncation_dimension,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "retrieval_config": retrieval_config or {},
        "reranking_config": reranking_config,
        "backend": backend,
        "hardware": hardware,
        "dependency_versions": dependency_versions or {},
        "parent_run_id": parent_run_id,
    }
    assert tuple(identity) == _IDENTITY_FIELDS  # keep in sync; see compute_run_id

    manifest = RunManifest(
        schema_version=schema_version,
        run_id=compute_run_id(identity),
        created_at=created_at,
        parent_run_id=parent_run_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        prompt=prompt,
        pooling=pooling,
        normalization=normalization,
        distance_metric=distance_metric,
        precision=precision,
        truncation_dimension=truncation_dimension,
        dataset_revision=dataset_revision,
        seed=seed,
        retrieval_config=retrieval_config or {},
        reranking_config=reranking_config,
        backend=backend,
        hardware=hardware,
        dependency_versions=dependency_versions or {},
    )
    manifest.validate()
    return manifest


def write_run(
    root: str | Path,
    manifest: RunManifest,
    *,
    metrics: Mapping[str, Any] | None = None,
    rankings: Iterable[Mapping[str, Any]] = (),
    errors: Iterable[Mapping[str, Any]] = (),
) -> Path:
    """Write a complete, immutable run artifact bundle under ``root``.

    Creates ``<root>/<manifest.run_id>/`` and writes ``manifest.json``
    (always), plus ``metrics.json``/``rankings.jsonl``/``errors.jsonl`` when
    the corresponding content is supplied. Raises
    :class:`RunArtifactExistsError` if the run directory already exists --
    this function never mutates a prior run's directory in place.

    Returns the run directory path.
    """
    manifest.validate()
    run_dir = Path(root) / manifest.run_id
    if run_dir.exists():
        raise RunArtifactExistsError(
            f"run artifact directory already exists and is immutable: {run_dir}. "
            "Emit a new run (with parent_run_id set to this run's id if it repeats "
            "this configuration) instead of overwriting it."
        )

    run_dir.mkdir(parents=True)

    _write_json(run_dir / MANIFEST_FILENAME, manifest.to_dict())
    _write_json(run_dir / METRICS_FILENAME, dict(metrics) if metrics is not None else {})
    _write_jsonl(run_dir / RANKINGS_FILENAME, rankings)
    _write_jsonl(run_dir / ERRORS_FILENAME, errors)

    return run_dir


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o444)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(row), sort_keys=True) for row in rows]
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o444)


def read_manifest(run_dir: str | Path) -> RunManifest:
    """Reconstruct a :class:`RunManifest` from a run directory on disk.

    Uses only this module and the stdlib -- no other part of ``ieeval`` is
    imported -- so a separate reader process (e.g. ieeVisualizer) can depend
    on this module alone.
    """
    manifest_path = Path(run_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {MANIFEST_FILENAME} found in run directory: {run_dir}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return RunManifest.from_dict(data)


def read_metrics(run_dir: str | Path) -> dict[str, Any]:
    """Read the aggregate/slice metrics document for a run, if present."""
    metrics_path = Path(run_dir) / METRICS_FILENAME
    if not metrics_path.exists():
        return {}
    result: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file (``rankings.jsonl``/``errors.jsonl``) into a list."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    text = file_path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
