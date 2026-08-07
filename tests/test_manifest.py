from dataclasses import fields, replace

import pytest

from ieeval.manifest import (
    ERRORS_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    RANKINGS_FILENAME,
    ManifestValidationError,
    RunArtifactExistsError,
    RunManifest,
    build_manifest,
    read_manifest,
    read_metrics,
    write_run,
)


def _toy_manifest(**overrides):
    kwargs = dict(
        created_at="2026-08-04T00:00:00Z",
        model_revision="bge-m3@abc123",
        tokenizer_revision="bge-m3-tokenizer@abc123",
        pooling="cls",
        normalization="l2",
        distance_metric="cosine",
        precision="float32",
        dataset_revision="toy-dataset@1",
        seed=42,
        backend="sentence-transformers",
        hardware="cpu",
        retrieval_config={"top_k": 10},
        dependency_versions={"python": "3.12.3"},
    )
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def test_build_manifest_is_deterministic_for_identical_inputs():
    first = _toy_manifest()
    second = _toy_manifest()
    assert first.run_id == second.run_id


def test_build_manifest_run_id_changes_with_parent_run_id():
    first = _toy_manifest()
    second = _toy_manifest(parent_run_id=first.run_id)
    assert first.run_id != second.run_id
    assert second.parent_run_id == first.run_id


def test_build_manifest_run_id_changes_with_seed():
    first = _toy_manifest()
    second = _toy_manifest(seed=43)
    assert first.run_id != second.run_id


def test_golden_artifact_round_trip(tmp_path):
    manifest = _toy_manifest()

    run_dir = write_run(
        tmp_path,
        manifest,
        metrics={"aggregate": {"recall@10": 0.9}},
        rankings=[{"query_id": "q-1", "ranked_artifact_ids": ["a-1", "a-2"]}],
        errors=[{"query_id": "q-1", "error_type": "none"}],
    )

    assert run_dir == tmp_path / manifest.run_id
    assert (run_dir / MANIFEST_FILENAME).exists()
    assert (run_dir / METRICS_FILENAME).exists()
    assert (run_dir / RANKINGS_FILENAME).exists()
    assert (run_dir / ERRORS_FILENAME).exists()

    reloaded = read_manifest(run_dir)

    # Field-by-field equality, not just dataclass `==`, per the golden-artifact
    # acceptance criterion.
    for f in fields(RunManifest):
        assert getattr(reloaded, f.name) == getattr(manifest, f.name), f.name

    assert reloaded == manifest
    assert read_metrics(run_dir) == {"aggregate": {"recall@10": 0.9}}


def test_write_run_never_mutates_an_existing_run_directory(tmp_path):
    manifest = _toy_manifest()
    write_run(tmp_path, manifest)

    with pytest.raises(RunArtifactExistsError):
        write_run(tmp_path, manifest)


def test_write_run_with_explicit_lineage_produces_a_sibling_directory(tmp_path):
    first = _toy_manifest()
    first_dir = write_run(tmp_path, first)

    second = _toy_manifest(parent_run_id=first.run_id)
    second_dir = write_run(tmp_path, second)

    assert first_dir != second_dir
    assert read_manifest(second_dir).parent_run_id == first.run_id


def test_read_manifest_missing_directory_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "does-not-exist")


def test_manifest_rejects_missing_required_field():
    payload = _toy_manifest().to_dict()
    del payload["model_revision"]

    with pytest.raises(ManifestValidationError, match="model_revision"):
        RunManifest.from_dict(payload)


def test_manifest_rejects_future_schema_version():
    payload = _toy_manifest().to_dict()
    payload["schema_version"] = 999

    with pytest.raises(ManifestValidationError, match="schema_version"):
        RunManifest.from_dict(payload)


def test_manifest_rejects_non_positive_truncation_dimension():
    with pytest.raises(ManifestValidationError, match="truncation_dimension"):
        _toy_manifest(truncation_dimension=0)


@pytest.mark.parametrize(
    "malicious_run_id",
    [
        "../../../../tmp/x",
        "../escaped",
        "run-tooshort",
        "run-0123456789abcdeff",  # 17 hex chars, one too many
        "run-0123456789ABCDEF",  # uppercase hex not produced by compute_run_id
        "/etc/passwd",
        "run-0123456789abcdef/../../escaped",
    ],
)
def test_manifest_rejects_run_id_not_shaped_like_compute_run_id(malicious_run_id):
    payload = _toy_manifest().to_dict()
    payload["run_id"] = malicious_run_id

    with pytest.raises(ManifestValidationError, match="run_id"):
        RunManifest.from_dict(payload)


def test_write_run_rejects_path_traversal_run_id_and_does_not_escape_root(tmp_path):
    # Simulates a foreign/tampered manifest.json (e.g. reloaded from disk or
    # constructed directly) whose run_id was crafted to escape the sandbox.
    good = _toy_manifest()
    malicious = replace(good, run_id="../../../../tmp/ieeval_traversal_poc")

    root = tmp_path / "runs"
    root.mkdir()

    with pytest.raises(ManifestValidationError, match="run_id"):
        write_run(root, malicious)

    # Nothing should have been written inside the configured root, and (since
    # validation happens before any filesystem access) nothing was created
    # anywhere else either.
    assert list(root.iterdir()) == []
