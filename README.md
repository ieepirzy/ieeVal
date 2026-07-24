# ieeVal

Reproducible evaluation and diagnostics for embedding-backed retrieval systems.

> **Status:** planning and bootstrap. The public interfaces described here are a
> target, not a compatibility promise yet.

ieeVal answers a narrower and more useful question than “which embedding model
tops a leaderboard?”:

> Which retrieval configuration works for this corpus, for which tasks and
> languages, why does it fail, and is the evidence strong enough to ship it?

It uses established evaluation ecosystems such as
[MTEB](https://github.com/embeddings-benchmark/mteb) and MMTEB rather than
reimplementing their benchmark orchestration. ieeVal adds the layers a real
semantic system needs around them: corpus-specific retrieval fixtures, lexical
and hybrid baselines, hard-negative analysis, multilingual calibration,
embedding-geometry diagnostics, threshold experiments, error slices, and
immutable experiment provenance.

The initial consumer is [Loimi](https://github.com/ieepirzy/Loimi). Loimi's
embedding model and relevance thresholds are deliberately provisional until an
ieeVal vertical slice provides evidence for them.

## Project boundary

- **ieeVal** is a headless Python library and CLI for datasets, experiments,
  metrics, diagnostics, and stable run artifacts.
- **ieeVisualizer** is a separate interactive application that consumes those
  artifacts for query exploration, model comparison, calibration, and failure
  inspection.
- **MTEB/MMTEB** remain the execution substrate for standardized embedding
  benchmarks.

ieeVal is not intended to become a universal leaderboard, duplicate MTEB, turn
two-dimensional projections into proof of quality, or collapse every use case
into one “embedding quality” number.

## What ieeVal will evaluate

The first-class comparison is between complete retrieval configurations, not
model names in isolation:

- lexical/FTS, dense, hybrid, and optionally reranked retrieval;
- query-to-document retrieval, symmetric similarity, cross-lingual matching,
  clustering, and centroid-based policies as separate tasks;
- random negatives and topically plausible hard negatives as separate slices;
- Finnish, English, Swedish, and cross-language behavior;
- full-precision, quantized, and truncated embeddings where relevant;
- global, per-language, and per-namespace threshold policies.

Standard outputs include Recall@k, Precision@k, MRR, MAP, nDCG@k, hit rate,
relevant-item rank, confidence intervals, score distributions, precision-recall
curves, and per-example errors. Geometry diagnostics will cover norms,
anisotropy, principal-component concentration, intrinsic dimensionality,
hubness, density, and neighborhood stability.

## Dataset contract

A local retrieval fixture is expected to describe at least:

```yaml
queries:
  - id: ocr-lineage-fi
    text: Miksi OCR-tuloksen pitäisi korvata alkuperäinen artefakti?
    language: fi
    query_type: retrieval
    namespace: architecture
    positive_artifact_ids: [artifact-123]
    hard_negative_ids: [artifact-456]
    expected_tags: [ocr, lineage]
    provenance: manually curated
    notes: Tests exact intent against a topical near miss.
```

Hard negatives are explicit and retain their source (lexical mining, dense
mining, cross-model mining, metadata-constrained sampling, or manual/adversarial
selection). High-value mined negatives should be human-confirmed to limit false
negatives.

## Run artifacts

Every run should be reproducible and immutable. Its manifest records the model
and tokenizer revision, prompt/instruction, pooling, normalization, distance
metric, precision, truncation dimension, dataset revision, seed, retrieval and
reranking configuration, backend, hardware, and dependency versions.

A stable artifact bundle will contain machine-readable manifests, aggregate and
slice metrics, per-query rankings and errors, calibration data, and optional
embedding/geometry outputs. ieeVisualizer should need no privileged access to
the experiment process—only this bundle.

## First vertical slice

The first milestone will compare:

- BGE-M3 (the current **provisional** Loimi bootstrap candidate);
- multilingual-e5-large;
- Qwen3-Embedding-0.6B;
- lexical/FTS, dense-only, and hybrid retrieval.

The curated Loimi fixture will include Finnish, English, Swedish, cross-language
queries, paraphrases, lexically disjoint positives, lexical traps, and
topically-similar hard negatives. Retrieval, symmetric similarity, and
clustering/tag-centroid suitability will be reported independently.

The resulting recommendation must identify which configuration owns each
pipeline role, justify any additional model's operational cost, validate the
FTS fallback independently, and expose representative failures for inspection
in ieeVisualizer. No global cosine threshold or automatic tag-promotion policy
is considered validated before this gate passes.

## Development

Phase 0 scaffolding is in place: a minimal Python package (`ieeval`) with a
placeholder CLI, lint/type-check config, and a CI workflow. The versioned
schemas, adapter protocols, and fixtures described above are still to come -
this is just the scaffold sliver of Phase 0.

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

See [ROADMAP.md](ROADMAP.md) for sequencing and acceptance criteria.

## Design principles

1. **Reuse the benchmark ecosystem.** Integrate MTEB-compatible tasks and result
   formats; build the missing diagnostic layer.
2. **Always keep a lexical baseline.** Dense retrieval must demonstrate value
   against FTS/BM25 and hybrid alternatives.
3. **Evaluate roles separately.** A retrieval winner is not automatically the
   best clustering, cross-lingual, or thresholding model.
4. **Make difficult errors visible.** Hard negatives and per-slice failures are
   more informative than easy random negatives and aggregate scores alone.
5. **Specialization is earned.** Add another model or reranker only when measured
   gains exceed its deployment, caching, versioning, and re-embedding cost.
6. **Artifacts are the interface.** Runs are immutable, reproducible, and usable
   by other tools without hidden state.

## License

MIT - see [LICENSE](LICENSE).
