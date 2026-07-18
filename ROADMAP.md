# ieeVal roadmap

This roadmap turns the current product planning into deliverable gates. Dates
are intentionally omitted until the repository has an executable baseline;
phases are ordered by dependency and evidence value.

## Phase 0 — contracts and scaffold

**Goal:** establish the smallest stable core on which experiments and
ieeVisualizer can depend.

- Scaffold the Python package, CLI, linting, type checking, and tests.
- Define versioned schemas for corpora, queries, positives, hard negatives, and
  hard-negative provenance.
- Define the immutable run manifest and artifact-directory contract.
- Define adapter protocols for embedders, lexical retrievers, dense retrievers,
  fusion strategies, and rerankers.
- Add a tiny deterministic fixture and golden artifact tests.
- Document privacy rules for exporting real Loimi material.

**Exit gate:** a toy experiment runs end to end, validates its inputs, and emits
a deterministic artifact bundle that can be read back without importing runtime
state.

## Phase 1 — retrieval baseline

**Goal:** make system-level retrieval comparisons trustworthy.

- Implement standard retrieval metrics: Recall@k, Precision@k, MRR, MAP,
  nDCG@k, hit rate, and relevant-item rank.
- Add lexical/FTS, dense, and hybrid comparison paths.
- Add Sentence Transformers/Hugging Face-compatible model adapters.
- Integrate selected MTEB/MMTEB tasks and compatible result import/export.
- Emit per-query rankings and error records, not only aggregates.
- Add bootstrap confidence intervals and slice reporting.

**Exit gate:** the same fixture can compare lexical, dense, and hybrid runs under
identical relevance judgments, with reproducible manifests and actionable
per-query errors.

## Phase 2 — Loimi evaluation gate

**Goal:** produce the evidence required to bless or replace Loimi's provisional
embedding tier.

- Curate a compact, reviewable Loimi corpus with Finnish, English, Swedish, and
  cross-language queries.
- Include paraphrases, lexically disjoint positives, lexical traps, and
  topically similar but intent-wrong hard negatives.
- Record hard-negative origin and human-confirm high-value mined candidates.
- Compare BGE-M3, multilingual-e5-large, and Qwen3-Embedding-0.6B.
- Compare FTS/lexical, dense-only, and hybrid configurations.
- Evaluate query-document retrieval separately from symmetric similarity and
  clustering/tag-centroid suitability.
- Validate graceful FTS fallback independently of embedding availability.

**Exit gate:** a reproducible recommendation assigns models/configurations to
pipeline roles, documents representative failures, and either validates or
rejects the provisional BGE-M3 choice. Automatic relevance enforcement remains
off until this gate and the threshold gate both pass.

## Phase 3 — multilingual and threshold laboratory

**Goal:** replace guessed cosine thresholds with calibrated policies.

- Report positive, random-negative, and hard-negative score distributions.
- Add precision-recall curves and operating-point selection.
- Slice calibration by language, language pair, namespace, and query type.
- Measure monolingual retrieval, cross-lingual retrieval, language
  separability, and cross-language neighborhood overlap.
- Test centroid stability, corpus-growth sensitivity, and model-change drift.
- Surface false promotions and missed positives for manual review.
- Export candidate policies for tag suggestion, promotion/rejection, duplicate
  detection, and related similarity decisions.

**Exit gate:** every proposed threshold has an explicit objective, confidence
evidence, conditioned scope, and failure examples. A single global threshold is
accepted only if it performs adequately across all required slices.

## Phase 4 — geometry and robustness diagnostics

**Goal:** explain representation behavior beyond ranking metrics.

- Add norm, anisotropy, principal-component concentration, intrinsic
  dimensionality, hubness, and density diagnostics.
- Measure neighborhood stability across model revisions, seeds, quantization,
  and truncation dimensions.
- Add duplicate and near-duplicate sensitivity tests.
- Add cross-model neighborhood-change outputs for visualization.
- Establish regression budgets for important metrics and slices.

**Exit gate:** a run can identify material geometry or neighborhood regressions
and link them to affected retrieval examples.

## Phase 5 — ieeVisualizer handoff

**Goal:** make the artifact contract sufficient for interactive diagnosis.

- Stabilize and version the run-artifact reader contract.
- Supply reference artifacts and conformance tests to ieeVisualizer.
- Support projection with original-space neighbor inspection.
- Support query comparison across FTS, dense, hybrid, and reranked runs.
- Support model-difference, language-pair, calibration, hubness/density, and
  grouped error views.
- Define compatibility and migration policy for older artifacts.

**Exit gate:** ieeVisualizer can render all required views from exported ieeVal
artifacts alone, including failures grouped by topical near miss, lexical false
friend, negation, entity confusion, temporal mismatch, language mismatch, OCR
corruption, and namespace bleed.

## Later candidates

These are intentionally outside the first Loimi gate:

- cross-model and active-learning loops for hard-negative discovery;
- reranker evaluation when first-stage recall is strong but ordering is weak;
- task-subset selection informed by inter-task correlation;
- experiment registry and multi-run comparison service;
- CI regression checks on curated, privacy-safe fixtures;
- additional model roles for bitext matching or duplicate detection when gains
  justify their operational complexity.

Generative summarization is not an embedding responsibility. If Loimi evaluates
summaries, ieeVal should assess the chosen generative system separately rather
than treating an embedding benchmark's summarization task as a production
architecture prescription.

## Ongoing rules

- Do not duplicate general benchmark orchestration already provided by MTEB.
- Do not approve a model from aggregate scores alone.
- Do not compare dense retrieval without lexical and hybrid baselines.
- Do not hide hard-negative, multilingual, or namespace-specific regressions in
  a universal score.
- Do not add a specialized model unless measured gains exceed its operational
  complexity tax.
- Do not mutate completed run artifacts; emit a new run with explicit lineage.
