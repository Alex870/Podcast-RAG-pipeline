# Roadmap

Updated: 2026-07-17

`Podcast-RAG-pipeline` converts contract-valid transcripts into evidence-linked knowledge artifacts. Schema `2.1`, deterministic representations, stable evidence closure, delta backfill, topic indexes, retrieval evaluation, and downstream contract fixtures are implemented. The next value lies in correction-aware incremental processing and measured retrieval improvements.

## Product Direction

- Keep display text, retrieval representations, and primary evidence distinct.
- Preserve stable document and source-span identity through every hierarchy layer.
- Recompute only artifacts whose inputs or dependencies changed.
- Use RAGScope and the shared evaluation pack as the authority for promotion.
- Prefer simple measured retrieval gains before adding more synthesis or larger models.

## Current Foundation

- Speaker-aware ingestion, leaf chunks, hierarchical summaries, episode theses, position cards, topic indexes, checkpoints, and reports.
- Backward-readable processed-cache `2.0` and evidence-preserving schema `2.1`.
- Versioned display, dense, and lexical representations with deterministic fingerprints.
- No-LLM backfill, `page-content-v1` baseline export, source hashes, stable IDs, and hierarchy/evidence validation.
- Dependency-light judged retrieval metrics and consumer-vendored transcription fixtures.
- Capability-aware local LLM support with strict structured-output boundaries.

## Value-Ordered Priorities

### 1. Establish the real processing and retrieval baseline

- Consume the approved local evaluation pack and bind every run to its corpus and pack fingerprint.
- Produce the first aligned `page-content-v1` dense result set for RAGScope.
- Diagnose failures by query class, speaker/date constraint, evidence level, duplicate pressure, and hierarchy path.
- Define release-critical processing fixtures and retrieval queries.

### 2. Consume human correction change sets

- Accept transcription correction manifests and classify affected episodes, spans, leaf nodes, ancestors, topics, and position cards.
- Preview recomputation scope and preserve old artifacts until the replacement validates.
- Emit a processed-cache delta manifest that Chroma Import can reconcile directly.
- Flag stale judgments when evidence IDs or source hashes change.

### 3. Improve retrieval representations through paired experiments

- Compare dense baseline, lexical fusion, MMR, and lightweight reranking on identical judged queries.
- Tune chunk size, overlap, context headers, hierarchy depth, and direct-evidence windows independently.
- Evaluate leaf evidence, summaries, position cards, temporal synthesis, and cross-speaker results separately.
- Add contradiction candidates and claim/evidence confidence only when their retrieval or answer value is measurable.

### 4. Strengthen temporal and comparative knowledge

- Build explicit time-aware topic and position trajectories linked to primary evidence.
- Represent uncertainty, reversals, and missing intervals rather than forcing a smooth narrative.
- Provide downstream query hints for evolution and speaker comparison without embedding answer prose into the contract.

### 5. Improve runtime and operations

- Expand backend capability discovery, concurrency control, prompt budgeting, retries, and cancellation telemetry.
- Add corpus release manifests, migration tooling, retention rules, and disk/runtime estimates.
- Quarantine invalid model output and retain diagnostic payloads without making them authoritative.
- Add one-shot changed-episode processing suitable for ecosystem orchestration.

## Sequencing

1. Bind the approved evaluation pack and record the dense baseline.
2. Implement correction-manifest invalidation and processed-cache delta output.
3. Compare lexical fusion, MMR, and lightweight reranking downstream.
4. Tune representations and hierarchy from per-query evidence.
5. Add temporal/contradiction structures only where baseline failures justify them.
6. Harden orchestration, migrations, and target-machine operations.

The ecosystem-level sequence and promotion rules live in `../PODCAST_ECOSYSTEM_ROADMAP.md` when these repositories share a workspace.
## Phases 0–2 implementation status (2026-07-17)

Correction-aware delta planning/application, canonical identities, fixtures, and campaign-bound baseline export are implemented. Real baseline execution awaits the approved private evaluation pack.
