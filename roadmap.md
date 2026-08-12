# Roadmap

Updated: 2026-08-12

`Podcast-RAG-pipeline` converts contract-valid transcripts into evidence-linked knowledge artifacts. Schema `2.1`, deterministic representations, stable evidence closure, correction-scoped deltas, measured retrieval exports, and optional evidence-bound temporal research artifacts are implemented. The next value lies in private-corpus promotion evidence and measured tuning.

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

- Implemented: accept validated v1/v2 transcription correction notifications and bind deltas to affected episodes and source spans.
- Implemented: fail closed when candidate documents contain unrelated corpus drift.
- Implemented: emit deterministic processed-cache deltas for explicit approval and downstream reconciliation.
- Next operational proof: execute the flow against the approved private pack and corpus release.

### 3. Improve retrieval representations through paired experiments

- Compare dense baseline, lexical fusion, MMR, and lightweight reranking on identical judged queries.
- Tune chunk size, overlap, context headers, hierarchy depth, and direct-evidence windows independently.
- Evaluate leaf evidence, summaries, position cards, temporal synthesis, and cross-speaker results separately.
- Add contradiction candidates and claim/evidence confidence only when their retrieval or answer value is measurable.

### 4. Strengthen temporal and comparative knowledge

- Implemented: deterministic evidence-bound claims, temporal adjacency, missing intervals, optional trajectories, and review-qualified contradiction/reversal candidates.
- Implemented: generated wording does not define stable claim identity; source spans and extraction identity do.
- Next: promote individual temporal artifacts only when private timeline/comparison slices show measurable value.

### 5. Improve runtime and operations

- Expand backend capability discovery, concurrency control, prompt budgeting, retries, and cancellation telemetry.
- Add corpus release manifests, migration tooling, retention rules, and disk/runtime estimates.
- Quarantine invalid model output and retain diagnostic payloads without making them authoritative.
- Add one-shot changed-episode processing suitable for ecosystem orchestration.

## Sequencing

1. Bind the approved evaluation pack and record the dense baseline.
2. Run the implemented correction-manifest-scoped delta path against the approved private corpus.
3. Promote the smallest M2 retrieval profile that passes the private campaign.
4. Evaluate implemented temporal/contradiction artifacts on dedicated M3 slices.
5. Tune representations and hierarchy from per-query component failures.
6. Harden orchestration, migrations, and target-machine operations.

The ecosystem-level sequence and promotion rules live in `../PODCAST_ECOSYSTEM_ROADMAP.md` when these repositories share a workspace.
## Phases 0–2 implementation status (2026-08-11)

Correction-aware delta planning/application, notification-scoped production CLI integration, canonical identities, fixtures, and campaign-bound baseline export are implemented. Real baseline execution awaits the approved private evaluation pack.
