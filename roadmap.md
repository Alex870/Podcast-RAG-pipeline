# Roadmap

Updated: 2026-09-05

`Podcast-RAG-pipeline` converts contract-valid transcripts into evidence-linked knowledge artifacts. Schema `2.1`, deterministic representations, stable evidence closure, correction-scoped deltas, measured retrieval exports, optional evidence-bound temporal research artifacts, advanced-retrieval sidecars, and processing hardening are implemented. The next value lies in private-corpus promotion evidence and measured tuning.

The retrieval-specific execution plan, including downstream ownership, experiment gates, and LLM-automation boundaries, is maintained in [`retrieval-modernization-action-plan.md`](retrieval-modernization-action-plan.md).

The consumer-boundary redesign is specified in [`docs/transcription-handoff-contract.md`](docs/transcription-handoff-contract.md). New ingestion and incremental-reuse work should implement that contract rather than extend unscoped parent-directory scanning.

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
- Partition-aware corpus identity is prepared upstream; consume transcription `partition_id` metadata when the partition rollout lands.
- Partition identity is now preserved on transformed document metadata so downstream exports can remain corpus-scoped.
- The detailed producer-to-consumer handoff contract is documented; package-based handoff publication, handoff-bound run state, and operational per-stage reuse reporting remain implementation work.
- Dependency-light judged retrieval metrics and consumer-vendored transcription fixtures.
- Capability-aware local LLM support with strict structured-output boundaries.

## Value-Ordered Priorities

### 1. Establish the real processing and retrieval baseline — implementation present; acceptance pending

- Consume the approved local evaluation pack and bind every run to its corpus and pack fingerprint.
- Replace the checked-in eight-query draft template with reviewed private-corpus judgments; the existing template contains no promotable ground truth.
- Produce the first aligned `page-content-v1` dense result set for RAGScope.
- Diagnose failures by query class, speaker/date constraint, evidence level, duplicate pressure, and hierarchy path.
- Define release-critical processing fixtures and retrieval queries.

### 2. Consume human correction change sets

- Implemented: accept validated v1/v2 transcription correction notifications and bind deltas to affected episodes and source spans.
- Implemented: fail closed when candidate documents contain unrelated corpus drift.
- Implemented: emit deterministic processed-cache deltas for explicit approval and downstream reconciliation.
- Next operational proof: execute the flow against the approved private pack and corpus release.

### 3. Improve retrieval representations through paired experiments — implemented profiles; promotion pending

- Compare dense baseline, lexical fusion, MMR, and lightweight reranking on identical judged queries.
- Treat normalized lexical-text production as implemented; complete BM25/sparse indexing, fusion, and reranking in the downstream retrieval applications.
- Compare a revision-pinned modern embedding model against `BAAI/bge-large-en-v1.5` in separate indexes without rebuilding hierarchy or mixing vector spaces.
- Evaluate the implemented `context-header-v1` before investing in true late-chunk vectors; use the late-chunk alignment sidecar only after a measured entry gate passes.
- Tune chunk size, overlap, context headers, hierarchy depth, and direct-evidence windows independently.
- Evaluate leaf evidence, summaries, position cards, temporal synthesis, and cross-speaker results separately.
- Add contradiction candidates and claim/evidence confidence only when their retrieval or answer value is measurable.

### 4. Strengthen temporal and comparative knowledge — implemented artifacts; promotion pending

- Implemented: deterministic evidence-bound claims, temporal adjacency, missing intervals, optional trajectories, and review-qualified contradiction/reversal candidates.
- Implemented: generated wording does not define stable claim identity; source spans and extraction identity do.
- Next: promote individual temporal artifacts only when private timeline/comparison slices show measurable value.

### 5. Evaluate and mature graph-assisted retrieval

- Implemented as an opt-in prototype: release-bound evidence graph generation, advanced sidecar packaging, and bounded Chat graph expansion with traceable fallback.
- The prototype is not a promoted retrieval path: require graph-positive failures that remain after hybrid retrieval and reranking before further graph investment.
- Next: establish graph-positive private query slices and compare dense, hybrid, reranked, and graph-assisted profiles on identical releases.
- Harden typed edge provenance, graph-quality validation, stable speaker relationships, correction supersession, and delta-scoped graph rebuilding.
- Promote graph expansion independently for cross-episode, evolution, contradiction, associative, and evidence-lineage query classes only when it passes the documented quality and latency gates.
- Keep vector/hybrid retrieval as the seed and fallback; defer a graph database until portable sidecar limits are demonstrated.
- Design and promotion details: [`docs/graph-rag-design-augmentation.md`](docs/graph-rag-design-augmentation.md).

### 6. Improve runtime and operations — partially implemented; operational proof pending

- Implemented in part: hardening preflight, cache/state integrity checks, concurrency checks, free-space estimates, deterministic offline diagnostics, and backup/restore evidence.
- Expand backend capability discovery, concurrency control, prompt budgeting, retries, and cancellation telemetry where still missing.
- Add corpus release manifests, migration tooling, retention rules, and disk/runtime estimates.
- Quarantine invalid model output and retain diagnostic payloads without making them authoritative.
- Add one-shot changed-episode processing suitable for ecosystem orchestration.

## Sequencing

1. Bind the approved evaluation pack and record the dense baseline.
2. Run the implemented correction-manifest-scoped delta path against the approved private corpus.
3. Promote the smallest M2 retrieval profile that passes the private campaign.
4. Evaluate implemented temporal/contradiction artifacts on dedicated M3 slices.
5. Evaluate bounded graph assistance on graph-positive query slices and promote only passing classes.
6. Tune representations and hierarchy from per-query component failures.
7. Harden orchestration, migrations, and target-machine operations.

For the retrieval-modernization workstream, expand step 3 above in this strict order: dense baseline, dense-plus-lexical hybrid, reranking, alternate embedding model, contextual header, optional true late chunking, and finally query-class-gated graph assistance. Correction-delta and temporal-artifact proofs remain parallel release requirements; they must not be silently folded into a retrieval experiment.

The ecosystem-level sequence and promotion rules live in `../PODCAST_ECOSYSTEM_ROADMAP.md` when these repositories share a workspace.
## Status as of 2026-09-05

Correction-aware delta planning/application, notification-scoped production CLI integration, canonical identities, fixtures, campaign-bound baseline export, schema `2.1` lexical/contextual representations, advanced-retrieval producers, and hardening preflight are implemented. The deterministic unit suite passes 62 tests in this checkout. Real baseline execution, paired private retrieval evidence, downstream hybrid/reranking runs, alternate-model indexes, and all promotion decisions await the approved private evaluation pack.
