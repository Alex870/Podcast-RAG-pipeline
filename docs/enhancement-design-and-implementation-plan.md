# Enhancement Design and Implementation Plan

_Based on the July 2026 state-of-the-art review and the project roadmap._

## 1. Purpose

This document converts the project roadmap and the research comparison into an implementable engineering program for `Podcast-RAG-pipeline` and its downstream ecosystem.

The goal is not to add every recent RAG technique. The goal is to improve retrieval quality, speaker attribution, temporal reasoning, evidence traceability, and operational reliability while preserving:

- Local-first Windows operation.
- Backward-readable processed caches.
- Deterministic and inspectable artifacts.
- Compatibility with `Chroma DB Import`, `PodCast Chat`, and `RAGScope`.
- A modest-hardware baseline.
- The ability to prove that a change improved the system before making it a default.

## 2. Design Conclusions

The current pipeline should remain a preprocessing and knowledge-structuring system. It should not absorb query serving, vector database ownership, chat orchestration, or visualization.

The most sensible upgrade sequence is:

1. Freeze and test the cross-project artifact contract.
2. Establish a judged retrieval baseline.
3. Emit richer retrieval representations without changing current behavior.
4. Add hybrid retrieval and reranking in downstream applications.
5. Evaluate modern embeddings and contextual chunk representations.
6. Extend position cards into stable temporal claim structures.
7. Add lightweight graph traversal over existing evidence links.
8. Consider adaptive retrieval and domain adaptation only after simpler methods plateau.

The existing RAPTOR-style hierarchy should be retained. It is already aligned with a strong research direction. The immediate need is to measure and improve how hierarchy nodes are represented and retrieved.

## 3. Scope and Repository Ownership

### Podcast-RAG-pipeline

Owns:

- Transcript normalization and validation.
- Leaf chunk creation.
- Hierarchical clustering and summarization.
- Position and claim extraction.
- Topic contribution and index generation.
- Retrieval-oriented text representations.
- Processed-cache schemas, provenance, and migrations.
- Offline evaluation dataset definitions and preprocessing ablations.

Does not own:

- Persistent vector database insertion.
- Query-time candidate retrieval.
- Rank fusion or reranking execution in production chat.
- Answer generation or chat history.
- Retrieval visualization.

### Chroma DB Import

Owns:

- Embedding generated cache documents for a selected representation.
- Dense index creation and updates.
- Optional lexical/sparse side-index construction.
- Embedding compatibility metadata.
- Import manifests and update semantics.

### PodCast Chat

Owns:

- Query interpretation and filters.
- Dense, lexical, hierarchy, and topic retrieval orchestration.
- Candidate fusion, reranking, and context selection.
- Answer generation, citations, and abstention behavior.
- Optional bounded corrective retrieval.

### RAGScope

Owns:

- Retrieval experiments and comparisons.
- Evaluation query-set management.
- Metrics, score distributions, hierarchy traces, and failure analysis.
- A/B comparison of embeddings, retrieval modes, and rerankers.
- Promotion reports for proposed defaults.

## 4. Architectural Principles

### 4.1 Evaluation-Gated Defaults

Every new representation or retrieval strategy starts behind an explicit feature flag. It becomes a default only when a recorded evaluation run demonstrates improvement against agreed metrics without unacceptable latency, storage, or hardware regression.

### 4.2 Additive Schema Evolution

New optional fields should be added without changing the meaning of existing fields. Readers must continue accepting older schema `2.x` caches. A major schema version is reserved for breaking changes.

### 4.3 Separate Source Text from Retrieval Text

`page_content` remains the human-readable source or generated document displayed and cited to users. New retrieval representations must not overwrite it.

The cache should distinguish:

- Display/citation text.
- Dense embedding text.
- Lexical search text.
- Optional contextual header.
- Generated summary or claim text.

### 4.4 Provenance for Every Derived Representation

Any generated text, embedding representation, topic label, or claim relationship must record:

- Method and version.
- Model and provider when applicable.
- Config fingerprint.
- Source document IDs.
- Whether the result is deterministic, model-generated, or manually reviewed.

### 4.5 Stable IDs Across Rebuilds

Stable IDs should derive from source identity and normalized semantic role, not transient cluster numbering. Reprocessing unchanged source material with the same representation version should preserve IDs.

### 4.6 Bounded Local Inference

All LLM operations must use discovered context limits, explicit prompt budgets, reserved completion capacity, bounded retries, and durable checkpoints. Frontier features must degrade cleanly to the existing baseline.

## 5. Target Data Contract

## 5.1 Cache Envelope

Retain the current cache envelope and add a versioned `representations` manifest:

```json
{
  "schema_version": "2.1",
  "source_file": "episode_speaker_transcript.json",
  "source_fingerprint": "...",
  "episode_title": "...",
  "representations": {
    "display_text": "1.0",
    "dense_text": "context-header-v1",
    "lexical_text": "normalized-lexical-v1",
    "claims": "position-card-v2"
  },
  "documents": []
}
```

Readers must treat a missing `representations` object as the current `2.0` behavior.

## 5.2 Document Representation Fields

Add optional top-level fields beside `page_content` and `metadata`:

```json
{
  "page_content": "Text shown to users and used for citations.",
  "embedding_text": "Deterministic contextual header followed by the document text.",
  "lexical_text": "Normalized text used for sparse or BM25 indexing.",
  "metadata": {}
}
```

Rules:

- `page_content` remains required.
- `embedding_text` falls back to `page_content` when absent.
- `lexical_text` falls back to `page_content` when absent.
- Importers select the representation explicitly and record the selection.
- The chat UI never displays `embedding_text` unless debug mode requests it.

## 5.3 Contextual Header Version 1

Generate a bounded deterministic header using available metadata:

```text
Podcast: {podcast_name}
Episode: {episode_title}
Date: {episode_date}
Speaker: {speaker or speaker list}
Document type: {node_type}
Hierarchy: {level}
Topic hints: {bounded topic tags}
```

The header must:

- Be deterministic.
- Exclude empty fields.
- Use a stable field order.
- Be capped by character or token budget.
- Avoid adding claims that are not in source metadata.
- Be independently versioned.

This is the low-risk precursor to true late chunking.

## 5.4 Lexical Representation Version 1

`lexical_text` should contain:

- Original document text.
- Episode title.
- Podcast name.
- Speaker names.
- ISO date and human-readable date tokens.
- Topic tags.
- Claim and stance text for position cards.
- Normalized aliases where explicitly known.

Do not remove uncommon words, names, acronyms, or quotations. Lexical retrieval benefits from preserving exact language.

## 5.5 Position Card Version 2

Extend position cards additively with:

```json
{
  "claim_id": "stable cross-episode claim identity",
  "claim_canonical": "normalized proposition",
  "attribution_confidence": 0.0,
  "evidence_strength": 0.0,
  "ambiguity": "",
  "relationship_candidates": [
    {
      "target_claim_id": "...",
      "relation": "supports|contradicts|qualifies|supersedes|reaffirms",
      "confidence": 0.0
    }
  ],
  "first_observed_date": "YYYY-MM-DD",
  "latest_observed_date": "YYYY-MM-DD"
}
```

Relationship candidates remain explicitly provisional until deterministic checks or review accept them. The source evidence IDs remain authoritative.

## 6. Proposed Package Structure

Keep `pipeline.py` as orchestration and move new behavior into focused modules:

```text
src/podcast_rag/
  evaluation/
    __init__.py
    dataset.py
    metrics.py
    runner.py
    reports.py
  representations/
    __init__.py
    contextual.py
    lexical.py
    manifest.py
  claims/
    __init__.py
    identity.py
    relationships.py
    temporal.py
  embeddings/
    __init__.py
    base.py
    sentence_transformer.py
    capabilities.py
  migrations/
    __init__.py
    cache_v2_0_to_v2_1.py
  config.py
  pipeline.py
  schema.py
```

Do not introduce these directories empty. Add each package only when its first functional unit is implemented.

## 7. Core Interfaces

## 7.1 Retrieval Representation Builder

```python
from typing import Protocol

class RetrievalRepresentationBuilder(Protocol):
    name: str
    version: str

    def build_embedding_text(self, document: dict) -> str: ...
    def build_lexical_text(self, document: dict) -> str: ...
```

The default implementation produces current behavior. `context-header-v1` is opt-in until evaluated.

## 7.2 Embedding Provider

```python
class EmbeddingProvider(Protocol):
    provider_id: str
    model_id: str
    dimension: int | None

    def discover_capabilities(self) -> dict: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
```

The first adapter wraps the current LangChain/Sentence Transformers path. A later BGE-M3 adapter may expose dense, sparse, and multi-vector capabilities without forcing callers to assume all providers support them.

## 7.3 Evaluation Retriever Adapter

```python
class EvaluationRetriever(Protocol):
    strategy_id: str

    def retrieve(self, query: str, filters: dict, top_k: int) -> list[dict]: ...
```

Adapters can execute against Chroma exports, lexical indexes, or captured retrieval-run JSON. Keeping this protocol outside the production pipeline prevents evaluation code from becoming a second chat implementation.

## 8. Phase 0: Baseline Capture and Contract Freeze

Implementation status: **implemented in the pipeline repository** for schema `2.1`, backward-compatible fixtures, representation provenance, and validation. Cross-repository fixture adoption remains downstream work.

### Objective

Create a reproducible baseline before changing representations.

### Podcast-RAG-pipeline Work

1. Reconcile the root roadmap, shared contract, current schema validator, and actual cache output.
2. Define schema `2.1` as an additive extension of `2.0`.
3. Add golden cache fixtures for:
   - Leaf chunks.
   - Cluster summaries.
   - Episode theses.
   - Position cards.
   - Multi-speaker nodes.
   - Missing optional metadata.
4. Add compatibility tests proving that:
   - A `2.0` cache remains readable.
   - A `2.1` cache falls back correctly in an old-style import path.
   - Stable IDs remain unchanged when optional representations are added.
5. Record a baseline processing report for a fixed set of episodes.

### Cross-Project Work

- Add the same golden cache fixture to importer contract tests.
- Validate an imported test database from PodCast Chat and RAGScope.
- Record current retrieval results before re-embedding anything.

### Acceptance Criteria

- One authoritative contract describes all required and optional fields.
- The same fixtures pass in the pipeline, importer, and RAGScope.
- No current database rebuild is required.
- Baseline output IDs and document counts are recorded.

## 9. Phase 1: Judged Retrieval Evaluation

Implementation status: **evaluation infrastructure implemented**. The included query set is intentionally a draft template; corpus-specific human judgments remain to be authored and reviewed in RAGScope.

### Objective

Build the measurement foundation required to approve later upgrades.

### Query-Set Format

Store versioned JSONL under `evaluation/query_sets`:

```json
{
  "query_id": "tfm-001",
  "query": "What did TFM say about multipolarity?",
  "category": "speaker_position",
  "expected_speakers": ["TFM"],
  "date_range": null,
  "relevance": {
    "stable-doc-id-1": 3,
    "stable-doc-id-2": 2
  },
  "acceptable_node_types": ["leaf_chunk", "position_card"],
  "answerable": true,
  "notes": ""
}
```

Use relevance grades:

- `3`: directly answers with attributable evidence.
- `2`: strongly supports part of the answer.
- `1`: useful background.
- `0`: not relevant.

### Minimum Initial Coverage

Create at least 50 queries across at least five episodes, including:

- 10 direct factual queries.
- 10 speaker-position queries.
- 8 exact-term or named-entity queries.
- 8 multi-passage queries.
- 6 temporal/evolution queries.
- 4 contradiction or qualification queries.
- 4 unanswerable queries.

This is sufficient to expose major regressions but not sufficient for domain fine-tuning.

### Metrics

Retrieval metrics:

- Recall@5, @10, and @20.
- MRR@10.
- nDCG@10.
- Speaker constraint precision and recall.
- Date-filter accuracy.
- Evidence coverage.
- Node-type contribution.
- Duplicate/equivalent evidence rate.
- Source and episode diversity.

Operational metrics:

- Median and p95 retrieval latency.
- Reranking latency.
- Index size.
- Embedding generation time.
- Peak VRAM and system RAM.

Generation metrics belong downstream:

- Supported-claim ratio.
- Citation precision.
- Citation coverage.
- Correct abstention rate.
- Human preference on correctness and completeness.

### RAGScope Integration

RAGScope should become the primary workbench for:

- Editing relevance judgments.
- Running retrieval strategies against the same query set.
- Inspecting false negatives and false positives.
- Comparing score distributions and hierarchy paths.
- Exporting a Markdown/JSON promotion report.

### Acceptance Criteria

- Runs are deterministic where the retrieval method is deterministic.
- Metrics can be reproduced from a saved run manifest.
- Every returned ID maps to a cache document and evidence path.
- Human labels and LLM-judge outputs are stored separately.

## 10. Phase 2: Retrieval-Ready Representations

Implementation status: **implemented behind configuration**. `page-content-v1` remains the dense default; `context-header-v1` is available for evaluation. Lexical text is emitted using `normalized-lexical-v1`.

### Objective

Emit contextual dense text and lexical text without changing displayed or cited content.

### Implementation

1. Add `representations/contextual.py` and `representations/lexical.py`.
2. Add config values:

```json
{
  "embedding_text_mode": "page_content",
  "lexical_text_mode": "normalized-lexical-v1",
  "contextual_header_max_chars": 700
}
```

3. Allow `embedding_text_mode` values:
   - `page_content`
   - `context-header-v1`
4. Populate optional `embedding_text` and `lexical_text` in serialized documents.
5. Add representation versions to the cache manifest and config fingerprint.
6. Extend schema validation with warnings for malformed optional fields, not errors for their absence.
7. Update importer selection rules and metadata before changing any default.

### Tests

- Header field ordering is stable.
- Empty metadata does not create empty labels.
- Header caps are respected.
- `page_content` is unchanged.
- Exact names, dates, and acronyms survive lexical normalization.
- Stable IDs do not depend on optional retrieval representations.

### Promotion Gate

`context-header-v1` becomes the default only if it improves aggregate nDCG or Recall@k and does not materially reduce exact-term performance. The report must also show node-type and query-category breakdowns.

## 11. Phase 3: Hybrid Retrieval and Reranking

### Objective

Improve query-time recall and precision using existing source documents before adopting more invasive representations.

### Importer Design

The importer should create:

- The existing Chroma dense collection.
- A lexical side index keyed by the same stable document IDs.
- A manifest declaring dense model, dense dimension, lexical method, and representation versions.

The first lexical implementation should be BM25 or an equivalently transparent local index. Learned sparse vectors can be evaluated later.

### Chat Retrieval Pipeline

```text
User query and filters
  -> dense retrieval top N
  -> lexical retrieval top N
  -> reciprocal rank fusion
  -> metadata constraint validation
  -> optional hierarchy expansion
  -> cross-encoder reranking
  -> redundancy-aware context selection
  -> answer prompt
```

### Initial Defaults for Evaluation

- Dense candidates: 50.
- Lexical candidates: 50.
- Reciprocal rank fusion constant: configurable, default 60.
- Rerank candidates: 40.
- Final context documents: determined by token budget, with a configurable upper bound.

These are experimental starting values, not production truths.

### Hierarchy Diversity Rules

- Avoid returning a parent summary and several nearly identical children unless the query benefits from both abstraction levels.
- Preserve at least one direct-evidence leaf when a summary or position card is selected.
- Apply per-node-type quotas only as a fallback; prefer score and evidence coverage.
- Record why each candidate survived or was removed.

### Acceptance Criteria

- Hybrid retrieval beats dense-only on exact-term queries without reducing aggregate retrieval quality.
- Reranking improves nDCG@10 or MRR without unacceptable p95 latency.
- Debug output exposes dense rank, lexical rank, fused rank, reranker score, and exclusion reason.
- Dense-only remains available as a one-click baseline and rollback mode.

## 12. Phase 4: Embedding Experiments

### Objective

Evaluate modern embeddings without coupling the pipeline to one model family.

### Work

1. Implement the embedding provider protocol around the current model.
2. Record discovered dimension, max input tokens, normalization behavior, device, and supported output modes.
3. Add BGE-M3 as an experimental provider.
4. Create separate exports for each model; never mix embedding spaces in one collection.
5. Evaluate:
   - Current BGE dense baseline.
   - BGE-M3 dense.
   - BGE-M3 dense plus BM25.
   - BGE-M3 dense plus learned sparse, if supported by the selected index.
6. Compare processing time, VRAM, storage, and retrieval metrics.

### Non-Goals

- Do not implement multi-vector retrieval in this phase.
- Do not replace Chroma solely to accommodate one experiment.
- Do not migrate existing speaker databases without an explicit rebuild and manifest change.

### Promotion Gate

A new dense default must improve the judged benchmark across multiple query categories. A small aggregate gain that substantially harms speaker-position or exact-term queries is insufficient.

## 13. Phase 5: Contextual Embedding Experiment

### Objective

Determine whether conversational references benefit from true late chunking beyond deterministic headers.

### Experiment Design

1. Select a bounded episode window rather than attempting to encode an entire multi-hour transcript.
2. Preserve exact mappings from source tokens to leaf chunk boundaries.
3. Encode the window with a compatible long-context embedding model.
4. Pool token embeddings separately for each leaf chunk.
5. Store the method, context window, overlap, model, and pooling version in the manifest.
6. Compare:
   - Plain chunk embedding.
   - Deterministic contextual header.
   - True late chunking.

### Focus Query Categories

- Pronoun and antecedent questions.
- Topic callbacks.
- Short ambiguous utterances.
- Questions whose key entity appears immediately before the relevant chunk.

### Stop Condition

Do not proceed to full-corpus implementation unless late chunking produces a meaningful gain over contextual headers. The custom pooling and rebuild complexity must earn its place.

## 14. Phase 6: Temporal Claims and Lightweight Graph Retrieval

### Objective

Improve cross-episode viewpoint, contradiction, and evolution questions using the graph already implicit in project artifacts.

### Claim Identity Pipeline

1. Normalize each position card into a bounded canonical claim.
2. Generate a deterministic candidate key from speaker, normalized topic, stance category, and claim text.
3. Retrieve semantically similar existing claims for the same speaker.
4. Ask the LLM only to classify candidate relationships, not to search the entire corpus.
5. Validate that every relationship points to existing claims and evidence.
6. Persist uncertain relationships as candidates rather than facts.

### Lightweight Graph

Use stable IDs and typed edges:

```text
episode -> contains -> document
summary -> summarizes -> child document
position -> supported_by -> evidence chunk
speaker -> holds -> position
position -> reaffirms|qualifies|contradicts|supersedes -> position
topic -> evidenced_by -> document
```

Store a portable adjacency artifact first. Do not require a graph database for the initial implementation.

### Retrieval Flow

1. Retrieve seed nodes using dense and lexical search.
2. Expand only allowed edge types within a bounded hop count.
3. Score expanded nodes using edge weight, source score, recency relevance, and evidence strength.
4. Rerank the combined candidate set.
5. Return graph paths in debug and citation metadata.

### Acceptance Criteria

- Graph expansion improves multi-hop and temporal query metrics.
- Simple factual queries retain their baseline path and latency.
- Every graph result has an explainable path to direct evidence.
- A bad relationship extraction cannot delete or overwrite source evidence.

## 15. Phase 7: Adaptive Retrieval

### Objective

Add bounded query planning only for queries that need it.

### Initial Scope

Support one optional planning step that may produce:

- A normalized primary query.
- Zero to three subqueries.
- Speaker constraints.
- Date constraints.
- Desired evidence types.

Always execute the original query as one retrieval channel.

After retrieval, allow at most one corrective retry when deterministic evidence checks show:

- No result exceeds a configured relevance threshold.
- Required speaker/date constraints are absent.
- A decomposed query has no evidence coverage.

### Guardrails

- Bounded number of model calls.
- Saved planner input and output.
- Strict structured schema.
- Original-query candidate preservation.
- User-visible indication when expanded retrieval was used.
- Ability to disable adaptive retrieval globally.

### Promotion Gate

Adaptive retrieval must improve hard-query coverage enough to justify added p95 latency and variability. It should remain off for direct factual queries when deterministic routing can identify them.

## 16. Configuration Strategy

Organize new settings by responsibility:

```json
{
  "representations": {
    "embedding_text_mode": "page_content",
    "lexical_text_mode": "normalized-lexical-v1",
    "contextual_header_max_chars": 700
  },
  "embedding": {
    "provider": "sentence_transformers",
    "model": "BAAI/bge-large-en-v1.5",
    "mode": "dense"
  },
  "evaluation": {
    "query_set": "evaluation/query_sets/podcast-baseline-v1.jsonl",
    "top_k": [5, 10, 20]
  }
}
```

For backward compatibility, existing flat config fields remain readable. New writes may retain the flat form until a planned config migration is implemented. Do not silently reinterpret an existing field.

## 17. Telemetry and Experiment Manifests

Every evaluation or ablation run should produce a manifest containing:

- Run ID and timestamp.
- Git commit IDs for participating repositories.
- Query-set ID and hash.
- Database/export ID.
- Cache schema and representation versions.
- Embedding provider, model, dimension, and device.
- Dense, lexical, fusion, reranking, graph, and planner configuration.
- Model endpoints and model IDs where LLMs participate.
- Metrics by category and overall.
- Latency, memory, storage, and processing cost.
- Failures, fallbacks, and excluded queries.

Promotion reports should compare a candidate against a named baseline and include both aggregate and per-category deltas.

## 18. Testing Strategy

### Unit Tests

- Representation rendering and caps.
- Lexical normalization.
- Stable ID behavior.
- Schema validation.
- Claim relationship validation.
- Metric calculations with known rankings.
- Reciprocal rank fusion with deterministic fixtures.

### Contract Tests

- Pipeline cache to importer.
- Importer manifest to PodCast Chat.
- Imported collection to RAGScope.
- Old cache versions through current readers.
- Missing optional representations and capabilities.

### Integration Tests

- Process a small transcript fixture with fake LLM responses.
- Import the resulting cache.
- Run dense and lexical retrieval against known queries.
- Verify stable IDs and evidence paths end to end.

### Regression Tests

- Snapshot document counts and node-type counts.
- Verify no speaker/date metadata loss.
- Verify no orphan child or evidence IDs.
- Run the judged query set and enforce configurable non-regression thresholds.

### Performance Tests

- Embedding throughput.
- Index build time.
- Retrieval and reranking p50/p95 latency.
- Peak RAM and VRAM.
- Database and lexical-index size.

## 19. Rollout and Rollback

Each promoted feature must have:

- A feature flag.
- A baseline-compatible fallback.
- A representation or strategy version.
- A migration note.
- A rollback test.

Do not mutate an existing database in place when changing embedding models or dimensionality. Build a separate export, validate it, then switch the configured database.

Do not delete old processed caches during schema migration. Write migrated output separately until validation succeeds.

## 20. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Evaluation set overfits one podcast or speaker | Include multiple episodes, query types, dates, and speakers; maintain a held-out set. |
| LLM summaries or claims distort evidence | Preserve source leaves, link every derived node to evidence, and score derived nodes separately. |
| Hybrid retrieval returns repetitive hierarchy nodes | Add equivalence grouping, node diversity, and direct-evidence requirements. |
| New embedding model breaks existing databases | Record model and dimension, prohibit mixed spaces, and rebuild into a separate export. |
| Generic reranker harms podcast relevance | Compare per-category metrics and retain fused pre-rerank scores for rollback. |
| Graph extraction creates false relationships | Store candidates with confidence, require evidence links, and never overwrite source facts. |
| Agentic retrieval becomes slow or unpredictable | Bound calls, retries, subqueries, and graph hops; keep deterministic retrieval available. |
| Schema evolution fragments the ecosystem | Maintain one shared contract, fixtures, compatibility tests, and explicit migrations. |

## 21. Recommended First Implementation Increment

The first increment should be deliberately narrow and unlock later work:

1. Add the versioned query-set schema and metric library.
2. Add a small manually judged baseline query set.
3. Add `embedding_text` and `lexical_text` as optional cache fields.
4. Implement deterministic `context-header-v1` and `normalized-lexical-v1` builders.
5. Extend cache provenance and validation for representation versions.
6. Add golden compatibility fixtures.
7. Update the shared contract.
8. Provide RAGScope with enough manifest information to compare `page_content` and contextual-header exports.

This increment does not change the default embedding text, does not require a database migration, and does not add query-time dependencies. It creates the measurement and data-contract foundation for hybrid retrieval and modern embedding experiments.

## 22. Definition of Done for the Upgrade Program

The enhancement program is successful when:

- Retrieval changes are approved using a versioned judged query set.
- The pipeline emits display, dense, and lexical representations without conflating them.
- Dense-only, hybrid, and reranked retrieval can be compared reproducibly.
- Every answer context can be traced to speaker, episode, time, hierarchy, and direct evidence.
- Embedding changes are versioned and cannot silently corrupt query compatibility.
- Cross-episode claims and relationships remain evidence-linked and uncertainty-aware.
- Advanced retrieval can be disabled without making current databases unusable.
- The baseline remains practical on the target Windows/NVIDIA environment.

The guiding rule remains simple: promote measured improvements, preserve evidence, and keep experimental intelligence behind stable, inspectable interfaces.
