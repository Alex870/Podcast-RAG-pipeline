# Retrieval Modernization Action Plan

_Prepared 2026-09-05 from `state-of-the-art-comparison.md`, `roadmap.md`, and the checked-out repository._

## Outcome

Modernize retrieval in measured increments while keeping `Podcast-RAG-pipeline` focused on evidence-preserving preprocessing. The promotion order is:

1. approve a real evaluation corpus and record the existing dense baseline;
2. enable lexical/dense hybrid retrieval downstream;
3. add downstream reranking;
4. compare a modern embedding model with the existing BGE baseline;
5. compare contextual headers, then true late chunking only if justified;
6. enable graph assistance only for query classes with a demonstrated residual gap.

No candidate becomes a default because it is newer or more complex. Every experiment must use the same immutable corpus release, reviewed query pack, stable document identities, and promotion policy.

## Cross-check: roadmap and repository state

The six recommendations remain directionally correct, but several are already partly implemented. The main blocker is operational evidence, not missing producer scaffolding.

| Recommendation | Repository evidence | Actual status and remaining work |
| --- | --- | --- |
| Evaluation corpus | `evaluation/query_sets/podcast-baseline-v1.jsonl`, `src/podcast_rag/evaluation/`, and campaign-bound export support | **Scoring is implemented; corpus acceptance is not.** The checked-in pack contains eight draft templates with empty relevance maps. There are no checked-in evaluation results or production corpus caches. Real questions, judgments, and downstream ranked runs are still required. |
| Lexical plus dense retrieval | Schema 2.1, `embedding_text`, `lexical_text`, `normalized-lexical-v1`, representation fingerprints, and representation-corpus export | **Producer work is substantially implemented.** Chroma DB Import still needs a lexical index, and PodCast Chat/RAGScope must execute and capture fusion runs. |
| Reranking | Evaluation format can identify strategy, reranker, latency, and ranked output | **Not implemented in this repository by design.** Candidate generation, reranking, and context packing belong in PodCast Chat/RAGScope. |
| Modern embeddings | Current config identifies `BAAI/bge-large-en-v1.5`; exports separate retrieval text from display text | **Experiment contract is ready; alternate retrieval indexes are not.** A new model must use a separate, revision-pinned index. Sparse/multi-vector modes require downstream capability changes. |
| Contextual or late-chunked embeddings | Opt-in `context-header-v1` is implemented. `late-chunk-alignment-1.0` can emit exact token/character alignment behind the M5 entry gate. | **Low-cost contextual representation is ready to test. True late-chunk vectors are not produced.** The current late-chunk artifact is a prototype alignment sidecar with `vectors_included: false`. |
| Graph retrieval | Opt-in `evidence-graph-1.0`, entry-gate validation, release binding, and a graph design/promotion document exist | **Prototype producer support exists; promotion evidence does not.** Quality reporting, delta maintenance, hardened provenance, and selective downstream use remain future work. |

This agrees with `roadmap.md`: bind the private evaluation pack first, record the dense baseline, promote the smallest useful M2 retrieval profile, evaluate temporal artifacts, and only then evaluate bounded graph assistance. The roadmap's current graph prototype does not invalidate the comparison's sequencing: it may exist as research code, but it must not be promoted before simpler approaches have been tested.

The deterministic unit suite currently passes 62 tests. That confirms the checked-in contracts and helpers; it does not establish retrieval quality on the private corpus.

## Rules that apply to every experiment

Before Step 1, write and version a promotion policy containing:

- immutable corpus release ID and corpus fingerprint;
- evaluation-pack ID, file hash, judgment version, and reviewer status;
- exact representation, embedding model and revision, index build configuration, retriever parameters, filters, candidate counts, reranker, and hardware;
- Recall@5/10/20, MRR@10, nDCG@10, evidence coverage, speaker/date/node constraint accuracy, duplicate rate, source diversity, p50/p95 latency, index size, and peak memory;
- results overall and by query class, with paired per-query deltas rather than aggregate scores alone;
- a no-regression rule for direct factual retrieval, speaker/date constraints, evidence traceability, and abstention;
- an explicit latency, storage, and hardware budget;
- rollback target and stop condition.

Use direct transcript evidence as the highest relevance grade. Summaries and position cards can be relevant, but should not substitute for attributable leaf evidence when a question calls for a quotation or factual claim. Keep answer-generation evaluation separate from retrieval evaluation.

## Step 1 — Establish the evaluation corpus and dense baseline

### 1.1 Freeze the evaluation target

1. Select one immutable private corpus release and export all stable document IDs and metadata.
2. Validate schema/evidence closure and representation coverage before authoring judgments.
3. Record the corpus fingerprint and do not silently refresh the corpus during the campaign. Corrections create a new release and identify stale judgments.
4. Export the `page-content-v1` dense corpus. This is the control; do not enable contextual headers yet.

### 1.2 Build a stratified query pack

Start with a pilot large enough to expose workflow defects, then expand only after reviewer agreement is acceptable:

- pilot: 40–60 answerable questions and 10–20 unanswerable controls;
- promotion pack: at least 150 reviewed questions, increasing the count until each release-critical query class has enough examples to inspect separately;
- required classes: direct fact, exact name/quotation/acronym, speaker position, date-bounded fact, multi-passage within one episode, cross-episode comparison, temporal evolution, contradiction/qualification, episode summary, and unanswerable;
- include easy, medium, and hard questions and avoid near-duplicate phrasings split across development and holdout subsets.

For each answerable query, record the exact question, class, relevance grades 0–3, expected speaker(s), date range where applicable, acceptable node types, and notes identifying the minimum direct evidence needed. Grade all independently useful passages for multi-passage questions rather than only the first match.

### 1.3 Review and lock judgments

1. Have a human reviewer verify every grade against transcript evidence, speaker identity, and episode date.
2. Double-review a representative sample and all release-critical, contradiction, and temporal questions.
3. Resolve disagreements and record the adjudication rule.
4. Freeze a development set for tuning and a holdout set for final promotion. Do not tune against the holdout.
5. Change records to `judged` only after review; retain unanswerable records with an empty relevance map for abstention diagnostics.

### 1.4 Record the baseline

1. Import `page-content-v1` using the existing `BAAI/bge-large-en-v1.5` retrieval model in a separately named index.
2. Capture top-100 ranked results, even if user-facing top-k is smaller, so later rerankers receive identical candidate pools.
3. Run the repository evaluator and campaign-bound export.
4. Publish the aggregate, per-class, and per-query failure report in RAGScope.
5. Tag release-critical failures: missed evidence, wrong speaker/date, wrong hierarchy level, duplicate pressure, vocabulary mismatch, or insufficient corpus evidence.

### Exit gate

- The corpus release and evaluation pack are immutable and fingerprinted.
- All release-critical queries are reviewed, and the planned sample of other queries is adjudicated.
- The dense baseline is reproducible and has complete metadata/latency capture.
- Failures are assigned to actionable query classes.

### LLM automation boundary

**LLM-assisted:** propose diverse questions from transcript passages; generate paraphrases; identify candidate positive and hard-negative document IDs; prefill likely speakers/dates/node types; summarize disagreement clusters; draft failure-taxonomy labels and reports.

**Must remain human:** approve query usefulness; verify that a question is genuinely answerable; confirm speaker/date attribution; assign final relevance grades; distinguish contradiction from nuance; approve unanswerable controls; freeze the pack and promotion policy. LLM suggestions must retain evidence links and never become ground truth without review.

**Better as deterministic automation, not an LLM:** stable-ID extraction, hashes, schema validation, leakage/duplicate checks, metric calculation, run binding, and regression comparison.

## Step 2 — Enable lexical fields and dense/sparse hybrid retrieval

### 2.1 Complete producer verification

1. Backfill schema 2.1 representations for the chosen release without rerunning summarization.
2. Export the representation corpus and require 100% non-empty display, dense, and lexical fields for eligible documents.
3. Audit `lexical_text` samples for exact names, acronyms, quotations, episode titles, ISO dates, speaker names, topic tags, and position-card claim fields.
4. Add aliases only from an approved deterministic alias table; do not allow generated aliases to alter authoritative metadata.

### 2.2 Implement downstream hybrid retrieval

1. In Chroma DB Import, preserve the dense index and add a BM25-compatible lexical side index over `lexical_text`.
2. Bind both indexes to the same corpus release, stable IDs, representation fingerprints, and filters.
3. In PodCast Chat/RAGScope, retrieve separate dense and lexical candidate lists and fuse ranks with Reciprocal Rank Fusion (RRF).
4. Preserve speaker/date/podcast filters in both channels before fusion.
5. Deduplicate stable document IDs and cap repeated hierarchy nodes or near-identical evidence.
6. Capture at least these profiles on the same query pack: dense only, lexical only, dense+lexical RRF, and hybrid with diversity/MMR.
7. Tune candidate counts and RRF constant on the development set; run the selected configuration once on holdout.

### Exit gate

- Exact-term/quotation Recall@20 and overall nDCG@10 improve over dense-only.
- Speaker/date constraint accuracy and direct-evidence coverage do not regress beyond the approved tolerance.
- Index size and p95 latency stay within budget.
- The dense-only fallback remains available.

### LLM automation boundary

**LLM-assisted:** find likely aliases and spelling variants for human approval; label failure patterns such as vocabulary mismatch; explain per-query hybrid gains/losses.

**Must remain human:** approve aliases, promotion tolerances, and regressions on sensitive speaker/date queries.

**Better as deterministic automation:** lexical normalization, BM25 indexing, filters, RRF, deduplication, MMR, sweep execution, metrics, and artifact binding.

## Step 3 — Add a reranking stage downstream

### 3.1 Implement a bounded reranker

1. Start with a local, revision-pinned cross-encoder. Do not start with an LLM reranker.
2. Feed it the same top 30, 50, and 100 fused candidates to measure the quality/latency curve.
3. Rerank query-to-retrieval text, but keep `page_content` and evidence metadata for citation and display.
4. Preserve hard filters; a reranker may reorder eligible candidates but may not restore filtered-out evidence.
5. Add deterministic tie-breaking, timeouts, maximum text length, batch size, and dense/hybrid fallback.
6. Record pre-rerank rank, reranker score, post-rerank rank, truncation, model revision, and latency for every result.
7. Compare dense, hybrid, dense+reranker, and hybrid+reranker. Do not attribute a hybrid gain to the reranker or vice versa.

### 3.2 Tune context selection separately

After ranking, test direct-evidence preference, duplicate suppression, hierarchy quotas, and source diversity as separate profiles. Do not hide context-packing changes inside a reranker experiment.

### Exit gate

- nDCG@10/MRR@10 and direct-evidence coverage improve on holdout.
- Exact-term and speaker/date slices do not regress materially.
- p95 latency and local memory use stay within the declared budget.
- Timeout or model failure reproduces the promoted non-reranked result list.

### LLM automation boundary

**LLM-assisted:** analyze false promotions/demotions; create candidate training pairs after human judgments exist; optionally act as an offline diagnostic judge, calibrated against reviewers.

**Must remain human:** decide whether relevance/latency tradeoffs are acceptable and review serious attribution regressions. An LLM reranker should not be promoted without a separate determinism, cost, and calibration study.

**Better as deterministic/model-serving automation:** cross-encoder inference, batching, truncation, fallbacks, traces, parameter sweeps, and paired scoring.

## Step 4 — Evaluate a modern multi-function embedding model

### 4.1 Compare dense retrieval first

1. Pin a specific model and immutable revision; BGE-M3 is the comparison's first candidate.
2. Build a new retrieval index from the same `page-content-v1` export. Do not mix vector spaces or overwrite the existing BGE index.
3. Keep chunking, hierarchy, query pack, candidate count, filters, fusion, reranker, and context selection fixed.
4. Compare `bge-large-en-v1.5` dense against BGE-M3 dense, first without reranking and then with the already selected reranker.
5. Record dimension, precision/quantization, device, batch size, build time, query latency, peak memory, and index size.

This experiment concerns downstream retrieval embeddings. Do not rebuild the podcast hierarchy merely to change the query index model; the pipeline also uses its embedding setting for semantic clustering, which would confound the comparison.

### 4.2 Consider sparse and multi-vector modes conditionally

1. Compare BGE-M3 learned sparse retrieval with the BM25 baseline only after dense comparison is complete.
2. Fuse dense plus learned sparse and compare it with dense plus BM25.
3. Consider BGE-M3 multi-vector or ColBERT-style retrieval only if hybrid plus cross-encoder reranking still has a measured fine-grained recall gap and storage/engine changes are affordable.

### Exit gate

- Promote a dense replacement only if holdout quality improves with acceptable resource cost and no important query-class regression.
- Promote learned sparse only if it beats or materially complements BM25.
- Do not promote multi-vector retrieval without a unique measured gain that justifies its index and serving cost.

### LLM automation boundary

**LLM-assisted:** summarize qualitative differences and cluster failures by conversational phenomenon or terminology.

**Must remain human:** choose acceptable hardware/cost tradeoffs and approve a full re-indexing migration and rollback plan.

**Better as deterministic automation:** model downloads/revision verification, embedding, isolated index builds, benchmark execution, resource telemetry, and statistical comparison.

## Step 5 — Test contextual and late-chunked representations

### 5.1 Test the already implemented contextual header

1. Export `context-header-v1` for the same corpus without changing `page_content` or stable document IDs.
2. Build a separate index using the currently promoted embedding model.
3. Compare it with `page-content-v1` on pronoun/callback, speaker-position, topic-transition, and exact-quotation slices.
4. Inspect whether contextual metadata helps disambiguation or causes neighboring chunks and summary nodes to become less distinct.
5. Tune only the bounded header fields/length on development data; preserve a fixed versioned format for holdout.

### 5.2 Test true late chunking only if a gap remains

1. Complete an `advanced-retrieval-entry-gate-1.0` naming the failures that contextual headers, hybrid retrieval, and reranking did not solve.
2. Select a long-context encoder/tokenizer with immutable revisions and define episode/window boundaries. Do not cross episode boundaries.
3. Use the current `late-chunk-alignment-1.0` sidecar as alignment input, but independently validate its tokenizer offsets against the selected model tokenizer.
4. Implement long-context encoding plus token pooling into retrieval chunks downstream or in a dedicated embedding builder. The current repository sidecar does not contain vectors.
5. Preserve stable document IDs, source spans, and citation text; record truncation and uncovered spans.
6. Compare page-content, contextual-header, and late-chunk indexes with identical retrieval/reranking settings.

### Exit gate

- Contextual or late-chunk representation improves the targeted contextual-reference slice and overall evidence retrieval without degrading quotations, speaker/date constraints, or citation traceability.
- Build time, peak memory, index size, and incremental update cost fit the operating budget.
- If the contextual header closes the gap, stop and do not build true late chunking.

### LLM automation boundary

**LLM-assisted:** identify likely pronouns, callbacks, ellipses, and transitions for reviewer sampling; explain contextual retrieval wins/losses.

**Must remain human:** confirm referents and judge whether added context preserves the speaker's meaning. Generated contextual summaries must not be inserted into evidence text without explicit review and separate provenance.

**Better as deterministic automation:** header generation, token alignment, pooling, coverage validation, separate indexing, and paired benchmarks.

## Step 6 — Add graph retrieval only after a demonstrated gap

### 6.1 Enforce the entry gate

Proceed only when the reviewed benchmark contains graph-positive cross-episode, temporal-evolution, contradiction, associative, or evidence-lineage queries for which promoted hybrid+reranking misses relevant evidence. Record concrete query IDs, simpler methods attempted, expected gain, budgets, minimum reviewed coverage, accepted baseline, and stop condition.

### 6.2 Harden the existing prototype before promotion

1. Generate the release-bound evidence graph from existing hierarchy, episode, topic, claim/evidence, speaker, and temporal links.
2. Add the graph-quality report described in `docs/graph-rag-design-augmentation.md`: dangling/stale edges, evidence closure, degree/hub diagnostics, edge-type coverage, provenance, and lifecycle state.
3. Require approved stable speaker identities for cross-episode speaker edges.
4. Consume processed deltas and correction supersession so one changed episode cannot leave stale active paths.
5. Package the graph with the exact corpus/index release and fail closed to the promoted non-graph baseline on mismatch or corruption.

### 6.3 Evaluate bounded graph assistance

1. Use promoted dense/hybrid retrieval as the seed.
2. Route only eligible query classes to graph expansion.
3. Start with one hop, strict edge allowlists, fan-out/hub caps, unchanged hard filters, and a small candidate budget.
4. Fuse or rerank the expanded union; never present a graph-derived claim without a path to direct transcript evidence.
5. Capture seed ranks, traversed/rejected paths, graph-only discoveries, post-rerank ranks, packed evidence, stop reason, fallback, and latency.
6. Compare non-graph and graph-assisted profiles on identical releases. Promote independently by query class.

### Exit gate

Use the provisional graph gate already documented in the repository: at least a five-percentage-point Recall@10 or nDCG@10 improvement on the targeted graph-positive slice; robust paired gain; no more than one-point regression on direct factual retrieval; no loss of direct-evidence coverage or unsupported-answer performance; 100% hard-filter preservation; acceptable p95 latency; and a valid direct-evidence path for every accepted graph result.

If the gate fails, retain the sidecar as an experiment and stop graph investment. Do not introduce a graph database until portable sidecar size, load time, update cost, or traversal latency demonstrates a concrete storage-engine need.

### LLM automation boundary

**LLM-assisted:** propose claim relationships, query-class routing candidates, and graph-positive questions; summarize false paths; prioritize edge types for review.

**Must remain human:** approve cross-episode entity/speaker identity, contradictions and reversals, relationship lifecycle, graph-positive ground truth, and per-class promotion. LLM-proposed edges remain candidates and cannot become authoritative evidence.

**Better as deterministic automation:** stable node/edge IDs, graph construction from approved artifacts, traversal bounds, filter preservation, provenance validation, deltas, packaging, fallbacks, and metric capture.

## Execution backlog and ownership

| Order | Deliverable | Primary repository/system | Depends on | Completion evidence |
| --- | --- | --- | --- | --- |
| 0 | Versioned promotion policy and immutable corpus release | Cross-project/RAGScope | Approved private data access | Policy, release ID, fingerprints, budgets |
| 1 | Reviewed development and holdout evaluation pack | RAGScope, with Podcast-RAG IDs | Order 0 | Adjudicated judgments and review report |
| 2 | Reproducible BGE `page-content-v1` dense baseline | Chroma DB Import + RAGScope | Order 1 | Captured top-100 run and scored report |
| 3 | Verified schema 2.1 representation corpus | Podcast-RAG | Order 0 | Coverage/contract report |
| 4 | BM25 side index and RRF profiles | Chroma DB Import + PodCast Chat/RAGScope | Orders 2–3 | Dense/lexical/hybrid paired report |
| 5 | Cross-encoder reranking profiles | PodCast Chat/RAGScope | Order 4 | Quality/latency curve and fallback tests |
| 6 | BGE-M3 dense, then optional learned-sparse profiles | Chroma DB Import + RAGScope | Order 5 | Isolated index and holdout comparison |
| 7 | `context-header-v1` profile | Podcast-RAG export + downstream systems | Selected embedding/reranker | Context-slice paired report |
| 8 | True late-chunk profile, if entry gate passes | Dedicated embedding builder + downstream systems | Order 7 residual gap | Alignment coverage, index, paired report |
| 9 | Hardened bounded graph profile, if entry gate passes | All four ecosystem repositories | Promoted simpler baseline and graph-positive slice | Graph quality, trace, delta, and promotion reports |

## Immediate next actions

1. Obtain the approved private evaluation pack/corpus release; this is the current roadmap blocker.
2. Run representation validation and export `page-content-v1` plus schema 2.1 representation coverage for that exact release.
3. Replace the eight draft query templates with a reviewed pilot pack, retaining the template categories.
4. Capture the top-100 `bge-large-en-v1.5` dense baseline in RAGScope and publish per-class failures.
5. Open downstream work items for BM25 indexing/RRF and cross-encoder reranking; no additional Podcast-RAG representation feature is required before those experiments.
6. Defer BGE-M3, true late chunking, and graph hardening until the prior stage produces a specific measured failure hypothesis.

The fastest legitimate use of an LLM is to reduce evaluation-authoring and diagnosis effort while keeping humans responsible for truth and promotion. The largest engineering savings outside that workflow will come from conventional deterministic automation—not from asking an LLM to perform indexing, fusion, reranking inference, hashing, or metric computation.
