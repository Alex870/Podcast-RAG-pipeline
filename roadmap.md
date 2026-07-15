# Roadmap

`Podcast-RAG-pipeline` converts speaker-labeled transcripts into structured, RAG-ready processed caches. The next stage is a reproducible processing and evaluation system that can use stronger local models when available while preserving deterministic, compatible output for every supported runtime.

## Principles

- Keep the baseline workflow usable on modest local hardware.
- Treat model, context, and backend as discovered capabilities, not GPU-brand profiles.
- Make artifacts self-describing, versioned, and backward-readable.
- Prefer measured retrieval and attribution gains over larger prompts.
- Never silently accept malformed structured output into a processed cache.

## Current Foundation

- Speaker-aware ingestion, hierarchical documents, topic indexes, manifests, checkpoints, validation, and run reports.
- Config, runtime, schema, LLM-support, and test modules.
- Processed-cache schema `2.1` with backward-readable `2.0` fixtures and versioned display, dense, and lexical representations.
- Deterministic contextual-header and lexical-text builders that do not alter display text or stable document IDs.
- A dependency-light retrieval evaluation runner with JSONL judgments, ranked-result input, standard metrics, and JSON/Markdown reports.

## Completed Foundation Work

- The pipeline-side contract and compatibility portion of Priorities 1 and 2 is implemented.
- Retrieval query templates and scoring infrastructure are implemented; real corpus judgments still require human authoring and review.
- Retrieval-ready contextual and lexical fields are emitted additively. Dense retrieval remains on `page-content-v1` by default.
- Cross-project consumption by Chroma DB Import, PodCast Chat, and RAGScope remains follow-on work.

## Priority 1: Contracts And Provenance

- Adopt the versioned processed-cache contract and golden fixtures in the importer, chat client, and RAGScope.
- Record source transcript hash, pipeline/config fingerprint, prompt version, model/backend, and deterministic-versus-LLM stage provenance.
- Preserve per-node evidence links: episode, timestamps, speaker, parent/child IDs, and source spans.
- Make schema migration explicit and test old-cache read compatibility.
- Quarantine or retry invalid structured output; permissive JSON recovery is diagnostic-only.

## Priority 2: Evaluation Before More Synthesis

- Replace the draft query templates with a human-reviewed podcast query set containing graded chunks and speaker/date constraints.
- Capture comparable dense, hybrid, and reranked result runs from downstream retrievers.
- Extend current Recall@k, MRR, nDCG, speaker/date, node-type, and evidence-coverage reports with source diversity and answer-grounding evaluation.
- Evaluate leaf retrieval, summaries, position cards, and temporal synthesis separately.
- Run chunking, overlap, hierarchy-depth, prompt-budget, and model ablations.
- Calibrate any LLM judge against human labels; it is a measurement aid, not ground truth.

## Priority 3: Capability-Based Local Inference

- Use a capability record: effective context, structured output, concurrency, latency, and loaded-model identity.
- Support LM Studio and OpenAI-compatible servers first; add vLLM-specific optimizations only when detected.
- Use native structured output with schema validation and bounded retries.
- Budget prompts from actual context limits and reserve completion tokens explicitly.
- Profile prefix-stable prompts and batching before making them defaults.

## Priority 4: Evidence-Preserving Processing

- Add claims, evidence IDs, confidence, ambiguity, and contradiction candidates to position cards.
- Preserve speaker and date attribution through every hierarchy layer.
- Tune direct-evidence window sizes against the evaluation set, not intuition.

## Priority 5: Operations And Tests

- Emit throughput, token, retry, fallback, schema-failure, and cache-reuse telemetry.
- Add malformed-output, context-overflow, contradictory-evidence, and old-cache fixtures.
- Add end-to-end contract tests with Chroma DB Import and RAGScope.

## Sequencing

1. Adopt schema `2.1`, representation selection, and compatibility fixtures downstream.
2. Author and review the first real judged query set.
3. Capture a `page-content-v1` dense baseline.
4. Implement and compare lexical fusion and reranking downstream.
5. Add capability discovery and strict structured output.
6. Run representation, chunking, and hierarchy ablations.
7. Promote only measured improvements to defaults.
