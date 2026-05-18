# Roadmap

This roadmap captures practical feature upgrades for `Podcast-RAG-pipeline`, the preprocessing stage that converts speaker-labeled transcript JSON into reusable processed RAG documents.

## Highest-Impact Improvements

Implementation status: all roadmap areas below have an initial implementation in the current codebase. The work adds processed-cache schema validation, intra-file checkpoints, prompt/cache manifests, deterministic fallback improvements, position-card quality checks, topic tags, grouping/reduction options, run reports, snapshots, cache inspection, config doctor, model evaluation, fake-LLM validation mode, and focused unit tests. Future work should tune thresholds against real batches and expand fixtures as more failure examples are collected.

- Add a formal processed-cache schema and validator. Every cache should be checked for required metadata, valid speaker/date fields, non-empty text, valid parent/child links, unique `node_id` values, and expected `node_type` coverage.
- Add resumability within a file. The pipeline currently resumes at the completed-file/cache level; long episodes would benefit from persisting completed leaf chunks, cluster summaries, and position extraction batches as checkpoints.
- Add a model-evaluation harness for comparing LM Studio models on the same transcript slices. Score missing-context responses, malformed JSON, compression quality, speaker attribution, position-card quality, token usage, and throughput.
- Add a prompt/version manifest to each processed cache. This should record pipeline version, config fingerprint, model name, prompt templates, token maxima, processing time, and fallback counts.
- Add deterministic fallback quality improvements. When LLM reduction stalls or returns malformed output, the fallback should preserve speaker/date/time metadata and produce concise retrieval-oriented bullets rather than generic clipped text.

## Retrieval-Oriented Data Quality

- Add richer `position_card` quality checks: speaker must be attributable, claim must be substantive, evidence node IDs must exist, date must be populated when available, and counterpoints/supporting evidence should be normalized to strings.
- Add explicit topic tags for cluster summaries and position cards using deterministic keyword extraction, optional LLM extraction, or both.
- Add duplicate and near-duplicate summary detection so repeated overlapping leaf chunks do not overpopulate higher levels.
- Add optional quote/evidence excerpts to position cards to help downstream chat produce cited answers.
- Add confidence flags for fallback-generated summaries versus model-generated summaries so retrieval consumers can treat them differently.

## Clustering And Summarization

- Make dimensionality reduction configurable. Current clustering uses PCA before HDBSCAN; consider adding optional UMAP back in for semantic neighborhood quality comparisons.
- Add cluster-quality telemetry: number of clusters, noise rate, cluster sizes, duplicate rate, and summary compression ratios at every level.
- Add alternate grouping modes for long episodes: chronological windows, speaker-first grouping, topic-first clustering, and hybrid topic/time grouping.
- Add a maximum prompt-token planner that estimates prompt size before calling the LLM and splits work earlier when a cluster is likely to exceed the configured context window.
- Add configurable summary objectives, such as speaker-belief extraction, timeline extraction, topic synopsis, or neutral episode overview.

## Operations And Observability

- Add a structured run report at the end of each batch, saved as JSON and Markdown. Include file counts, elapsed time, failures, fallbacks, token maxima, model throughput, document counts, position-card counts, and skipped cached files.
- Add periodic run-state snapshots that can be consumed by a future UI or dashboard.
- Add clearer debug-output classification: missing-context, empty response, malformed JSON, reduction stall, context overflow, retry exhaustion, and deterministic fallback.
- Add a cache inspector command for summarizing existing `processed_data` without reprocessing.
- Add a config doctor that checks LM Studio model identity, context length, max token settings, prompt budget, and expected throughput before a batch starts.

## Pipeline Integration

- Share executable schema tests with `podcast-host-transcription-pipeline`, `Chroma DB Import`, `PodCast Chat`, and `RAGScope`.
- Emit import-ready manifests so `Chroma DB Import` can detect changed caches, changed models, and changed prompt versions.
- Record source transcript fingerprint and source transcript schema version in every processed cache.
- Add stable document IDs that can survive path moves when the transcript content is unchanged.

## Testing And Quality

- Add unit tests for prompt splitting, fallback summaries, position parsing, token tracking, processed-cache validation, and state transitions.
- Add a small transcript fixture that exercises single-speaker, multi-speaker, missing date, long episode, malformed LLM response, and fallback paths.
- Add regression tests using stored debug-output examples so previously observed model failure modes stay fixed.
- Add a no-LM Studio test mode that uses deterministic fake model responses for CI-style validation.
