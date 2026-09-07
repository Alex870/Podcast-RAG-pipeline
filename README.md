# Podcast RAG Pipeline

This project builds pre-processed podcast RAG documents from JSON transcript files produced by `podcast_transcribe_host.py` (in the [Podcast-Host-Transcription-Pipeline](https://github.com/Alex870/Podcast-Host-Transcription-Pipeline) repository). It creates leaf chunks, RAPTOR-style rollup summaries, episode thesis summaries, and durable position cards, then saves them as processed JSON caches.

It also derives a lightweight topic index from those processed caches so downstream tools such as `PodCast Chat` can browse recurring themes, estimate topic depth, and ask evolution-over-time questions without re-running the full expensive transcript pipeline.

The output of this process is then loaded into a Vector database in the `Chroma DB Import` project. This division keeps expensive LLM preprocessing separate from Chroma database rebuilds.

The default workflow targets a local LM Studio server running on Windows 11 using LM Studio's OpenAI-compatible API. The included example config defaults to `http://127.0.0.1:1234/v1` with `mistral-small-3.2-24b-instruct-2506`, but both values are configurable.

The shared transcript, processed-cache, Chroma metadata, and `podcast.json` expectations are documented in [`docs/podcast_pipeline_contract.md`](docs/podcast_pipeline_contract.md).

Managed podcasts and work-meeting corpora use isolated processing partitions and the upstream manifest handoff described in [`docs/partition-management.md`](docs/partition-management.md) and the root [`transcription-handoff-contract.md`](transcription-handoff-contract.md).

## Clean-machine packaging

The Windows environment file installs the exact direct dependency pins in `podcast_rag_requirements.txt`. Use `scripts/Test-PodcastRagEnvironment.ps1` before running delta processing, migration, or benchmarks; missing Conda or model-provider access is reported as a prerequisite diagnostic rather than hidden.

## Local Versus Cloud

For overnight batch processing, local LM Studio processing is the sensible default. A cloud RTX 6000 Pro with 96 GB VRAM may let you run a larger model, a longer context, or more concurrent work, but this pipeline is designed to reduce long transcripts into bounded chunks and summaries. Unless you have a specific larger model in mind that materially improves the summaries, the expected quality gain is probably smaller than the operational cost for routine batches.

## Batch Performance Snapshot

The current local batch benchmark used LM Studio with `mistral-small-3.2-24b-instruct-2506` on an RTX 5070 Ti. Across 19 completed processed-data caches from one batch run, the pipeline processed about 88.4 hours of podcast audio in 16.8 hours of recorded processing time, or about 11.4 minutes of processing per podcast hour. That is roughly 5.3x faster than real time for this workload.

This benchmark is based on `state/podcast_rag_state.json` elapsed seconds and episode durations inferred from `leaf_chunk` start/end metadata in `processed_data`. The same run produced 9,991 documents and 379 position cards. The maximum observed request size was 3,736 total tokens, so a minimum LM Studio context length of 4,096 tokens is advisable for this workload.

## Repository Layout

- `src/podcast_rag/`: package source, split into config, runtime, transcript ingestion, cache/state IO, orchestration, and CLI modules
- `Run Podcast RAG Pipeline.ps1`: root bootstrap launcher for the most common setup, validation, control, and run actions
- `scripts/`: PowerShell launchers and diagnostics for Windows-first operation
- `scripts/Migrate-LegacyPodcastRagState.ps1`: guided migration assistant for importing config, caches, state, and repo-local input data from a legacy working directory
- `examples/`: editable runtime configuration template
- `docs/`: schema and architecture notes
- `tests/`: focused unit coverage for schema validation and helper behavior
- `podcast_rag_pipeline.py`: compatibility wrapper for `python .\podcast_rag_pipeline.py`
- `podcast_rag_requirements.txt` and `environment.yml`: Python environment definitions

## Architecture

- `config.py`: runtime settings, environment overrides, and config fingerprints
- `runtime.py`: lazy imports, stop control, live concurrency control, and run telemetry
- `transcript.py`: transcript JSON normalization and episode metadata extraction
- `text_utils.py`: deterministic text, date, token, and fallback helpers shared across stages
- `state.py`: durable state, checkpoints, processed-cache IO, and run-report writing
- `topics.py`: cache-only topic contribution extraction, delta tracking, and aggregate topic index generation
- `pipeline.py`: the `PodcastRagPipeline` orchestration class for chunking, summarization, clustering, and position extraction
- `cli.py`: batch execution, cache inspection, config doctor, and model evaluation entry points
- `representations/`: deterministic dense and lexical retrieval text builders with versioned manifests
- `evaluation/`: dependency-light judged query loading, retrieval metrics, and JSON/Markdown reports

## Retrieval Representations

Processed cache schema `2.1` keeps `page_content` as the display and citation source while adding optional `embedding_text` and `lexical_text` fields. The default dense representation remains `page-content-v1`, so upgrading does not silently change existing embedding behavior. `normalized-lexical-v1` preserves source wording and adds searchable episode, date, speaker, topic, claim, and keyword metadata for downstream lexical indexes.

Set `embedding_text_mode` to `context-header-v1` for an experimental deterministic header before the source text. This representation should be evaluated against the default before it is used for a production database.

Schema `2.0` caches remain supported.

## Retrieval Evaluation

The repository includes a versioned JSONL query-set format and computes Recall@5/10/20, MRR@10, nDCG@10, evidence coverage, speaker/date constraint measurements, and node-type composition from captured ranked results. The included query set is a draft template rather than fabricated ground truth; replace it with real questions and stable document relevance judgments.

```powershell
python .\podcast_rag_pipeline.py `
  --retrieval-eval `
  --retrieval-results .\path\to\captured_results.json `
  --query-set .\evaluation\query_sets\podcast-baseline-v1.jsonl
```

The expected captured-results shape is shown in `examples/retrieval_results.example.json`. Reports are written as JSON and Markdown under `evaluation/results` by default. Retrieval execution remains the responsibility of Chroma DB Import, PodCast Chat, or RAGScope; this command scores a captured run without requiring Chroma or LM Studio. See [`docs/retrieval-evaluation.md`](docs/retrieval-evaluation.md) for query authoring, result capture, metrics, and report interpretation.

The pipeline now emits the fields needed for downstream contextual and lexical retrieval, but existing versions of Chroma DB Import and PodCast Chat must be upgraded separately before they use those fields. Until then, their established `page_content` dense-vector workflow remains unchanged.

For an additive upgrade of existing processed caches, use `--backfill-representations`; it reuses valid hierarchy and position outputs and writes deterministic provenance/representation fingerprints without calling the LLM. Use `--export-dense-baseline` to emit a downstream-ready `page-content-v1` export containing stable IDs, dense text, citation text, metadata, and source-cache manifests.

New configuration fields:

| Field | Default | Purpose |
|---|---|---|
| `embedding_cache_dir` | `""` | Optional explicit cache directory; empty uses the default Hugging Face/sentence-transformers cache. |
| `embedding_local_files_only` | `true` | Prevents embedding startup from contacting Hugging Face; requires the model to already be cached. |
| `embedding_text_mode` | `page-content-v1` | Selects plain display text or experimental `context-header-v1` for the emitted dense representation. |
| `lexical_text_mode` | `normalized-lexical-v1` | Selects the emitted lexical representation; use `page-content-v1` to disable metadata enrichment. |
| `contextual_header_max_chars` | `700` | Bounds the deterministic header when contextual dense text is enabled. |
| `retrieval_evaluation_query_set` | `evaluation/query_sets/podcast-baseline-v1.jsonl` | Default judged-query JSONL path. |
| `retrieval_evaluation_output_dir` | `evaluation/results` | Default JSON and Markdown evaluation-report directory. |
| `hierarchy_algorithm_version` | `adaptive-v2` | Version of adaptive hierarchy rollup policy. |
| `hierarchy_min_parent_docs` | `2` | Minimum current-node count required to form a parent. |
| `hierarchy_max_dominant_cluster_fraction` | `0.60` | Maximum accepted semantic dominant-cluster fraction. |
| `hierarchy_max_noise_rate` | `0.25` | Maximum accepted semantic noise rate. |
| `hierarchy_summary_cluster_size_divisor` | `12` | Divisor used to adapt HDBSCAN cluster size for summary levels. |
| `hierarchy_summary_min_cluster_size` | `3` | Minimum HDBSCAN cluster size for summary levels. |
| `hierarchy_summary_max_cluster_size` | `6` | Maximum HDBSCAN cluster size for summary levels. |
| `hierarchy_summary_min_samples` | `2` | Explicit HDBSCAN `min_samples` for summary levels. |
| `hierarchy_fallback_mode` | `chronological` | Safe deterministic grouping strategy after a semantic quality-gate failure. |

## First-Time Setup

1. Start LM Studio and load your model.
2. Enable the local OpenAI-compatible server in LM Studio.
3. Create the Conda environment:

```powershell
.\Run Podcast RAG Pipeline.ps1
```

Choose `6` to create or refresh the Conda environment. The underlying launcher uses the `podcast-rag-pipeline` Conda environment by default. Run the script in `scripts` directly with `-CondaEnvName` if you want a different name.

4. Copy the example config:

```powershell
Copy-Item .\examples\podcast_rag_config.example.json .\podcast_rag_config.json
```

5. Put transcript JSON files under `data`, or point `input_dir` at the output directory from your transcription project.

## Running

Before a batch run, verify the local environment:

```powershell
.\Run Podcast RAG Pipeline.ps1
```

Choose `5` for environment validation.

To start the main batch pipeline:

```powershell
.\Run Podcast RAG Pipeline.ps1
```

Choose `1` to process or resume a managed partition. The menu displays partitions by friendly name, shows handoff and episode status, and reuses valid processed caches and within-file checkpoints automatically. Choose `8` for the legacy flat-input workflow.

The interactive menu remains open after each operation. Choose `2` for the partition management center, where you can create partitions with a friendly name, inspect status, select the active partition, archive or restore partitions, validate setup, and control a running partition. Managed stop requests and concurrency settings are written to the selected partition's own `state` directory.

The main pipeline launcher still creates `podcast_rag_config.json` from the example if needed, applies optional command-line overrides, checks Python dependencies, and runs the pipeline. Existing direct `-Action` calls remain available for automation.

For a direct Conda invocation, use live-streaming and unbuffered Python so
progress is displayed as it is produced rather than when the process exits:

```powershell
conda run --live-stream -n podcast-rag-pipeline python -u .\podcast_rag_pipeline.py process --partition <partition-id>
```

`--live-stream` is Conda's alias for `--no-capture-output`. The supplied
PowerShell launchers already use that mode, and the pipeline also configures
line-buffered output for direct entry-point calls.

To build or refresh the topic index from already-processed caches without re-running the main pipeline:

```powershell
.\Run Podcast RAG Pipeline.ps1
```

Choose `7` and then `2` for `Build or refresh the topic index`.

That topic refresh path is intentionally incremental. It scans the existing `processed_data` caches, rebuilds only new or changed per-episode topic contributions, and updates the aggregate `state/topic_index.json` catalog without redoing the expensive transcript summarization work.

If you want the topic refresh to run an extra cache-only cleanup pass over ambiguous topic rows, set `enable_llm_topic_label_curation` to `true` in `podcast_rag_config.json` or run:

```powershell
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json --build-topic-index --curate-topic-labels
```

That pass persists its decisions in `state/topic_label_blacklist.json` and `state/topic_label_whitelist.json`, so future topic-index rebuilds can reuse them without re-asking the model on every run.
Each rebuild also writes `state/topic_label_curation_report.json`, which shows what got filtered by deterministic rules, what the optional LLM pass explicitly kept or dropped, and which topics survived into the final index.

If you are starting from a fresh pull but already have an older working directory with expensive caches and state, the bootstrap also exposes a migration assistant:

```powershell
.\Run Podcast RAG Pipeline.ps1
```

Choose `9` for `Migrate settings and state from a legacy directory`.

If you prefer to skip the menu, the root launcher also supports direct actions:

```powershell
.\Run Podcast RAG Pipeline.ps1 -Action Debug
.\Run Podcast RAG Pipeline.ps1 -Action Run
.\Run Podcast RAG Pipeline.ps1 -Action CacheCheck
.\Run Podcast RAG Pipeline.ps1 -Action SetControl -MaxParallelModelRequests 2
.\Run Podcast RAG Pipeline.ps1 -Action CreateStopFile
.\Run Podcast RAG Pipeline.ps1 -Action ClearStopFile
.\Run Podcast RAG Pipeline.ps1 -Action CreateCondaEnv
.\Run Podcast RAG Pipeline.ps1 -Action BuildTopicIndex
.\Run Podcast RAG Pipeline.ps1 -Action Migrate
```

The underlying PowerShell scripts remain available in `scripts\` when you want to bypass the menu and call a specific launcher directly.

For backwards compatibility, `-Action Run` remains the direct legacy
flat-input action.  Managed partition processing is available from the
interactive menu or through the underlying `-Managed -Partition` parameters
shown below.

### Hugging Face embedding cache

The embedding model is loaded from the local sentence-transformers/Hugging Face
cache and `embedding_local_files_only` defaults to `true`. This means normal
pipeline starts do not contact Hugging Face after the model has been cached.
The default empty `embedding_cache_dir` keeps the library's normal per-user
cache; set it to a local path when you want an explicit cache location.

On a fresh machine, warm the cache once while network access is available:

```powershell
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json --cache-embedding-model
```

That explicit cache command is the only normal operation that is expected to
contact Hugging Face. If the model is missing while offline mode is enabled,
the pipeline stops with an actionable message instead of reaching out to the
Hub.

Useful launcher parameters:

```powershell
.\scripts\Run-PodcastRagPipeline.ps1 -Managed -Partition "podcast-history"
.\scripts\Run-PodcastRagPipeline.ps1 -Managed -Partition "podcast-history" -OneFile
.\scripts\Run-PodcastRagPipeline.ps1 -Managed -Partition "podcast-history" -Episode "episode-01"
.\scripts\Run-PodcastRagPipeline.ps1 -Managed -Partition "podcast-history" -Episode "episode-01" -ForceReprocess
.\scripts\Run-PodcastRagPipeline.ps1 -InputDir "C:\path\to\transcripts"
.\scripts\Run-PodcastRagPipeline.ps1 -Model "mistral-small-3.2-24b-instruct-2506"
.\scripts\Run-PodcastRagPipeline.ps1 -BaseUrl "http://127.0.0.1:1234/v1"
.\scripts\Run-PodcastRagPipeline.ps1 -MaxParallelModelRequests 2
.\scripts\Run-PodcastRagPipeline.ps1 -CreateStopFile
.\scripts\Run-PodcastRagPipeline.ps1 -ClearStopFile
.\scripts\Run-PodcastRagPipeline.ps1 -CondaEnvName "podcast-rag-pipeline"
```

## Live Tuning

At startup, the pipeline initializes `state/pipeline_control.json` from `max_parallel_model_requests` in `podcast_rag_config.json` or the `-MaxParallelModelRequests` launcher override. While a batch is running, change how many new model requests can run in parallel with:

```powershell
.\Run Podcast RAG Pipeline.ps1 -Action SetControl -MaxParallelModelRequests 1
.\Run Podcast RAG Pipeline.ps1 -Action SetControl -MaxParallelModelRequests 3
```

The new value is applied before the pipeline launches additional model requests. Already-running LM Studio requests are allowed to finish.

## Stopping After The Current File

The pipeline checks for `state/stop_after_current.txt` between files and watches `Ctrl+C` while work is running. To request a clean stop from another PowerShell window, create the stop file while the batch is running:

```powershell
.\Run Podcast RAG Pipeline.ps1 -Action CreateStopFile
```

On `Ctrl+C`, the pipeline stops launching new model requests, waits for in-flight request(s) to finish, saves state, and exits. A partially processed file is marked `interrupted` and will be retried on the next run. Completed files are skipped using `state/podcast_rag_state.json`.

The stop file is intentionally left in place so the request is visible. Remove it before the next full run:

```powershell
.\Run Podcast RAG Pipeline.ps1 -Action ClearStopFile
```

## State And Resume

Progress is tracked in `state/podcast_rag_state.json`. Completed files are skipped on later runs using a stable fingerprint derived from file path, size, and modification time. If a file changes, it is treated as new work.

Processed document caches are stored in `processed_data` using the same file fingerprint. When a matching cache exists, the pipeline validates it and skips LLM processing for that transcript. If every pending file has a cache, LM Studio model verification is skipped because no model generation is needed.

The topic index is intentionally built on top of those processed caches. You can backfill topic coverage for a library that already took days to preprocess by running a cheap cache-only topic refresh instead of rerunning LM Studio summarization for every episode. Per-episode topic contributions are stored in `state/topic_contributions`, the aggregate catalog lives at `state/topic_index.json`, and delta tracking lives at `state/topic_index_manifest.json`.

Topic-label curation lives beside them in `state/topic_label_blacklist.json` and `state/topic_label_whitelist.json`. Deterministic filtering removes obvious junk like stopwords and speaker placeholders first, and the optional LM Studio curation pass can then persist decisions for the harder borderline labels.

The migration assistant is designed around the same principle. It can copy forward `podcast_rag_config.json`, `processed_data`, configured state artifacts, processed-input archives, optional debug outputs, and repo-local input transcripts from a legacy repo without making you rebuild the expensive caches. When the legacy config used absolute paths inside the old repository tree, the migrator rewrites those values to portable repo-relative paths in the new `podcast_rag_config.json`.

To insert or reinsert processed caches into Chroma, use the separate `Chroma DB Import` project.

New processed caches use schema version `2.1`; schema `2.0` remains readable. Each cache includes a prompt/version manifest, representation manifest, config fingerprint, model and embedding names, source transcript fingerprint and schema version, stable document IDs, cluster telemetry, fallback counts, token maxima, validation counts, and an import manifest for downstream Chroma import.

The hierarchy settings and prompt/pipeline versions participate in the
generation and processing fingerprints. Changing them creates a new versioned
cache key; existing cache files remain available for rollback or comparison.
The managed force command bypasses the selected canonical cache and normal
checkpoints without changing cache identity:

```powershell
python .\podcast_rag_pipeline.py process --partition <partition-id> --episode <episode-id> --force-reprocess
```

Before the rebuild, the command writes `state\reprocess_backups\<run-id>\`
with the prior cache and errata pair when present, original paths, SHA-256
checksums, artifact presence/absence, prior state metadata, and replacement
status. Replacement files are written atomically; failed promotion restores
the previous canonical artifacts. Diagnosis is advisory only: its allowed
actions are `code`, `config`, `data`, `rerun`, and `human_review`, and failed
diagnoses retain bounded errors, invalid action values by index, digests, and a
bounded response excerpt—not a raw or unbounded model response.

The upstream contract reader accepts both `correction-manifest-v1` and `correction-manifest-v2`. V2 approvals expose stable correction and affected-span identities while preserving source-hash and before-value validation. When the transcription project is configured with this repository path, notifications arrive under `state/transcription_corrections`; these are inputs to the processed-delta workflow, not a reason to rerun unrelated episodes.

Use the notification-aware planner after producing candidate replacement documents for the corrected episode:

```powershell
podcast-rag-delta plan-notification-delta `
  --project-root . `
  --old .\state\old-documents.json `
  --new .\state\new-documents.json `
  --parent-corpus-id corpus-current `
  --processing-fingerprint processing-current `
  --representation-fingerprint representation-current `
  --output .\state\correction-delta.json
```

The command requires exactly one validated ready notification unless `--correction-set-id` selects one explicitly. It scopes the delta to affected episodes/source spans and refuses unrelated old/new corpus drift rather than hiding it in a correction run.

Within a single file, completed `leaf_chunks`, `hierarchy`, and `positions` stages are checkpointed under `state/file_checkpoints`. If a long episode is interrupted after an expensive stage, the next run can resume from the checkpoint instead of starting over. Set `resume_within_file` to `false` to disable this.

At the end of each batch, structured reports are written to `state/run_reports` as JSON and Markdown. A live dashboard-friendly snapshot is refreshed at `state/current_run_snapshot.json`.

By default, a successful batch also refreshes the topic index automatically. Set `auto_refresh_topic_index` to `false` if you want to keep topic aggregation as a separate step. The optional `podcast_id` and `podcast_name` config values let you override how this corpus is labeled inside the topic index when multiple podcast libraries are merged downstream.

To scan existing caches for missing-context LLM responses:

```powershell
.\Run Podcast RAG Pipeline.ps1 -Action CacheCheck
```

The built-in cache inspector gives a broader schema and document summary:

```powershell
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json --inspect-cache
```

Before a run, the config doctor checks model identity, context and prompt budget settings, parallelism, and the processed-cache schema:

```powershell
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json --config-doctor
```

For model comparison, the evaluation harness runs the configured model against the same transcript slices and records throughput, missing-context responses, compression ratio, and topic tags:

```powershell
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json --model-eval --model-eval-limit 3
```

For CI-style validation without LM Studio generation, use deterministic fake responses:

```powershell
python .\podcast_rag_pipeline.py --config .\examples\podcast_rag_config.example.json --config-doctor --fake-llm
```

Rejected LLM responses and fallback events are written to `debug_output` as JSON files containing the label, source prompt text, model response, and error reason. These files are ignored by Git.

For Qwen reasoning models in LM Studio, keep `llm_max_tokens` high enough for hidden reasoning plus final answer text. If you switch away from the default Mistral model to a Qwen reasoning model, `4096` has been more reliable than `2048` even when `/no_think` is present.

By default, the final `episode_thesis` document is a deterministic bounded overview instead of another LLM reduction pass. This avoids throwing away speaker/date resolution at the end of a long episode. The retrieval-grade evidence remains in `leaf_chunk`, `cluster_summary`, and `position_card` documents. Set `episode_thesis_reduce_with_llm` to `true` if you want the older LLM-generated thesis behavior.

Clustering can use `clustering_reduction: "pca"` or `"umap"` when `umap-learn` is installed. `grouping_mode` supports `semantic`, `chronological`, `speaker_first`, and `hybrid`/`topic_time` grouping. Cluster summaries and position cards get deterministic topic tags, fallback/model confidence metadata, compression telemetry, evidence excerpts, and stricter position-card validation.

Hierarchy construction uses `hierarchy_algorithm_version: "adaptive-v2"`. A level with zero or one current node stops; two through eleven nodes receive one forced parent; twelve or more nodes use semantic clustering with dominant-cluster, noise-rate, group-count, and `max_clusters` quality gates. Summary levels use a less conservative adaptive HDBSCAN profile: cluster size is bounded between `hierarchy_summary_min_cluster_size` and `hierarchy_summary_max_cluster_size` using `hierarchy_summary_cluster_size_divisor`, and `hierarchy_summary_min_samples` is explicit. Failed semantic gates use deterministic chronological groups of `group_fallback_size` and continue to the next level. Each processed cache records a file-local `hierarchy_manifest` and `cluster_telemetry`, including the strategy, clustering parameters, counts, quality metrics, fallback reasons, and stop reason. The episode thesis keeps structural hierarchy roots in `child_ids` and records its separate generation inputs in optional `generation_source_node_ids`.

By default, input JSON files are not moved after processing. Set `move_processed_files` to `true` if you prefer the older workflow where processed files are moved to `processed`.

## Direct Python Usage

```powershell
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json
python .\podcast_rag_pipeline.py --input-dir "C:\path\to\transcripts" --one-file
python .\podcast_rag_pipeline.py --create-stop-file
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json --build-topic-index
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json --build-topic-index --curate-topic-labels
python .\podcast_rag_pipeline.py --inspect-cache
python .\podcast_rag_pipeline.py --config-doctor
python .\podcast_rag_pipeline.py --model-eval --model-eval-limit 3
python .\podcast_rag_pipeline.py --retrieval-eval --retrieval-results .\path\to\captured_results.json --query-set .\evaluation\query_sets\podcast-baseline-v1.jsonl
```

The same commands also work as a module entry point after installation:

```powershell
python -m podcast_rag --config .\podcast_rag_config.json
```

For a bounded repair of one managed episode after a model-quality issue, select the episode explicitly and force only that episode to rebuild:

```powershell
python -m podcast_rag process --partition <partition-id> --episode TFM_20260425 --force-reprocess
```

The command preserves the existing cache and errata pair under the partition's `state\reprocess_backups\` directory before atomically publishing replacement output. It refuses unscoped force-reprocess requests.
