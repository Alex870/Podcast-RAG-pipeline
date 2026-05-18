# Podcast RAG Pipeline

This project builds pre-processed podcast RAG documents from JSON transcript files produced by `podcast_transcribe_host.py` (in the [Podcast-Host-Transcription-Pipeline](https://github.com/Alex870/Podcast-Host-Transcription-Pipeline) repository). It creates leaf chunks, RAPTOR-style rollup summaries, episode thesis summaries, and durable position cards, then saves them as processed JSON caches.

The output of this process is then loaded into a Vector database in the `Chroma DB Import` project. This division keeps expensive LLM preprocessing separate from Chroma database rebuilds.

The default workflow targets a local LM Studio server running on Windows 11 using LM Studio's OpenAI-compatible API. The included example config defaults to `http://127.0.0.1:1234/v1` with `unsloth/qwen3.6-35b-a3b`, but both values are configurable.

The shared transcript, processed-cache, Chroma metadata, and `podcast.json` expectations are documented in [`docs/podcast_pipeline_contract.md`](docs/podcast_pipeline_contract.md).

## Local Versus Cloud

For overnight batch processing, local LM Studio processing is the sensible default. A cloud RTX 6000 Pro with 96 GB VRAM may let you run a larger model, a longer context, or more concurrent work, but this pipeline is designed to reduce long transcripts into bounded chunks and summaries. Unless you have a specific larger model in mind that materially improves the summaries, the expected quality gain is probably smaller than the operational cost for routine batches.

## Batch Performance Snapshot

The current local batch benchmark used LM Studio with `mistral-small-3.2-24b-instruct-2506` on an RTX 5070 Ti. Across 19 completed processed-data caches from one batch run, the pipeline processed about 88.4 hours of podcast audio in 16.8 hours of recorded processing time, or about 11.4 minutes of processing per podcast hour. That is roughly 5.3x faster than real time for this workload.

This benchmark is based on `state/podcast_rag_state.json` elapsed seconds and episode durations inferred from `leaf_chunk` start/end metadata in `processed_data`. The same run produced 9,991 documents and 379 position cards. The maximum observed request size was 3,736 total tokens, so a minimum LM Studio context length of 4,096 tokens is advisable for this workload.

## Repository Layout

- `src/podcast_rag/`: package source, split into config, runtime, transcript ingestion, cache/state IO, orchestration, and CLI modules
- `Run Podcast RAG Pipeline.ps1`: root bootstrap launcher for the most common setup, validation, control, and run actions
- `scripts/`: PowerShell launchers and diagnostics for Windows-first operation
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
- `pipeline.py`: the `PodcastRagPipeline` orchestration class for chunking, summarization, clustering, and position extraction
- `cli.py`: batch execution, cache inspection, config doctor, and model evaluation entry points

## First-Time Setup

1. Start LM Studio and load your model.
2. Enable the local OpenAI-compatible server in LM Studio.
3. Create the Conda environment:

```powershell
.\Run Podcast RAG Pipeline.ps1
```

Choose `7` to create or refresh the Conda environment. The underlying launcher uses the `podcast-rag-pipeline` Conda environment by default. Run the script in `scripts` directly with `-CondaEnvName` if you want a different name.

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

```powershell
.\Run Podcast RAG Pipeline.ps1
```

Use the menu to choose between environment validation, the main batch pipeline, processed-cache inspection, live control updates, stop-file management, and Conda environment creation. The main pipeline launcher still creates `podcast_rag_config.json` from the example if needed, applies optional command-line overrides, checks Python dependencies, and runs the pipeline.

If you prefer to skip the menu, the root launcher also supports direct actions:

```powershell
.\Run Podcast RAG Pipeline.ps1 -Action Debug
.\Run Podcast RAG Pipeline.ps1 -Action Run
.\Run Podcast RAG Pipeline.ps1 -Action CacheCheck
.\Run Podcast RAG Pipeline.ps1 -Action SetControl -MaxParallelModelRequests 2
.\Run Podcast RAG Pipeline.ps1 -Action CreateStopFile
.\Run Podcast RAG Pipeline.ps1 -Action ClearStopFile
.\Run Podcast RAG Pipeline.ps1 -Action CreateCondaEnv
```

The underlying scripts remain available in `scripts\` for more specific options.

Useful launcher parameters:

```powershell
.\scripts\Run-PodcastRagPipeline.ps1 -InputDir "C:\path\to\transcripts"
.\scripts\Run-PodcastRagPipeline.ps1 -Model "unsloth/qwen3.6-35b-a3b"
.\scripts\Run-PodcastRagPipeline.ps1 -BaseUrl "http://127.0.0.1:1234/v1"
.\scripts\Run-PodcastRagPipeline.ps1 -OneFile
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

To insert or reinsert processed caches into Chroma, use the separate `Chroma DB Import` project.

Processed caches now use schema version `2.0`. Each cache includes a prompt/version manifest, config fingerprint, model and embedding names, source transcript fingerprint and schema version, stable document IDs, cluster telemetry, fallback counts, token maxima, validation counts, and an import manifest for downstream Chroma import.

Within a single file, completed `leaf_chunks`, `hierarchy`, and `positions` stages are checkpointed under `state/file_checkpoints`. If a long episode is interrupted after an expensive stage, the next run can resume from the checkpoint instead of starting over. Set `resume_within_file` to `false` to disable this.

At the end of each batch, structured reports are written to `state/run_reports` as JSON and Markdown. A live dashboard-friendly snapshot is refreshed at `state/current_run_snapshot.json`.

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

For Qwen reasoning models in LM Studio, keep `llm_max_tokens` high enough for hidden reasoning plus final answer text. With `unsloth/qwen3.6-35b-a3b`, `4096` has been more reliable than `2048` even when `/no_think` is present.

By default, the final `episode_thesis` document is a deterministic bounded overview instead of another LLM reduction pass. This avoids throwing away speaker/date resolution at the end of a long episode. The retrieval-grade evidence remains in `leaf_chunk`, `cluster_summary`, and `position_card` documents. Set `episode_thesis_reduce_with_llm` to `true` if you want the older LLM-generated thesis behavior.

Clustering can use `clustering_reduction: "pca"` or `"umap"` when `umap-learn` is installed. `grouping_mode` supports `semantic`, `chronological`, `speaker_first`, and `hybrid`/`topic_time` grouping. Cluster summaries and position cards get deterministic topic tags, fallback/model confidence metadata, compression telemetry, evidence excerpts, and stricter position-card validation.

By default, input JSON files are not moved after processing. Set `move_processed_files` to `true` if you prefer the older workflow where processed files are moved to `processed`.

## Direct Python Usage

```powershell
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json
python .\podcast_rag_pipeline.py --input-dir "C:\path\to\transcripts" --one-file
python .\podcast_rag_pipeline.py --create-stop-file
python .\podcast_rag_pipeline.py --inspect-cache
python .\podcast_rag_pipeline.py --config-doctor
python .\podcast_rag_pipeline.py --model-eval --model-eval-limit 3
```

The same commands also work as a module entry point after installation:

```powershell
python -m podcast_rag --config .\podcast_rag_config.json
```
