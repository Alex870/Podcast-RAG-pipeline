# Podcast RAG Pipeline

This project builds a retrieval-oriented podcast knowledge base from JSON transcript files produced by `podcast_transcribe_host.py`. It creates leaf chunks, RAPTOR-style rollup summaries, episode thesis summaries, and durable position cards, then stores them in a persistent Chroma collection.

The default workflow targets a local LM Studio server on Windows 11 using LM Studio's OpenAI-compatible API. The included example config defaults to `http://127.0.0.1:1234/v1` with `unsloth/qwen3.6-35b-a3b`, but both values are configurable.

## Local Versus Cloud

For overnight batch processing, local LM Studio processing is the sensible default. A cloud RTX 6000 Pro with 96 GB VRAM may let you run a larger model, a longer context, or more concurrent work, but this pipeline is designed to reduce long transcripts into bounded chunks and summaries. Unless you have a specific larger model in mind that materially improves the summaries, the expected quality gain is probably smaller than the operational cost for routine batches.

## Repository Contents

- `podcast_rag_pipeline.py`: main restartable Python pipeline
- `Run Podcast RAG Pipeline.ps1`: Windows PowerShell launcher
- `Test Podcast RAG Environment.ps1`: dependency and runtime diagnostic script
- `Set Podcast RAG Control.ps1`: live control helper for active batch runs
- `podcast_rag_config.example.json`: editable runtime configuration template
- `environment.yml`: Miniconda/Conda environment definition
- `podcast_rag_requirements.txt`: Python dependency list
- `LICENSE`: GPL-3.0 license

## First-Time Setup

1. Start LM Studio and load your model.
2. Enable the local OpenAI-compatible server in LM Studio.
3. Create the Conda environment:

```powershell
.\Run Podcast RAG Pipeline.ps1 -CreateCondaEnv
```

The launcher uses the `podcast-rag-pipeline` Conda environment by default. Override it with `-CondaEnvName` if you want a different name.

4. Copy the example config:

```powershell
Copy-Item .\podcast_rag_config.example.json .\podcast_rag_config.json
```

5. Put transcript JSON files under `data`, or point `input_dir` at the output directory from your transcription project.

## Running

Before a batch run, verify the local environment:

```powershell
.\Test Podcast RAG Environment.ps1
```

```powershell
.\Run Podcast RAG Pipeline.ps1
```

The launcher creates `podcast_rag_config.json` from the example if needed, applies optional command-line overrides, checks Python dependencies, and runs the pipeline.

Useful launcher parameters:

```powershell
.\Run Podcast RAG Pipeline.ps1 -InputDir "C:\path\to\transcripts"
.\Run Podcast RAG Pipeline.ps1 -Model "unsloth/qwen3.6-35b-a3b"
.\Run Podcast RAG Pipeline.ps1 -BaseUrl "http://127.0.0.1:1234/v1"
.\Run Podcast RAG Pipeline.ps1 -OneFile
.\Run Podcast RAG Pipeline.ps1 -MaxParallelModelRequests 2
.\Run Podcast RAG Pipeline.ps1 -CreateStopFile
.\Run Podcast RAG Pipeline.ps1 -ClearStopFile
.\Run Podcast RAG Pipeline.ps1 -CondaEnvName "podcast-rag-pipeline"
```

## Live Tuning

At startup, the pipeline initializes `state/pipeline_control.json` from `max_parallel_model_requests` in `podcast_rag_config.json` or the `-MaxParallelModelRequests` launcher override. While a batch is running, change how many new model requests can run in parallel with:

```powershell
.\Set Podcast RAG Control.ps1 -MaxParallelModelRequests 1
.\Set Podcast RAG Control.ps1 -MaxParallelModelRequests 3
```

The new value is applied before the pipeline launches additional model requests. Already-running LM Studio requests are allowed to finish.

## Stopping After The Current File

The pipeline checks for `state/stop_after_current.txt` between files and watches `Ctrl+C` while work is running. To request a clean stop from another PowerShell window, create the stop file while the batch is running:

```powershell
.\Run Podcast RAG Pipeline.ps1 -CreateStopFile
```

On `Ctrl+C`, the pipeline stops launching new model requests, waits for in-flight request(s) to finish, saves state, and exits. A partially processed file is marked `interrupted` and will be retried on the next run. Completed files are skipped using `state/podcast_rag_state.json`.

The stop file is intentionally left in place so the request is visible. Remove it before the next full run:

```powershell
.\Run Podcast RAG Pipeline.ps1 -ClearStopFile
```

## State And Resume

Progress is tracked in `state/podcast_rag_state.json`. Completed files are skipped on later runs using a stable fingerprint derived from file path, size, and modification time. If a file changes, it is treated as new work.

By default, input JSON files are not moved after processing. Set `move_processed_files` to `true` if you prefer the older workflow where processed files are moved to `processed`.

## Direct Python Usage

```powershell
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json
python .\podcast_rag_pipeline.py --input-dir "C:\path\to\transcripts" --one-file
python .\podcast_rag_pipeline.py --create-stop-file
```
