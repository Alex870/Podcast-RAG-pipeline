from __future__ import annotations

import argparse
import datetime as dt
import signal
import time
from collections import Counter
from pathlib import Path

import podcast_rag.runtime as runtime
from podcast_rag.config import PipelineConfig, apply_env_overrides, load_config, resolve_path
from podcast_rag.llm_support import test_model_inference, verify_model_available
from podcast_rag.pipeline import PodcastRagPipeline
from podcast_rag.runtime import PIPELINE_VERSION, PROMPT_VERSION, PipelineInterrupted, RunStats, RuntimeControl, request_stop
from podcast_rag.schema import dumps_schema_summary, validate_processed_documents
from podcast_rag.state import (
    load_state,
    mark_state,
    maybe_move_processed,
    processed_data_cache_path,
    quarantine_invalid_cache,
    read_json_file,
    save_state,
    write_json_file,
    write_run_reports,
    write_run_snapshot,
    should_skip_file,
)
from podcast_rag.text_utils import (
    deterministic_topic_tags,
    format_duration,
    estimate_remaining_seconds,
    file_fingerprint,
    is_missing_context_response,
    token_estimate,
)
from podcast_rag.topics import refresh_topic_index
from podcast_rag.transcript import iter_transcript_files, load_transcript_json

def run_batch(config: PipelineConfig, project_dir: Path, one_file: bool) -> int:
    """Process every pending transcript, reusing caches and checkpoints when possible."""
    runtime.load_runtime_deps()

    input_dir = resolve_path(project_dir, config.input_dir)
    processed_dir = resolve_path(project_dir, config.processed_dir)
    processed_data_dir = resolve_path(project_dir, config.processed_data_dir)
    state_path = resolve_path(project_dir, config.state_path)
    stop_file = resolve_path(project_dir, config.stop_file)
    snapshot_path = resolve_path(project_dir, config.run_snapshot_path)
    report_dir = resolve_path(project_dir, config.run_report_dir)

    input_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    control = RuntimeControl(config, project_dir)
    control.initialize_file_for_run()

    state = load_state(state_path)
    files = iter_transcript_files(input_dir, config.file_glob)
    pending = []
    for path in files:
        fingerprint = file_fingerprint(path)
        cache_path = processed_data_cache_path(processed_data_dir, fingerprint, path)
        if cache_path.exists() or not should_skip_file(state, fingerprint):
            pending.append((path, fingerprint))

    print(f"Found {len(files)} matching files; {len(pending)} pending.")
    if not pending:
        return 0

    cached_pending = [processed_data_cache_path(processed_data_dir, fingerprint, path).exists() for path, fingerprint in pending]
    needs_llm_processing = not all(cached_pending)
    if needs_llm_processing:
        if config.fake_llm:
            print("Fake LLM mode enabled; skipping LM Studio model verification.")
        elif config.verify_model:
            verify_model_available(config)
        if not config.fake_llm and config.test_inference:
            test_model_inference(config)
    else:
        print("All pending files have processed data caches; skipping LM Studio model verification.")

    pipeline = PodcastRagPipeline(config, project_dir, control)
    batch_started_at = time.time()
    completed_files_this_run = 0
    stats = RunStats()
    stats.files_total = len(pending)
    write_run_snapshot(snapshot_path, stats, pipeline.performance)

    for idx, (path, fingerprint) in enumerate(pending, 1):
        if runtime.STOP_REQUESTED or stop_file.exists():
            print("Stop requested before starting next file.")
            break

        cache_path = processed_data_cache_path(processed_data_dir, fingerprint, path)
        file_eta = format_duration(estimate_remaining_seconds(completed_files_this_run, len(pending), time.time() - batch_started_at))
        print(f"\nFile {idx}/{len(pending)} eta_files={file_eta}")

        try:
            if cache_path.exists():
                try:
                    result = pipeline.validate_cached_file(path, fingerprint, cache_path)
                    stats.cached_files += 1
                except ValueError as exc:
                    quarantined_path = quarantine_invalid_cache(cache_path, str(exc))
                    print(f"  Invalid processed data cache moved to: {quarantined_path}")
                    print("  Rebuilding processed data from transcript.")
                    mark_state(state, fingerprint, path, "in_progress")
                    save_state(state_path, state)
                    result = pipeline.process_file(path)
                    stats.llm_files += 1
                    if result["status"] != "completed":
                        mark_state(state, fingerprint, path, result["status"], result)
                        save_state(state_path, state)
                        continue
                    docs = result.pop("documents", [])
                    pipeline.save_cached_documents(cache_path, path, fingerprint, docs)
                    result["cache_path"] = str(cache_path)
                    result["quarantined_cache_path"] = str(quarantined_path)
                    print(f"  Saved processed data cache: {cache_path}")
            else:
                mark_state(state, fingerprint, path, "in_progress")
                save_state(state_path, state)
                result = pipeline.process_file(path)
                stats.llm_files += 1
                if result["status"] != "completed":
                    mark_state(state, fingerprint, path, result["status"], result)
                    save_state(state_path, state)
                    continue
                docs = result.pop("documents", [])
                pipeline.save_cached_documents(cache_path, path, fingerprint, docs)
                result["cache_path"] = str(cache_path)
                print(f"  Saved processed data cache: {cache_path}")

            moved_to = None
            if result["status"] == "completed" and config.move_processed_files:
                moved_to = maybe_move_processed(path, processed_dir)
                print(f"  Moved to {moved_to}")
            if moved_to:
                result["moved_to"] = moved_to
            mark_state(state, fingerprint, path, result["status"], result)
            save_state(state_path, state)
            if result["status"] in {"completed", "skipped"}:
                completed_files_this_run += 1
            if result["status"] == "completed":
                stats.files_completed += 1
            elif result["status"] == "skipped":
                stats.files_skipped += 1
            stats.documents += int(result.get("nodes") or 0)
            stats.position_cards += int(result.get("position_cards") or 0)
            stats.fallbacks = pipeline.fallback_count
            stats.files.append({"path": str(path), **{key: value for key, value in result.items() if key != "documents"}})
            write_run_snapshot(snapshot_path, stats, pipeline.performance)
        except PipelineInterrupted as exc:
            mark_state(state, fingerprint, path, "interrupted", {"error": str(exc)})
            save_state(state_path, state)
            stats.files_failed += 1
            stats.failures.append({"path": str(path), "error": str(exc), "type": "interrupted"})
            write_run_snapshot(snapshot_path, stats, pipeline.performance)
            print("Stop request handled. Progress state was saved; this file will be retried on the next run.")
            break
        except Exception as exc:
            mark_state(state, fingerprint, path, "failed", {"error": f"{type(exc).__name__}: {exc}"})
            save_state(state_path, state)
            stats.files_failed += 1
            stats.failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}", "type": "exception"})
            write_run_snapshot(snapshot_path, stats, pipeline.performance)
            raise

        if one_file:
            print("Processed one file; stopping because --one-file was set.")
            break

        if runtime.STOP_REQUESTED or stop_file.exists():
            print("Stop request detected. Batch will resume with the next pending file on the next run.")
            break

    pipeline.performance.final_report()
    json_report, md_report = write_run_reports(report_dir, stats, pipeline.performance, config)
    print(f"Run reports saved: {json_report} and {md_report}")
    if config.auto_refresh_topic_index:
        topic_summary = refresh_topic_index(config, project_dir)
        print(
            "Topic index refreshed: "
            f"{topic_summary['topic_count']} topics across {topic_summary['episode_count']} episode contribution(s) "
            f"({topic_summary['reused_contributions']} reused, {topic_summary['rebuilt_contributions']} rebuilt)."
        )
        if topic_summary["llm_curated_keep"] or topic_summary["llm_curated_drop"]:
            print(
                "Topic label curation updated: "
                f"{topic_summary['llm_curated_keep']} kept, {topic_summary['llm_curated_drop']} dropped "
                f"(whitelist={topic_summary['whitelist_size']}, blacklist={topic_summary['blacklist_size']})."
            )
        print(f"Topic index path: {topic_summary['topic_index_path']}")
        print(f"Topic curation report: {topic_summary['topic_curation_report_path']}")
    print("\nBatch run complete.")
    return 0

def build_topic_index(config: PipelineConfig, project_dir: Path) -> int:
    """Build or incrementally refresh the cache-only topic index from processed_data."""
    summary = refresh_topic_index(config, project_dir)
    print(
        "Topic index refreshed: "
        f"{summary['topic_count']} topics across {summary['episode_count']} episode contribution(s); "
        f"{summary['reused_contributions']} reused, {summary['rebuilt_contributions']} rebuilt, "
        f"{summary['removed_contributions']} removed."
    )
    if summary["llm_curated_keep"] or summary["llm_curated_drop"]:
        print(
            "Topic label curation updated: "
            f"{summary['llm_curated_keep']} kept, {summary['llm_curated_drop']} dropped "
            f"(whitelist={summary['whitelist_size']}, blacklist={summary['blacklist_size']})."
        )
    print(f"Topic index path: {summary['topic_index_path']}")
    print(f"Topic curation report: {summary['topic_curation_report_path']}")
    return 0

def inspect_processed_cache(config: PipelineConfig, project_dir: Path) -> int:
    """Validate cached processed-document files without running the model."""
    processed_data_dir = resolve_path(project_dir, config.processed_data_dir)
    files = sorted(processed_data_dir.glob("*.processed_documents.json"))
    totals = Counter()
    invalid = 0
    print(f"Inspecting {len(files)} processed cache file(s) in {processed_data_dir}")
    for cache_path in files:
        try:
            payload = read_json_file(cache_path)
            docs = payload.get("documents") or []
            validation = validate_processed_documents(docs)
            counts = Counter(validation.counts)
            totals.update(counts)
            status = "valid" if validation.valid else "invalid"
            if not validation.valid:
                invalid += 1
            print(
                f"- {cache_path.name}: {status}, docs={len(docs)}, "
                f"positions={counts.get('position_card', 0)}, schema={payload.get('schema_version', 'unknown')}"
            )
            for warning in validation.warnings[:3]:
                print(f"    warning: {warning}")
            for error in validation.errors[:3]:
                print(f"    error: {error}")
        except Exception as exc:
            invalid += 1
            print(f"- {cache_path.name}: unreadable ({type(exc).__name__}: {exc})")
    print(f"Totals: {dict(totals)} invalid_files={invalid}")
    return 1 if invalid else 0

def config_doctor(config: PipelineConfig, project_dir: Path) -> int:
    """Check the effective config against model, token, and cache expectations."""
    print("Config doctor")
    print(f"  pipeline_version: {PIPELINE_VERSION}")
    print(f"  prompt_version: {PROMPT_VERSION}")
    print(f"  model: {config.lm_studio_model}")
    print(f"  base_url: {config.lm_studio_base_url}")
    print(f"  context_window_tokens: {config.context_window_tokens}")
    print(f"  prompt_token_budget: {config.prompt_token_budget}")
    print(f"  llm_max_tokens: {config.llm_max_tokens}")
    print(f"  rollup_char_budget: {config.rollup_char_budget}")
    print(f"  estimated rollup prompt tokens: {token_estimate('x' * config.rollup_char_budget, config.prompt_token_chars_per_token)}")
    if config.prompt_token_budget + config.llm_max_tokens > config.context_window_tokens:
        print("  warning: prompt_token_budget + llm_max_tokens exceeds context_window_tokens")
    if config.max_parallel_model_requests > 1:
        print("  note: parallel requests can reduce wall time but may increase LM Studio context pressure")
    if not config.fake_llm and config.verify_model:
        runtime.load_runtime_deps()
        verify_model_available(config)
    if not config.fake_llm and config.test_inference:
        runtime.load_runtime_deps()
        test_model_inference(config)
    print("Processed cache schema:")
    print(dumps_schema_summary())
    return 0

def evaluate_model(config: PipelineConfig, project_dir: Path, limit: int = 3) -> int:
    """Benchmark the configured model on a small deterministic transcript sample."""
    runtime.load_runtime_deps()
    input_dir = resolve_path(project_dir, config.input_dir)
    output_dir = resolve_path(project_dir, config.model_eval_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = iter_transcript_files(input_dir, config.file_glob)[: max(1, limit)]
    control = RuntimeControl(config, project_dir)
    control.initialize_file_for_run()
    pipeline = PodcastRagPipeline(config, project_dir, control)
    results = []
    for path in files:
        docs = load_transcript_json(path)
        leaf_chunks = pipeline.build_leaf_chunks(docs, str(path))[:3]
        sample_text = "\n\n".join(pipeline.render_doc_for_rollup(doc) for doc in leaf_chunks)
        started = time.time()
        summary = pipeline.invoke_llm(pipeline.summary_chain, sample_text, f"model eval {path.name}")
        elapsed = time.time() - started
        results.append(
            {
                "path": str(path),
                "elapsed_seconds": round(elapsed, 3),
                "missing_context": is_missing_context_response(summary),
                "empty": not bool(summary.strip()),
                "compression_ratio": round(len(summary) / max(1, len(sample_text)), 4),
                "topic_tags": deterministic_topic_tags(summary, config.deterministic_topic_count),
            }
        )
    report_path = output_dir / f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.model_eval.json"
    write_json_file(
        report_path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": config.lm_studio_model,
            "performance": pipeline.performance.snapshot(),
            "results": results,
        },
    )
    print(f"Model evaluation report saved: {report_path}")
    return 0

def create_stop_file(config: PipelineConfig, project_dir: Path) -> int:
    stop_file = resolve_path(project_dir, config.stop_file)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text(f"Stop requested at {dt.datetime.now().isoformat()}\n", encoding="utf-8")
    print(f"Created stop file: {stop_file}")
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a podcast RAG knowledge base from transcript JSON files.")
    parser.add_argument("--config", default="podcast_rag_config.json", help="Path to the JSON config file.")
    parser.add_argument("--input-dir", help="Override config input_dir.")
    parser.add_argument("--file-glob", help="Override config file_glob.")
    parser.add_argument("--model", help="Override config lm_studio_model.")
    parser.add_argument("--base-url", help="Override config lm_studio_base_url.")
    parser.add_argument("--max-parallel-model-requests", type=int, help="Override initial max_parallel_model_requests.")
    parser.add_argument("--one-file", action="store_true", help="Process only one pending file.")
    parser.add_argument("--create-stop-file", action="store_true", help="Create the configured stop file and exit.")
    parser.add_argument("--inspect-cache", action="store_true", help="Inspect processed_data caches without processing.")
    parser.add_argument("--config-doctor", action="store_true", help="Validate operational config and LM Studio settings before a batch.")
    parser.add_argument("--model-eval", action="store_true", help="Run the model-evaluation harness on transcript slices.")
    parser.add_argument("--model-eval-limit", type=int, default=3, help="Maximum transcript files to sample for --model-eval.")
    parser.add_argument("--build-topic-index", action="store_true", help="Build or refresh the cache-only topic index from processed_data.")
    parser.add_argument("--curate-topic-labels", action="store_true", help="Run the optional LM Studio topic-label curation pass during topic-index refresh.")
    parser.add_argument("--fake-llm", action="store_true", help="Use deterministic fake LLM responses for no-LM Studio validation.")
    return parser.parse_args()

def main() -> int:
    """CLI entry point for batch processing, diagnostics, and model evaluation."""
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    args = parse_args()
    config_path = Path(args.config).expanduser()
    project_dir = config_path.resolve().parent if config_path.exists() else Path.cwd()
    config = apply_env_overrides(load_config(config_path))

    if args.input_dir:
        config.input_dir = args.input_dir
    if args.file_glob:
        config.file_glob = args.file_glob
    if args.model:
        config.lm_studio_model = args.model
    if args.base_url:
        config.lm_studio_base_url = args.base_url
    if args.max_parallel_model_requests:
        config.max_parallel_model_requests = args.max_parallel_model_requests
    if args.fake_llm:
        config.fake_llm = True
        config.verify_model = False
        config.test_inference = False
    if args.curate_topic_labels:
        config.enable_llm_topic_label_curation = True

    if args.create_stop_file:
        return create_stop_file(config, project_dir)
    if args.inspect_cache:
        return inspect_processed_cache(config, project_dir)
    if args.config_doctor:
        return config_doctor(config, project_dir)
    if args.model_eval:
        return evaluate_model(config, project_dir, args.model_eval_limit)
    if args.build_topic_index:
        return build_topic_index(config, project_dir)

    return run_batch(config, project_dir, args.one_file)
