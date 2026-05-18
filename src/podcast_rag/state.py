from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import podcast_rag.runtime as runtime
from podcast_rag.config import PipelineConfig, config_fingerprint, resolve_path
from podcast_rag.runtime import PIPELINE_VERSION, PROMPT_VERSION, PerformanceTracker, RunStats
from podcast_rag.schema import serialize_document, validate_processed_documents
from podcast_rag.text_utils import format_duration, source_schema_version, stable_episode_id

def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"version": 1, "files": {}}

    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup_path = state_path.with_suffix(f".corrupt.{int(time.time())}.json")
        shutil.copy2(state_path, backup_path)
        print(f"State file was invalid JSON. Backed it up to {backup_path} and starting fresh.")
        return {"version": 1, "files": {}}

def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(state_path)

def checkpoint_path(config: PipelineConfig, project_dir: Path, source_path: Path, fingerprint: str, stage: str) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_path.stem).strip("._") or "transcript"
    return resolve_path(project_dir, config.checkpoint_dir) / f"{safe_stem}.{fingerprint}.{stage}.json"

def read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temp.replace(path)

def document_payloads(docs: list[Any], source_fingerprint_value: str = "") -> list[dict[str, Any]]:
    return [serialize_document(doc, source_fingerprint_value) for doc in docs]

def docs_from_payloads(payloads: list[dict[str, Any]]) -> list[Any]:
    runtime.load_runtime_deps()
    Document = runtime.Document
    return [
        Document(page_content=str(item.get("page_content", "")), metadata=dict(item.get("metadata") or {}))
        for item in payloads
        if isinstance(item, dict)
    ]

def write_run_snapshot(path: Path, stats: RunStats, performance: PerformanceTracker | None = None) -> None:
    write_json_file(path, stats.snapshot(performance))

def write_run_reports(report_dir: Path, stats: RunStats, performance: PerformanceTracker, config: PipelineConfig) -> tuple[Path, Path]:
    """Persist machine-readable and human-readable summaries for a completed batch."""
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = stats.snapshot(performance)
    payload["pipeline_version"] = PIPELINE_VERSION
    payload["prompt_version"] = PROMPT_VERSION
    payload["config_fingerprint"] = config_fingerprint(config)
    json_path = report_dir / f"{stamp}.run_report.json"
    md_path = report_dir / f"{stamp}.run_report.md"
    write_json_file(json_path, payload)
    lines = [
        "# Podcast RAG Run Report",
        "",
        f"- Pipeline version: {PIPELINE_VERSION}",
        f"- Prompt version: {PROMPT_VERSION}",
        f"- Elapsed: {format_duration(payload.get('elapsed_seconds'))}",
        f"- Files: {payload['files_completed']} completed, {payload['files_skipped']} skipped, {payload['files_failed']} failed of {payload['files_total']}",
        f"- Cached files: {payload['cached_files']}",
        f"- LLM-processed files: {payload['llm_files']}",
        f"- Documents: {payload['documents']}",
        f"- Position cards: {payload['position_cards']}",
        f"- Fallbacks: {payload['fallbacks']}",
        f"- Requests: {payload['performance']['requests']}",
        f"- Failures: {payload['performance']['failures']}",
        f"- Max tokens: {payload['performance']['run_max_total_tokens'] or 'unknown'}",
        "",
        "## Files",
        "",
    ]
    for item in payload["files"]:
        lines.append(
            f"- {Path(str(item.get('path', item.get('source', 'unknown')))).name}: "
            f"{item.get('status')} nodes={item.get('nodes', 0)} positions={item.get('position_cards', 0)} "
            f"source={item.get('source', '')}"
        )
    if payload["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in payload["failures"]:
            lines.append(f"- {failure.get('path')}: {failure.get('error')}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path

def should_skip_file(state: dict[str, Any], fingerprint: str) -> bool:
    entry = state.get("files", {}).get(fingerprint)
    return bool(entry and entry.get("status") in {"completed", "skipped"})

def processed_data_cache_path(processed_data_dir: Path, fingerprint: str, source_path: Path) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_path.stem).strip("._") or "transcript"
    return processed_data_dir / f"{safe_stem}.{fingerprint}.processed_documents.json"

def quarantine_invalid_cache(cache_path: Path, reason: str) -> Path:
    quarantine_dir = cache_path.parent / "invalid"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = quarantine_dir / f"{cache_path.name}.invalid.{stamp}"
    counter = 1
    while dest.exists():
        dest = quarantine_dir / f"{cache_path.name}.invalid.{stamp}.{counter}"
        counter += 1
    shutil.move(str(cache_path), str(dest))
    reason_path = dest.with_suffix(dest.suffix + ".reason.txt")
    reason_path.write_text(reason, encoding="utf-8")
    return dest

def mark_state(state: dict[str, Any], fingerprint: str, path: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "path": str(path),
        "status": status,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    state.setdefault("files", {})[fingerprint] = payload

def maybe_move_processed(path: Path, processed_dir: Path) -> str | None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / f"{path.name}.{int(time.time())}"
    shutil.move(str(path), str(dest))
    return str(dest)
