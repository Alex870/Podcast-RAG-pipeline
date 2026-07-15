from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import podcast_rag.runtime as runtime
from podcast_rag.config import PipelineConfig, config_fingerprint, generation_config_fingerprint, resolve_path
from podcast_rag.runtime import PIPELINE_VERSION, PROMPT_VERSION, PerformanceTracker, RunStats
from podcast_rag.schema import serialize_document, validate_processed_cache, validate_processed_documents
from podcast_rag.representations import RepresentationBuilder
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

def document_payloads(
    docs: list[Any],
    source_fingerprint_value: str = "",
    representation_builder: RepresentationBuilder | None = None,
) -> list[dict[str, Any]]:
    payloads = []
    for doc in docs:
        page_content = str(getattr(doc, "page_content", "") or "")
        metadata = dict(getattr(doc, "metadata", {}) or {})
        embedding_text = lexical_text = None
        if representation_builder is not None:
            embedding_text, lexical_text = representation_builder.build(page_content, metadata)
        payload = serialize_document(
            doc,
            source_fingerprint_value,
            embedding_text=embedding_text,
            lexical_text=lexical_text,
        )
        if representation_builder is not None:
            payload["metadata"]["representation_fingerprints"] = representation_builder.fingerprints(
                source_fingerprint_value,
                payload["metadata"],
                page_content,
            )
        payloads.append(payload)
    return payloads


def _cache_source_fingerprint(payload: dict[str, Any]) -> str:
    value = str(payload.get("source_fingerprint") or "").strip()
    if value:
        return value
    canonical = json.dumps(payload.get("documents") or [], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def backfill_cache_payload(payload: dict[str, Any], config: PipelineConfig) -> dict[str, Any]:
    """Add deterministic provenance and representations without invoking the LLM."""
    before = validate_processed_cache(payload)
    before.raise_for_errors("existing processed cache")
    source_fingerprint_value = _cache_source_fingerprint(payload)
    builder = RepresentationBuilder(
        embedding_text_mode=config.embedding_text_mode,
        lexical_text_mode=config.lexical_text_mode,
        contextual_header_max_chars=config.contextual_header_max_chars,
    )
    documents = []
    for raw_item in payload.get("documents") or []:
        if not isinstance(raw_item, dict):
            raise ValueError("processed cache contains a non-object document")
        metadata = dict(raw_item.get("metadata") or {})
        node_type = str(metadata.get("node_type") or "unknown")
        if node_type == "leaf_chunk" and not metadata.get("source_segment_ids") and not metadata.get("source_spans"):
            indices = metadata.get("segment_indices") or []
            if indices and metadata.get("episode_id"):
                metadata["source_segment_ids"] = [
                    f"{metadata['episode_id']}:segment:{index}" for index in indices if index is not None
                ]
            elif metadata.get("start_time") is None or metadata.get("end_time") is None:
                raise ValueError(
                    f"cannot backfill {metadata.get('node_id', 'unknown')}: source segment IDs or exact timestamps are missing"
                )
        if node_type == "leaf_chunk" and not metadata.get("source_spans"):
            if metadata.get("start_time") is not None and metadata.get("end_time") is not None:
                metadata["source_spans"] = [{
                    "start_time": metadata["start_time"],
                    "end_time": metadata["end_time"],
                    "segment_ids": metadata.get("source_segment_ids") or [],
                }]
        page_content = str(raw_item.get("page_content") or "")
        source_node_payload = {
            "source_fingerprint": source_fingerprint_value,
            "node_id": metadata.get("node_id"),
            "node_type": node_type,
            "page_content": page_content,
            "source_segment_ids": metadata.get("source_segment_ids") or [],
            "source_spans": metadata.get("source_spans") or [],
        }
        metadata["source_node_fingerprint"] = hashlib.sha256(
            json.dumps(source_node_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        stable = serialize_document(
            type("CacheDocument", (), {"page_content": page_content, "metadata": metadata})(),
            source_fingerprint_value,
            *builder.build(page_content, metadata),
        )
        stable["metadata"]["representation_fingerprints"] = builder.fingerprints(
            source_fingerprint_value, stable["metadata"], page_content
        )
        documents.append(stable)

    updated = dict(payload)
    updated["schema_version"] = "2.1"
    updated["source_fingerprint"] = source_fingerprint_value
    updated["representations"] = builder.manifest()
    updated["representation_config_fingerprint"] = builder.manifest()["config_fingerprint"]
    updated["config_fingerprint"] = config_fingerprint(config)
    existing_generation_fingerprint = str(payload.get("generation_config_fingerprint") or "")
    generation_matches = not existing_generation_fingerprint or existing_generation_fingerprint == generation_config_fingerprint(config)
    if existing_generation_fingerprint:
        updated["generation_config_fingerprint"] = existing_generation_fingerprint
    updated["delta"] = {
        "mode": "deterministic_backfill",
        "reused_nodes": len(documents),
        "llm_stages_reused": ["hierarchy", "positions"] if generation_matches else [],
        "llm_reuse_verified": generation_matches,
        "llm_stages_requiring_rebuild": [] if generation_matches else ["hierarchy", "positions"],
        "representations_rebuilt": [builder.manifest()["dense_text"], builder.manifest()["lexical_text"]],
    }
    updated["document_count"] = len(documents)
    updated["documents"] = documents
    result = validate_processed_cache(updated)
    result.raise_for_errors("backfilled processed cache")
    return updated


def backfill_cache_file(cache_path: Path, config: PipelineConfig, output_path: Path | None = None) -> dict[str, Any]:
    payload = read_json_file(cache_path)
    updated = backfill_cache_payload(payload, config)
    destination = output_path or cache_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    temp_path.write_text(json.dumps(updated, indent=2, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(destination)
    return {
        "cache_path": str(destination),
        "schema_version": updated["schema_version"],
        "document_count": updated["document_count"],
        "delta": updated["delta"],
    }


def export_dense_baseline(processed_data_dir: Path, output_path: Path, representation_id: str = "page-content-v1") -> dict[str, Any]:
    """Export stable IDs and dense text for downstream retrieval evaluation."""
    records = []
    manifests = []
    for cache_path in sorted(processed_data_dir.glob("*.processed_documents.json")):
        payload = read_json_file(cache_path)
        validation = validate_processed_cache(payload)
        validation.raise_for_errors(f"cache {cache_path}")
        representations = payload.get("representations") or {}
        dense_representation = representations.get("dense_text") or (
            "page-content-v1" if str(payload.get("schema_version") or "2.0") == "2.0" else ""
        )
        if dense_representation != representation_id:
            continue
        export_manifest = dict(representations)
        export_manifest.setdefault("dense_text", dense_representation)
        manifests.append({
            "cache_path": str(cache_path),
            "source_fingerprint": payload.get("source_fingerprint"),
            "schema_version": payload.get("schema_version"),
            "representation": export_manifest,
        })
        for item in payload.get("documents") or []:
            metadata = dict(item.get("metadata") or {})
            records.append({
                "document_id": metadata.get("stable_document_id"),
                "embedding_text": item.get("embedding_text") or item.get("page_content") or "",
                "page_content": item.get("page_content") or "",
                "metadata": metadata,
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export = {
        "schema_version": "retrieval-baseline-v1",
        "representation_id": representation_id,
        "document_count": len(records),
        "caches": manifests,
        "documents": records,
    }
    output_path.write_text(json.dumps(export, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"output_path": str(output_path), "document_count": len(records), "cache_count": len(manifests)}

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
