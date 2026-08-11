"""Processed-delta-v1 planning and correction-aware cache replacement."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


CONTRACT = "processed-delta-v1"
MUTABLE = {"notes", "display_label", "ui_state", "delta_id"}


class DeltaError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def identity(value: Mapping[str, Any], prefix: str = "delta") -> str:
    payload = {k: deepcopy(v) for k, v in value.items() if k not in MUTABLE}
    return f"{prefix}_{hashlib.sha256(canonical_json(payload)).hexdigest()}"


def document_id(document: Mapping[str, Any]) -> str:
    payload = {
        "contract_version": document.get("contract_version", "page-content-v1"),
        "producer": document.get("producer", "Podcast-RAG-pipeline"),
        "parent_ids": document.get("parent_ids", []),
        "content": document.get("content", document.get("page_content", "")),
        "metadata": {k: v for k, v in document.get("metadata", {}).items() if k not in MUTABLE},
        "processing_fingerprint": document.get("processing_fingerprint"),
        "representation_fingerprint": document.get("representation_fingerprint"),
    }
    return f"doc_{hashlib.sha256(canonical_json(payload)).hexdigest()}"


_TRANSCRIPT_SUFFIXES = (
    "_cleaned_speaker_transcript",
    "_reviewed_speaker_transcript",
    "_corrected_speaker_transcript",
    "_speaker_transcript",
)


def _episode_aliases(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    aliases = {text.casefold()}
    stem = Path(text.replace("\\", "/")).stem
    aliases.add(stem.casefold())
    for suffix in _TRANSCRIPT_SUFFIXES:
        if stem.casefold().endswith(suffix):
            aliases.add(stem[: -len(suffix)].casefold())
    return aliases


def _document_evidence(document: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), Mapping) else {}
    episodes: set[str] = set()
    spans: set[str] = set()
    for key in ("episode_id", "source_episode_id", "episode_title", "source", "source_file"):
        episodes.update(_episode_aliases(document.get(key)))
        episodes.update(_episode_aliases(metadata.get(key)))
    for container in (document, metadata):
        for key in ("source_span_id", "source_segment_id", "segment_id"):
            if container.get(key) not in (None, ""):
                spans.add(str(container[key]))
        for key in ("source_span_ids", "source_segment_ids", "segment_ids"):
            values = container.get(key)
            if isinstance(values, list):
                spans.update(str(value) for value in values if value not in (None, ""))
        source_spans = container.get("source_spans")
        if isinstance(source_spans, list):
            for source_span in source_spans:
                if not isinstance(source_span, Mapping):
                    continue
                if source_span.get("source_span_id") not in (None, ""):
                    spans.add(str(source_span["source_span_id"]))
                values = source_span.get("segment_ids")
                if isinstance(values, list):
                    spans.update(str(value) for value in values if value not in (None, ""))
    return episodes, spans


def _document_matches_scope(
    document: Mapping[str, Any], *, affected_episode_ids: set[str], affected_source_span_ids: set[str],
) -> bool:
    episodes, spans = _document_evidence(document)
    episode_aliases = set().union(*(_episode_aliases(value) for value in affected_episode_ids)) if affected_episode_ids else set()
    if episode_aliases and episodes.intersection(episode_aliases):
        return True
    if affected_source_span_ids and spans.intersection(affected_source_span_ids):
        return True
    return False


def plan_delta(
    old_documents: Mapping[str, Mapping[str, Any]],
    new_documents: Mapping[str, Mapping[str, Any]],
    *, parent_corpus_id: str, correction_set_id: str,
    processing_fingerprint: str, representation_fingerprint: str,
    affected_episode_ids: list[str] | None = None,
    affected_source_span_ids: list[str] | None = None,
) -> dict[str, Any]:
    old_ids, new_ids = set(old_documents), set(new_documents)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    common = old_ids & new_ids
    changed = sorted(key for key in common if document_id(old_documents[key]) != document_id(new_documents[key]))
    unchanged = sorted(common - set(changed))
    episode_scope = {str(value) for value in (affected_episode_ids or []) if value}
    span_scope = {str(value) for value in (affected_source_span_ids or []) if value}
    if episode_scope or span_scope:
        matched = {
            key
            for key in old_ids | new_ids
            if _document_matches_scope(
                old_documents.get(key, new_documents.get(key, {})),
                affected_episode_ids=episode_scope,
                affected_source_span_ids=span_scope,
            )
            or _document_matches_scope(
                new_documents.get(key, old_documents.get(key, {})),
                affected_episode_ids=episode_scope,
                affected_source_span_ids=span_scope,
            )
        }
        if not matched:
            raise DeltaError("correction notification scope did not match any corpus documents")
        drift = sorted((set(added) | set(removed) | set(changed)) - matched)
        if drift:
            raise DeltaError(f"out-of-scope corpus changes detected: {drift}")
        added = sorted(set(added) & matched)
        removed = sorted(set(removed) & matched)
        changed = sorted(set(changed) & matched)
        unchanged = sorted((old_ids & new_ids) - set(changed))
    mappings = {key: document_id(new_documents[key]) for key in changed}
    reasons = {key: "content_or_authoritative_metadata_changed" for key in changed}
    reasons.update({key: "source_document_removed_advisory" for key in removed})
    affected = changed + removed
    invalidated = {
        "ancestors": sorted({str(old_documents[k].get("parent_id")) for k in affected if old_documents[k].get("parent_id")}),
        "topics": sorted({str(old_documents[k].get("topic_id")) for k in affected if old_documents[k].get("topic_id")}),
        "positions": sorted({str(old_documents[k].get("position_id")) for k in affected if old_documents[k].get("position_id")}),
    }
    delta: dict[str, Any] = {
        "contract_version": CONTRACT,
        "producer": {"name": "Podcast-RAG-pipeline", "contract_version": "1"},
        "parent_corpus_id": parent_corpus_id,
        "parent_cache_ids": sorted({str(v.get("cache_id")) for v in old_documents.values() if v.get("cache_id")}),
        "correction_set_ids": [correction_set_id],
        "affected_episode_ids": sorted(episode_scope),
        "affected_source_span_ids": sorted(span_scope),
        "processing_fingerprint": processing_fingerprint,
        "representation_fingerprint": representation_fingerprint,
        "added_document_ids": added,
        "changed_document_ids": changed,
        "unchanged_document_ids": unchanged,
        "removed_document_ids": removed,
        "removals_advisory": True,
        "invalidated": invalidated,
        "old_to_new_evidence": mappings,
        "stale_judgment_ids": [],
        "reasons": reasons,
        "validation": {"evidence_closure": True},
        "timings_ms": {},
        "failures": [],
    }
    delta["delta_id"] = identity(delta)
    validate_delta(delta)
    return delta


def validate_delta(delta: Mapping[str, Any]) -> None:
    if delta.get("contract_version") != CONTRACT:
        raise DeltaError("unsupported processed delta contract")
    if delta.get("delta_id") != identity(delta):
        raise DeltaError("processed delta identity mismatch")
    reasons = delta.get("reasons", {})
    affected = list(delta.get("changed_document_ids", [])) + list(delta.get("removed_document_ids", []))
    missing = [item for item in affected if item not in reasons]
    if missing:
        raise DeltaError(f"changed/removed documents require reasons: {missing}")


def apply_delta(
    delta: Mapping[str, Any], old_documents: Mapping[str, Mapping[str, Any]],
    new_documents: Mapping[str, Mapping[str, Any]], *, approved_correction_set_id: str,
) -> dict[str, Mapping[str, Any]]:
    validate_delta(delta)
    if approved_correction_set_id not in delta.get("correction_set_ids", []):
        raise DeltaError("approved correction-set identity does not match delta")
    result = deepcopy(dict(old_documents))
    for key in delta.get("added_document_ids", []) + delta.get("changed_document_ids", []):
        if key not in new_documents:
            raise DeltaError(f"replacement document missing: {key}")
        result[key] = deepcopy(new_documents[key])
    for key in delta.get("removed_document_ids", []):
        result.pop(key, None)
    return result


def write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
