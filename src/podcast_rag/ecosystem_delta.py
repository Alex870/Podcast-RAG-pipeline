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


def plan_delta(
    old_documents: Mapping[str, Mapping[str, Any]],
    new_documents: Mapping[str, Mapping[str, Any]],
    *, parent_corpus_id: str, correction_set_id: str,
    processing_fingerprint: str, representation_fingerprint: str,
) -> dict[str, Any]:
    old_ids, new_ids = set(old_documents), set(new_documents)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    common = old_ids & new_ids
    changed = sorted(key for key in common if document_id(old_documents[key]) != document_id(new_documents[key]))
    unchanged = sorted(common - set(changed))
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
