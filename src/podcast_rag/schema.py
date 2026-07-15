from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


PROCESSED_CACHE_SCHEMA_VERSION = "2.1"
SUPPORTED_PROCESSED_CACHE_SCHEMA_VERSIONS = {"2.0", "2.1"}
REQUIRED_NODE_TYPES = {"leaf_chunk", "episode_thesis"}
SUMMARY_NODE_TYPES = {"cluster_summary", "episode_thesis", "position_card"}
REQUIRED_METADATA_FIELDS = {
    "node_id",
    "node_type",
    "level",
    "source",
    "episode_id",
    "episode_title",
    "source_type",
    "speaker_scope",
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    counts: dict[str, int]

    def raise_for_errors(self, label: str) -> None:
        if self.errors:
            preview = "; ".join(self.errors[:10])
            if len(self.errors) > 10:
                preview += f"; and {len(self.errors) - 10} more"
            raise ValueError(f"{label} produced invalid processed-cache documents: {preview}")


def stable_document_id(source_fingerprint: str, node_type: str, node_id: str, content: str) -> str:
    compact = re.sub(r"\s+", " ", content or "").strip()
    key = f"{source_fingerprint}|{node_type}|{node_id}|{compact}"
    import hashlib

    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


def serialize_document(
    doc: Any,
    source_fingerprint: str = "",
    embedding_text: str | None = None,
    lexical_text: str | None = None,
) -> dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    page_content = str(getattr(doc, "page_content", "") or "")
    if "stable_document_id" not in metadata:
        metadata["stable_document_id"] = stable_document_id(
            source_fingerprint,
            str(metadata.get("node_type") or "unknown"),
            str(metadata.get("node_id") or ""),
            page_content,
        )
    if "source_node_fingerprint" not in metadata:
        import hashlib

        source_node = {
            "source_fingerprint": source_fingerprint,
            "node_id": metadata.get("node_id"),
            "node_type": metadata.get("node_type"),
            "page_content": page_content,
            "source_segment_ids": metadata.get("source_segment_ids") or [],
            "source_spans": metadata.get("source_spans") or [],
        }
        metadata["source_node_fingerprint"] = hashlib.sha256(
            json.dumps(source_node, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
        ).hexdigest()
    payload = {"page_content": page_content, "metadata": metadata}
    if embedding_text is not None:
        payload["embedding_text"] = str(embedding_text)
    if lexical_text is not None:
        payload["lexical_text"] = str(lexical_text)
    return payload


def normalize_document_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return {
            "page_content": str(item.get("page_content", "") or ""),
            "embedding_text": item.get("embedding_text"),
            "lexical_text": item.get("lexical_text"),
            "metadata": dict(item.get("metadata") or {}),
        }
    return {
        "page_content": str(getattr(item, "page_content", "") or ""),
        "metadata": dict(getattr(item, "metadata", {}) or {}),
    }


def validate_processed_documents(items: list[Any], require_provenance: bool = False) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    normalized = [normalize_document_item(item) for item in items]
    counts = Counter(str(item["metadata"].get("node_type") or "unknown") for item in normalized)
    node_ids: dict[str, str] = {}
    child_refs: list[tuple[str, str]] = []

    for index, item in enumerate(normalized):
        content = re.sub(r"\s+", " ", item["page_content"]).strip()
        metadata = item["metadata"]
        node_id = str(metadata.get("node_id") or f"index_{index}")
        node_type = str(metadata.get("node_type") or "unknown")
        label = f"{node_id} ({node_type})"

        if not content:
            errors.append(f"{label} has empty page_content")
        for representation_field in ("embedding_text", "lexical_text"):
            value = item.get(representation_field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                warnings.append(f"{label} has an empty or invalid {representation_field}")
        if node_id in node_ids:
            errors.append(f"{label} duplicates node_id used by {node_ids[node_id]}")
        node_ids[node_id] = label

        for field in REQUIRED_METADATA_FIELDS:
            if metadata.get(field) in (None, ""):
                errors.append(f"{label} is missing metadata.{field}")

        if not metadata.get("episode_date"):
            warnings.append(f"{label} is missing episode_date")
        if node_type == "position_card":
            if metadata.get("speaker_scope") != "single" or not metadata.get("speaker"):
                errors.append(f"{label} must have an attributable single speaker")
            if not metadata.get("claim"):
                errors.append(f"{label} is missing claim metadata")
            if not metadata.get("child_ids"):
                warnings.append(f"{label} has no evidence child_ids")

        if require_provenance:
            if node_type == "leaf_chunk" and not (
                metadata.get("source_segment_ids")
                or metadata.get("source_spans")
                or (metadata.get("start_time") is not None and metadata.get("end_time") is not None)
            ):
                errors.append(f"{label} has no source segment IDs or exact source span/timestamps")
            if node_type != "leaf_chunk" and not metadata.get("child_ids"):
                errors.append(f"{label} has no child evidence links")

        for child_id in metadata.get("child_ids") or []:
            if isinstance(child_id, str) and child_id:
                child_refs.append((node_id, child_id))

    missing_types = REQUIRED_NODE_TYPES.difference(counts)
    for node_type in sorted(missing_types):
        errors.append(f"cache is missing required node_type={node_type}")
    if counts.get("cluster_summary", 0) == 0:
        warnings.append("cache has no cluster_summary nodes")
    for parent_id, child_id in child_refs:
        if child_id not in node_ids:
            errors.append(f"{parent_id} references missing child_id {child_id}")

    if require_provenance:
        by_id = {str(item["metadata"].get("node_id")): item["metadata"] for item in normalized}
        for node_id, metadata in by_id.items():
            parent_id = metadata.get("parent_id")
            if parent_id and parent_id not in by_id:
                errors.append(f"{node_id} references missing parent_id {parent_id}")
            for child_id in metadata.get("child_ids") or []:
                if (
                    metadata.get("node_type") != "position_card"
                    and child_id in by_id
                    and by_id[child_id].get("parent_id") != node_id
                ):
                    errors.append(f"{node_id} child {child_id} has inconsistent parent_id")

            if metadata.get("node_type") == "leaf_chunk":
                continue
            frontier = list(metadata.get("child_ids") or [])
            visited: set[str] = set()
            reaches_leaf = False
            while frontier:
                child_id = str(frontier.pop())
                if child_id in visited:
                    errors.append(f"{node_id} evidence graph contains a cycle at {child_id}")
                    break
                visited.add(child_id)
                child = by_id.get(child_id)
                if child is None:
                    break
                if child.get("node_type") == "leaf_chunk":
                    reaches_leaf = True
                    continue
                frontier.extend(child.get("child_ids") or [])
            if not reaches_leaf:
                errors.append(f"{node_id} does not close to a leaf evidence node")

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings, counts=dict(counts))


def validate_processed_cache(payload: Any) -> ValidationResult:
    """Validate the cache envelope while remaining backward compatible with schema 2.0."""
    if not isinstance(payload, dict):
        return ValidationResult(False, ["cache payload must be a JSON object"], [], {})

    schema_version = str(payload.get("schema_version") or "2.0")
    errors: list[str] = []
    warnings: list[str] = []
    if schema_version not in SUPPORTED_PROCESSED_CACHE_SCHEMA_VERSIONS:
        errors.append(
            f"unsupported schema_version={schema_version}; supported={sorted(SUPPORTED_PROCESSED_CACHE_SCHEMA_VERSIONS)}"
        )

    documents = payload.get("documents")
    if not isinstance(documents, list):
        errors.append("cache documents must be an array")
        return ValidationResult(False, errors, warnings, {})

    document_result = validate_processed_documents(documents, require_provenance=schema_version == "2.1")
    errors.extend(document_result.errors)
    warnings.extend(document_result.warnings)

    representations = payload.get("representations")
    if schema_version == "2.1":
        if not isinstance(representations, dict):
            errors.append("schema 2.1 cache is missing the representations manifest")
        else:
            for required in ("display_text", "dense_text", "lexical_text"):
                if not str(representations.get(required) or "").strip():
                    errors.append(f"representations manifest is missing {required}")
    elif representations is not None and not isinstance(representations, dict):
        warnings.append("cache representations manifest is not an object")

    return ValidationResult(not errors, errors, warnings, document_result.counts)


def schema_summary() -> dict[str, Any]:
    return {
        "schema_version": PROCESSED_CACHE_SCHEMA_VERSION,
        "supported_schema_versions": sorted(SUPPORTED_PROCESSED_CACHE_SCHEMA_VERSIONS),
        "required_node_types": sorted(REQUIRED_NODE_TYPES),
        "required_metadata_fields": sorted(REQUIRED_METADATA_FIELDS),
        "summary_node_types": sorted(SUMMARY_NODE_TYPES),
    }


def dumps_schema_summary() -> str:
    return json.dumps(schema_summary(), indent=2, ensure_ascii=True)

