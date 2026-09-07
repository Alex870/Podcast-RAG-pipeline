from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
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
    issues: list[dict[str, Any]] = field(default_factory=list)

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
    issues: list[dict[str, Any]] = []

    def add_error(
        message: str,
        *,
        code: str = "validation_error",
        node_id: str | None = None,
        evidence_path: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        errors.append(message)
        issue: dict[str, Any] = {"code": code, "message": message}
        if node_id:
            issue["node_ids"] = [node_id]
        if evidence_path:
            issue["evidence_paths"] = [evidence_path]
        if details:
            issue["details"] = details
        issues.append(issue)

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
            add_error(f"{label} has empty page_content", code="empty_page_content", node_id=node_id)
        for representation_field in ("embedding_text", "lexical_text"):
            value = item.get(representation_field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                warnings.append(f"{label} has an empty or invalid {representation_field}")
        if node_id in node_ids:
            add_error(
                f"{label} duplicates node_id used by {node_ids[node_id]}",
                code="duplicate_node_id",
                node_id=node_id,
            )
        node_ids[node_id] = label

        for field in REQUIRED_METADATA_FIELDS:
            if metadata.get(field) in (None, ""):
                add_error(
                    f"{label} is missing metadata.{field}",
                    code="missing_required_metadata",
                    node_id=node_id,
                    details={"field": field},
                )

        if not metadata.get("episode_date"):
            warnings.append(f"{label} is missing episode_date")
        if node_type == "position_card":
            if metadata.get("speaker_scope") != "single" or not metadata.get("speaker"):
                add_error(
                    f"{label} must have an attributable single speaker",
                    code="position_missing_speaker",
                    node_id=node_id,
                )
            if not metadata.get("claim"):
                add_error(f"{label} is missing claim metadata", code="position_missing_claim", node_id=node_id)
            if not metadata.get("child_ids"):
                warnings.append(f"{label} has no evidence child_ids")

        if require_provenance:
            if node_type == "leaf_chunk" and not (
                metadata.get("source_segment_ids")
                or metadata.get("source_spans")
                or (metadata.get("start_time") is not None and metadata.get("end_time") is not None)
            ):
                add_error(
                    f"{label} has no source segment IDs or exact source span/timestamps",
                    code="leaf_missing_source_provenance",
                    node_id=node_id,
                )
            if node_type != "leaf_chunk" and not metadata.get("child_ids"):
                add_error(
                    f"{label} has no child evidence links",
                    code="derived_missing_child_links",
                    node_id=node_id,
                )

        for child_id in metadata.get("child_ids") or []:
            if isinstance(child_id, str) and child_id:
                child_refs.append((node_id, child_id))

        generation_refs = metadata.get("generation_source_node_ids")
        if generation_refs is not None:
            if not isinstance(generation_refs, list):
                add_error(
                    f"{label} has non-array generation_source_node_ids",
                    code="invalid_generation_source_references",
                    node_id=node_id,
                )
            else:
                for generation_id in generation_refs:
                    if not isinstance(generation_id, str) or not generation_id:
                        add_error(
                            f"{label} has an invalid generation source reference",
                            code="invalid_generation_source_reference",
                            node_id=node_id,
                        )

    missing_types = REQUIRED_NODE_TYPES.difference(counts)
    for node_type in sorted(missing_types):
        add_error(
            f"cache is missing required node_type={node_type}",
            code="missing_required_node_type",
            details={"node_type": node_type},
        )
    if counts.get("cluster_summary", 0) == 0:
        warnings.append("cache has no cluster_summary nodes")
    for parent_id, child_id in child_refs:
        if child_id not in node_ids:
            add_error(
                f"{parent_id} references missing child_id {child_id}",
                code="missing_child_reference",
                node_id=parent_id,
                evidence_path=[parent_id, child_id],
            )

    for item in normalized:
        metadata = item["metadata"]
        node_id = str(metadata.get("node_id") or "")
        generation_refs = metadata.get("generation_source_node_ids")
        for generation_id in generation_refs if isinstance(generation_refs, list) else []:
            if isinstance(generation_id, str) and generation_id not in node_ids:
                add_error(
                    f"{node_id} references missing generation source node {generation_id}",
                    code="missing_generation_source_reference",
                    node_id=node_id,
                    evidence_path=[node_id, generation_id],
                )

    if require_provenance:
        by_id = {str(item["metadata"].get("node_id")): item["metadata"] for item in normalized}

        def validate_evidence_closure(
            root_id: str,
            metadata: dict[str, Any],
            *,
            structural_only: bool,
        ) -> None:
            """Require a leaf and reject only back-edges on the current path.

            The path is carried per frontier entry instead of using one global
            visited set. This accepts shared evidence and direct-plus-transitive
            citations while still rejecting an actual back-edge.
            """
            frontier = [
                (str(child_id), (root_id,))
                for child_id in metadata.get("child_ids") or []
            ]
            reaches_leaf = False
            while frontier:
                child_id, path = frontier.pop()
                if child_id in path:
                    cycle_path = list(path) + [child_id]
                    add_error(
                        f"{root_id} evidence graph contains a cycle at {child_id}: {' -> '.join(cycle_path)}",
                        code="evidence_graph_cycle",
                        node_id=root_id,
                        evidence_path=cycle_path,
                        details={"cycle_node_id": child_id, "traversal_path": cycle_path},
                    )
                    continue
                child = by_id.get(child_id)
                if child is None:
                    continue
                if structural_only and child.get("node_type") == "position_card":
                    add_error(
                        f"{root_id} structurally references position card {child_id}",
                        code="structural_position_reference",
                        node_id=root_id,
                        evidence_path=list(path) + [child_id],
                    )
                    continue
                if child.get("node_type") == "leaf_chunk":
                    reaches_leaf = True
                    continue
                child_path = path + (child_id,)
                frontier.extend(
                    (str(grandchild_id), child_path)
                    for grandchild_id in child.get("child_ids") or []
                )
            if not reaches_leaf:
                add_error(
                    f"{root_id} does not close to a leaf evidence node",
                    code="evidence_missing_leaf_closure",
                    node_id=root_id,
                    evidence_path=[root_id],
                )

        for node_id, metadata in by_id.items():
            parent_id = metadata.get("parent_id")
            if parent_id and parent_id not in by_id:
                add_error(
                    f"{node_id} references missing parent_id {parent_id}",
                    code="missing_parent_reference",
                    node_id=node_id,
                    evidence_path=[node_id, str(parent_id)],
                )
            for child_id in metadata.get("child_ids") or []:
                if (
                    metadata.get("node_type") != "position_card"
                    and child_id in by_id
                    and by_id[child_id].get("parent_id") != node_id
                ):
                    add_error(
                        f"{node_id} child {child_id} has inconsistent parent_id",
                        code="inconsistent_structural_parent",
                        node_id=node_id,
                        evidence_path=[node_id, str(child_id)],
                        details={"expected_parent_id": node_id, "actual_parent_id": by_id[child_id].get("parent_id")},
                    )

            if metadata.get("node_type") == "leaf_chunk":
                continue
            validate_evidence_closure(
                node_id,
                metadata,
                structural_only=metadata.get("node_type") != "position_card",
            )

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings, counts=dict(counts), issues=issues)


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
    issues = list(document_result.issues)

    representations = payload.get("representations")
    if schema_version == "2.1":
        if not isinstance(representations, dict):
            errors.append("schema 2.1 cache is missing the representations manifest")
            issues.append({"code": "missing_representations_manifest", "message": errors[-1]})
        else:
            for required in ("display_text", "dense_text", "lexical_text"):
                if not str(representations.get(required) or "").strip():
                    message = f"representations manifest is missing {required}"
                    errors.append(message)
                    issues.append({"code": "missing_representation", "message": message, "details": {"field": required}})
    elif representations is not None and not isinstance(representations, dict):
        warnings.append("cache representations manifest is not an object")

    hierarchy_manifest = payload.get("hierarchy_manifest")
    if hierarchy_manifest is not None:
        if not isinstance(hierarchy_manifest, dict):
            errors.append("cache hierarchy_manifest must be an object when present")
            issues.append({"code": "invalid_hierarchy_manifest", "message": errors[-1]})
        elif hierarchy_manifest.get("levels") is not None and not isinstance(hierarchy_manifest.get("levels"), list):
            errors.append("cache hierarchy_manifest.levels must be an array when present")
            issues.append({"code": "invalid_hierarchy_manifest_levels", "message": errors[-1]})

    return ValidationResult(not errors, errors, warnings, document_result.counts, issues)


def schema_summary() -> dict[str, Any]:
    return {
        "schema_version": PROCESSED_CACHE_SCHEMA_VERSION,
        "supported_schema_versions": sorted(SUPPORTED_PROCESSED_CACHE_SCHEMA_VERSIONS),
        "required_node_types": sorted(REQUIRED_NODE_TYPES),
        "required_metadata_fields": sorted(REQUIRED_METADATA_FIELDS),
        "summary_node_types": sorted(SUMMARY_NODE_TYPES),
        "optional_metadata_fields": ["generation_source_node_ids"],
        "cache_manifests": ["representations", "hierarchy_manifest"],
    }


def dumps_schema_summary() -> str:
    return json.dumps(schema_summary(), indent=2, ensure_ascii=True)

