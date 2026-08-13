"""M5 prototype artifacts for evidence graphs and late chunking.

These artifacts are deliberately representation sidecars. They do not replace the
dense/hybrid baseline and cannot be marked promoted without an external evaluation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

ENTRY_GATE_CONTRACT = "advanced-retrieval-entry-gate-1.0"
GRAPH_CONTRACT = "evidence-graph-1.0"
LATE_CHUNK_CONTRACT = "late-chunk-alignment-1.0"
REQUIRED_GATE_FIELDS = (
    "failing_query_slices",
    "failure_examples",
    "simpler_methods_attempted",
    "hypothesis",
    "expected_gain",
    "budgets",
    "accepted_baseline",
    "minimum_reviewed_coverage",
    "stop_condition",
)


class AdvancedRetrievalError(ValueError):
    pass


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_entry_gate(gate: dict[str, Any]) -> dict[str, Any]:
    encountered = str(gate.get("contract_version") or "")
    if (
        not encountered.startswith("advanced-retrieval-entry-gate-")
        or encountered.rsplit("-", 1)[-1].split(".", 1)[0] != "1"
    ):
        raise AdvancedRetrievalError(
            f"unsupported M5 entry gate contract: {gate.get('contract_version')}"
        )
    missing = [field for field in REQUIRED_GATE_FIELDS if not gate.get(field)]
    if missing:
        raise AdvancedRetrievalError(
            "M5 entry gate is incomplete: " + ", ".join(missing)
        )
    result = dict(gate)
    result["gate_id"] = "m5_gate_" + _hash(
        {k: v for k, v in result.items() if k != "gate_id"}
    )
    result["disposition"] = "prototype"
    return result


def _documents(
    processed_data_dir: Path,
) -> Iterable[tuple[str, dict[str, Any], dict[str, Any]]]:
    for cache_path in sorted(processed_data_dir.glob("*.processed_documents.json")):
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        source = str(payload.get("source_fingerprint") or cache_path.name)
        for document in payload.get("documents") or []:
            metadata = dict(document.get("metadata") or {})
            document_id = str(
                metadata.get("stable_document_id")
                or metadata.get("node_id")
                or document.get("id")
                or ""
            )
            if document_id:
                yield source, document, metadata


def build_evidence_graph(
    processed_data_dir: Path,
    temporal_artifact_path: Path,
    output_path: Path,
    *,
    corpus_release_id: str,
    entry_gate: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic evidence graph with only inspectable edge types."""
    gate = validate_entry_gate(entry_gate)
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def node(node_id: str, kind: str, **metadata: Any) -> None:
        current = nodes.get(node_id)
        if current is None or (
            current.get("node_type") == "document_reference" and kind == "document"
        ):
            nodes[node_id] = {"node_id": node_id, "node_type": kind, **metadata}

    def edge(
        source: str, target: str, edge_type: str, provenance: dict[str, Any]
    ) -> None:
        if source and target and source != target:
            edges[(source, target, edge_type)] = {
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "provenance": provenance,
                "supersession_state": "active",
            }

    for source_fingerprint, document, metadata in _documents(processed_data_dir):
        document_id = str(
            metadata.get("stable_document_id")
            or metadata.get("node_id")
            or document.get("id")
        )
        node(
            document_id,
            "document",
            primary_evidence=str(metadata.get("node_type")) == "leaf_chunk",
            text=str(document.get("page_content") or ""),
            metadata=metadata,
        )
        provenance = {
            "document_id": document_id,
            "source_fingerprint": source_fingerprint,
        }
        parent = str(metadata.get("parent_id") or metadata.get("parent_node_id") or "")
        if parent:
            node(parent, "document_reference")
            edge(parent, document_id, "parent_child", provenance)
        speaker = str(metadata.get("speaker") or "")
        episode = str(metadata.get("episode_id") or "")
        if speaker and episode:
            speaker_id, episode_id = "speaker_" + _hash(speaker), "episode_" + _hash(
                episode
            )
            node(speaker_id, "speaker", label=speaker)
            node(episode_id, "episode", label=episode)
            edge(speaker_id, episode_id, "speaker_episode", provenance)
            edge(episode_id, document_id, "episode_document", provenance)
        for topic in sorted(map(str, metadata.get("topic_tags") or [])):
            topic_id = "topic_" + _hash(topic)
            node(topic_id, "topic", label=topic)
            edge(topic_id, document_id, "topic_document", provenance)

    temporal = (
        json.loads(temporal_artifact_path.read_text(encoding="utf-8"))
        if temporal_artifact_path.is_file()
        else {}
    )
    for claim in temporal.get("claims") or []:
        claim_id, document_id = str(claim.get("claim_id") or ""), str(
            claim.get("document_id") or ""
        )
        if claim_id and document_id in nodes:
            node(claim_id, "claim", source_span_ids=claim.get("source_span_ids") or [])
            edge(
                claim_id,
                document_id,
                "claim_evidence",
                {"temporal_artifact_id": temporal.get("artifact_id")},
            )
    for item in temporal.get("temporal_adjacency") or []:
        before, after = str(item.get("before_claim_id") or ""), str(
            item.get("after_claim_id") or ""
        )
        if before in nodes and after in nodes:
            edge(
                before,
                after,
                "temporal_adjacency",
                {"temporal_artifact_id": temporal.get("artifact_id")},
            )

    dangling = sorted(
        {
            endpoint
            for item in edges.values()
            for endpoint in (item["source"], item["target"])
            if endpoint not in nodes
        }
    )
    if dangling:
        raise AdvancedRetrievalError(
            "evidence graph has dangling endpoints: " + ", ".join(dangling[:5])
        )
    document_count = sum(item["node_type"] == "document" for item in nodes.values())
    covered = {
        item["target"]
        for item in edges.values()
        if nodes[item["target"]]["node_type"] == "document"
    }
    artifact = {
        "contract_version": GRAPH_CONTRACT,
        "disposition": "prototype",
        "parent_corpus_release_id": corpus_release_id,
        "entry_gate_id": gate["gate_id"],
        "nodes": sorted(nodes.values(), key=lambda item: item["node_id"]),
        "edges": sorted(
            edges.values(),
            key=lambda item: (item["source"], item["target"], item["edge_type"]),
        ),
        "coverage": {
            "document_count": document_count,
            "linked_document_count": len(covered),
            "ratio": len(covered) / max(1, document_count),
        },
        "allowed_edge_types": [
            "parent_child",
            "claim_evidence",
            "speaker_episode",
            "episode_document",
            "topic_document",
            "temporal_adjacency",
        ],
    }
    artifact["graph_id"] = "evidence_graph_" + _hash(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        output_path.exists()
        and json.loads(output_path.read_text(encoding="utf-8")) != artifact
    ):
        raise FileExistsError(f"immutable evidence graph already exists: {output_path}")
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _tokens(text: str) -> list[dict[str, Any]]:
    return [
        {"token": match.group(0), "start": match.start(), "end": match.end()}
        for match in re.finditer(r"\S+", text)
    ]


def build_late_chunk_alignment(
    processed_data_dir: Path,
    output_path: Path,
    *,
    corpus_release_id: str,
    entry_gate: dict[str, Any],
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    pooling: str = "mean-token-pooling",
    window_tokens: int = 512,
    overlap_tokens: int = 64,
    truncation: str = "none",
) -> dict[str, Any]:
    """Record exact token/source alignment for an external long-context encoder."""
    gate = validate_entry_gate(entry_gate)
    if not model_revision or model_revision == "unresolved":
        raise AdvancedRetrievalError(
            "late chunking requires an immutable model revision"
        )
    if window_tokens <= overlap_tokens or overlap_tokens < 0:
        raise AdvancedRetrievalError(
            "window_tokens must be greater than overlap_tokens"
        )
    rows = []
    for source, document, metadata in _documents(processed_data_dir):
        document_id = str(
            metadata.get("stable_document_id")
            or metadata.get("node_id")
            or document.get("id")
        )
        text = str(document.get("page_content") or "")
        tokens = _tokens(text)
        windows = []
        step = window_tokens - overlap_tokens
        for start in range(0, len(tokens), step):
            batch = tokens[start : start + window_tokens]
            if not batch:
                break
            windows.append(
                {
                    "token_start": start,
                    "token_end": start + len(batch),
                    "char_start": batch[0]["start"],
                    "char_end": batch[-1]["end"],
                }
            )
            if start + window_tokens >= len(tokens):
                break
        rows.append(
            {
                "document_id": document_id,
                "source_fingerprint": source,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "character_count": len(text),
                "token_count": len(tokens),
                "tokens": tokens,
                "windows": windows,
            }
        )
    artifact = {
        "contract_version": LATE_CHUNK_CONTRACT,
        "disposition": "prototype",
        "parent_corpus_release_id": corpus_release_id,
        "entry_gate_id": gate["gate_id"],
        "encoder": {
            "model_id": model_id,
            "model_revision": model_revision,
            "tokenizer_id": tokenizer_id,
            "pooling": pooling,
            "window_tokens": window_tokens,
            "overlap_tokens": overlap_tokens,
            "truncation": truncation,
        },
        "documents": sorted(rows, key=lambda item: item["document_id"]),
        "incremental_rebuild_key": "document_id+text_sha256+encoder",
        "vectors_included": False,
    }
    artifact["alignment_id"] = "late_chunk_" + _hash(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        output_path.exists()
        and json.loads(output_path.read_text(encoding="utf-8")) != artifact
    ):
        raise FileExistsError(
            f"immutable late-chunk alignment already exists: {output_path}"
        )
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact
