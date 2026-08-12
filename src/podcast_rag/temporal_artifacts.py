"""Deterministic, evidence-bound temporal research artifacts."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "temporal-research-1.0"
EXTRACTION_VERSION = "evidence-claim-extraction-1.0"


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evidence_ids(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("source_span_ids") or metadata.get("source_segment_ids") or []
    if metadata.get("source_span_id"):
        values = [*values, metadata["source_span_id"]]
    return sorted({str(value) for value in values if str(value)})


def build_temporal_artifacts(
    processed_data_dir: Path,
    output_path: Path,
    *,
    include_trajectories: bool = False,
    include_contradiction_candidates: bool = False,
    missing_interval_days: int = 180,
) -> dict[str, Any]:
    """Build additive research artifacts without assigning identity from generated wording."""
    claims: list[dict[str, Any]] = []
    source_fingerprints: list[str] = []
    for cache_path in sorted(processed_data_dir.glob("*.processed_documents.json")):
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        source_fingerprint = str(payload.get("source_fingerprint") or cache_path.name)
        source_fingerprints.append(source_fingerprint)
        for document in payload.get("documents") or []:
            metadata = dict(document.get("metadata") or {})
            if str(metadata.get("node_type") or "") not in {"leaf_chunk", "position_card"}:
                continue
            document_id = str(metadata.get("stable_document_id") or metadata.get("node_id") or "")
            evidence_ids = _evidence_ids(metadata)
            if not document_id or not evidence_ids:
                continue
            extraction_identity = _hash({
                "version": EXTRACTION_VERSION,
                "source_fingerprint": source_fingerprint,
                "document_id": document_id,
                "evidence_ids": evidence_ids,
            })
            claim_id = "claim_" + _hash({"extraction_identity": extraction_identity, "evidence_ids": evidence_ids})
            text = str(metadata.get("claim") or document.get("page_content") or "").strip()
            claims.append({
                "claim_id": claim_id,
                "extraction_identity": extraction_identity,
                "extraction_version": EXTRACTION_VERSION,
                "document_id": document_id,
                "source_span_ids": evidence_ids,
                "speaker": str(metadata.get("speaker") or ""),
                "episode_id": str(metadata.get("episode_id") or ""),
                "episode_date": str(metadata.get("episode_date") or ""),
                "topic_ids": sorted(map(str, metadata.get("topic_tags") or [])),
                "display_text": text,
                "uncertainty": str(metadata.get("uncertainty") or "unreviewed_extraction"),
                "primary_evidence": str(metadata.get("node_type") or "") == "leaf_chunk",
            })
    claims.sort(key=lambda item: (item["speaker"], item["episode_date"], item["claim_id"]))
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        by_speaker[claim["speaker"]].append(claim)
    adjacency: list[dict[str, Any]] = []
    missing_intervals: list[dict[str, Any]] = []
    from datetime import date
    for speaker, items in sorted(by_speaker.items()):
        dated = [item for item in items if item["episode_date"]]
        for left, right in zip(dated, dated[1:]):
            adjacency.append({"speaker": speaker, "before_claim_id": left["claim_id"], "after_claim_id": right["claim_id"], "before_date": left["episode_date"], "after_date": right["episode_date"]})
            try:
                gap = (date.fromisoformat(right["episode_date"][:10]) - date.fromisoformat(left["episode_date"][:10])).days
            except ValueError:
                gap = 0
            if gap > missing_interval_days:
                missing_intervals.append({"speaker": speaker, "after": left["episode_date"], "before": right["episode_date"], "days": gap, "qualification": "no_continuity_inference"})
    trajectories = []
    if include_trajectories:
        trajectories = [{"trajectory_id": "trajectory_" + _hash({"speaker": speaker, "claims": [item["claim_id"] for item in items]}), "speaker": speaker, "claim_ids": [item["claim_id"] for item in items], "source_span_ids": sorted({span for item in items for span in item["source_span_ids"]}), "missing_intervals": [gap for gap in missing_intervals if gap["speaker"] == speaker], "qualification": "candidate_unreviewed"} for speaker, items in sorted(by_speaker.items())]
    contradictions = []
    if include_contradiction_candidates:
        for speaker, items in sorted(by_speaker.items()):
            for left, right in zip(items, items[1:]):
                left_negated = any(token in left["display_text"].casefold().split() for token in ("not", "never", "no"))
                right_negated = any(token in right["display_text"].casefold().split() for token in ("not", "never", "no"))
                if left_negated != right_negated and set(left["topic_ids"]).intersection(right["topic_ids"]):
                    contradictions.append({"candidate_id": "contradiction_" + _hash({"left": left["claim_id"], "right": right["claim_id"]}), "speaker": speaker, "claim_ids": [left["claim_id"], right["claim_id"]], "source_span_ids": sorted(set(left["source_span_ids"] + right["source_span_ids"])), "confidence_basis": "opposed_negation_shared_topic", "qualification": "candidate_requires_human_review"})
    for trajectory in trajectories:
        claim_ids=set(trajectory["claim_ids"])
        trajectory["reversal_candidate_ids"]=[item["candidate_id"] for item in contradictions if claim_ids.intersection(item["claim_ids"])]
    artifact = {"contract_version": CONTRACT_VERSION, "producer": "podcast-rag-pipeline", "configuration": {"include_trajectories": include_trajectories, "include_contradiction_candidates": include_contradiction_candidates, "missing_interval_days": missing_interval_days}, "source_fingerprints": sorted(source_fingerprints), "claims": claims, "temporal_adjacency": adjacency, "missing_intervals": missing_intervals, "trajectories": trajectories, "contradiction_candidates": contradictions}
    artifact["artifact_id"] = "temporal_" + _hash(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"artifact_id": artifact["artifact_id"], "claim_count": len(claims), "output_path": str(output_path)}
