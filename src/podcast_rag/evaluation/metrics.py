from __future__ import annotations

import math
from statistics import mean
from typing import Any

from podcast_rag.evaluation.dataset import QueryRecord


DEFAULT_CUTOFFS = (5, 10, 20)


def _document_id(result: dict[str, Any]) -> str:
    return str(result.get("document_id") or result.get("id") or result.get("stable_document_id") or "")


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _speaker_values(result: dict[str, Any]) -> set[str]:
    metadata = result.get("metadata") or {}
    values = metadata.get("speakers")
    if not isinstance(values, list):
        values = [metadata.get("speaker")] if metadata.get("speaker") else []
    return {str(value).strip().casefold() for value in values if str(value).strip()}


def _date_matches(result: dict[str, Any], date_range: dict[str, str | None] | None) -> bool:
    if not date_range:
        return True
    value = str((result.get("metadata") or {}).get("episode_date") or "")
    if not value:
        return False
    start = str(date_range.get("start") or "")
    end = str(date_range.get("end") or "")
    return (not start or value >= start) and (not end or value <= end)


def score_query(query: QueryRecord, results: list[dict[str, Any]], cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS) -> dict[str, Any]:
    ranked_ids = [_document_id(result) for result in results]
    relevant = {document_id: grade for document_id, grade in query.relevance.items() if grade > 0}
    metrics: dict[str, Any] = {}
    for cutoff in cutoffs:
        found = {document_id for document_id in ranked_ids[:cutoff] if document_id in relevant}
        metrics[f"recall@{cutoff}"] = len(found) / len(relevant) if relevant else 0.0

    metrics["mrr@10"] = next(
        (1.0 / index for index, document_id in enumerate(ranked_ids[:10], start=1) if document_id in relevant),
        0.0,
    )
    actual_grades = [query.relevance.get(document_id, 0) for document_id in ranked_ids[:10]]
    ideal_grades = sorted(relevant.values(), reverse=True)[:10]
    ideal_dcg = _dcg(ideal_grades)
    metrics["ndcg@10"] = _dcg(actual_grades) / ideal_dcg if ideal_dcg else 0.0
    total_grade = sum(relevant.values())
    covered_grade = sum(query.relevance.get(document_id, 0) for document_id in set(ranked_ids[:10]))
    metrics["evidence_coverage@10"] = covered_grade / total_grade if total_grade else 0.0

    top_ten = results[:10]
    if query.expected_speakers:
        expected = {speaker.casefold() for speaker in query.expected_speakers}
        matches = sum(1 for result in top_ten if _speaker_values(result).intersection(expected))
        metrics["speaker_constraint_precision@10"] = matches / len(top_ten) if top_ten else 0.0
    if query.date_range:
        matches = sum(1 for result in top_ten if _date_matches(result, query.date_range))
        metrics["date_constraint_accuracy@10"] = matches / len(top_ten) if top_ten else 0.0

    node_types = [str((result.get("metadata") or {}).get("node_type") or "unknown") for result in top_ten]
    metrics["node_type_counts@10"] = {node_type: node_types.count(node_type) for node_type in sorted(set(node_types))}
    metrics["returned_ids"] = ranked_ids
    return metrics


def aggregate_metrics(per_query: list[dict[str, Any]]) -> dict[str, float]:
    numeric_keys = sorted(
        {
            key
            for item in per_query
            for key, value in item.get("metrics", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {
        key: mean(float(item["metrics"][key]) for item in per_query if key in item.get("metrics", {}))
        for key in numeric_keys
    }
