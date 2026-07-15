from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from podcast_rag.evaluation.dataset import load_query_set
from podcast_rag.evaluation.metrics import aggregate_by_category, aggregate_metrics, score_query
from podcast_rag.evaluation.reports import write_markdown_report


def _query_set_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_retrieval_run(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid retrieval-results JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"retrieval-results record at {path}:{line_number} must be an object")
            records.append(record)
        return {"queries": records, "run_id": path.stem, "strategy_id": "jsonl"}
    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        return payload
    if isinstance(payload, dict) and payload.get("query_id"):
        return {"queries": [payload], "run_id": path.stem, "strategy_id": "jsonl"}
    raise ValueError("retrieval results must be an object containing a queries array or JSONL query records")


def evaluate_retrieval_run(query_set_path: Path, retrieval_results_path: Path, output_dir: Path) -> dict[str, Any]:
    queries = load_query_set(query_set_path)
    run = _load_retrieval_run(retrieval_results_path)

    results_by_query: dict[str, list[dict[str, Any]]] = {}
    for item in run["queries"]:
        if not isinstance(item, dict):
            continue
        query_id = str(item.get("query_id") or "")
        results = item.get("results") or []
        if query_id and isinstance(results, list):
            results_by_query[query_id] = [result for result in results if isinstance(result, dict)]

    per_query = []
    abstention_outcomes = []
    skipped = 0
    for query in queries:
        if not query.evaluable:
            skipped += 1
            continue
        result_item = next(
            (
                item
                for item in run["queries"]
                if isinstance(item, dict) and str(item.get("query_id") or "") == query.query_id
            ),
            {},
        )
        results = results_by_query.get(query.query_id, [])
        abstained = bool(result_item.get("abstained")) or bool(str(result_item.get("abstention_reason") or "").strip())
        abstention = {
            "expected": not query.answerable,
            "observed": abstained,
            "correct": abstained == (not query.answerable),
            "reason": str(result_item.get("abstention_reason") or ""),
        }
        if not query.answerable:
            abstention_outcomes.append({"query_id": query.query_id, **abstention})
        metrics = score_query(
            query,
            results,
            latency_ms=result_item.get("latency_ms")
            if result_item.get("latency_ms") is not None
            else (
                float(result_item["latency_seconds"]) * 1000
                if result_item.get("latency_seconds") is not None
                else None
            ),
        )
        per_query.append({
            "query_id": query.query_id,
            "query": query.query,
            "category": query.category,
            "answerable": query.answerable,
            "acceptable_node_types": list(query.acceptable_node_types),
            "abstention": abstention,
            "metrics": metrics,
        })

    run_id = str(run.get("run_id") or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    report = {
        "schema_version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "strategy_id": str(run.get("strategy_id") or "unknown"),
        "query_set": str(query_set_path),
        "query_set_sha256": _query_set_hash(query_set_path),
        "retrieval_results": str(retrieval_results_path),
        "judged_query_count": len(per_query),
        "skipped_query_count": skipped,
        "aggregate": aggregate_metrics(per_query),
        "aggregate_by_category": aggregate_by_category(per_query),
        "abstention": {
            "query_count": len(abstention_outcomes),
            "correct_count": sum(1 for item in abstention_outcomes if item["correct"]),
            "accuracy": (
                sum(1 for item in abstention_outcomes if item["correct"]) / len(abstention_outcomes)
                if abstention_outcomes
                else None
            ),
            "queries": abstention_outcomes,
        },
        "queries": per_query,
        "run_manifest": run.get("manifest") or {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}.retrieval_evaluation.json"
    markdown_path = output_dir / f"{run_id}.retrieval_evaluation.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    write_markdown_report(markdown_path, report)
    report["json_report_path"] = str(json_path)
    report["markdown_report_path"] = str(markdown_path)
    return report
