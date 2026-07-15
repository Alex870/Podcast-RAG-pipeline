from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from podcast_rag.evaluation.dataset import load_query_set
from podcast_rag.evaluation.metrics import aggregate_metrics, score_query
from podcast_rag.evaluation.reports import write_markdown_report


def _query_set_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_retrieval_run(query_set_path: Path, retrieval_results_path: Path, output_dir: Path) -> dict[str, Any]:
    queries = load_query_set(query_set_path)
    run = json.loads(retrieval_results_path.read_text(encoding="utf-8"))
    if not isinstance(run, dict) or not isinstance(run.get("queries"), list):
        raise ValueError("retrieval results must be an object containing a queries array")

    results_by_query: dict[str, list[dict[str, Any]]] = {}
    for item in run["queries"]:
        if not isinstance(item, dict):
            continue
        query_id = str(item.get("query_id") or "")
        results = item.get("results") or []
        if query_id and isinstance(results, list):
            results_by_query[query_id] = [result for result in results if isinstance(result, dict)]

    per_query = []
    skipped = 0
    for query in queries:
        if not query.judged:
            skipped += 1
            continue
        per_query.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "category": query.category,
                "metrics": score_query(query, results_by_query.get(query.query_id, [])),
            }
        )

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
