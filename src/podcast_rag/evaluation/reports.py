from __future__ import annotations

from pathlib import Path
from typing import Any


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"- Run ID: {payload.get('run_id', 'unknown')}",
        f"- Strategy: {payload.get('strategy_id', 'unknown')}",
        f"- Query set: {payload.get('query_set', 'unknown')}",
        f"- Judged queries: {payload.get('judged_query_count', 0)}",
        f"- Draft or unjudged queries skipped: {payload.get('skipped_query_count', 0)}",
        "",
        "## Aggregate Metrics",
        "",
    ]
    aggregate = payload.get("aggregate") or {}
    if aggregate:
        lines.extend(f"- {key}: {float(value):.4f}" for key, value in aggregate.items())
    else:
        lines.append("No judged queries were available.")

    lines.extend(["", "## Queries", ""])
    for item in payload.get("queries") or []:
        lines.extend([f"### {item['query_id']}: {item['query']}", "", f"Category: `{item['category']}`", ""])
        for key, value in item.get("metrics", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lines.append(f"- {key}: {float(value):.4f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(payload), encoding="utf-8")
