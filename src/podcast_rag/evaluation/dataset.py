from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VALID_RELEVANCE_GRADES = {0, 1, 2, 3}


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    query: str
    category: str
    relevance: dict[str, int] = field(default_factory=dict)
    expected_speakers: tuple[str, ...] = ()
    date_range: dict[str, str | None] | None = None
    acceptable_node_types: tuple[str, ...] = ()
    answerable: bool = True
    status: str = "judged"
    notes: str = ""

    @property
    def judged(self) -> bool:
        return self.status == "judged" and bool(self.relevance)

    @property
    def evaluable(self) -> bool:
        """Include judged unanswerable queries for abstention diagnostics, not recall."""
        return self.status == "judged" and (bool(self.relevance) or not self.answerable)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], line_number: int | None = None) -> "QueryRecord":
        location = f" on line {line_number}" if line_number else ""
        required = {key: str(payload.get(key) or "").strip() for key in ("query_id", "query", "category")}
        if not all(required.values()):
            raise ValueError(f"query_id, query, and category are required{location}")

        raw_relevance = payload.get("relevance") or {}
        if not isinstance(raw_relevance, dict):
            raise ValueError(f"relevance must be an object{location}")
        relevance: dict[str, int] = {}
        for document_id, raw_grade in raw_relevance.items():
            try:
                grade = int(raw_grade)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid relevance grade for {document_id}{location}") from exc
            if grade not in VALID_RELEVANCE_GRADES:
                raise ValueError(f"relevance grade for {document_id} must be 0-3{location}")
            relevance[str(document_id)] = grade

        date_range = payload.get("date_range")
        if date_range is not None and not isinstance(date_range, dict):
            raise ValueError(f"date_range must be null or an object{location}")
        speakers = tuple(str(value).strip() for value in payload.get("expected_speakers") or [] if str(value).strip())
        node_types = tuple(str(value).strip() for value in payload.get("acceptable_node_types") or [] if str(value).strip())
        return cls(
            query_id=required["query_id"],
            query=required["query"],
            category=required["category"],
            relevance=relevance,
            expected_speakers=speakers,
            date_range=date_range,
            acceptable_node_types=node_types,
            answerable=bool(payload.get("answerable", True)),
            status=str(payload.get("status") or "judged"),
            notes=str(payload.get("notes") or ""),
        )


def load_query_set(path: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"query record at {path}:{line_number} must be an object")
        record = QueryRecord.from_dict(payload, line_number)
        if record.query_id in seen:
            raise ValueError(f"duplicate query_id={record.query_id} at {path}:{line_number}")
        seen.add(record.query_id)
        records.append(record)
    return records
