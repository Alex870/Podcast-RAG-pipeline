from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import podcast_rag.runtime as runtime
from podcast_rag.text_utils import compact_episode_date, episode_sort_key, parse_episode_date, primary_speaker_from_record, safe_float

def iter_transcript_files(input_dir: Path, file_glob: str) -> list[Path]:
    pattern = str(input_dir / file_glob)
    return [Path(path) for path in sorted(glob.glob(pattern, recursive=True)) if Path(path).is_file()]

def first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None

def extract_segment_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("segments", "transcript", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []

def extract_episode_metadata(payload: Any, path: Path) -> dict[str, Any]:
    source: dict[str, Any] = payload if isinstance(payload, dict) else {}
    nested = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    episode_date = (
        parse_episode_date(first_present(source, ["episode_date", "show_date", "recording_date", "published_date", "date"]))
        or parse_episode_date(first_present(nested, ["episode_date", "show_date", "recording_date", "published_date", "date"]))
        or parse_episode_date(path.name)
    )
    return {
        "episode_date": episode_date,
        "episode_date_compact": first_present(source, ["episode_date_compact"])
        or first_present(nested, ["episode_date_compact"])
        or compact_episode_date(episode_date),
        "episode_sort_key": first_present(source, ["episode_sort_key"])
        or first_present(nested, ["episode_sort_key"])
        or episode_sort_key(episode_date),
    }

def load_transcript_json(path: Path) -> list[Document]:
    """Normalize a transcript JSON payload into segment-level LangChain documents."""
    runtime.load_runtime_deps()
    Document = runtime.Document
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = extract_segment_records(payload)
    episode_metadata = extract_episode_metadata(payload, path)
    docs = []

    for idx, record in enumerate(records):
        text = first_present(record, ["text", "content", "transcript", "sentence"])
        if not text or not str(text).strip():
            continue

        record_episode_date = parse_episode_date(first_present(record, ["episode_date", "show_date", "recording_date", "published_date", "date"]))
        speaker = primary_speaker_from_record(record)
        metadata = {
            "source": str(path),
            "level": "leaf",
            "start_time": safe_float(first_present(record, ["start", "start_time", "timestamp_start"])),
            "end_time": safe_float(first_present(record, ["end", "end_time", "timestamp_end"])),
            "speaker": speaker,
            "segment_index": first_present(record, ["id", "segment_id", "seek"]) or idx,
            "source_type": "json_transcript",
            "episode_date": record_episode_date or episode_metadata["episode_date"],
            "episode_date_compact": first_present(record, ["episode_date_compact"]) or episode_metadata["episode_date_compact"],
            "episode_sort_key": first_present(record, ["episode_sort_key"]) or episode_metadata["episode_sort_key"],
        }
        docs.append(Document(page_content=str(text).strip(), metadata=metadata))

    if docs:
        return docs

    fallback_text = payload.get("text") if isinstance(payload, dict) else None
    if fallback_text:
        return [
            Document(
                page_content=str(fallback_text).strip(),
                metadata={
                    "source": str(path),
                    "level": "leaf",
                    "start_time": None,
                    "end_time": None,
                    "speaker": None,
                    "segment_index": 0,
                    "source_type": "json_transcript",
                    "episode_date": episode_metadata["episode_date"],
                    "episode_date_compact": episode_metadata["episode_date_compact"],
                    "episode_sort_key": episode_metadata["episode_sort_key"],
                },
            )
        ]

    return []
