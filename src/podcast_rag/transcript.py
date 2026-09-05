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
    partition = nested.get("partition") if isinstance(nested.get("partition"), dict) else source.get("partition")
    partition = partition if isinstance(partition, dict) else {}
    partition_id = (
        first_present(source, ["partition_id"])
        or first_present(nested, ["partition_id"])
        or first_present(partition, ["partition_id"])
    )
    episode_id = (
        first_present(source, ["episode_id"])
        or first_present(nested, ["episode_id"])
        or path.stem
    )
    return {
        "episode_id": episode_id,
        "episode_uid": first_present(source, ["episode_uid"])
        or first_present(nested, ["episode_uid"]),
        "episode_date": episode_date,
        "episode_date_compact": first_present(source, ["episode_date_compact"])
        or first_present(nested, ["episode_date_compact"])
        or compact_episode_date(episode_date),
        "episode_sort_key": first_present(source, ["episode_sort_key"])
        or first_present(nested, ["episode_sort_key"])
        or episode_sort_key(episode_date),
        "partition_id": partition_id,
        "corpus_id": partition_id
        or first_present(source, ["corpus_id"])
        or first_present(nested, ["corpus_id"])
        or first_present(partition, ["corpus_id"]),
        "partition_display_name": first_present(source, ["partition_display_name"])
        or first_present(nested, ["partition_display_name"])
        or first_present(partition, ["partition_display_name"]),
        "context_type": first_present(source, ["context_type"])
        or first_present(nested, ["context_type"])
        or first_present(partition, ["context_type"]),
        "workflow_profile": first_present(source, ["workflow_profile"])
        or first_present(nested, ["workflow_profile"])
        or first_present(partition, ["workflow_profile"]),
        "partition_config_fingerprint": first_present(source, ["partition_config_fingerprint"])
        or first_present(nested, ["partition_config_fingerprint"])
        or first_present(partition, ["partition_config_fingerprint"]),
        "correction_set_id": first_present(source, ["correction_set_id"])
        or first_present(nested, ["correction_set_id"]),
        "selected_variant": first_present(source, ["text_version", "selected_variant"])
        or first_present(nested, ["text_version", "selected_variant"]),
        "source_audio_fingerprint": first_present(source, ["source_audio_fingerprint"])
        or first_present(nested, ["source_audio_fingerprint"]),
    }


def partition_scope(files: list[Path]) -> dict[str, Any]:
    """Inspect a transcript root and require one processing-space identity."""
    identities: dict[str, dict[str, Any]] = {}
    legacy_files: list[str] = []
    errors: list[str] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Leave malformed input for the per-file validation/errata boundary;
            # it has no partition identity that could create a false scope mix.
            legacy_files.append(str(path))
            continue
        metadata = extract_episode_metadata(payload, path)
        identity = {
            key: metadata[key]
            for key in (
                "episode_id",
                "episode_uid",
                "partition_id",
                "corpus_id",
                "partition_display_name",
                "context_type",
                "workflow_profile",
                "partition_config_fingerprint",
                "correction_set_id",
                "selected_variant",
                "source_audio_fingerprint",
            )
            if metadata.get(key) not in (None, "")
        }
        key = str(identity.get("partition_id") or identity.get("corpus_id") or "")
        if key:
            identities.setdefault(key, identity)
        else:
            legacy_files.append(str(path))
    if len(identities) > 1:
        errors.append("transcript input contains multiple processing spaces: " + ", ".join(sorted(identities)))
    if identities and legacy_files:
        errors.append("transcript input mixes partition-aware and legacy files: " + ", ".join(legacy_files[:5]))
    return {
        "valid": not errors,
        "partition": next(iter(identities.values()), {}),
        "partition_ids": sorted(identities),
        "legacy_file_count": len(legacy_files),
        "errors": errors,
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
            **episode_metadata,
            "source": str(path),
            "level": "leaf",
            "start_time": safe_float(first_present(record, ["start", "start_time", "timestamp_start"])),
            "end_time": safe_float(first_present(record, ["end", "end_time", "timestamp_end"])),
            "speaker": speaker,
            "segment_index": first_present(record, ["id", "segment_id", "seek"]) or idx,
            "source_segment_id": first_present(record, ["source_span_id", "segment_id", "id"]) or f"{episode_metadata['episode_id']}:segment:{idx}",
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
                    **episode_metadata,
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
