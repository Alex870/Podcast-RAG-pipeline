from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import podcast_rag.runtime as runtime


TOPIC_STOPWORDS = {
    "about",
    "after",
    "against",
    "because",
    "before",
    "being",
    "could",
    "every",
    "going",
    "host",
    "maybe",
    "other",
    "really",
    "should",
    "their",
    "there",
    "these",
    "thing",
    "those",
    "under",
    "where",
    "which",
    "would",
}

def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def stable_episode_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]

def new_node_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None

def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    return str(dt.timedelta(seconds=int(max(0, seconds))))

def estimate_remaining_seconds(completed: int, total: int, elapsed_seconds: float) -> float | None:
    if completed <= 0 or total <= completed:
        return None
    return elapsed_seconds / completed * (total - completed)

def parse_episode_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    match = re.search(r"(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return text

def compact_episode_date(value: Any) -> str | None:
    episode_date = parse_episode_date(value)
    if not episode_date:
        return None
    match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", episode_date)
    if not match:
        return None
    return "".join(match.groups())

def episode_sort_key(value: Any) -> int | None:
    compact = compact_episode_date(value)
    if not compact:
        return None
    return int(compact)

def short_text(text: str, max_chars: int = 280) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."

def extract_json_payload(text: str) -> Any:
    text = text.strip()
    if not text:
        return {}

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = min([idx for idx in [text.find("{"), text.find("[")] if idx != -1], default=-1)
    if start == -1:
        return {}

    for end in range(len(text), start, -1):
        snippet = text[start:end]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return {}

def extract_position_objects_from_partial_json(text: str) -> list[dict[str, Any]]:
    match = re.search(r'"positions"\s*:\s*\[', text or "")
    if not match:
        return []

    decoder = json.JSONDecoder()
    idx = match.end()
    positions = []
    while idx < len(text):
        while idx < len(text) and text[idx] in " \r\n\t,":
            idx += 1
        if idx >= len(text) or text[idx] == "]":
            break
        if text[idx] != "{":
            idx += 1
            continue
        try:
            payload, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            break
        if isinstance(payload, dict) and payload.get("claim"):
            positions.append(payload)
        idx += end
    return positions

def with_retry(func, label: str, retries: int = 3, delay: int = 1):
    for attempt in range(retries):
        if runtime.STOP_REQUESTED:
            raise runtime.PipelineInterrupted("Stop requested before retrying a model request.")
        try:
            return func()
        except Exception:
            if attempt == retries - 1:
                raise
            print(f"{label} retry {attempt + 1}")
            time.sleep(delay)

def episode_title_from_source(source: str) -> str:
    return Path(source).stem

def merge_speaker_values(values) -> list[str]:
    return [value for value in dict.fromkeys(v for v in values if v)]

def primary_speaker_from_record(record: dict[str, Any]) -> str | None:
    words = record.get("words")
    if isinstance(words, list):
        counts: dict[str, int] = {}
        for word in words:
            if not isinstance(word, dict):
                continue
            speaker = word.get("speaker")
            if not speaker:
                continue
            counts[str(speaker)] = counts.get(str(speaker), 0) + 1
        if counts:
            named_counts = {speaker: count for speaker, count in counts.items() if not re.fullmatch(r"SPEAKER_\d+", speaker)}
            selected = named_counts or counts
            return max(selected.items(), key=lambda item: item[1])[0]
    speaker = first_present(record, ["speaker", "speaker_name", "speaker_id", "voice", "who"])
    return str(speaker) if speaker else None

def speaker_scope(speakers: list[str]) -> tuple[str, str]:
    if len(speakers) == 1:
        return speakers[0], "single"
    if len(speakers) > 1:
        return "", "multi"
    return "", "unknown"

def has_substantive_text(text: str, min_chars: int = 40) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip()
    return len(compact) >= min_chars

def is_missing_context_response(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip().lower()
    if not compact:
        return False
    asks_for_missing_input = (
        ("please provide" in compact or "please share" in compact or "send me" in compact)
        and any(term in compact for term in ["transcript", "source text", "source material", "podcast text", "material"])
    )
    deferred_until_shared = any(pattern in compact for pattern in ["once shared", "once you share", "when you provide"])
    return asks_for_missing_input or deferred_until_shared

def fallback_summary_from_text(text: str, label: str, max_chars: int = 1800) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        raise ValueError(f"{label} had no source text for fallback summary.")
    lines = []
    seen = set()
    current_header = ""
    for raw_line in re.split(r"\n+|(?<=\.)\s+", text):
        line = re.sub(r"\s+", " ", raw_line or "").strip()
        if line.startswith("[") and line.endswith("]"):
            current_header = line.strip("[]")
            continue
        if not has_substantive_text(line, min_chars=30):
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        speaker = ""
        time_span = ""
        match = re.match(r"^\[([^|\]]+)\]\s*(.+)$", line)
        if match:
            header = match.group(1)
            line = match.group(2).strip()
        else:
            header = current_header
        if header:
            speaker_match = re.search(r"speakers?=([^|]+)", header)
            time_match = re.search(r"time=([^|]+)", header)
            if speaker_match:
                speaker = speaker_match.group(1).strip()
            if time_match:
                time_span = time_match.group(1).strip()
        prefix = "Fallback summary"
        if speaker:
            prefix += f" | speaker={speaker}"
        if time_span:
            prefix += f" | time={time_span}"
        bullet = f"- {prefix}: {clip_text(line, max_chars=360)}"
        lines.append(bullet)
        if sum(len(item) + 1 for item in lines) >= max_chars:
            break
        if len(lines) >= 8:
            break
    if not lines:
        lines = [f"- Fallback summary: {short_text(compact, max_chars=max_chars)}"]
    return "\n".join(lines)

def clip_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= max_chars:
        return compact
    boundary = max(compact.rfind(". ", 0, max_chars), compact.rfind("; ", 0, max_chars), compact.rfind(", ", 0, max_chars))
    if boundary < int(max_chars * 0.55):
        boundary = max_chars
    return compact[:boundary].rstrip(" .,;:") + "..."

def compact_reduced_summaries(summaries: list[str], label: str, max_chars: int) -> str:
    parts = [re.sub(r"\s+", " ", summary or "").strip() for summary in summaries if has_substantive_text(summary, min_chars=1)]
    if not parts:
        raise ValueError(f"{label} had no reduced summaries to compact.")
    combined = "\n".join(parts)
    return fallback_summary_from_text(combined, f"deterministic compaction for {label}", max_chars=max_chars)

def extract_summary_bullets(text: str) -> list[str]:
    bullets = []
    current = ""
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^[-*]\s+", line):
            if current:
                bullets.append(current.strip())
            current = re.sub(r"^[-*]\s+", "- ", line)
        elif current:
            current = f"{current} {line}"
        else:
            bullets.append(f"- {line}")
    if current:
        bullets.append(current.strip())
    return bullets

def deterministic_episode_overview(docs: list[Any], max_chars: int) -> str:
    max_chars = max(800, int(max_chars or 3200))
    heading = "Deterministic episode overview:\n"
    selected = []
    seen = set()

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        prefix_parts = []
        if metadata.get("episode_date_compact"):
            prefix_parts.append(str(metadata["episode_date_compact"]))
        if metadata.get("speaker_scope") == "single" and metadata.get("speaker"):
            prefix_parts.append(str(metadata["speaker"]))
        elif metadata.get("speakers"):
            prefix_parts.append(", ".join(str(speaker) for speaker in metadata["speakers"][:4]))
        prefix = f"[{'; '.join(prefix_parts)}] " if prefix_parts else ""

        for bullet in extract_summary_bullets(getattr(doc, "page_content", "")):
            bullet = re.sub(r"\s+", " ", bullet).strip()
            if not bullet:
                continue
            if not bullet.startswith("- "):
                bullet = f"- {bullet}"
            normalized = re.sub(r"[^a-z0-9]+", " ", bullet.lower()).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            selected.append(f"- {prefix}{bullet[2:].strip()}")

    if not selected:
        return compact_reduced_summaries(
            [getattr(doc, "page_content", "") for doc in docs],
            "episode overview",
            max_chars=max_chars,
        )

    output_lines = []
    used_chars = len(heading)
    for line in selected:
        next_chars = len(line) + 1
        if output_lines and used_chars + next_chars > max_chars:
            break
        if not output_lines and used_chars + next_chars > max_chars:
            line = f"- {clip_text(line[2:], max_chars=max_chars - used_chars - 4)}"
        output_lines.append(line)
        used_chars += next_chars

    return heading + "\n".join(output_lines)

def deterministic_topic_tags(text: str, max_tags: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text or "")
    counts: Counter[str] = Counter()
    for word in words:
        lowered = word.lower().strip("_-")
        if lowered in TOPIC_STOPWORDS or lowered.isdigit():
            continue
        counts[lowered] += 1
    return [word for word, _count in counts.most_common(max(1, max_tags))]

def normalized_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

def token_set_similarity(left: str, right: str) -> float:
    left_tokens = set(normalized_text_key(left).split())
    right_tokens = set(normalized_text_key(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

def token_estimate(text: str, chars_per_token: float = 4.0) -> int:
    chars_per_token = max(1.0, float(chars_per_token or 4.0))
    return int(len(text or "") / chars_per_token) + 1

def source_schema_version(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return coerce_text(payload.get("schema_version") or payload.get("transcript_schema_version") or payload.get("version")) or None
    return None

def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(coerce_text(item) for item in value if coerce_text(item)).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value).strip()

def coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [coerce_text(item) for item in value if coerce_text(item)]
    text = coerce_text(value)
    return [text] if text else []

def text_fingerprint(values: list[str]) -> str:
    payload = "\n\n".join(re.sub(r"\s+", " ", value or "").strip() for value in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
