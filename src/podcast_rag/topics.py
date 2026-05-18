from __future__ import annotations

import datetime as dt
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from podcast_rag.config import PipelineConfig, config_fingerprint, resolve_path
from podcast_rag.state import read_json_file, write_json_file
from podcast_rag.text_utils import compact_episode_date, episode_sort_key, parse_episode_date, short_text


TOPIC_INDEX_SCHEMA_VERSION = "1.0"
TOPIC_INDEX_LOGIC_VERSION = "2026-05-18"
TOPIC_SOURCE_NODE_TYPES = {"position_card", "cluster_summary", "episode_thesis"}
TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "apologizes",
    "asks",
    "asked",
    "because",
    "coherent",
    "day",
    "episode",
    "general",
    "good",
    "great",
    "homebrew",
    "how",
    "issues",
    "like",
    "maybe",
    "more",
    "other",
    "plan",
    "podcast",
    "quest",
    "really",
    "says",
    "show",
    "speaker",
    "speakers",
    "start",
    "starting",
    "talk",
    "talked",
    "talking",
    "technical",
    "the",
    "their",
    "there",
    "thing",
    "things",
    "this",
    "unknown",
    "very",
    "voodoo",
}
TOPIC_ALIASES = {
    "artificial intelligence": "ai",
    "federal reserve": "fed",
    "fed reserve": "fed",
    "federal reserve inflation": "fed",
    "russia ukraine": "ukraine war",
    "russia ukraine war": "ukraine war",
    "ukraine": "ukraine war",
    "us": "u.s.",
    "u s": "u.s.",
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "topic"


def _cache_file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def _canonical_episode_key(source_path: str) -> str:
    stem = Path(source_path).stem
    stem = re.sub(r"_(cleaned_)?speaker_transcript$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(cleaned_)?host_only$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or Path(source_path).stem


def _cache_priority(source_path: str) -> tuple[int, int]:
    name = Path(source_path).name.lower()
    if "_cleaned_speaker_transcript" in name:
        return (3, 0)
    if "_speaker_transcript" in name:
        return (2, 0)
    if "_cleaned_host_only" in name:
        return (1, 0)
    return (0, 0)


def _infer_podcast_name(source_path: str) -> str:
    base = _canonical_episode_key(source_path)
    match = re.match(r"^(.*?)(?:\s+20\d{2}(?:[-_.]?\d{2}){2}|\s+\d{8})$", base)
    if match:
        base = match.group(1)
    return re.sub(r"\s+", " ", base).strip() or "Podcast"


def _infer_podcast_id(name: str) -> str:
    return _slug(name)


def _normalize_topic_label(label: str) -> tuple[str | None, str | None]:
    raw = re.sub(r"[_/]+", " ", str(label or "")).strip()
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return None, None
    normalized = re.sub(r"[^a-z0-9\s&-]", " ", raw.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip(" -")
    normalized = TOPIC_ALIASES.get(normalized, normalized)
    if not normalized:
        return None, None
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return None, None
    if len(tokens) == 1 and (tokens[0] in TOPIC_STOPWORDS or len(tokens[0]) < 3 and tokens[0] not in {"ai", "uk", "us"}):
        return None, None
    if all(token in TOPIC_STOPWORDS for token in tokens):
        return None, None
    display = raw if raw.isupper() and len(raw) <= 8 else normalized.title()
    if normalized == "ai":
        display = "AI"
    elif normalized == "u.s.":
        display = "U.S."
    return normalized, display


def _extract_doc_topics(doc: dict[str, Any]) -> list[str]:
    metadata = dict(doc.get("metadata") or {})
    labels: list[str] = []
    for field in ("topic_tags", "keywords"):
        for value in metadata.get(field) or []:
            if isinstance(value, str):
                labels.append(value)
    claim = str(metadata.get("claim") or "").strip()
    if claim:
        labels.extend(re.split(r"[;,/]|(?:\s+-\s+)", claim))
    return labels


def _doc_excerpt(doc: dict[str, Any], max_chars: int = 320) -> str:
    metadata = dict(doc.get("metadata") or {})
    page_content = str(doc.get("page_content", "") or "").strip()
    claim = str(metadata.get("claim") or "").strip()
    if claim:
        return short_text(claim, max_chars)
    return short_text(page_content, max_chars)


def _doc_weight(node_type: str) -> float:
    if node_type == "position_card":
        return 3.0
    if node_type == "episode_thesis":
        return 2.0
    return 1.0


def _make_sample_questions(label: str) -> list[str]:
    return [
        f"What are the host's views on {label}?",
        f"How have the host's views on {label} changed over time?",
        f"What arguments recur in the corpus about {label}?",
    ]


def _days_between(start: str | None, end: str | None) -> int:
    if not start or not end:
        return 0
    try:
        return max(0, (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days)
    except ValueError:
        return 0


def _depth_label(score: float) -> str:
    if score >= 0.66:
        return "High"
    if score >= 0.33:
        return "Medium"
    return "Low"


def _summarize_topic(topic: dict[str, Any]) -> str:
    keywords = ", ".join(topic.get("top_keywords") or []) or "mixed cues"
    return (
        f"Covered in {topic['episode_count']} episode(s) across {topic['document_count']} supporting document(s) "
        f"from {topic.get('first_episode_date') or 'unknown'} to {topic.get('latest_episode_date') or 'unknown'}, "
        f"with recurring emphasis on {keywords}."
    )


def _topic_state_paths(config: PipelineConfig, project_dir: Path) -> tuple[Path, Path, Path]:
    contribution_dir = resolve_path(project_dir, config.topic_contribution_dir)
    index_path = resolve_path(project_dir, config.topic_index_path)
    manifest_path = resolve_path(project_dir, config.topic_index_manifest_path)
    return contribution_dir, index_path, manifest_path


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": TOPIC_INDEX_SCHEMA_VERSION, "logic_version": TOPIC_INDEX_LOGIC_VERSION, "episodes": {}}
    payload = read_json_file(path)
    if not isinstance(payload, dict):
        return {"schema_version": TOPIC_INDEX_SCHEMA_VERSION, "logic_version": TOPIC_INDEX_LOGIC_VERSION, "episodes": {}}
    payload.setdefault("episodes", {})
    return payload


def _selected_cache_records(processed_data_dir: Path, config: PipelineConfig) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for cache_path in sorted(processed_data_dir.glob("*.processed_documents.json")):
        try:
            payload = read_json_file(cache_path)
        except Exception:
            continue
        source_path = str(payload.get("source_path") or cache_path.name)
        canonical_episode = _canonical_episode_key(source_path)
        record = {
            "cache_path": cache_path,
            "payload": payload,
            "source_path": source_path,
            "canonical_episode": canonical_episode,
            "priority": _cache_priority(source_path),
        }
        current = chosen.get(canonical_episode)
        if current is None:
            chosen[canonical_episode] = record
            continue
        if record["priority"] > current["priority"]:
            chosen[canonical_episode] = record
            continue
        if record["priority"] == current["priority"] and cache_path.stat().st_mtime_ns > current["cache_path"].stat().st_mtime_ns:
            chosen[canonical_episode] = record
    return chosen


def build_episode_topic_contribution(config: PipelineConfig, cache_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    source_path = str(payload.get("source_path") or cache_path.name)
    podcast_name = config.podcast_name.strip() or _infer_podcast_name(source_path)
    podcast_id = config.podcast_id.strip() or _infer_podcast_id(podcast_name)
    canonical_episode = _canonical_episode_key(source_path)
    source_fingerprint = str(payload.get("source_fingerprint") or "")
    docs = [doc for doc in payload.get("documents") or [] if isinstance(doc, dict)]

    episode_date = parse_episode_date(payload.get("episode_date") or "")
    if not episode_date:
        for doc in docs:
            metadata = dict(doc.get("metadata") or {})
            episode_date = parse_episode_date(metadata.get("episode_date"))
            if episode_date:
                break
    episode_title = canonical_episode

    topic_map: dict[str, dict[str, Any]] = {}
    for doc in docs:
        metadata = dict(doc.get("metadata") or {})
        node_type = str(metadata.get("node_type") or "unknown")
        if node_type not in TOPIC_SOURCE_NODE_TYPES:
            continue
        raw_topics = _extract_doc_topics(doc)
        if not raw_topics:
            continue
        for raw_topic in raw_topics:
            normalized, display = _normalize_topic_label(raw_topic)
            if not normalized or not display:
                continue
            topic = topic_map.setdefault(
                normalized,
                {
                    "topic_key": _slug(normalized),
                    "canonical_label": display,
                    "aliases": Counter(),
                    "score": 0.0,
                    "document_count": 0,
                    "node_type_counts": Counter(),
                    "speaker_counts": Counter(),
                    "keyword_counts": Counter(),
                    "evidence": [],
                },
            )
            topic["aliases"][display] += 1
            topic["score"] += _doc_weight(node_type)
            topic["document_count"] += 1
            topic["node_type_counts"][node_type] += 1
            speaker = str(metadata.get("speaker") or "").strip()
            if speaker:
                topic["speaker_counts"][speaker] += 1
            for keyword in metadata.get("keywords") or []:
                norm_keyword, disp_keyword = _normalize_topic_label(str(keyword))
                if norm_keyword and disp_keyword:
                    topic["keyword_counts"][disp_keyword] += 1
            topic["evidence"].append(
                {
                    "stable_document_id": str(metadata.get("stable_document_id") or ""),
                    "node_id": str(metadata.get("node_id") or ""),
                    "node_type": node_type,
                    "speaker": speaker,
                    "speaker_scope": str(metadata.get("speaker_scope") or ""),
                    "episode_date": parse_episode_date(metadata.get("episode_date")) or episode_date or "",
                    "episode_title": str(metadata.get("episode_title") or episode_title),
                    "source_path": str(metadata.get("source") or source_path),
                    "excerpt": _doc_excerpt(doc),
                    "topic_tags": [str(item) for item in metadata.get("topic_tags") or [] if isinstance(item, str)],
                    "keywords": [str(item) for item in metadata.get("keywords") or [] if isinstance(item, str)],
                }
            )

    topics = []
    for normalized, topic in topic_map.items():
        aliases = [label for label, _count in topic["aliases"].most_common()]
        label = aliases[0] if aliases else normalized.title()
        evidence = sorted(topic["evidence"], key=lambda item: (_doc_weight(item["node_type"]), len(item["excerpt"] or "")), reverse=True)[:6]
        topics.append(
            {
                "topic_key": topic["topic_key"],
                "normalized_label": normalized,
                "label": label,
                "aliases": aliases,
                "score": round(float(topic["score"]), 4),
                "document_count": int(topic["document_count"]),
                "position_count": int(topic["node_type_counts"].get("position_card", 0)),
                "cluster_summary_count": int(topic["node_type_counts"].get("cluster_summary", 0)),
                "episode_thesis_count": int(topic["node_type_counts"].get("episode_thesis", 0)),
                "top_speakers": [speaker for speaker, _count in topic["speaker_counts"].most_common(4)],
                "top_keywords": [keyword for keyword, _count in topic["keyword_counts"].most_common(6)],
                "evidence": evidence,
            }
        )

    topics.sort(key=lambda item: (-item["score"], -item["document_count"], item["label"].lower()))
    return {
        "schema_version": TOPIC_INDEX_SCHEMA_VERSION,
        "logic_version": TOPIC_INDEX_LOGIC_VERSION,
        "cache_path": str(cache_path),
        "cache_signature": _cache_file_signature(cache_path),
        "source_fingerprint": source_fingerprint,
        "source_path": source_path,
        "canonical_episode_key": canonical_episode,
        "podcast_id": podcast_id,
        "podcast_name": podcast_name,
        "episode_title": episode_title,
        "episode_date": episode_date or "",
        "episode_date_compact": compact_episode_date(episode_date) or "",
        "episode_sort_key": episode_sort_key(episode_date) or 0,
        "topic_count": len(topics),
        "topics": topics,
    }


def _aggregate_topic_index(config: PipelineConfig, project_dir: Path, contributions: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_map: dict[tuple[str, str], dict[str, Any]] = {}
    episode_counts_by_podcast: Counter[str] = Counter()
    podcast_names: dict[str, str] = {}

    for contribution in contributions:
        podcast_id = str(contribution.get("podcast_id") or "")
        podcast_name = str(contribution.get("podcast_name") or podcast_id or "Podcast")
        podcast_names[podcast_id] = podcast_name
        episode_counts_by_podcast[podcast_id] += 1
        for topic in contribution.get("topics") or []:
            key = (podcast_id, str(topic.get("normalized_label") or topic.get("label") or ""))
            bucket = bucket_map.setdefault(
                key,
                {
                    "podcast_id": podcast_id,
                    "podcast_name": podcast_name,
                    "topic_key": str(topic.get("topic_key") or ""),
                    "aliases": Counter(),
                    "episode_keys": set(),
                    "scores": [],
                    "document_count": 0,
                    "position_count": 0,
                    "cluster_summary_count": 0,
                    "episode_thesis_count": 0,
                    "keyword_counts": Counter(),
                    "speaker_counts": Counter(),
                    "evidence": [],
                    "first_episode_date": "",
                    "latest_episode_date": "",
                    "representative_episode_title": "",
                },
            )
            bucket["aliases"].update(topic.get("aliases") or [])
            bucket["episode_keys"].add(str(contribution.get("canonical_episode_key") or contribution.get("source_fingerprint") or ""))
            bucket["scores"].append(float(topic.get("score") or 0.0))
            bucket["document_count"] += int(topic.get("document_count") or 0)
            bucket["position_count"] += int(topic.get("position_count") or 0)
            bucket["cluster_summary_count"] += int(topic.get("cluster_summary_count") or 0)
            bucket["episode_thesis_count"] += int(topic.get("episode_thesis_count") or 0)
            bucket["keyword_counts"].update(topic.get("top_keywords") or [])
            bucket["speaker_counts"].update(topic.get("top_speakers") or [])
            episode_date = str(contribution.get("episode_date") or "")
            if episode_date and (not bucket["first_episode_date"] or episode_date < bucket["first_episode_date"]):
                bucket["first_episode_date"] = episode_date
            if episode_date and (not bucket["latest_episode_date"] or episode_date > bucket["latest_episode_date"]):
                bucket["latest_episode_date"] = episode_date
                bucket["representative_episode_title"] = str(contribution.get("episode_title") or "")
            for evidence in topic.get("evidence") or []:
                entry = dict(evidence)
                entry["canonical_episode_key"] = contribution.get("canonical_episode_key")
                bucket["evidence"].append(entry)

    max_episode_count = max((len(bucket["episode_keys"]) for bucket in bucket_map.values()), default=1)
    max_document_count = max((bucket["document_count"] for bucket in bucket_map.values()), default=1)
    max_position_count = max((bucket["position_count"] for bucket in bucket_map.values()), default=1)
    max_timespan_days = max((_days_between(bucket["first_episode_date"], bucket["latest_episode_date"]) for bucket in bucket_map.values()), default=1)

    topics = []
    for (_podcast_id, _normalized_label), bucket in bucket_map.items():
        aliases = [label for label, _count in bucket["aliases"].most_common()]
        label = aliases[0] if aliases else bucket["topic_key"].replace("-", " ").title()
        episode_count = len(bucket["episode_keys"])
        timespan_days = _days_between(bucket["first_episode_date"], bucket["latest_episode_date"])
        depth_score = (
            0.35 * (episode_count / max_episode_count)
            + 0.30 * (bucket["document_count"] / max_document_count)
            + 0.20 * (timespan_days / max_timespan_days if max_timespan_days else 0.0)
            + 0.15 * (bucket["position_count"] / max_position_count if max_position_count else 0.0)
        )
        evidence = sorted(
            bucket["evidence"],
            key=lambda item: (
                item.get("episode_date") or "",
                _doc_weight(str(item.get("node_type") or "")),
                len(str(item.get("excerpt") or "")),
            ),
        )
        earliest = evidence[0] if evidence else {}
        latest = evidence[-1] if evidence else {}
        topic = {
            "podcast_id": bucket["podcast_id"],
            "podcast_name": bucket["podcast_name"],
            "topic_key": bucket["topic_key"],
            "label": label,
            "aliases": aliases,
            "depth_score": round(depth_score, 4),
            "depth_label": _depth_label(depth_score),
            "episode_count": episode_count,
            "document_count": int(bucket["document_count"]),
            "position_count": int(bucket["position_count"]),
            "cluster_summary_count": int(bucket["cluster_summary_count"]),
            "episode_thesis_count": int(bucket["episode_thesis_count"]),
            "first_episode_date": bucket["first_episode_date"],
            "latest_episode_date": bucket["latest_episode_date"],
            "timespan_days": timespan_days,
            "top_keywords": [keyword for keyword, _count in bucket["keyword_counts"].most_common(8)],
            "top_speakers": [speaker for speaker, _count in bucket["speaker_counts"].most_common(5)],
            "representative_episode_title": bucket["representative_episode_title"],
            "representative_excerpt": earliest.get("excerpt") or latest.get("excerpt") or "",
            "earliest_evidence": earliest,
            "latest_evidence": latest,
            "sample_questions": _make_sample_questions(label),
            "related_doc_ids": [item.get("stable_document_id") for item in evidence if item.get("stable_document_id")][:12],
            "evidence": evidence[:12],
        }
        topic["summary"] = _summarize_topic(topic)
        topics.append(topic)

    topics.sort(key=lambda item: (-item["depth_score"], -item["episode_count"], item["label"].lower()))
    podcasts = [
        {
            "podcast_id": podcast_id,
            "podcast_name": podcast_names.get(podcast_id, podcast_id or "Podcast"),
            "episode_count": int(episode_counts_by_podcast[podcast_id]),
            "topic_count": sum(1 for topic in topics if topic["podcast_id"] == podcast_id),
        }
        for podcast_id in sorted(podcast_names, key=lambda item: podcast_names[item].lower())
    ]
    return {
        "schema_version": TOPIC_INDEX_SCHEMA_VERSION,
        "logic_version": TOPIC_INDEX_LOGIC_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint(config),
        "processed_data_dir": str(resolve_path(project_dir, config.processed_data_dir)),
        "topic_contribution_dir": str(resolve_path(project_dir, config.topic_contribution_dir)),
        "podcasts": podcasts,
        "topic_count": len(topics),
        "topics": topics,
    }


def refresh_topic_index(config: PipelineConfig, project_dir: Path) -> dict[str, Any]:
    processed_data_dir = resolve_path(project_dir, config.processed_data_dir)
    contribution_dir, index_path, manifest_path = _topic_state_paths(config, project_dir)
    contribution_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_cache_records(processed_data_dir, config)
    manifest = _load_manifest(manifest_path)
    next_manifest = {
        "schema_version": TOPIC_INDEX_SCHEMA_VERSION,
        "logic_version": TOPIC_INDEX_LOGIC_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint(config),
        "episodes": {},
    }

    contributions = []
    reused = 0
    rebuilt = 0
    removed = 0

    existing_episode_keys = set(manifest.get("episodes", {}))
    selected_episode_keys = set(selected)
    stale_episode_keys = existing_episode_keys.difference(selected_episode_keys)

    for episode_key, record in selected.items():
        cache_path = record["cache_path"]
        payload = record["payload"]
        cache_signature = _cache_file_signature(cache_path)
        old_entry = (manifest.get("episodes") or {}).get(episode_key) or {}
        contribution_name = f"{_slug(episode_key)}.{str(payload.get('source_fingerprint') or 'unknown')[:12]}.topic_contribution.json"
        contribution_path = contribution_dir / contribution_name

        if (
            old_entry.get("cache_signature") == cache_signature
            and old_entry.get("logic_version") == TOPIC_INDEX_LOGIC_VERSION
            and contribution_path.exists()
        ):
            contribution = read_json_file(contribution_path)
            reused += 1
        else:
            contribution = build_episode_topic_contribution(config, cache_path, payload)
            write_json_file(contribution_path, contribution)
            rebuilt += 1

        contributions.append(contribution)
        next_manifest["episodes"][episode_key] = {
            "cache_path": str(cache_path),
            "cache_signature": cache_signature,
            "source_path": str(payload.get("source_path") or ""),
            "source_fingerprint": str(payload.get("source_fingerprint") or ""),
            "contribution_path": str(contribution_path),
            "logic_version": TOPIC_INDEX_LOGIC_VERSION,
        }

    for episode_key in stale_episode_keys:
        stale_entry = (manifest.get("episodes") or {}).get(episode_key) or {}
        stale_path = Path(str(stale_entry.get("contribution_path") or ""))
        if stale_path and stale_path.exists():
            stale_path.unlink()
            removed += 1

    topic_index = _aggregate_topic_index(config, project_dir, contributions)
    write_json_file(index_path, topic_index)
    write_json_file(manifest_path, next_manifest)
    return {
        "topic_index_path": str(index_path),
        "topic_count": int(topic_index.get("topic_count") or 0),
        "podcast_count": len(topic_index.get("podcasts") or []),
        "episode_count": len(contributions),
        "reused_contributions": reused,
        "rebuilt_contributions": rebuilt,
        "removed_contributions": removed,
    }
