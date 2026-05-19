from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import podcast_rag.runtime as runtime
from podcast_rag.config import PipelineConfig, config_fingerprint, resolve_path
from podcast_rag.llm_support import extract_llm_text
from podcast_rag.state import read_json_file, write_json_file
from podcast_rag.text_utils import compact_episode_date, episode_sort_key, parse_episode_date, short_text


TOPIC_INDEX_SCHEMA_VERSION = "1.1"
TOPIC_INDEX_LOGIC_VERSION = "2026-05-18-curated"
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
    "only",
    "maybe",
    "more",
    "other",
    "over",
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
    "than",
    "unknown",
    "very",
    "voodoo",
    "were",
    "when",
    "will",
    "without",
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
GENERIC_TOPIC_LABELS = {
    "acknowledges",
    "advocate",
    "advocates",
    "argue",
    "argues",
    "assert",
    "asserts",
    "believe",
    "believes",
    "calls",
    "comments",
    "compares",
    "criticize",
    "criticizes",
    "criticise",
    "criticises",
    "describes",
    "discuss",
    "discusses",
    "expresses",
    "explains",
    "introduces",
    "mention",
    "mentions",
    "notes",
    "oppose",
    "opposes",
    "question",
    "questions",
    "recommends",
    "responds",
    "says",
    "suggests",
    "support",
    "supports",
    "talks",
    "thinks",
    "uses",
    "warns",
    "despite",
    "lack",
    "using",
    "over",
    "only",
    "than",
    "when",
    "were",
    "will",
    "without",
}
META_TOPIC_QUESTION_TEMPLATES = {
    "advocate": [
        "What subjects does the host advocate for most strongly?",
        "What positions does the host push most consistently?",
        "Where does the host sound most committed or forceful?",
    ],
    "criticize": [
        "What subjects does the host criticize most strongly?",
        "What recurring objections does the host make?",
        "Where does the host sound most dismissive or hostile?",
    ],
    "question": [
        "What subjects does the host question or challenge most often?",
        "What assumptions does the host push back against?",
        "Where does the host sound most skeptical?",
    ],
    "discuss": [
        "What subjects does the host spend the most time discussing?",
        "What themes recur most often in the corpus?",
        "What topics keep resurfacing across episodes?",
    ],
    "believe": [
        "What beliefs does the host return to most often?",
        "What core assumptions seem to shape the host's worldview?",
        "What positions appear most consistent across episodes?",
    ],
}
TOPIC_LABEL_NOISE_PATTERNS = (
    re.compile(r"^speaker\s+\d+$"),
    re.compile(r"^speaker[_\s-]*[a-z0-9]+$"),
)


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


def _topic_label_tokens(label: str) -> list[str]:
    return [
        token
        for token in "".join(character.lower() if character.isalnum() else " " for character in label).split()
        if token
    ]


def _normalize_topic_key(label: str) -> str:
    raw = re.sub(r"[_/]+", " ", str(label or "")).strip()
    raw = re.sub(r"\s+", " ", raw)
    normalized = re.sub(r"[^a-z0-9\s&.-]", " ", raw.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip(" -.")
    return TOPIC_ALIASES.get(normalized, normalized)


def _topic_label_verdict(label: str) -> tuple[str | None, str | None, str | None]:
    raw = re.sub(r"[_/]+", " ", str(label or "")).strip()
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return None, None, "empty"
    normalized = _normalize_topic_key(raw)
    if not normalized:
        return None, None, "empty"
    if any(pattern.fullmatch(normalized) for pattern in TOPIC_LABEL_NOISE_PATTERNS):
        return None, None, "speaker_placeholder"
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return None, None, "empty"
    if len(tokens) == 1 and (tokens[0] in TOPIC_STOPWORDS or len(tokens[0]) < 3 and tokens[0] not in {"ai", "uk", "us"}):
        return None, None, "stopword_or_too_short"
    if len(tokens) == 1 and tokens[0] in GENERIC_TOPIC_LABELS:
        return None, None, "generic_reporting_verb"
    if len(tokens) == 1 and tokens[0].endswith("s") and tokens[0][:-1] in GENERIC_TOPIC_LABELS:
        return None, None, "generic_reporting_verb"
    if len(tokens) <= 2 and any(token == "speaker" for token in tokens):
        return None, None, "speaker_placeholder"
    if all(token in TOPIC_STOPWORDS for token in tokens):
        return None, None, "all_stopwords"
    display = raw if raw.isupper() and len(raw) <= 8 else normalized.title()
    if normalized == "ai":
        display = "AI"
    elif normalized == "u.s.":
        display = "U.S."
    return normalized, display, None


def _topic_meta_question_key(label: str) -> str | None:
    tokens = _topic_label_tokens(label)
    if not tokens:
        return None
    primary = tokens[0]
    if primary in {"advocate", "advocates", "support", "supports", "promote", "promotes"}:
        return "advocate"
    if primary in {"criticize", "criticizes", "criticise", "criticises", "oppose", "opposes"}:
        return "criticize"
    if primary in {"question", "questions", "challenge", "challenges", "doubt", "doubts"}:
        return "question"
    if primary in {"discuss", "discusses", "mention", "mentions", "talk", "talks", "describe", "describes"}:
        return "discuss"
    if primary in {"believe", "believes", "think", "thinks", "argue", "argues", "assert", "asserts"}:
        return "believe"
    return None


def _infer_topic_kind(label: str, top_keywords: list[str] | None = None) -> str:
    tokens = _topic_label_tokens(label)
    if not tokens:
        return "subject"
    if _topic_meta_question_key(label):
        return "meta_pattern"
    if len(tokens) == 1 and tokens[0].isupper():
        return "entity"
    if any(keyword.lower() in {"policy", "regulation", "law", "tax", "tariff", "fed"} for keyword in (top_keywords or [])):
        return "policy_theme"
    return "subject"


def _question_templates_for_topic(label: str, topic_kind: str) -> list[str]:
    meta_key = _topic_meta_question_key(label)
    if meta_key:
        return list(META_TOPIC_QUESTION_TEMPLATES[meta_key])
    if topic_kind == "policy_theme":
        return [
            f"What policy positions does the host take on {label}?",
            f"How has the host's framing of {label} changed over time?",
            f"What tradeoffs or objections recur in the corpus about {label}?",
        ]
    return [
        f"What are the host's views on {label}?",
        f"How have the host's views on {label} changed over time?",
        f"What arguments recur in the corpus about {label}?",
    ]


def _build_query_hints(label: str, aliases: list[str], top_keywords: list[str], topic_kind: str) -> list[str]:
    hints: list[str] = [label, *aliases[:4], *top_keywords[:6]]
    meta_key = _topic_meta_question_key(label)
    if meta_key == "advocate":
        hints.extend(["what the host advocates for", "what the host supports most strongly"])
    elif meta_key == "criticize":
        hints.extend(["what the host criticizes", "what the host objects to most often"])
    elif meta_key == "question":
        hints.extend(["what the host questions", "what the host is skeptical about"])
    elif meta_key == "discuss":
        hints.extend(["recurring themes", "most discussed subjects"])
    elif meta_key == "believe":
        hints.extend(["core beliefs", "recurring assumptions"])
    elif topic_kind == "policy_theme":
        hints.extend([f"{label} policy", f"{label} regulation"])
    seen: set[str] = set()
    normalized_hints: list[str] = []
    for hint in hints:
        compact = re.sub(r"\s+", " ", str(hint or "")).strip()
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized_hints.append(compact)
    return normalized_hints[:12]


def _normalize_topic_label(label: str) -> tuple[str | None, str | None]:
    normalized, display, _reason = _topic_label_verdict(label)
    return normalized, display


def _is_obvious_topic_noise(normalized_label: str) -> bool:
    normalized, _display, _reason = _topic_label_verdict(normalized_label)
    return normalized is None


def _extract_doc_topics(doc: dict[str, Any]) -> list[str]:
    metadata = dict(doc.get("metadata") or {})
    labels: list[str] = []
    for field in ("topic_tags", "keywords"):
        for value in metadata.get(field) or []:
            if isinstance(value, str):
                labels.append(value)
    claim = str(metadata.get("claim") or "").strip()
    if (
        claim
        and not labels
        and len(claim) <= 80
        and len(claim.split()) <= 6
        and not re.search(r"[.!?]", claim)
    ):
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


def _make_sample_questions(label: str, topic_kind: str) -> list[str]:
    return _question_templates_for_topic(label, topic_kind)


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


def _topic_curation_paths(config: PipelineConfig, project_dir: Path) -> tuple[Path, Path]:
    return (
        resolve_path(project_dir, config.topic_blacklist_path),
        resolve_path(project_dir, config.topic_whitelist_path),
    )


def _topic_curation_report_path(config: PipelineConfig, project_dir: Path) -> Path:
    return resolve_path(project_dir, config.topic_curation_report_path)


def _load_label_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = read_json_file(path)
    if isinstance(payload, dict):
        values = payload.get("labels") or payload.get("topics") or []
    else:
        values = payload
    labels: set[str] = set()
    if not isinstance(values, list):
        return labels
    for value in values:
        normalized = _normalize_topic_key(str(value or ""))
        if normalized:
            labels.add(normalized)
    return labels


def _write_label_set(path: Path, labels: set[str]) -> None:
    write_json_file(
        path,
        {
            "schema_version": "1.0",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "labels": sorted(labels),
        },
    )


def _load_topic_curation(config: PipelineConfig, project_dir: Path) -> tuple[set[str], set[str]]:
    blacklist_path, whitelist_path = _topic_curation_paths(config, project_dir)
    return _load_label_set(blacklist_path), _load_label_set(whitelist_path)


def _topic_is_curated_out(normalized_label: str, blacklist: set[str], whitelist: set[str]) -> bool:
    if normalized_label in whitelist:
        return False
    return normalized_label in blacklist


def _collect_topic_review_candidates(
    contributions: list[dict[str, Any]],
    blacklist: set[str],
    whitelist: set[str],
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for contribution in contributions:
        episode_key = str(contribution.get("canonical_episode_key") or contribution.get("source_fingerprint") or "")
        for topic in contribution.get("topics") or []:
            normalized = str(topic.get("normalized_label") or "").strip()
            if not normalized or normalized in blacklist or normalized in whitelist:
                continue
            entry = candidates.setdefault(
                normalized,
                {
                    "label": str(topic.get("label") or normalized.title()),
                    "topic_kind": str(topic.get("topic_kind") or "subject"),
                    "aliases": Counter(),
                    "episode_keys": set(),
                    "document_count": 0,
                    "keyword_counts": Counter(),
                },
            )
            entry["aliases"].update(topic.get("aliases") or [])
            entry["episode_keys"].add(episode_key)
            entry["document_count"] += int(topic.get("document_count") or 0)
            entry["keyword_counts"].update(topic.get("top_keywords") or [])
    return candidates


def _needs_llm_topic_review(candidate: dict[str, Any]) -> bool:
    label = str(candidate.get("label") or "")
    tokens = _topic_label_tokens(label)
    if not tokens:
        return False
    if any(pattern.fullmatch(_normalize_topic_key(label)) for pattern in TOPIC_LABEL_NOISE_PATTERNS):
        return False
    if len(tokens) == 1:
        return True
    if candidate.get("topic_kind") == "meta_pattern":
        return True
    if len(tokens) == 2 and all(len(token) <= 5 for token in tokens):
        return True
    return False


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in topic curation response.")
    return json.loads(text[start : end + 1])


def _curate_topic_labels_with_llm(
    config: PipelineConfig,
    candidates: dict[str, dict[str, Any]],
) -> tuple[set[str], set[str]]:
    if not config.enable_llm_topic_label_curation or config.fake_llm:
        return set(), set()
    reviewable = [
        {
            "normalized_label": normalized,
            "label": candidate["label"],
            "topic_kind": candidate["topic_kind"],
            "episode_count": len(candidate["episode_keys"]),
            "document_count": candidate["document_count"],
            "aliases": [alias for alias, _count in candidate["aliases"].most_common(4)],
            "top_keywords": [keyword for keyword, _count in candidate["keyword_counts"].most_common(6)],
        }
        for normalized, candidate in candidates.items()
        if _needs_llm_topic_review(candidate)
    ]
    if not reviewable:
        return set(), set()

    runtime.load_runtime_deps()
    client = runtime.OpenAI(base_url=config.lm_studio_base_url, api_key=config.lm_studio_api_key)
    keep: set[str] = set()
    drop: set[str] = set()
    limit = max(0, int(config.llm_topic_label_curation_limit or 0))
    batches = reviewable[:limit] if limit else reviewable

    for offset in range(0, len(batches), 40):
        batch = batches[offset : offset + 40]
        response = client.chat.completions.create(
            model=config.lm_studio_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You curate candidate podcast topic labels for a browsing UI. "
                        "Keep only labels that are meaningful browseable subjects, entities, policy themes, "
                        "or stable rhetorical/meta patterns. Drop function words, grammar leftovers, generic verbs, "
                        "speaker placeholders, and vague fragments that would make bad topic rows. "
                        "Return JSON only with keys keep and drop, each containing normalized_label values."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"topics": batch}, ensure_ascii=True),
                },
            ],
            max_tokens=800,
        )
        message = response.choices[0].message if getattr(response, "choices", None) else ""
        text = extract_llm_text(message) or getattr(message, "content", "") or ""
        payload = _extract_json_object(str(text))
        for value in payload.get("keep") or []:
            normalized = _normalize_topic_key(str(value or ""))
            if normalized:
                keep.add(normalized)
        for value in payload.get("drop") or []:
            normalized = _normalize_topic_key(str(value or ""))
            if normalized:
                drop.add(normalized)
    drop.difference_update(keep)
    return keep, drop


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
    filtered_topic_labels: list[dict[str, Any]] = []
    for doc in docs:
        metadata = dict(doc.get("metadata") or {})
        node_type = str(metadata.get("node_type") or "unknown")
        if node_type not in TOPIC_SOURCE_NODE_TYPES:
            continue
        raw_topics = _extract_doc_topics(doc)
        if not raw_topics:
            continue
        for raw_topic in raw_topics:
            normalized, display, rejection_reason = _topic_label_verdict(raw_topic)
            if not normalized or not display:
                filtered_topic_labels.append(
                    {
                        "raw_label": str(raw_topic or ""),
                        "normalized_label": _normalize_topic_key(str(raw_topic or "")),
                        "reason": rejection_reason or "filtered",
                        "node_type": node_type,
                        "stable_document_id": str(metadata.get("stable_document_id") or ""),
                    }
                )
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
                "topic_kind": _infer_topic_kind(label, [keyword for keyword, _count in topic["keyword_counts"].most_common(6)]),
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
        "filtered_topic_labels": filtered_topic_labels,
    }


def _aggregate_topic_index(
    config: PipelineConfig,
    project_dir: Path,
    contributions: list[dict[str, Any]],
    blacklist: set[str] | None = None,
    whitelist: set[str] | None = None,
) -> dict[str, Any]:
    blacklist = blacklist or set()
    whitelist = whitelist or set()
    bucket_map: dict[tuple[str, str], dict[str, Any]] = {}
    episode_counts_by_podcast: Counter[str] = Counter()
    podcast_names: dict[str, str] = {}

    for contribution in contributions:
        podcast_id = str(contribution.get("podcast_id") or "")
        podcast_name = str(contribution.get("podcast_name") or podcast_id or "Podcast")
        podcast_names[podcast_id] = podcast_name
        episode_counts_by_podcast[podcast_id] += 1
        for topic in contribution.get("topics") or []:
            normalized_label = str(topic.get("normalized_label") or topic.get("label") or "")
            if (
                not normalized_label
                or _is_obvious_topic_noise(normalized_label)
                or _topic_is_curated_out(normalized_label, blacklist, whitelist)
            ):
                continue
            key = (podcast_id, normalized_label)
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
        topic_kind = _infer_topic_kind(label, [keyword for keyword, _count in bucket["keyword_counts"].most_common(8)])
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
            "topic_kind": topic_kind,
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
            "sample_questions": _make_sample_questions(label, topic_kind),
            "question_templates": _question_templates_for_topic(label, topic_kind),
            "query_hints": _build_query_hints(
                label,
                aliases,
                [keyword for keyword, _count in bucket["keyword_counts"].most_common(8)],
                topic_kind,
            ),
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
    llm_curated_keep = 0
    llm_curated_drop = 0
    deterministic_filtered_counts: Counter[str] = Counter()
    deterministic_filtered_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

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
        for filtered in contribution.get("filtered_topic_labels") or []:
            normalized_label = str(filtered.get("normalized_label") or "").strip()
            if not normalized_label:
                continue
            deterministic_filtered_counts[normalized_label] += 1
            if len(deterministic_filtered_examples[normalized_label]) < 3:
                deterministic_filtered_examples[normalized_label].append(
                    {
                        "raw_label": str(filtered.get("raw_label") or ""),
                        "reason": str(filtered.get("reason") or "filtered"),
                        "episode_key": episode_key,
                        "node_type": str(filtered.get("node_type") or ""),
                    }
                )
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

    blacklist_path, whitelist_path = _topic_curation_paths(config, project_dir)
    blacklist, whitelist = _load_topic_curation(config, project_dir)
    try:
        keep_labels, drop_labels = _curate_topic_labels_with_llm(
            config,
            _collect_topic_review_candidates(contributions, blacklist, whitelist),
        )
    except Exception as exc:
        print(f"Topic label curation warning: {type(exc).__name__}: {exc}")
        keep_labels, drop_labels = set(), set()
    if keep_labels or drop_labels or not blacklist_path.exists() or not whitelist_path.exists():
        whitelist.update(keep_labels)
        blacklist.update(drop_labels)
        blacklist.difference_update(whitelist)
        _write_label_set(blacklist_path, blacklist)
        _write_label_set(whitelist_path, whitelist)
        llm_curated_keep = len(keep_labels)
        llm_curated_drop = len(drop_labels)

    topic_index = _aggregate_topic_index(config, project_dir, contributions, blacklist=blacklist, whitelist=whitelist)
    report_path = _topic_curation_report_path(config, project_dir)
    write_json_file(
        report_path,
        {
            "schema_version": "1.0",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "logic_version": TOPIC_INDEX_LOGIC_VERSION,
            "config_fingerprint": config_fingerprint(config),
            "blacklist_path": str(blacklist_path),
            "whitelist_path": str(whitelist_path),
            "deterministic_filtered": [
                {
                    "normalized_label": label,
                    "count": int(deterministic_filtered_counts[label]),
                    "examples": deterministic_filtered_examples[label],
                }
                for label, _count in deterministic_filtered_counts.most_common()
            ],
            "llm_whitelisted": sorted(keep_labels),
            "llm_blacklisted": sorted(drop_labels),
            "persisted_whitelist": sorted(whitelist),
            "persisted_blacklist": sorted(blacklist),
            "surviving_topics": [
                {
                    "topic_key": topic.get("topic_key"),
                    "label": topic.get("label"),
                    "topic_kind": topic.get("topic_kind"),
                    "episode_count": topic.get("episode_count"),
                    "depth_label": topic.get("depth_label"),
                }
                for topic in topic_index.get("topics") or []
            ],
        },
    )
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
        "blacklist_size": len(blacklist),
        "whitelist_size": len(whitelist),
        "llm_curated_keep": llm_curated_keep,
        "llm_curated_drop": llm_curated_drop,
        "topic_curation_report_path": str(report_path),
    }
