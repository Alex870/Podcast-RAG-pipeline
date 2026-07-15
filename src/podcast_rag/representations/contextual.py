from __future__ import annotations

import re
from typing import Any


CONTEXTUAL_HEADER_VERSION = "context-header-v1"


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_contextual_header(metadata: dict[str, Any], max_chars: int = 700) -> str:
    """Build deterministic retrieval context without inventing source claims."""
    speakers = metadata.get("speakers")
    if isinstance(speakers, (list, tuple)):
        speaker_text = ", ".join(_clean(value) for value in speakers if _clean(value))
    else:
        speaker_text = _clean(metadata.get("speaker"))

    topic_tags = metadata.get("topic_tags")
    if isinstance(topic_tags, (list, tuple)):
        topic_text = ", ".join(_clean(value) for value in topic_tags if _clean(value))
    else:
        topic_text = _clean(topic_tags)

    fields = (
        ("Podcast", metadata.get("podcast_name")),
        ("Episode", metadata.get("episode_title")),
        ("Date", metadata.get("episode_date")),
        ("Speaker", speaker_text),
        ("Document type", metadata.get("node_type")),
        ("Hierarchy", metadata.get("level")),
        ("Topic hints", topic_text),
    )
    lines = [f"{label}: {_clean(value)}" for label, value in fields if _clean(value)]
    header = "\n".join(lines)
    if max_chars > 0 and len(header) > max_chars:
        header = header[:max_chars].rsplit(" ", 1)[0].rstrip(" ,:;-")
    return header


def contextual_embedding_text(page_content: str, metadata: dict[str, Any], max_chars: int = 700) -> str:
    header = build_contextual_header(metadata, max_chars=max_chars)
    content = str(page_content or "").strip()
    return f"{header}\n\n{content}" if header else content
