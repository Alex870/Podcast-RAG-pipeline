from __future__ import annotations

import re
from typing import Any


LEXICAL_TEXT_VERSION = "normalized-lexical-v1"


def _values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if value not in (None, "") else []


def build_lexical_text(page_content: str, metadata: dict[str, Any]) -> str:
    """Preserve exact source language while adding high-value searchable metadata."""
    parts = [str(page_content or "")]
    for key in (
        "podcast_name",
        "episode_title",
        "episode_date",
        "episode_date_compact",
        "speaker",
        "speakers",
        "topic_tags",
        "claim",
        "stance_category",
        "keywords",
    ):
        parts.extend(_values(metadata.get(key)))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()
