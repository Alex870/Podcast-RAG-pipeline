from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from podcast_rag.representations.contextual import CONTEXTUAL_HEADER_VERSION, contextual_embedding_text
from podcast_rag.representations.lexical import LEXICAL_TEXT_VERSION, build_lexical_text


DISPLAY_TEXT_VERSION = "page-content-v1"


def representation_manifest(embedding_text_mode: str, lexical_text_mode: str) -> dict[str, str]:
    dense_version = CONTEXTUAL_HEADER_VERSION if embedding_text_mode == CONTEXTUAL_HEADER_VERSION else DISPLAY_TEXT_VERSION
    lexical_version = LEXICAL_TEXT_VERSION if lexical_text_mode == LEXICAL_TEXT_VERSION else DISPLAY_TEXT_VERSION
    return {
        "display_text": DISPLAY_TEXT_VERSION,
        "dense_text": dense_version,
        "lexical_text": lexical_version,
    }


@dataclass(frozen=True)
class RepresentationBuilder:
    embedding_text_mode: str = DISPLAY_TEXT_VERSION
    lexical_text_mode: str = LEXICAL_TEXT_VERSION
    contextual_header_max_chars: int = 700

    def build(self, page_content: str, metadata: dict[str, Any]) -> tuple[str, str]:
        if self.embedding_text_mode == CONTEXTUAL_HEADER_VERSION:
            embedding_text = contextual_embedding_text(
                page_content,
                metadata,
                max_chars=self.contextual_header_max_chars,
            )
        else:
            embedding_text = str(page_content or "")

        if self.lexical_text_mode == LEXICAL_TEXT_VERSION:
            lexical_text = build_lexical_text(page_content, metadata)
        else:
            lexical_text = str(page_content or "")
        return embedding_text, lexical_text

    def manifest(self) -> dict[str, str]:
        return representation_manifest(self.embedding_text_mode, self.lexical_text_mode)
