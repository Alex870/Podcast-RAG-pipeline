from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from podcast_rag.representations.contextual import CONTEXTUAL_HEADER_VERSION, contextual_embedding_text
from podcast_rag.representations.lexical import LEXICAL_TEXT_VERSION, build_lexical_text


DISPLAY_TEXT_VERSION = "page-content-v1"
REPRESENTATION_BUILDER_VERSION = "2026-07-15"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def representation_config_fingerprint(embedding_text_mode: str, lexical_text_mode: str, contextual_header_max_chars: int) -> str:
    payload = {
        "builder_version": REPRESENTATION_BUILDER_VERSION,
        "embedding_text_mode": embedding_text_mode,
        "lexical_text_mode": lexical_text_mode,
        "contextual_header_max_chars": int(contextual_header_max_chars),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def representation_fingerprint(
    source_fingerprint: str,
    metadata: dict[str, Any],
    representation_id: str,
    text: str,
    config_fingerprint: str,
) -> str:
    """Fingerprint one deterministic representation without changing document identity."""
    source_node = {
        "source_fingerprint": source_fingerprint,
        "node_id": metadata.get("node_id"),
        "node_type": metadata.get("node_type"),
        "page_content": metadata.get("source_node_content_hash") or metadata.get("stable_document_id"),
        "source_segment_ids": metadata.get("source_segment_ids") or [],
        "source_spans": metadata.get("source_spans") or [],
        "start_time": metadata.get("start_time"),
        "end_time": metadata.get("end_time"),
    }
    payload = {
        "builder_version": REPRESENTATION_BUILDER_VERSION,
        "config_fingerprint": config_fingerprint,
        "representation_id": representation_id,
        "source_node": source_node,
        "text": text,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def representation_manifest(embedding_text_mode: str, lexical_text_mode: str, contextual_header_max_chars: int = 700) -> dict[str, str]:
    dense_version = CONTEXTUAL_HEADER_VERSION if embedding_text_mode == CONTEXTUAL_HEADER_VERSION else DISPLAY_TEXT_VERSION
    lexical_version = LEXICAL_TEXT_VERSION if lexical_text_mode == LEXICAL_TEXT_VERSION else DISPLAY_TEXT_VERSION
    return {
        "display_text": DISPLAY_TEXT_VERSION,
        "dense_text": dense_version,
        "lexical_text": lexical_version,
        "builder_version": REPRESENTATION_BUILDER_VERSION,
        "config_fingerprint": representation_config_fingerprint(
            embedding_text_mode,
            lexical_text_mode,
            contextual_header_max_chars,
        ),
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
        return representation_manifest(
            self.embedding_text_mode,
            self.lexical_text_mode,
            self.contextual_header_max_chars,
        )

    def fingerprints(self, source_fingerprint: str, metadata: dict[str, Any], page_content: str) -> dict[str, str]:
        embedding_text, lexical_text = self.build(page_content, metadata)
        config_fingerprint = self.manifest()["config_fingerprint"]
        return {
            "display_text": representation_fingerprint(
                source_fingerprint,
                metadata,
                DISPLAY_TEXT_VERSION,
                str(page_content or ""),
                config_fingerprint,
            ),
            "dense_text": representation_fingerprint(
                source_fingerprint,
                metadata,
                self.manifest()["dense_text"],
                embedding_text,
                config_fingerprint,
            ),
            "lexical_text": representation_fingerprint(
                source_fingerprint,
                metadata,
                self.manifest()["lexical_text"],
                lexical_text,
                config_fingerprint,
            ),
        }
