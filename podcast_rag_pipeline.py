from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


STOP_REQUESTED = False
RUNTIME_DEPS_LOADED = False


def load_runtime_deps() -> None:
    global RUNTIME_DEPS_LOADED
    global hdbscan, np, PCA, Document, StrOutputParser, ChatPromptTemplate
    global HuggingFaceEmbeddings, ChatOpenAI, RecursiveCharacterTextSplitter, OpenAI, normalize

    if RUNTIME_DEPS_LOADED:
        return

    import hdbscan
    import numpy as np
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from openai import OpenAI
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    RUNTIME_DEPS_LOADED = True


@dataclass
class PipelineConfig:
    input_dir: str = "data"
    file_glob: str = "**/*_speaker_transcript.json"
    processed_dir: str = "processed"
    state_path: str = "state/podcast_rag_state.json"
    stop_file: str = "state/stop_after_current.txt"
    control_file: str = "state/pipeline_control.json"
    processed_data_dir: str = "processed_data"
    debug_output_dir: str = "debug_output"
    move_processed_files: bool = False
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    lm_studio_model: str = "unsloth/qwen3.6-35b-a3b"
    verify_model: bool = True
    test_inference: bool = True
    max_threads: int = 2
    max_parallel_model_requests: int = 2
    performance_report_interval_seconds: int = 30
    max_levels: int = 4
    max_clusters: int = 300
    min_docs_to_cluster: int = 12
    group_fallback_size: int = 6
    rollup_char_budget: int = 6000
    leaf_chunk_size: int = 1800
    leaf_chunk_overlap: int = 250
    max_position_source_docs: int = 40
    embedding_batch_size: int = 64
    llm_max_tokens: int = 4096
    max_reduction_rounds: int = 8
    summary_target_chars: int = 1600
    position_extraction_batch_char_budget: int = 8000
    position_passage_max_chars: int = 900


class PipelineInterrupted(Exception):
    pass


class MissingContextResponse(Exception):
    pass


class EmptyLLMResponse(Exception):
    pass


class PerformanceTracker:
    def __init__(self, report_interval_seconds: int):
        self.report_interval_seconds = max(5, int(report_interval_seconds or 30))
        self.started_at = time.time()
        self.last_report_at = self.started_at
        self.requests = 0
        self.failures = 0
        self.total_seconds = 0.0
        self.approx_output_tokens = 0

    def record_llm_result(self, label: str, elapsed: float, text: str) -> None:
        self.requests += 1
        self.total_seconds += elapsed
        self.approx_output_tokens += max(1, len(text or "") // 4)
        self.maybe_report(label)

    def record_failure(self) -> None:
        self.failures += 1
        self.maybe_report("failure")

    def maybe_report(self, label: str, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_report_at < self.report_interval_seconds:
            return

        wall = max(0.001, now - self.started_at)
        model_seconds = max(0.001, self.total_seconds)
        req_per_min = self.requests / wall * 60.0
        approx_tok_per_sec = self.approx_output_tokens / model_seconds
        print(
            "  perf: "
            f"requests={self.requests}, failures={self.failures}, "
            f"avg_request_seconds={self.total_seconds / max(1, self.requests):.1f}, "
            f"requests_per_min={req_per_min:.2f}, "
            f"approx_output_tokens_per_sec={approx_tok_per_sec:.1f}, "
            f"last={label}"
        )
        self.last_report_at = now


class RuntimeControl:
    def __init__(self, config: PipelineConfig, project_dir: Path):
        self.config = config
        self.path = resolve_path(project_dir, config.control_file)
        self.default_parallel = max(1, int(config.max_parallel_model_requests or config.max_threads or 1))
        self.last_read_at = 0.0
        self.cached_parallel = self.default_parallel

    def initialize_file_for_run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_parallel_model_requests": self.default_parallel,
            "initialized_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "note": "Initialized from config at pipeline startup. Edit max_parallel_model_requests while the pipeline runs; new requests use the updated value.",
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.cached_parallel = self.default_parallel
        print(f"Live control initialized: {self.path} max_parallel_model_requests={self.default_parallel}")

    def max_parallel_model_requests(self) -> int:
        now = time.time()
        if now - self.last_read_at < 2:
            return self.cached_parallel
        self.last_read_at = now

        try:
            if self.path.exists():
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                value = int(payload.get("max_parallel_model_requests", self.default_parallel))
                self.cached_parallel = max(1, min(64, value))
        except Exception as exc:
            print(f"  control file read failed; using {self.cached_parallel}: {exc}")
        return self.cached_parallel


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nStop requested. No new model requests will be started; waiting for in-flight request(s) to finish.")


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def load_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        return PipelineConfig()

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Invalid JSON in config file: {config_path}\n"
            f"Line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    allowed = {field.name for field in fields(PipelineConfig)}
    values = {key: value for key, value in payload.items() if key in allowed}
    return PipelineConfig(**values)


def apply_env_overrides(config: PipelineConfig) -> PipelineConfig:
    config.embedding_model = os.getenv("EMBEDDING_MODEL", config.embedding_model)
    config.lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", config.lm_studio_base_url)
    config.lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY", config.lm_studio_api_key)
    config.lm_studio_model = os.getenv("LM_STUDIO_MODEL", config.lm_studio_model)
    return config


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"version": 1, "files": {}}

    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup_path = state_path.with_suffix(f".corrupt.{int(time.time())}.json")
        shutil.copy2(state_path, backup_path)
        print(f"State file was invalid JSON. Backed it up to {backup_path} and starting fresh.")
        return {"version": 1, "files": {}}


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(state_path)


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


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
        if STOP_REQUESTED:
            raise PipelineInterrupted("Stop requested before retrying a model request.")
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
    return (
        f"Fallback extractive summary for {label}: "
        f"{short_text(compact, max_chars=max_chars)}"
    )


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
    combined = " ".join(parts)
    return f"Deterministic compacted summary for {label}: {clip_text(combined, max_chars)}"


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


def extract_llm_text(response: Any) -> str:
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        joined = "\n".join(part for part in parts if part.strip())
        if joined.strip():
            return joined

    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    for key in ("reasoning_content", "reasoning", "thinking", "thoughts"):
        value = additional_kwargs.get(key)
        if isinstance(value, str) and value.strip():
            return value

    response_metadata = getattr(response, "response_metadata", {}) or {}
    for key in ("reasoning_content", "reasoning", "thinking", "thoughts"):
        value = response_metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def serialize_llm_response(response: Any) -> Any:
    if isinstance(response, (str, int, float, bool)) or response is None:
        return response
    payload = {
        "type": type(response).__name__,
        "content": getattr(response, "content", None),
        "additional_kwargs": getattr(response, "additional_kwargs", None),
        "response_metadata": getattr(response, "response_metadata", None),
    }
    return payload


def verify_model_available(config: PipelineConfig) -> None:
    client = OpenAI(base_url=config.lm_studio_base_url, api_key=config.lm_studio_api_key)
    models = [model.id for model in client.models.list().data]
    if config.lm_studio_model not in models:
        raise RuntimeError(f"Model not found: {config.lm_studio_model}\nAvailable: {models}")
    print(f"Model '{config.lm_studio_model}' is available.")


def test_model_inference(config: PipelineConfig) -> None:
    client = OpenAI(base_url=config.lm_studio_base_url, api_key=config.lm_studio_api_key)
    client.chat.completions.create(
        model=config.lm_studio_model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
    print("Inference test passed.")


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


class PodcastRagPipeline:
    def __init__(self, config: PipelineConfig, project_dir: Path, control: RuntimeControl):
        self.config = config
        self.project_dir = project_dir
        self.control = control
        self.debug_output_dir = resolve_path(project_dir, config.debug_output_dir)
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        self.performance = PerformanceTracker(config.performance_report_interval_seconds)
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self.llm = ChatOpenAI(
            model=config.lm_studio_model,
            temperature=0.0,
            max_tokens=config.llm_max_tokens,
            base_url=config.lm_studio_base_url,
            api_key=config.lm_studio_api_key,
        )
        self.leaf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.leaf_chunk_size,
            chunk_overlap=config.leaf_chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
        )
        self.rollup_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rollup_char_budget,
            chunk_overlap=400,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
        )
        self.summary_chain = self.make_chain(
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You create retrieval-oriented summaries for a long-form podcast knowledge base. "
                        "Emphasize durable beliefs, recurring arguments, values, causal explanations, disagreements, "
                        "speaker attribution, episode date, and the context needed to answer future questions accurately. Avoid filler. "
                        "Return a non-empty final answer in the assistant message content. Do not ask for more source text.",
                    ),
                    ("user", f"The source material to summarize is included below between delimiters.\n\n<<<SOURCE_MATERIAL>>>\n{{text}}\n<<<END_SOURCE_MATERIAL>>>\n\nSummarize only the provided source material for retrieval. Preserve who said what when speaker labels are present. Include the episode date when available. Return 5-10 dense bullets, no preamble, no repeated headings, and stay under {self.config.summary_target_chars} characters. Return the final summary now.{self.thinking_control_suffix()}"),
                ]
            )
        )
        self.thesis_chain = self.make_chain(
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are distilling an episode-level worldview summary. Extract the central theses, recurring positions, "
                        "normative commitments, policy preferences, key uncertainties, notable counterarguments, and speaker attribution. "
                        "Return a non-empty final answer in the assistant message content. Do not ask for more source text.",
                    ),
                    ("user", f"The episode source material is included below between delimiters.\n\n<<<SOURCE_MATERIAL>>>\n{{text}}\n<<<END_SOURCE_MATERIAL>>>\n\nCreate an episode thesis summary using only the provided source material. Preserve which speaker held each position when the evidence supports attribution, and include the episode date when available. Return dense bullets, no preamble, no repeated headings, and stay under {self.config.summary_target_chars * 2} characters. Return the final summary now.{self.thinking_control_suffix()}"),
                ]
            )
        )
        self.position_chain = self.make_chain(
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You extract durable positions from long-form podcasts. Return strict JSON only. "
                        "Focus on beliefs, philosophies, recurring preferences, normative claims, and causal models "
                        "that would matter across episodes. Prefer precision over volume. "
                        "Only attribute a position to a speaker when the provided evidence supports that attribution. "
                        "Every JSON field must use a string value except evidence_node_ids, evidence_timestamps, and keywords, which must be arrays of strings. "
                        "Return a non-empty final JSON object in the assistant message content. Do not ask for more source text.",
                    ),
                    (
                        "user",
                        'Return a JSON object with key "positions". Each position must be an object with keys: '
                        '"claim", "speaker", "episode_date", "stance_category", "confidence", "rationale", "counterpoints", '
                        '"evidence_node_ids", "evidence_timestamps", and "keywords".\n\n'
                        f"Use only evidence from the passages below. Prefer speaker-specific position cards over generic episode-level claims. If attribution is ambiguous, skip the claim instead of guessing. Return at most 5 positions. Keep each field concise. Return JSON only, with no markdown, no commentary, and no bullet list outside the JSON object.\n\n{{text}}{self.thinking_control_suffix()}",
                    ),
                ]
            )
        )

    def make_chain(self, prompt):
        return prompt | self.llm

    def thinking_control_suffix(self) -> str:
        model_name = (self.config.lm_studio_model or "").lower()
        if "qwen" in model_name:
            return "\n/no_think"
        return ""

    def write_llm_debug_event(
        self,
        label: str,
        event: str,
        prompt_text: str,
        response_text: str | None = None,
        error: str | None = None,
        raw_response: Any = None,
    ) -> Path:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or "llm"
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.debug_output_dir / f"{stamp}.{safe_label}.{uuid.uuid4().hex[:8]}.json"
        payload = {
            "version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": event,
            "label": label,
            "model": self.config.lm_studio_model,
            "base_url": self.config.lm_studio_base_url,
            "prompt_char_count": len(prompt_text or ""),
            "response_char_count": len(response_text or ""),
            "error": error,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "raw_response": serialize_llm_response(raw_response),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

    def invoke_llm(self, chain, text: str, label: str) -> str:
        if STOP_REQUESTED:
            raise PipelineInterrupted("Stop requested before starting another model request.")
        if not has_substantive_text(text):
            raise ValueError(f"{label} received empty or too-short source text.")

        start = time.time()
        try:
            def run_and_validate():
                raw_candidate = chain.invoke({"text": text})
                candidate = extract_llm_text(raw_candidate)
                if not has_substantive_text(candidate, min_chars=1):
                    debug_path = self.write_llm_debug_event(
                        label=label,
                        event="empty_response",
                        prompt_text=text,
                        response_text=candidate,
                        error="Model returned empty assistant message content.",
                        raw_response=raw_candidate,
                    )
                    print(f"  debug saved: {debug_path}")
                    raise EmptyLLMResponse(f"{label} returned an empty response.")
                if is_missing_context_response(candidate):
                    debug_path = self.write_llm_debug_event(
                        label=label,
                        event="missing_context_response",
                        prompt_text=text,
                        response_text=candidate,
                        error="Response looked like the model was asking for source text that was already provided.",
                        raw_response=raw_candidate,
                    )
                    print(f"  debug saved: {debug_path}")
                    raise MissingContextResponse(f"{label} returned a missing-context response instead of a summary.")
                return candidate

            result = with_retry(run_and_validate, label)
        except EmptyLLMResponse:
            self.performance.record_failure()
            if "position extraction" in label:
                print(f"  {label} returned empty responses; using empty position list.")
                return '{"positions": []}'
            print(f"  {label} returned empty responses; using fallback extractive summary.")
            result = fallback_summary_from_text(text, label)
            debug_path = self.write_llm_debug_event(
                label=label,
                event="fallback_after_empty_response",
                prompt_text=text,
                response_text=result,
                error="All retries returned empty assistant message content.",
            )
            print(f"  debug saved: {debug_path}")
        except MissingContextResponse:
            self.performance.record_failure()
            if "position extraction" in label:
                print(f"  {label} returned missing-context responses; using empty position list.")
                return '{"positions": []}'
            print(f"  {label} returned missing-context responses; using fallback extractive summary.")
            result = fallback_summary_from_text(text, label)
            debug_path = self.write_llm_debug_event(
                label=label,
                event="fallback_extractive_summary",
                prompt_text=text,
                response_text=result,
                error="All retries returned missing-context responses.",
            )
            print(f"  debug saved: {debug_path}")
        except Exception as exc:
            self.performance.record_failure()
            error_text = f"{type(exc).__name__}: {exc}"
            debug_path = self.write_llm_debug_event(
                label=label,
                event="llm_exception",
                prompt_text=text,
                error=error_text,
            )
            print(f"  debug saved: {debug_path}")
            if "position extraction" in label:
                print(f"  {label} failed after retries; using empty position list. error={error_text}")
                return '{"positions": []}'
            print(f"  {label} failed after retries; using fallback extractive summary. error={error_text}")
            result = fallback_summary_from_text(text, label)

        self.performance.record_llm_result(label, time.time() - start, result)
        return result

    def normalize_doc(self, doc: Document, source: str, index: int) -> Document:
        metadata = dict(doc.metadata or {})
        metadata["source"] = source
        metadata["episode_id"] = stable_episode_id(source)
        metadata["episode_title"] = metadata.get("episode_title") or episode_title_from_source(source)
        metadata["source_type"] = metadata.get("source_type") or "json_transcript"
        metadata["segment_index"] = metadata.get("segment_index", index)
        metadata["start_time"] = safe_float(metadata.get("start_time"))
        metadata["end_time"] = safe_float(metadata.get("end_time"))
        metadata["speaker"] = metadata.get("speaker")
        metadata["episode_date"] = parse_episode_date(metadata.get("episode_date"))
        metadata["episode_date_compact"] = metadata.get("episode_date_compact") or compact_episode_date(metadata.get("episode_date"))
        metadata["episode_sort_key"] = metadata.get("episode_sort_key") or episode_sort_key(metadata.get("episode_date"))
        return Document(page_content=doc.page_content.strip(), metadata=metadata)

    def build_leaf_chunks(self, docs: list[Document], source: str) -> list[Document]:
        normalized = [self.normalize_doc(doc, source, idx) for idx, doc in enumerate(docs)]
        normalized = [doc for doc in normalized if doc.page_content]
        normalized.sort(
            key=lambda doc: (
                doc.metadata.get("start_time") is None,
                doc.metadata.get("start_time") or 0.0,
                doc.metadata.get("segment_index", 0),
            )
        )

        if not normalized:
            return []

        first = normalized[0].metadata
        episode_id = first["episode_id"]
        episode_title = first["episode_title"]
        chunks = []
        current_docs = []
        current_chars = 0

        for doc in normalized:
            addition = len(doc.page_content) + 1
            if current_docs and current_chars + addition > self.config.leaf_chunk_size:
                chunks.append(self.make_leaf_chunk(current_docs, source, episode_id, episode_title))
                overlap_docs = current_docs[-2:] if len(current_docs) > 2 else current_docs[-1:]
                current_docs = overlap_docs + [doc]
                current_chars = sum(len(item.page_content) + 1 for item in current_docs)
            else:
                current_docs.append(doc)
                current_chars += addition

        if current_docs:
            chunks.append(self.make_leaf_chunk(current_docs, source, episode_id, episode_title))
        return chunks

    def make_leaf_chunk(self, docs: list[Document], source: str, episode_id: str, episode_title: str) -> Document:
        start_time = min((doc.metadata.get("start_time") for doc in docs if doc.metadata.get("start_time") is not None), default=None)
        end_time = max((doc.metadata.get("end_time") for doc in docs if doc.metadata.get("end_time") is not None), default=None)
        speakers = merge_speaker_values(doc.metadata.get("speaker") for doc in docs)
        speaker, scope = speaker_scope(speakers)
        first = docs[0].metadata
        text = "\n".join(
            f"[{doc.metadata.get('speaker') or 'unknown'} {format_seconds(doc.metadata.get('start_time'))}-{format_seconds(doc.metadata.get('end_time'))}] {doc.page_content}"
            for doc in docs
            if doc.page_content
        )
        return Document(
            page_content=text,
            metadata={
                "node_id": new_node_id("leaf"),
                "node_type": "leaf_chunk",
                "level": "leaf",
                "parent_id": None,
                "child_ids": [],
                "source": source,
                "episode_id": episode_id,
                "episode_title": episode_title,
                "episode_date": first.get("episode_date"),
                "episode_date_compact": first.get("episode_date_compact"),
                "episode_sort_key": first.get("episode_sort_key"),
                "source_type": "json_transcript",
                "segment_count": len(docs),
                "segment_indices": [doc.metadata.get("segment_index") for doc in docs],
                "start_time": start_time,
                "end_time": end_time,
                "speaker": speaker,
                "speaker_scope": scope,
                "speakers": speakers,
            },
        )

    def render_doc_for_rollup(self, doc: Document) -> str:
        metadata = doc.metadata
        time_span = f"{format_seconds(metadata.get('start_time'))}-{format_seconds(metadata.get('end_time'))}"
        speakers = ", ".join(metadata.get("speakers") or ([metadata["speaker"]] if metadata.get("speaker") else [])) or "unknown"
        return (
            f"[node_id={metadata.get('node_id')} | type={metadata.get('node_type')} | level={metadata.get('level')} "
            f"| episode_date={metadata.get('episode_date') or 'unknown'} | speaker_scope={metadata.get('speaker_scope') or 'unknown'} "
            f"| speakers={speakers} | time={time_span}]\n{doc.page_content}"
        )

    def reduce_text_blocks(self, blocks: list[str], chain, label: str) -> str:
        pending = []
        for block in blocks:
            if not has_substantive_text(block):
                continue
            if len(block) <= self.config.rollup_char_budget:
                pending.append(block)
            else:
                pending.extend(part for part in self.rollup_splitter.split_text(block) if has_substantive_text(part))

        if not pending:
            raise ValueError(f"{label} had no substantive text blocks to summarize.")

        original_text = "\n\n".join(pending)
        seen_fingerprints = {text_fingerprint(pending)}
        previous_total_chars = sum(len(block) for block in pending)

        for reduction_round in range(1, max(1, self.config.max_reduction_rounds) + 1):
            joined = "\n\n".join(pending)
            if len(pending) == 1 and len(joined) <= self.config.rollup_char_budget:
                return self.invoke_llm(chain, joined, label)

            batches = []
            current_batch = []
            current_size = 0

            for block in pending:
                block_len = len(block) + 2
                if current_batch and current_size + block_len > self.config.rollup_char_budget:
                    batches.append("\n\n".join(current_batch))
                    current_batch = [block]
                    current_size = block_len
                else:
                    current_batch.append(block)
                    current_size += block_len

            if current_batch:
                batches.append("\n\n".join(current_batch))

            reduced = []
            if len(batches) > 1:
                print(
                    f"  {label} reduction round {reduction_round}: "
                    f"{len(batches)} batch(es) from {len(pending)} block(s)"
                )
            for idx, batch in enumerate(batches):
                if STOP_REQUESTED:
                    raise PipelineInterrupted("Stop requested before starting another model request.")
                reduced.append(self.invoke_llm(chain, batch, f"{label} batch {idx + 1}"))
                if len(batches) > 1:
                    print(
                        f"  {label} reduction round {reduction_round}: "
                        f"completed batch {idx + 1}/{len(batches)}"
                    )

            if len(reduced) == 1:
                return reduced[0]

            reduced_total_chars = sum(len(block) for block in reduced)
            reduced_fingerprint = text_fingerprint(reduced)
            made_progress = len(reduced) < len(pending) or reduced_total_chars < previous_total_chars
            if reduced_fingerprint in seen_fingerprints or not made_progress:
                debug_path = self.write_llm_debug_event(
                    label=label,
                    event="reduction_stalled",
                    prompt_text=original_text,
                    response_text="\n\n".join(reduced),
                    error=(
                        f"Reduction stalled on round {reduction_round}: "
                        f"{len(pending)} block(s), {previous_total_chars} chars -> "
                        f"{len(reduced)} block(s), {reduced_total_chars} chars."
                    ),
                )
                print(f"  {label} reduction stalled; compacting reduced summaries deterministically.")
                print(f"  debug saved: {debug_path}")
                return compact_reduced_summaries(reduced, label, max_chars=min(self.config.rollup_char_budget, self.config.summary_target_chars * 2))

            seen_fingerprints.add(reduced_fingerprint)
            previous_total_chars = reduced_total_chars
            pending = reduced

        debug_path = self.write_llm_debug_event(
            label=label,
            event="reduction_round_limit",
            prompt_text=original_text,
            response_text="\n\n".join(pending),
            error=f"Reduction exceeded max_reduction_rounds={self.config.max_reduction_rounds}.",
        )
        print(f"  {label} reached reduction round limit; compacting reduced summaries deterministically.")
        print(f"  debug saved: {debug_path}")
        return compact_reduced_summaries(pending, label, max_chars=min(self.config.rollup_char_budget, self.config.summary_target_chars * 2))

    def summarize_documents(self, docs: list[Document], chain, label: str) -> str:
        source_docs = [doc for doc in docs if has_substantive_text(doc.page_content, min_chars=20)]
        if not source_docs:
            raise ValueError(f"{label} had no substantive document content to summarize.")
        blocks = [self.render_doc_for_rollup(doc) for doc in source_docs]
        return self.reduce_text_blocks(blocks, chain, label)

    def embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        results = []
        batch_size = self.config.embedding_batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self.embeddings.embed_documents(batch))
        return results

    def cluster_documents(self, documents: list[Document]) -> list[list[Document]]:
        if len(documents) < self.config.min_docs_to_cluster:
            return [documents]

        texts = [doc.page_content for doc in documents]
        batch_size = min(self.config.embedding_batch_size, max(8, len(texts)))
        embeds = normalize(np.array(self.embed_in_batches(texts[:]), dtype=float))

        n_components = min(5, len(documents) - 1, embeds.shape[1])
        if n_components >= 2:
            reduced = PCA(n_components=n_components, random_state=42).fit_transform(embeds)
        else:
            reduced = embeds

        min_cluster_size = max(3, min(8, len(documents) // 8))
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)

        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                clusters[f"noise_{idx}"] = [documents[idx]]
                continue
            clusters.setdefault(int(label), []).append(documents[idx])

        if len(clusters) > self.config.max_clusters:
            print(f"Too many clusters ({len(clusters)}), using fallback grouping")
            clusters = {
                f"fallback_{i}": documents[i : i + self.config.group_fallback_size]
                for i in range(0, len(documents), self.config.group_fallback_size)
            }
            print(f"Created {len(clusters)} fallback groups")

        return list(clusters.values())

    def summarize_cluster(self, level: int, docs: list[Document], source: str) -> Document:
        summary = self.summarize_documents(docs, self.summary_chain, f"cluster summary L{level}")
        first = docs[0].metadata
        node_id = new_node_id("summary")
        child_ids = [doc.metadata["node_id"] for doc in docs]
        start_time = min((doc.metadata.get("start_time") for doc in docs if doc.metadata.get("start_time") is not None), default=None)
        end_time = max((doc.metadata.get("end_time") for doc in docs if doc.metadata.get("end_time") is not None), default=None)
        speakers = merge_speaker_values(
            speaker
            for doc in docs
            for speaker in (doc.metadata.get("speakers") or ([doc.metadata["speaker"]] if doc.metadata.get("speaker") else []))
        )
        speaker, scope = speaker_scope(speakers)

        summary_doc = Document(
            page_content=summary,
            metadata={
                "node_id": node_id,
                "node_type": "cluster_summary",
                "level": f"summary_{level}",
                "parent_id": None,
                "child_ids": child_ids,
                "source": source,
                "episode_id": first["episode_id"],
                "episode_title": first["episode_title"],
                "episode_date": first.get("episode_date"),
                "episode_date_compact": first.get("episode_date_compact"),
                "episode_sort_key": first.get("episode_sort_key"),
                "source_type": first["source_type"],
                "start_time": start_time,
                "end_time": end_time,
                "speaker": speaker,
                "speaker_scope": scope,
                "speakers": speakers,
            },
        )

        for doc in docs:
            doc.metadata["parent_id"] = node_id

        return summary_doc

    def build_hierarchy(self, leaf_chunks: list[Document], source: str) -> tuple[list[Document], Document]:
        all_nodes = list(leaf_chunks)
        current_level_docs = list(leaf_chunks)
        latest_summaries = []

        for level in range(1, self.config.max_levels + 1):
            if len(current_level_docs) < self.config.min_docs_to_cluster:
                break

            clusters = self.cluster_documents(current_level_docs)
            if len(clusters) == 1 and len(clusters[0]) == len(current_level_docs):
                break

            summaries = []
            start_time = time.time()
            completed = 0
            pending_clusters = list(clusters)
            running = set()

            with ThreadPoolExecutor(max_workers=64) as executor:
                while pending_clusters or running:
                    while pending_clusters and not STOP_REQUESTED and len(running) < self.control.max_parallel_model_requests():
                        cluster_docs = pending_clusters.pop(0)
                        running.add(executor.submit(self.summarize_cluster, level, cluster_docs, source))

                    if not running:
                        break

                    done, running = wait(running, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        summaries.append(future.result())
                        completed += 1
                        elapsed = dt.timedelta(seconds=int(time.time() - start_time))
                        live_limit = self.control.max_parallel_model_requests()
                        print(
                            f"  [{completed:2d}/{len(clusters)}] built L{level} summary nodes "
                            f"elapsed={elapsed} in_flight={len(running)} live_parallel={live_limit}"
                        )
                        self.performance.maybe_report(f"L{level} summary", force=False)

                    if STOP_REQUESTED and not running:
                        raise PipelineInterrupted("Stop requested after in-flight model requests completed.")

            all_nodes.extend(summaries)
            latest_summaries = summaries
            current_level_docs = summaries

        thesis_inputs = latest_summaries or leaf_chunks
        thesis_text = self.summarize_documents(thesis_inputs, self.thesis_chain, "episode thesis")
        thesis_speakers = merge_speaker_values(
            speaker
            for doc in leaf_chunks
            for speaker in (doc.metadata.get("speakers") or ([doc.metadata["speaker"]] if doc.metadata.get("speaker") else []))
        )
        thesis_speaker, thesis_scope = speaker_scope(thesis_speakers)
        thesis_doc = Document(
            page_content=thesis_text,
            metadata={
                "node_id": new_node_id("thesis"),
                "node_type": "episode_thesis",
                "level": "episode",
                "parent_id": None,
                "child_ids": [doc.metadata["node_id"] for doc in thesis_inputs],
                "source": source,
                "episode_id": leaf_chunks[0].metadata["episode_id"],
                "episode_title": leaf_chunks[0].metadata["episode_title"],
                "episode_date": leaf_chunks[0].metadata.get("episode_date"),
                "episode_date_compact": leaf_chunks[0].metadata.get("episode_date_compact"),
                "episode_sort_key": leaf_chunks[0].metadata.get("episode_sort_key"),
                "source_type": leaf_chunks[0].metadata["source_type"],
                "start_time": min((doc.metadata.get("start_time") for doc in leaf_chunks if doc.metadata.get("start_time") is not None), default=None),
                "end_time": max((doc.metadata.get("end_time") for doc in leaf_chunks if doc.metadata.get("end_time") is not None), default=None),
                "speaker": thesis_speaker,
                "speaker_scope": thesis_scope,
                "speakers": thesis_speakers,
            },
        )

        for doc in thesis_inputs:
            doc.metadata["parent_id"] = thesis_doc.metadata["node_id"]

        all_nodes.append(thesis_doc)
        return all_nodes, thesis_doc

    def build_position_source_docs(self, all_nodes: list[Document], thesis_doc: Document) -> list[Document]:
        candidates = [doc for doc in all_nodes if doc.metadata["node_type"] == "cluster_summary"]
        candidates.sort(
            key=lambda doc: (
                doc.metadata.get("speaker_scope") != "single",
                doc.metadata.get("start_time") is None,
                doc.metadata.get("start_time") or 0.0,
                len(doc.page_content or ""),
            )
        )

        trimmed = candidates[: self.config.max_position_source_docs]
        if not trimmed:
            return [thesis_doc]
        return trimmed

    def render_position_passage(self, doc: Document) -> str:
        metadata = doc.metadata
        payload = {
            "node_id": metadata["node_id"],
            "node_type": metadata["node_type"],
            "episode_date": metadata.get("episode_date") or "",
            "time_range": f"{format_seconds(metadata.get('start_time'))}-{format_seconds(metadata.get('end_time'))}",
            "speaker_scope": metadata.get("speaker_scope") or "unknown",
            "speaker": metadata.get("speaker") or "",
            "speakers": metadata.get("speakers") or [],
            "text": short_text(doc.page_content, max_chars=max(300, int(self.config.position_passage_max_chars))),
        }
        return json.dumps(payload, ensure_ascii=True)

    def build_position_batches(self, source_docs: list[Document]) -> list[list[Document]]:
        budget = max(2000, int(self.config.position_extraction_batch_char_budget or 8000))
        batches = []
        current = []
        current_size = 0

        for doc in source_docs:
            rendered_size = len(self.render_position_passage(doc)) + 1
            if current and current_size + rendered_size > budget:
                batches.append(current)
                current = [doc]
                current_size = rendered_size
            else:
                current.append(doc)
                current_size += rendered_size

        if current:
            batches.append(current)
        return batches

    def parse_position_payload(self, raw: str, label: str) -> list[dict[str, Any]]:
        payload = extract_json_payload(raw)
        positions = payload.get("positions") if isinstance(payload, dict) else payload
        if isinstance(payload, dict) and not isinstance(positions, list):
            for key in ("claims", "position_cards", "items", "results"):
                if isinstance(payload.get(key), list):
                    positions = payload[key]
                    break
            if not isinstance(positions, list) and payload.get("claim"):
                positions = [payload]
        if not isinstance(positions, list):
            partial_positions = extract_position_objects_from_partial_json(raw)
            if partial_positions:
                print(f"{label} returned truncated JSON; recovered {len(partial_positions)} complete position object(s)")
                return partial_positions
            print(f"{label} returned non-list payload; skipping")
            debug_path = self.write_llm_debug_event(
                label=label,
                event="position_payload_not_list",
                prompt_text="",
                response_text=raw,
                error=f"Could not parse a list of positions from model response. Parsed payload type={type(payload).__name__}.",
            )
            print(f"  debug saved: {debug_path}")
            return []
        return [position for position in positions if isinstance(position, dict) and position.get("claim")]

    def extract_positions(self, all_nodes: list[Document], thesis_doc: Document) -> list[Document]:
        source_docs = self.build_position_source_docs(all_nodes, thesis_doc)
        batches = self.build_position_batches(source_docs)
        node_lookup = {doc.metadata.get("node_id"): doc for doc in all_nodes if doc.metadata.get("node_id")}

        positions = []
        for batch_idx, batch in enumerate(batches, 1):
            prompt_text = "\n".join(self.render_position_passage(doc) for doc in batch)
            label = "position extraction" if len(batches) == 1 else f"position extraction batch {batch_idx}"
            print(
                f"  {label}: {len(batch)} source doc(s), "
                f"{len(prompt_text)} prompt chars"
            )
            raw = self.invoke_llm(self.position_chain, prompt_text, label)
            positions.extend(self.parse_position_payload(raw, label))

        thesis_meta = thesis_doc.metadata
        docs = []
        seen_position_keys = set()
        for idx, position in enumerate(positions):
            claim = coerce_text(position.get("claim"))
            rationale = coerce_text(position.get("rationale"))
            counterpoints = coerce_text(position.get("counterpoints"))
            stance_category = coerce_text(position.get("stance_category")) or "unspecified"
            confidence = coerce_text(position.get("confidence")) or "unknown"
            evidence_ids = coerce_string_list(position.get("evidence_node_ids"))
            evidence_docs = [node_lookup[node_id] for node_id in evidence_ids if node_id in node_lookup]
            evidence_times = coerce_string_list(position.get("evidence_timestamps"))
            keywords = coerce_string_list(position.get("keywords"))
            evidence_start = min((doc.metadata.get("start_time") for doc in evidence_docs if doc.metadata.get("start_time") is not None), default=thesis_meta.get("start_time"))
            evidence_end = max((doc.metadata.get("end_time") for doc in evidence_docs if doc.metadata.get("end_time") is not None), default=thesis_meta.get("end_time"))
            episode_date = parse_episode_date(position.get("episode_date")) or thesis_meta.get("episode_date")
            position_speaker = coerce_text(position.get("speaker")) or "unknown"
            if position_speaker.lower() in {"unknown", "unclear", "ambiguous", "multiple", "mixed"}:
                continue
            if not claim:
                continue
            position_key = (
                position_speaker.lower(),
                (episode_date or "").lower(),
                re.sub(r"\s+", " ", claim.lower()),
            )
            if position_key in seen_position_keys:
                continue
            seen_position_keys.add(position_key)

            card_text = "\n".join(
                [
                    f"Claim: {claim}",
                    f"Speaker: {position_speaker}",
                    f"Episode Date: {episode_date or 'unknown'}",
                    f"Evidence Time: {format_seconds(evidence_start)}-{format_seconds(evidence_end)}",
                    f"Category: {stance_category}",
                    f"Confidence: {confidence}",
                    f"Rationale: {rationale}",
                    f"Counterpoints: {counterpoints}",
                    f"Keywords: {', '.join(keywords)}",
                ]
            ).strip()

            docs.append(
                Document(
                    page_content=card_text,
                    metadata={
                        "node_id": new_node_id("position"),
                        "node_type": "position_card",
                        "level": "position",
                        "parent_id": thesis_meta["node_id"],
                        "child_ids": evidence_ids,
                        "source": thesis_meta["source"],
                        "episode_id": thesis_meta["episode_id"],
                        "episode_title": thesis_meta["episode_title"],
                        "episode_date": episode_date,
                        "episode_date_compact": compact_episode_date(episode_date) or thesis_meta.get("episode_date_compact"),
                        "episode_sort_key": episode_sort_key(episode_date) or thesis_meta.get("episode_sort_key"),
                        "source_type": thesis_meta["source_type"],
                        "position_index": idx,
                        "claim": claim,
                        "speaker": position_speaker,
                        "speaker_scope": "single" if position_speaker != "unknown" else "unknown",
                        "stance_category": stance_category,
                        "confidence": confidence,
                        "evidence_timestamps": evidence_times,
                        "keywords": keywords,
                        "start_time": evidence_start,
                        "end_time": evidence_end,
                        "speakers": [position_speaker] if position_speaker != "unknown" else [],
                    },
                )
            )
        return docs

    def validate_documents_before_cache(self, docs: list[Document], label: str) -> None:
        bad = []
        for idx, doc in enumerate(docs):
            node_id = doc.metadata.get("node_id", f"index_{idx}")
            node_type = doc.metadata.get("node_type", "unknown")
            if not has_substantive_text(doc.page_content, min_chars=1):
                bad.append(f"{node_id} ({node_type}) has empty page_content")
            elif node_type in {"cluster_summary", "episode_thesis", "position_card"} and is_missing_context_response(doc.page_content):
                bad.append(f"{node_id} ({node_type}) has a missing-context response")
            if not doc.metadata.get("episode_date"):
                bad.append(f"{node_id} ({node_type}) is missing episode_date")
            if node_type in {"leaf_chunk", "cluster_summary", "episode_thesis", "position_card"} and not doc.metadata.get("speaker_scope"):
                bad.append(f"{node_id} ({node_type}) is missing speaker_scope")

        if bad:
            preview = "; ".join(bad[:10])
            if len(bad) > 10:
                preview += f"; and {len(bad) - 10} more"
            raise ValueError(f"{label} produced invalid documents: {preview}")

    def load_cached_documents(self, cache_path: Path) -> list[Document]:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        docs = []
        for item in payload.get("documents", []):
            if not isinstance(item, dict):
                continue
            docs.append(
                Document(
                    page_content=str(item.get("page_content", "")),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        return docs

    def save_cached_documents(self, cache_path: Path, source_path: Path, fingerprint: str, docs: list[Document]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.validate_documents_before_cache(docs, f"cache write {cache_path}")
        payload = {
            "version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_path": str(source_path),
            "source_fingerprint": fingerprint,
            "document_count": len(docs),
            "documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in docs
            ],
        }
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        temp_path.replace(cache_path)

    def validate_cached_file(self, path: Path, fingerprint: str, cache_path: Path) -> dict[str, Any]:
        print(f"\nValidating cached processed data: {cache_path}")
        docs = self.load_cached_documents(cache_path)
        if not docs:
            raise RuntimeError(f"Processed data cache was empty: {cache_path}")
        self.validate_documents_before_cache(docs, f"cache {cache_path}")
        print(f"  Cached processed data is valid for {path}: {len(docs)} documents")
        return {
            "status": "completed",
            "source": "processed_data_cache",
            "nodes": len(docs),
            "cache_path": str(cache_path),
        }

    def process_file(self, path: Path) -> dict[str, Any]:
        source = str(path)
        print(f"\nProcessing: {source}")
        docs = load_transcript_json(path)
        leaf_chunks = self.build_leaf_chunks(docs, source)

        if not leaf_chunks:
            print("  No usable text found; skipping")
            return {"status": "skipped", "nodes": 0}

        start = time.time()
        all_nodes, thesis_doc = self.build_hierarchy(leaf_chunks, source)
        position_docs = self.extract_positions(all_nodes, thesis_doc)
        all_nodes.extend(position_docs)
        self.validate_documents_before_cache(all_nodes, source)
        elapsed = dt.timedelta(seconds=int(time.time() - start))

        print(
            f"  Built {len(leaf_chunks)} leaf chunks, "
            f"{len([doc for doc in all_nodes if doc.metadata['node_type'] == 'cluster_summary'])} cluster summaries, "
            f"{len(position_docs)} position cards in {elapsed}"
        )

        self.performance.maybe_report("file complete", force=True)
        return {
            "status": "completed",
            "source": "llm_processing",
            "nodes": len(all_nodes),
            "position_cards": len(position_docs),
            "elapsed_seconds": int(time.time() - start),
            "documents": all_nodes,
        }


def should_skip_file(state: dict[str, Any], fingerprint: str) -> bool:
    entry = state.get("files", {}).get(fingerprint)
    return bool(entry and entry.get("status") in {"completed", "skipped"})


def processed_data_cache_path(processed_data_dir: Path, fingerprint: str, source_path: Path) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_path.stem).strip("._") or "transcript"
    return processed_data_dir / f"{safe_stem}.{fingerprint}.processed_documents.json"


def quarantine_invalid_cache(cache_path: Path, reason: str) -> Path:
    quarantine_dir = cache_path.parent / "invalid"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = quarantine_dir / f"{cache_path.name}.invalid.{stamp}"
    counter = 1
    while dest.exists():
        dest = quarantine_dir / f"{cache_path.name}.invalid.{stamp}.{counter}"
        counter += 1
    shutil.move(str(cache_path), str(dest))
    reason_path = dest.with_suffix(dest.suffix + ".reason.txt")
    reason_path.write_text(reason, encoding="utf-8")
    return dest


def mark_state(state: dict[str, Any], fingerprint: str, path: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "path": str(path),
        "status": status,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    state.setdefault("files", {})[fingerprint] = payload


def maybe_move_processed(path: Path, processed_dir: Path) -> str | None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / f"{path.name}.{int(time.time())}"
    shutil.move(str(path), str(dest))
    return str(dest)


def run_batch(config: PipelineConfig, project_dir: Path, one_file: bool) -> int:
    load_runtime_deps()

    input_dir = resolve_path(project_dir, config.input_dir)
    processed_dir = resolve_path(project_dir, config.processed_dir)
    processed_data_dir = resolve_path(project_dir, config.processed_data_dir)
    state_path = resolve_path(project_dir, config.state_path)
    stop_file = resolve_path(project_dir, config.stop_file)

    input_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    control = RuntimeControl(config, project_dir)
    control.initialize_file_for_run()

    state = load_state(state_path)
    files = iter_transcript_files(input_dir, config.file_glob)
    pending = []
    for path in files:
        fingerprint = file_fingerprint(path)
        cache_path = processed_data_cache_path(processed_data_dir, fingerprint, path)
        if cache_path.exists() or not should_skip_file(state, fingerprint):
            pending.append((path, fingerprint))

    print(f"Found {len(files)} matching files; {len(pending)} pending.")
    if not pending:
        return 0

    cached_pending = [processed_data_cache_path(processed_data_dir, fingerprint, path).exists() for path, fingerprint in pending]
    needs_llm_processing = not all(cached_pending)
    if needs_llm_processing:
        if config.verify_model:
            verify_model_available(config)
        if config.test_inference:
            test_model_inference(config)
    else:
        print("All pending files have processed data caches; skipping LM Studio model verification.")

    pipeline = PodcastRagPipeline(config, project_dir, control)

    for idx, (path, fingerprint) in enumerate(pending, 1):
        if STOP_REQUESTED or stop_file.exists():
            print("Stop requested before starting next file.")
            break

        cache_path = processed_data_cache_path(processed_data_dir, fingerprint, path)
        print(f"\nFile {idx}/{len(pending)}")

        try:
            if cache_path.exists():
                try:
                    result = pipeline.validate_cached_file(path, fingerprint, cache_path)
                except ValueError as exc:
                    quarantined_path = quarantine_invalid_cache(cache_path, str(exc))
                    print(f"  Invalid processed data cache moved to: {quarantined_path}")
                    print("  Rebuilding processed data from transcript.")
                    mark_state(state, fingerprint, path, "in_progress")
                    save_state(state_path, state)
                    result = pipeline.process_file(path)
                    if result["status"] != "completed":
                        mark_state(state, fingerprint, path, result["status"], result)
                        save_state(state_path, state)
                        continue
                    docs = result.pop("documents", [])
                    pipeline.save_cached_documents(cache_path, path, fingerprint, docs)
                    result["cache_path"] = str(cache_path)
                    result["quarantined_cache_path"] = str(quarantined_path)
                    print(f"  Saved processed data cache: {cache_path}")
            else:
                mark_state(state, fingerprint, path, "in_progress")
                save_state(state_path, state)
                result = pipeline.process_file(path)
                if result["status"] != "completed":
                    mark_state(state, fingerprint, path, result["status"], result)
                    save_state(state_path, state)
                    continue
                docs = result.pop("documents", [])
                pipeline.save_cached_documents(cache_path, path, fingerprint, docs)
                result["cache_path"] = str(cache_path)
                print(f"  Saved processed data cache: {cache_path}")

            moved_to = None
            if result["status"] == "completed" and config.move_processed_files:
                moved_to = maybe_move_processed(path, processed_dir)
                print(f"  Moved to {moved_to}")
            if moved_to:
                result["moved_to"] = moved_to
            mark_state(state, fingerprint, path, result["status"], result)
            save_state(state_path, state)
        except PipelineInterrupted as exc:
            mark_state(state, fingerprint, path, "interrupted", {"error": str(exc)})
            save_state(state_path, state)
            print("Stop request handled. Progress state was saved; this file will be retried on the next run.")
            break
        except Exception as exc:
            mark_state(state, fingerprint, path, "failed", {"error": f"{type(exc).__name__}: {exc}"})
            save_state(state_path, state)
            raise

        if one_file:
            print("Processed one file; stopping because --one-file was set.")
            break

        if STOP_REQUESTED or stop_file.exists():
            print("Stop request detected. Batch will resume with the next pending file on the next run.")
            break

    print("\nBatch run complete.")
    return 0


def create_stop_file(config: PipelineConfig, project_dir: Path) -> int:
    stop_file = resolve_path(project_dir, config.stop_file)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text(f"Stop requested at {dt.datetime.now().isoformat()}\n", encoding="utf-8")
    print(f"Created stop file: {stop_file}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a podcast RAG knowledge base from transcript JSON files.")
    parser.add_argument("--config", default="podcast_rag_config.json", help="Path to the JSON config file.")
    parser.add_argument("--input-dir", help="Override config input_dir.")
    parser.add_argument("--file-glob", help="Override config file_glob.")
    parser.add_argument("--model", help="Override config lm_studio_model.")
    parser.add_argument("--base-url", help="Override config lm_studio_base_url.")
    parser.add_argument("--max-parallel-model-requests", type=int, help="Override initial max_parallel_model_requests.")
    parser.add_argument("--one-file", action="store_true", help="Process only one pending file.")
    parser.add_argument("--create-stop-file", action="store_true", help="Create the configured stop file and exit.")
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    args = parse_args()
    config_path = Path(args.config).expanduser()
    project_dir = config_path.resolve().parent if config_path.exists() else Path.cwd()
    config = apply_env_overrides(load_config(config_path))

    if args.input_dir:
        config.input_dir = args.input_dir
    if args.file_glob:
        config.file_glob = args.file_glob
    if args.model:
        config.lm_studio_model = args.model
    if args.base_url:
        config.lm_studio_base_url = args.base_url
    if args.max_parallel_model_requests:
        config.max_parallel_model_requests = args.max_parallel_model_requests

    if args.create_stop_file:
        return create_stop_file(config, project_dir)

    return run_batch(config, project_dir, args.one_file)


if __name__ == "__main__":
    raise SystemExit(main())
