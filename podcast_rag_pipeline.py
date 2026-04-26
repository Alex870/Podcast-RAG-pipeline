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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


STOP_REQUESTED = False
RUNTIME_DEPS_LOADED = False


def load_runtime_deps() -> None:
    global RUNTIME_DEPS_LOADED
    global hdbscan, np, umap, Chroma, Document, StrOutputParser, ChatPromptTemplate
    global HuggingFaceEmbeddings, ChatOpenAI, RecursiveCharacterTextSplitter, OpenAI, normalize

    if RUNTIME_DEPS_LOADED:
        return

    import hdbscan
    import numpy as np
    import umap
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from openai import OpenAI
    from sklearn.preprocessing import normalize

    RUNTIME_DEPS_LOADED = True


@dataclass
class PipelineConfig:
    input_dir: str = "data"
    file_glob: str = "**/*_speaker_transcript.json"
    processed_dir: str = "processed"
    state_path: str = "state/podcast_rag_state.json"
    stop_file: str = "state/stop_after_current.txt"
    move_processed_files: bool = False
    persist_dir: str = "chroma_db_raptor_v2"
    collection_name: str = "whisper_rag_v2"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    lm_studio_model: str = "qwen3.6-35b-a3b"
    verify_model: bool = True
    test_inference: bool = True
    max_threads: int = 2
    max_levels: int = 4
    max_clusters: int = 300
    min_docs_to_cluster: int = 12
    group_fallback_size: int = 6
    rollup_char_budget: int = 12000
    leaf_chunk_size: int = 1800
    leaf_chunk_overlap: int = 250
    max_position_source_docs: int = 40
    embedding_batch_size: int = 64
    llm_max_tokens: int = 1024


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nStop requested. The current file will finish, then the batch will stop cleanly.")


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def load_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        return PipelineConfig()

    payload = json.loads(config_path.read_text(encoding="utf-8"))
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


def with_retry(func, label: str, retries: int = 3, delay: int = 1):
    for attempt in range(retries):
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


def load_transcript_json(path: Path) -> list[Document]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = extract_segment_records(payload)
    docs = []

    for idx, record in enumerate(records):
        text = first_present(record, ["text", "content", "transcript", "sentence"])
        if not text or not str(text).strip():
            continue

        metadata = {
            "source": str(path),
            "level": "leaf",
            "start_time": safe_float(first_present(record, ["start", "start_time", "timestamp_start"])),
            "end_time": safe_float(first_present(record, ["end", "end_time", "timestamp_end"])),
            "speaker": first_present(record, ["speaker", "speaker_name", "speaker_id", "voice", "who"]),
            "segment_index": first_present(record, ["id", "segment_id", "seek"]) or idx,
            "source_type": "json_transcript",
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
                },
            )
        ]

    return []


class PodcastRagPipeline:
    def __init__(self, config: PipelineConfig, project_dir: Path):
        self.config = config
        self.project_dir = project_dir
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
                        "and the context needed to answer future questions accurately. Avoid filler.",
                    ),
                    ("user", "Summarize this material for retrieval:\n\n{text}"),
                ]
            )
        )
        self.thesis_chain = self.make_chain(
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are distilling an episode-level worldview summary. Extract the central theses, recurring positions, "
                        "normative commitments, policy preferences, key uncertainties, and notable counterarguments.",
                    ),
                    ("user", "Create an episode thesis summary from this material:\n\n{text}"),
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
                        "that would matter across episodes. Prefer precision over volume.",
                    ),
                    (
                        "user",
                        'Return a JSON object with key "positions". Each position must be an object with keys: '
                        '"claim", "speaker", "stance_category", "confidence", "rationale", "counterpoints", '
                        '"evidence_node_ids", "evidence_timestamps", and "keywords".\n\n'
                        "Use only evidence from the passages below.\n\n{text}",
                    ),
                ]
            )
        )
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(resolve_path(project_dir, config.persist_dir)),
            collection_name=config.collection_name,
        )

    def make_chain(self, prompt):
        return prompt | self.llm | StrOutputParser()

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
        text = "\n".join(doc.page_content for doc in docs if doc.page_content)
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
                "source_type": "json_transcript",
                "segment_count": len(docs),
                "segment_indices": [doc.metadata.get("segment_index") for doc in docs],
                "start_time": start_time,
                "end_time": end_time,
                "speakers": speakers,
            },
        )

    def render_doc_for_rollup(self, doc: Document) -> str:
        metadata = doc.metadata
        time_span = f"{format_seconds(metadata.get('start_time'))}-{format_seconds(metadata.get('end_time'))}"
        speakers = ", ".join(metadata.get("speakers") or ([metadata["speaker"]] if metadata.get("speaker") else [])) or "unknown"
        return (
            f"[node_id={metadata.get('node_id')} | type={metadata.get('node_type')} | level={metadata.get('level')} "
            f"| speaker={speakers} | time={time_span}]\n{doc.page_content}"
        )

    def reduce_text_blocks(self, blocks: list[str], chain, label: str) -> str:
        pending = []
        for block in blocks:
            if len(block) <= self.config.rollup_char_budget:
                pending.append(block)
            else:
                pending.extend(self.rollup_splitter.split_text(block))

        while True:
            joined = "\n\n".join(pending)
            if len(pending) == 1 and len(joined) <= self.config.rollup_char_budget:
                return with_retry(lambda: chain.invoke({"text": joined}), label)

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
            for idx, batch in enumerate(batches):
                reduced.append(with_retry(lambda batch=batch: chain.invoke({"text": batch}), f"{label} batch {idx + 1}"))

            if len(reduced) == 1:
                return reduced[0]
            pending = reduced

    def summarize_documents(self, docs: list[Document], chain, label: str) -> str:
        blocks = [self.render_doc_for_rollup(doc) for doc in docs]
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

        reducer = umap.UMAP(
            n_neighbors=min(20, len(documents) - 1),
            n_components=5,
            metric="cosine",
            min_dist=0.01,
            spread=1.0,
            random_state=42,
        )
        reduced = reducer.fit_transform(embeds)

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
                "source_type": first["source_type"],
                "start_time": start_time,
                "end_time": end_time,
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
            with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
                futures = [executor.submit(self.summarize_cluster, level, cluster_docs, source) for cluster_docs in clusters]
                for idx, future in enumerate(as_completed(futures), 1):
                    summaries.append(future.result())
                    elapsed = dt.timedelta(seconds=int(time.time() - start_time))
                    print(f"  [{idx:2d}/{len(futures)}] built L{level} summary nodes elapsed={elapsed}")

            all_nodes.extend(summaries)
            latest_summaries = summaries
            current_level_docs = summaries

        thesis_inputs = latest_summaries or leaf_chunks
        thesis_text = self.summarize_documents(thesis_inputs, self.thesis_chain, "episode thesis")
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
                "source_type": leaf_chunks[0].metadata["source_type"],
                "start_time": min((doc.metadata.get("start_time") for doc in leaf_chunks if doc.metadata.get("start_time") is not None), default=None),
                "end_time": max((doc.metadata.get("end_time") for doc in leaf_chunks if doc.metadata.get("end_time") is not None), default=None),
                "speakers": merge_speaker_values(
                    speaker
                    for doc in leaf_chunks
                    for speaker in (doc.metadata.get("speakers") or ([doc.metadata["speaker"]] if doc.metadata.get("speaker") else []))
                ),
            },
        )

        for doc in thesis_inputs:
            doc.metadata["parent_id"] = thesis_doc.metadata["node_id"]

        all_nodes.append(thesis_doc)
        return all_nodes, thesis_doc

    def build_position_source_docs(self, all_nodes: list[Document], thesis_doc: Document) -> list[Document]:
        candidates = [doc for doc in all_nodes if doc.metadata["node_type"] in {"cluster_summary", "episode_thesis"}]
        candidates.sort(
            key=lambda doc: (
                doc.metadata["node_type"] != "episode_thesis",
                doc.metadata.get("start_time") is None,
                doc.metadata.get("start_time") or 0.0,
            )
        )

        trimmed = candidates[: self.config.max_position_source_docs]
        if thesis_doc not in trimmed:
            trimmed = [thesis_doc] + trimmed[: self.config.max_position_source_docs - 1]
        return trimmed

    def render_position_passage(self, doc: Document) -> str:
        metadata = doc.metadata
        payload = {
            "node_id": metadata["node_id"],
            "node_type": metadata["node_type"],
            "time_range": f"{format_seconds(metadata.get('start_time'))}-{format_seconds(metadata.get('end_time'))}",
            "speakers": metadata.get("speakers") or [],
            "text": short_text(doc.page_content, max_chars=1200),
        }
        return json.dumps(payload, ensure_ascii=True)

    def extract_positions(self, all_nodes: list[Document], thesis_doc: Document) -> list[Document]:
        source_docs = self.build_position_source_docs(all_nodes, thesis_doc)
        prompt_text = "\n".join(self.render_position_passage(doc) for doc in source_docs)

        raw = with_retry(lambda: self.position_chain.invoke({"text": prompt_text}), "position extraction")
        payload = extract_json_payload(raw)
        positions = payload.get("positions") if isinstance(payload, dict) else payload
        if not isinstance(positions, list):
            print("Position extraction returned non-list payload; skipping position cards")
            return []

        thesis_meta = thesis_doc.metadata
        docs = []
        for idx, position in enumerate(positions):
            if not isinstance(position, dict) or not position.get("claim"):
                continue

            evidence_ids = [item for item in position.get("evidence_node_ids", []) if isinstance(item, str)]
            evidence_times = [item for item in position.get("evidence_timestamps", []) if isinstance(item, str)]
            keywords = [item for item in position.get("keywords", []) if isinstance(item, str)]

            card_text = "\n".join(
                [
                    f"Claim: {position.get('claim', '').strip()}",
                    f"Speaker: {position.get('speaker', 'unknown')}",
                    f"Category: {position.get('stance_category', 'unspecified')}",
                    f"Confidence: {position.get('confidence', 'unknown')}",
                    f"Rationale: {position.get('rationale', '').strip()}",
                    f"Counterpoints: {position.get('counterpoints', '').strip()}",
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
                        "source_type": thesis_meta["source_type"],
                        "position_index": idx,
                        "claim": position.get("claim", "").strip(),
                        "speaker": position.get("speaker", "unknown"),
                        "stance_category": position.get("stance_category", "unspecified"),
                        "confidence": position.get("confidence", "unknown"),
                        "evidence_timestamps": evidence_times,
                        "keywords": keywords,
                        "start_time": thesis_meta.get("start_time"),
                        "end_time": thesis_meta.get("end_time"),
                        "speakers": thesis_meta.get("speakers", []),
                    },
                )
            )
        return docs

    def sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        clean = {}
        for key, value in metadata.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = json.dumps(value, ensure_ascii=True)
        return clean

    def sanitize_documents(self, docs: list[Document]) -> list[Document]:
        return [Document(page_content=doc.page_content, metadata=self.sanitize_metadata(doc.metadata)) for doc in docs]

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
        elapsed = dt.timedelta(seconds=int(time.time() - start))

        print(
            f"  Built {len(leaf_chunks)} leaf chunks, "
            f"{len([doc for doc in all_nodes if doc.metadata['node_type'] == 'cluster_summary'])} cluster summaries, "
            f"{len(position_docs)} position cards in {elapsed}"
        )

        ids = [doc.metadata["node_id"] for doc in all_nodes]
        self.vectorstore.add_documents(self.sanitize_documents(all_nodes), ids=ids)
        return {"status": "completed", "nodes": len(all_nodes), "position_cards": len(position_docs), "elapsed_seconds": int(time.time() - start)}


def should_skip_file(state: dict[str, Any], fingerprint: str) -> bool:
    entry = state.get("files", {}).get(fingerprint)
    return bool(entry and entry.get("status") in {"completed", "skipped"})


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
    state_path = resolve_path(project_dir, config.state_path)
    stop_file = resolve_path(project_dir, config.stop_file)

    input_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    stop_file.parent.mkdir(parents=True, exist_ok=True)

    if config.verify_model:
        verify_model_available(config)
    if config.test_inference:
        test_model_inference(config)

    state = load_state(state_path)
    files = iter_transcript_files(input_dir, config.file_glob)
    pending = []
    for path in files:
        fingerprint = file_fingerprint(path)
        if not should_skip_file(state, fingerprint):
            pending.append((path, fingerprint))

    print(f"Found {len(files)} matching files; {len(pending)} pending.")
    if not pending:
        return 0

    pipeline = PodcastRagPipeline(config, project_dir)

    for idx, (path, fingerprint) in enumerate(pending, 1):
        if STOP_REQUESTED or stop_file.exists():
            print("Stop requested before starting next file.")
            break

        print(f"\nFile {idx}/{len(pending)}")
        mark_state(state, fingerprint, path, "in_progress")
        save_state(state_path, state)

        try:
            result = pipeline.process_file(path)
            moved_to = None
            if result["status"] == "completed" and config.move_processed_files:
                moved_to = maybe_move_processed(path, processed_dir)
                print(f"  Moved to {moved_to}")
            if moved_to:
                result["moved_to"] = moved_to
            mark_state(state, fingerprint, path, result["status"], result)
            save_state(state_path, state)
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

    if args.create_stop_file:
        return create_stop_file(config, project_dir)

    return run_batch(config, project_dir, args.one_file)


if __name__ == "__main__":
    raise SystemExit(main())
