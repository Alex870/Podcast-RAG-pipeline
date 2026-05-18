from __future__ import annotations

import datetime as dt
import json
import re
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import podcast_rag.runtime as runtime
from podcast_rag.config import PipelineConfig, config_fingerprint, resolve_path
from podcast_rag.llm_support import extract_llm_text, extract_token_usage, serialize_llm_response
from podcast_rag.runtime import (
    PIPELINE_VERSION,
    PROMPT_VERSION,
    EmptyLLMResponse,
    FakeChain,
    MissingContextResponse,
    PerformanceTracker,
    PipelineInterrupted,
    RuntimeControl,
)
from podcast_rag.schema import serialize_document, validate_processed_documents
from podcast_rag.state import (
    checkpoint_path,
    docs_from_payloads,
    document_payloads,
    read_json_file,
    write_json_file,
)
from podcast_rag.text_utils import (
    clip_text,
    compact_episode_date,
    compact_reduced_summaries,
    coerce_string_list,
    coerce_text,
    deterministic_episode_overview,
    deterministic_topic_tags,
    estimate_remaining_seconds,
    episode_sort_key,
    episode_title_from_source,
    extract_json_payload,
    extract_position_objects_from_partial_json,
    extract_summary_bullets,
    fallback_summary_from_text,
    file_fingerprint,
    format_duration,
    format_seconds,
    has_substantive_text,
    is_missing_context_response,
    merge_speaker_values,
    new_node_id,
    normalized_text_key,
    parse_episode_date,
    safe_float,
    short_text,
    speaker_scope,
    stable_episode_id,
    source_schema_version,
    text_fingerprint,
    token_estimate,
    token_set_similarity,
    with_retry,
)
from podcast_rag.transcript import load_transcript_json

Document = None
HuggingFaceEmbeddings = None
ChatOpenAI = None
RecursiveCharacterTextSplitter = None
ChatPromptTemplate = None
OpenAI = None
np = None
hdbscan = None
PCA = None
normalize = None

def _refresh_runtime_symbols() -> None:
    global Document, HuggingFaceEmbeddings, ChatOpenAI, RecursiveCharacterTextSplitter, ChatPromptTemplate
    global OpenAI, np, hdbscan, PCA, normalize
    runtime.load_runtime_deps()
    Document = runtime.Document
    HuggingFaceEmbeddings = runtime.HuggingFaceEmbeddings
    ChatOpenAI = runtime.ChatOpenAI
    RecursiveCharacterTextSplitter = runtime.RecursiveCharacterTextSplitter
    ChatPromptTemplate = runtime.ChatPromptTemplate
    OpenAI = runtime.OpenAI
    np = runtime.np
    hdbscan = runtime.hdbscan
    PCA = runtime.PCA
    normalize = runtime.normalize

class PodcastRagPipeline:
    """Orchestrate chunking, hierarchical summarization, and position extraction."""

    def __init__(self, config: PipelineConfig, project_dir: Path, control: RuntimeControl):
        _refresh_runtime_symbols()
        self.config = config
        self.project_dir = project_dir
        self.control = control
        self.debug_output_dir = resolve_path(project_dir, config.debug_output_dir)
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        self.performance = PerformanceTracker(config.performance_report_interval_seconds)
        self.fallback_count = 0
        self.cluster_telemetry: list[dict[str, Any]] = []
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self.llm = FakeChain() if config.fake_llm else ChatOpenAI(
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
        summary_system = (
            "You create retrieval-oriented summaries for a long-form podcast knowledge base. "
            f"Objective: {config.summary_objective}. Emphasize durable beliefs, recurring arguments, values, "
            "causal explanations, disagreements, speaker attribution, episode date, and the context needed "
            "to answer future questions accurately. Avoid filler. Return a non-empty final answer in the assistant "
            "message content. Do not ask for more source text."
        )
        summary_user = (
            "The source material to summarize is included below between delimiters.\n\n"
            "<<<SOURCE_MATERIAL>>>\n{text}\n<<<END_SOURCE_MATERIAL>>>\n\n"
            f"Summarize only the provided source material for retrieval. Preserve who said what when speaker labels are present. "
            f"Include the episode date when available. Return 5-10 dense bullets, no preamble, no repeated headings, "
            f"and stay under {self.config.summary_target_chars} characters. Return the final summary now.{self.thinking_control_suffix()}"
        )
        thesis_system = (
            "You are distilling an episode-level worldview summary. Extract the central theses, recurring positions, "
            "normative commitments, policy preferences, key uncertainties, notable counterarguments, and speaker attribution. "
            "Return a non-empty final answer in the assistant message content. Do not ask for more source text."
        )
        thesis_user = (
            "The episode source material is included below between delimiters.\n\n"
            "<<<SOURCE_MATERIAL>>>\n{text}\n<<<END_SOURCE_MATERIAL>>>\n\n"
            f"Create an episode thesis summary using only the provided source material. Preserve which speaker held each position "
            f"when the evidence supports attribution, and include the episode date when available. Return dense bullets, no preamble, "
            f"no repeated headings, and stay under {self.config.summary_target_chars * 2} characters. Return the final summary now.{self.thinking_control_suffix()}"
        )
        position_system = (
            "You extract durable positions from long-form podcasts. Return strict JSON only. "
            "Focus on beliefs, philosophies, recurring preferences, normative claims, and causal models "
            "that would matter across episodes. Prefer precision over volume. "
            "Only attribute a position to a speaker when the provided evidence supports that attribution. "
            "Every JSON field must use a string value except evidence_node_ids, evidence_timestamps, and keywords, which must be arrays of strings. "
            "Return a non-empty final JSON object in the assistant message content. Do not ask for more source text."
        )
        position_user = (
            'Return a JSON object with key "positions". Each position must be an object with keys: '
            '"claim", "speaker", "episode_date", "stance_category", "confidence", "rationale", "counterpoints", '
            '"evidence_node_ids", "evidence_timestamps", and "keywords".\n\n'
            "Use only evidence from the passages below. Prefer speaker-specific position cards over generic episode-level claims. "
            "If attribution is ambiguous, skip the claim instead of guessing. Return at most 5 positions. Keep each field concise. "
            f"Return JSON only, with no markdown, no commentary, and no bullet list outside the JSON object.\n\n{{text}}{self.thinking_control_suffix()}"
        )
        self.prompt_manifest = {
            "prompt_version": PROMPT_VERSION,
            "summary_system": summary_system,
            "summary_user": summary_user,
            "thesis_system": thesis_system,
            "thesis_user": thesis_user,
            "position_system": position_system,
            "position_user": position_user,
        }
        self.summary_chain = self.make_chain(ChatPromptTemplate.from_messages([("system", summary_system), ("user", summary_user)]))
        self.thesis_chain = self.make_chain(ChatPromptTemplate.from_messages([("system", thesis_system), ("user", thesis_user)]))
        self.position_chain = self.make_chain(ChatPromptTemplate.from_messages([("system", position_system), ("user", position_user)]))

    def make_chain(self, prompt):
        if self.config.fake_llm:
            return FakeChain()
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
        if runtime.STOP_REQUESTED:
            raise PipelineInterrupted("Stop requested before starting another model request.")
        if not has_substantive_text(text):
            raise ValueError(f"{label} received empty or too-short source text.")
        prompt_tokens = token_estimate(text, self.config.prompt_token_chars_per_token)
        if prompt_tokens > int(self.config.prompt_token_budget or self.config.context_window_tokens):
            debug_path = self.write_llm_debug_event(
                label=label,
                event="context_overflow_preflight",
                prompt_text=text,
                error=(
                    f"Estimated prompt tokens {prompt_tokens} exceed prompt budget "
                    f"{self.config.prompt_token_budget}; caller should split earlier."
                ),
            )
            print(f"  debug saved: {debug_path}")
            raise ValueError(f"{label} estimated prompt tokens exceed configured prompt budget.")

        start = time.time()
        token_usage: dict[str, int] = {}
        try:
            def run_and_validate():
                nonlocal token_usage
                raw_candidate = chain.invoke({"text": text})
                token_usage = extract_token_usage(raw_candidate)
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
            self.fallback_count += 1
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
            self.fallback_count += 1
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
            self.fallback_count += 1
            result = fallback_summary_from_text(text, label)

        self.performance.record_llm_result(label, time.time() - start, result, token_usage)
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
                if runtime.STOP_REQUESTED:
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

    def grouping_documents(self, documents: list[Document]) -> list[list[Document]]:
        mode = (self.config.grouping_mode or "semantic").lower()
        if mode == "chronological":
            return [documents[i : i + self.config.group_fallback_size] for i in range(0, len(documents), self.config.group_fallback_size)]
        if mode == "speaker_first":
            groups: dict[str, list[Document]] = {}
            for doc in documents:
                key = doc.metadata.get("speaker") or "multi"
                groups.setdefault(str(key), []).append(doc)
            clusters = []
            for group in groups.values():
                clusters.extend(group[i : i + self.config.group_fallback_size] for i in range(0, len(group), self.config.group_fallback_size))
            return clusters
        if mode in {"topic_time", "hybrid"}:
            chronological = [documents[i : i + self.config.group_fallback_size] for i in range(0, len(documents), self.config.group_fallback_size)]
            return sorted(chronological, key=lambda group: (group[0].metadata.get("start_time") is None, group[0].metadata.get("start_time") or 0.0))
        return []

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

        grouped = self.grouping_documents(documents)
        if grouped:
            self.cluster_telemetry.append(
                {
                    "mode": self.config.grouping_mode,
                    "input_documents": len(documents),
                    "cluster_count": len(grouped),
                    "noise_rate": 0.0,
                    "cluster_sizes": [len(group) for group in grouped],
                }
            )
            return grouped

        texts = [doc.page_content for doc in documents]
        batch_size = min(self.config.embedding_batch_size, max(8, len(texts)))
        embeds = normalize(np.array(self.embed_in_batches(texts[:]), dtype=float))

        n_components = min(5, len(documents) - 1, embeds.shape[1])
        if n_components >= 2:
            if (self.config.clustering_reduction or "pca").lower() == "umap":
                try:
                    import umap

                    reduced = umap.UMAP(n_components=n_components, random_state=42, metric="cosine").fit_transform(embeds)
                except Exception as exc:
                    print(f"UMAP reduction unavailable ({type(exc).__name__}: {exc}); falling back to PCA.")
                    reduced = PCA(n_components=n_components, random_state=42).fit_transform(embeds)
            else:
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

        cluster_values = list(clusters.values())
        noise_count = sum(1 for label in labels if label == -1)
        duplicate_rate = 0.0
        normalized_keys = [normalized_text_key(doc.page_content) for doc in documents]
        if normalized_keys:
            duplicate_rate = 1.0 - (len(set(normalized_keys)) / len(normalized_keys))
        self.cluster_telemetry.append(
            {
                "mode": "semantic",
                "reduction": self.config.clustering_reduction,
                "input_documents": len(documents),
                "cluster_count": len(cluster_values),
                "noise_rate": round(noise_count / max(1, len(documents)), 4),
                "cluster_sizes": [len(group) for group in cluster_values],
                "duplicate_rate": round(duplicate_rate, 4),
            }
        )
        return cluster_values

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
                "topic_tags": deterministic_topic_tags(summary, self.config.deterministic_topic_count),
                "summary_generation": "fallback" if summary.lstrip().startswith("- Fallback") else "model",
                "fallback_generated": summary.lstrip().startswith("- Fallback"),
                "compression_ratio": round(len(summary) / max(1, sum(len(doc.page_content or "") for doc in docs)), 4),
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
                    while pending_clusters and not runtime.STOP_REQUESTED and len(running) < self.control.max_parallel_model_requests():
                        cluster_docs = pending_clusters.pop(0)
                        running.add(executor.submit(self.summarize_cluster, level, cluster_docs, source))

                    if not running:
                        break

                    done, running = wait(running, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        summaries.append(future.result())
                        completed += 1
                        elapsed_seconds = time.time() - start_time
                        elapsed = dt.timedelta(seconds=int(elapsed_seconds))
                        eta = format_duration(estimate_remaining_seconds(completed, len(clusters), elapsed_seconds))
                        live_limit = self.control.max_parallel_model_requests()
                        print(
                            f"  [{completed:2d}/{len(clusters)}] built L{level} summary nodes "
                            f"elapsed={elapsed} eta={eta} in_flight={len(running)} live_parallel={live_limit} "
                            f"file_max_tokens={self.performance.current_file_max_total_tokens or 'unknown'} "
                            f"run_max_tokens={self.performance.run_max_total_tokens or 'unknown'}"
                        )
                        self.performance.maybe_report(f"L{level} summary", force=False)

                    if runtime.STOP_REQUESTED and not running:
                        raise PipelineInterrupted("Stop requested after in-flight model requests completed.")

            unique_summaries = []
            seen_summary_texts = []
            for summary_doc in summaries:
                key = normalized_text_key(summary_doc.page_content)
                duplicate = any(
                    key == seen_key
                    or token_set_similarity(summary_doc.page_content, seen_text) >= float(self.config.near_duplicate_threshold)
                    for seen_key, seen_text in seen_summary_texts
                )
                if key and duplicate:
                    summary_doc.metadata["duplicate_summary"] = True
                    continue
                seen_summary_texts.append((key, summary_doc.page_content))
                summary_doc.metadata["duplicate_summary"] = False
                unique_summaries.append(summary_doc)
            all_nodes.extend(summaries)
            if len(unique_summaries) != len(summaries):
                print(f"  removed {len(summaries) - len(unique_summaries)} duplicate L{level} summary node(s) from rollup")
            summaries = unique_summaries
            latest_summaries = summaries
            current_level_docs = summaries

        thesis_inputs = latest_summaries or leaf_chunks
        if self.config.episode_thesis_reduce_with_llm:
            thesis_text = self.summarize_documents(thesis_inputs, self.thesis_chain, "episode thesis")
        else:
            thesis_text = deterministic_episode_overview(thesis_inputs, self.config.episode_thesis_max_chars)
            print(
                f"  built deterministic episode overview from {len(thesis_inputs)} source node(s); "
                "skipped lossy LLM thesis reduction"
            )
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
                "topic_tags": deterministic_topic_tags(thesis_text, self.config.deterministic_topic_count),
                "summary_generation": "model" if self.config.episode_thesis_reduce_with_llm else "deterministic",
                "fallback_generated": False,
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
            evidence_ids = [node_id for node_id in evidence_ids if node_id in node_lookup]
            evidence_docs = [node_lookup[node_id] for node_id in evidence_ids]
            evidence_times = coerce_string_list(position.get("evidence_timestamps"))
            keywords = coerce_string_list(position.get("keywords"))
            if not keywords:
                keywords = deterministic_topic_tags(f"{claim} {rationale}", self.config.deterministic_topic_count)
            evidence_start = min((doc.metadata.get("start_time") for doc in evidence_docs if doc.metadata.get("start_time") is not None), default=thesis_meta.get("start_time"))
            evidence_end = max((doc.metadata.get("end_time") for doc in evidence_docs if doc.metadata.get("end_time") is not None), default=thesis_meta.get("end_time"))
            episode_date = parse_episode_date(position.get("episode_date")) or thesis_meta.get("episode_date")
            position_speaker = coerce_text(position.get("speaker")) or "unknown"
            if position_speaker.lower() in {"unknown", "unclear", "ambiguous", "multiple", "mixed"}:
                continue
            if not claim:
                continue
            if not evidence_docs:
                continue
            evidence_excerpt = " ".join(clip_text(doc.page_content, self.config.position_quote_excerpt_chars) for doc in evidence_docs[:3])
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
                    f"Evidence Excerpt: {evidence_excerpt}",
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
                        "evidence_excerpts": [clip_text(doc.page_content, self.config.position_quote_excerpt_chars) for doc in evidence_docs[:3]],
                        "keywords": keywords,
                        "topic_tags": keywords,
                        "quality_flags": [
                            flag
                            for flag, present in {
                                "has_speaker": bool(position_speaker and position_speaker != "unknown"),
                                "has_claim": bool(claim),
                                "has_evidence": bool(evidence_docs),
                                "has_episode_date": bool(episode_date),
                            }.items()
                            if present
                        ],
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
        result = validate_processed_documents(docs)
        result.raise_for_errors(label)
        for warning in result.warnings[:5]:
            print(f"  cache validation warning: {warning}")

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
        validation = validate_processed_documents(docs)
        payload = {
            "version": 2,
            "schema_version": self.config.cache_schema_version,
            "pipeline_version": PIPELINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_path": str(source_path),
            "source_fingerprint": fingerprint,
            "source_schema_version": source_schema_version(source_path),
            "stable_source_id": stable_episode_id(fingerprint),
            "config_fingerprint": config_fingerprint(self.config),
            "model": self.config.lm_studio_model,
            "embedding_model": self.config.embedding_model,
            "prompt_manifest": self.prompt_manifest,
            "token_maxima": self.performance.snapshot(),
            "fallback_count": self.fallback_count,
            "cluster_telemetry": self.cluster_telemetry,
            "validation": {
                "counts": validation.counts,
                "warnings": validation.warnings,
            },
            "import_manifest": {
                "cache_path": str(cache_path),
                "cache_schema_version": self.config.cache_schema_version,
                "pipeline_version": PIPELINE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "source_fingerprint": fingerprint,
                "model": self.config.lm_studio_model,
                "config_fingerprint": config_fingerprint(self.config),
            },
            "document_count": len(docs),
            "documents": document_payloads(docs, fingerprint),
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

    def load_file_checkpoint(self, source_path: Path, fingerprint: str, stage: str) -> list[Document] | None:
        if not self.config.resume_within_file:
            return None
        path = checkpoint_path(self.config, self.project_dir, source_path, fingerprint, stage)
        if not path.exists():
            return None
        try:
            payload = read_json_file(path)
        except Exception:
            return None
        if payload.get("source_fingerprint") != fingerprint:
            return None
        documents = payload.get("documents")
        if not isinstance(documents, list):
            return None
        print(f"  checkpoint reused: {stage} ({len(documents)} document(s))")
        return docs_from_payloads(documents)

    def save_file_checkpoint(self, source_path: Path, fingerprint: str, stage: str, docs: list[Document]) -> None:
        if not self.config.resume_within_file:
            return
        path = checkpoint_path(self.config, self.project_dir, source_path, fingerprint, stage)
        write_json_file(
            path,
            {
                "stage": stage,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_path": str(source_path),
                "source_fingerprint": fingerprint,
                "documents": document_payloads(docs, fingerprint),
            },
        )

    def clear_file_checkpoints(self, source_path: Path, fingerprint: str) -> None:
        if not self.config.resume_within_file:
            return
        for stage in ("leaf_chunks", "hierarchy", "positions"):
            path = checkpoint_path(self.config, self.project_dir, source_path, fingerprint, stage)
            if path.exists():
                path.unlink()

    def process_file(self, path: Path) -> dict[str, Any]:
        """Build or resume all retrieval artifacts for a single transcript file."""
        source = str(path)
        fingerprint = file_fingerprint(path)
        print(f"\nProcessing: {source}")
        self.performance.start_file(source)
        docs = load_transcript_json(path)
        leaf_chunks = self.load_file_checkpoint(path, fingerprint, "leaf_chunks")
        if leaf_chunks is None:
            # Leaf chunks are deterministic, so they are the first cheap checkpoint.
            leaf_chunks = self.build_leaf_chunks(docs, source)
            self.save_file_checkpoint(path, fingerprint, "leaf_chunks", leaf_chunks)

        if not leaf_chunks:
            print("  No usable text found; skipping")
            self.performance.finish_file()
            return {"status": "skipped", "nodes": 0}

        start = time.time()
        hierarchy_checkpoint = self.load_file_checkpoint(path, fingerprint, "hierarchy")
        if hierarchy_checkpoint is None:
            # Hierarchy creation is the expensive summarization path, so we checkpoint it separately.
            all_nodes, thesis_doc = self.build_hierarchy(leaf_chunks, source)
            self.save_file_checkpoint(path, fingerprint, "hierarchy", all_nodes)
        else:
            all_nodes = hierarchy_checkpoint
            thesis_doc = next(doc for doc in all_nodes if doc.metadata.get("node_type") == "episode_thesis")
        position_docs = self.load_file_checkpoint(path, fingerprint, "positions")
        if position_docs is None:
            position_docs = self.extract_positions(all_nodes, thesis_doc)
            self.save_file_checkpoint(path, fingerprint, "positions", position_docs)
        all_nodes.extend(position_docs)
        self.validate_documents_before_cache(all_nodes, source)
        elapsed = dt.timedelta(seconds=int(time.time() - start))

        print(
            f"  Built {len(leaf_chunks)} leaf chunks, "
            f"{len([doc for doc in all_nodes if doc.metadata['node_type'] == 'cluster_summary'])} cluster summaries, "
            f"{len(position_docs)} position cards in {elapsed}"
        )

        self.performance.maybe_report("file complete", force=True)
        self.performance.finish_file()
        self.clear_file_checkpoints(path, fingerprint)
        return {
            "status": "completed",
            "source": "llm_processing",
            "nodes": len(all_nodes),
            "position_cards": len(position_docs),
            "elapsed_seconds": int(time.time() - start),
            "fallbacks": self.fallback_count,
            "documents": all_nodes,
        }
