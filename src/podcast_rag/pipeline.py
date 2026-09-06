from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import podcast_rag.runtime as runtime
from podcast_rag.config import PipelineConfig, config_fingerprint, generation_config_fingerprint, resolve_path
from podcast_rag.errata import DIAGNOSIS_ACTION_TYPES, ErrataRecorder, validate_diagnosis_payload
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
from podcast_rag.schema import serialize_document, validate_processed_cache, validate_processed_documents
from podcast_rag.representations import RepresentationBuilder
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

    def __init__(self, config: PipelineConfig, project_dir: Path, control: RuntimeControl, *, load_models: bool = True):
        _refresh_runtime_symbols()
        self.config = config
        self.project_dir = project_dir
        self.control = control
        self.debug_output_dir = resolve_path(project_dir, config.debug_output_dir)
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        self.performance = PerformanceTracker(config.performance_report_interval_seconds)
        # Keep the run total for aggregate reporting, but track the current
        # file separately so per-file result and cache metadata stay isolated.
        self.fallback_count = 0
        self._file_fallback_count = 0
        self.cluster_telemetry: list[dict[str, Any]] = []
        self.active_errata: ErrataRecorder | None = None
        self.active_context: dict[str, Any] | None = None
        if not load_models:
            # Cache validation needs the lightweight Document/schema runtime,
            # but must not load an embedding model or open an LLM chain.
            self.embeddings = None
            self.llm = None
            self.leaf_splitter = None
            self.rollup_splitter = None
            self.prompt_manifest = {}
            self.summary_chain = None
            self.thesis_chain = None
            self.position_chain = None
            self.diagnosis_chain = None
            return
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
            "to answer future questions accurately. Episode dates are historical source metadata, not deadlines "
            "or scheduling signals; never refuse or defer summarization because a date appears earlier or later "
            "than today's date. The source material is already present. Avoid filler. Return a non-empty final "
            "answer in the assistant "
            "message content. Do not ask for more source text."
        )
        summary_user = (
            "The source material to summarize is included below between delimiters.\n\n"
            "<<<SOURCE_MATERIAL>>>\n{text}\n<<<END_SOURCE_MATERIAL>>>\n\n"
            f"Summarize only the provided source material for retrieval. Preserve who said what when speaker labels are present. "
            f"Include the episode date when available. Return 5-10 dense bullets, no preamble, no repeated headings, "
            f"and stay under {self.config.summary_target_chars} characters. Return the final summary now."
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
            f"no repeated headings, and stay under {self.config.summary_target_chars * 2} characters. Return the final summary now."
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
            f"Return JSON only, with no markdown, no commentary, and no bullet list outside the JSON object.\n\n{{text}}"
        )
        diagnosis_system = (
            "You diagnose one podcast RAG file from a bounded deterministic review packet. "
            "Do not invent facts or recommend automatic code changes. Distinguish model output, "
            "pipeline logic, configuration, input data, and environment causes. Ground every root "
            "cause in finding IDs supplied in the packet. Return strict JSON only."
        )
        diagnosis_user = (
            "Analyze the bounded review packet below. Return a JSON object with exactly these keys: "
            "summary, root_causes, actions, uncertainties, human_review_questions. "
            "Each root_causes item must contain category, confidence, finding_ids, claim, and explanation. "
            "Each actions item must contain priority, type, action, and verification. "
            f"The action type must be exactly one of: {', '.join(DIAGNOSIS_ACTION_TYPES)}. "
            "Use human_review when uncertain. "
            "Use only finding IDs from the packet; if evidence is insufficient, say so.\n\n"
            "ERRATA_REVIEW_PACKET\n{text}"
        )
        self.prompt_manifest = {
            "prompt_version": PROMPT_VERSION,
            "request_controls": {
                "summary": {"chat_template_kwargs": {"enable_thinking": False}},
                "thesis": {"chat_template_kwargs": {"enable_thinking": False}},
                "position": {"chat_template_kwargs": {"enable_thinking": False}},
                "diagnosis": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            "summary_system": summary_system,
            "summary_user": summary_user,
            "thesis_system": thesis_system,
            "thesis_user": thesis_user,
            "position_system": position_system,
            "position_user": position_user,
            "diagnosis_system": diagnosis_system,
            "diagnosis_user": diagnosis_user,
        }
        self.summary_chain = self.make_chain(
            ChatPromptTemplate.from_messages([("system", summary_system), ("user", summary_user)]),
            enable_thinking=False,
        )
        self.thesis_chain = self.make_chain(
            ChatPromptTemplate.from_messages([("system", thesis_system), ("user", thesis_user)]),
            enable_thinking=False,
        )
        self.position_chain = self.make_chain(
            ChatPromptTemplate.from_messages([("system", position_system), ("user", position_user)]),
            enable_thinking=False,
        )
        self.diagnosis_chain = self.make_chain(
            ChatPromptTemplate.from_messages([("system", diagnosis_system), ("user", diagnosis_user)]),
            enable_thinking=False,
        )

    def make_chain(self, prompt, *, enable_thinking: bool | None = False):
        if self.config.fake_llm:
            return FakeChain()
        return prompt | self._llm_for_call(enable_thinking)

    def _llm_for_call(self, enable_thinking: bool | None):
        """Return the shared model with an optional request-scoped thinking control.

        The base ``self.llm`` remains unmodified so callers that need reasoning can
        continue to use the model normally. ``ChatOpenAI`` forwards ``extra_body``
        to the OpenAI-compatible server, where Qwen's chat template consumes the
        ``enable_thinking`` request parameter.
        """
        if enable_thinking is None:
            return self.llm
        return self.llm.bind(
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
        )

    def record_diagnostic(
        self,
        code: str,
        severity: str,
        stage: str,
        message: str,
        *,
        details: Any = None,
        node_ids: list[str] | None = None,
        evidence_paths: list[Any] | None = None,
        excerpts: list[str] | None = None,
    ) -> str | None:
        if self.active_errata is None:
            return None
        return self.active_errata.record(
            code,
            severity,
            stage,
            message,
            details=details,
            node_ids=node_ids,
            evidence_paths=evidence_paths,
            excerpts=excerpts,
        )

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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

    def diagnose_errata(self, recorder: ErrataRecorder) -> None:
        """Run the bounded advisory diagnosis pass without affecting file success."""
        if not recorder.has_anomalies() and not recorder.diagnosis_requested_explicitly:
            recorder.set_diagnosis({"status": "not_requested", "reason": "no_actionable_anomalies"})
            return
        if not self.config.errata_enabled or not self.config.errata_llm_on_anomaly:
            recorder.set_diagnosis({"status": "not_requested", "reason": "diagnosis_disabled"})
            return
        if recorder.has_model_service_failure():
            recorder.set_diagnosis({"status": "skipped", "reason": "model_service_unavailable"})
            return

        packet = recorder.review_packet()
        packet_text = json.dumps(packet, ensure_ascii=True, separators=(",", ":"))
        prompt_text = f"ERRATA_REVIEW_PACKET\n{packet_text}"
        recorder.diagnosis_input_digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        finding_ids = {str(item.get("finding_id")) for item in recorder.findings}
        recorder.record(
            "errata_diagnosis_requested",
            "info",
            "errata",
            "Requested an advisory LLM diagnosis for actionable file findings.",
            details={"finding_count": len(recorder.findings), "input_chars": len(prompt_text)},
        )
        response_text = ""
        token_usage: dict[str, int] = {}
        diagnosis_validation_details: dict[str, Any] = {}
        attempt_prompt_text = prompt_text
        start = time.time()
        try:
            def run_and_validate():
                nonlocal response_text, token_usage
                raw_response = self.diagnosis_chain.invoke({"text": attempt_prompt_text})
                token_usage = extract_token_usage(raw_response)
                response_text = extract_llm_text(raw_response)
                parsed = extract_json_payload(response_text)
                errors = validate_diagnosis_payload(parsed, finding_ids)
                if errors:
                    actions = parsed.get("actions") if isinstance(parsed, dict) else []
                    diagnosis_validation_details["validation_errors"] = errors[:8]
                    diagnosis_validation_details["invalid_action_types"] = [
                        {
                            "index": index,
                            "value": short_text(str(action.get("type")), 120),
                        }
                        for index, action in enumerate(actions or [])
                        if isinstance(action, dict) and action.get("type") not in DIAGNOSIS_ACTION_TYPES
                    ]
                    raise ValueError("; ".join(errors[:8]))
                return parsed

            def on_retry(attempt, exc):
                nonlocal attempt_prompt_text
                corrective_prompt_applied = bool(diagnosis_validation_details.get("validation_errors"))
                if corrective_prompt_applied:
                    feedback = "; ".join(diagnosis_validation_details["validation_errors"][:4])
                    attempt_prompt_text = (
                        f"{prompt_text}\n\nDIAGNOSIS VALIDATION FEEDBACK\n{feedback}\n"
                        "Return a corrected JSON object and use only the allowed action type values."
                    )
                recorder.record(
                    "errata_diagnosis_retry",
                    "warning",
                    "errata",
                    "The advisory diagnosis request required a retry.",
                    details={
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "corrective_prompt_applied": corrective_prompt_applied,
                    },
                )

            parsed = with_retry(
                run_and_validate,
                "errata diagnosis",
                on_retry=on_retry,
            )
            self.performance.record_llm_result(
                "errata diagnosis",
                time.time() - start,
                response_text,
                token_usage,
            )
            recorder.set_diagnosis(
                {
                    "status": "completed",
                    **parsed,
                    "model": self.config.lm_studio_model,
                    "input_digest": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                    "output_digest": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                    "token_usage": token_usage,
                }
            )
        except Exception as exc:
            self.performance.record_failure()
            error_text = f"{type(exc).__name__}: {exc}"
            debug_path = self.write_llm_debug_event(
                label="errata diagnosis",
                event="errata_diagnosis_failed",
                prompt_text=attempt_prompt_text,
                response_text=short_text(response_text, 1600),
                error=error_text,
            )
            recorder.add_debug_artifact(debug_path)
            recorder.record(
                "errata_diagnosis_failed",
                "warning",
                "errata",
                "The advisory LLM diagnosis failed or returned malformed/ungrounded JSON; file outcome is unchanged.",
                details={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "debug_path": str(debug_path),
                    **diagnosis_validation_details,
                },
                evidence_paths=[str(debug_path)],
                excerpts=[short_text(response_text, 800)],
            )
            recorder.set_diagnosis(
                {
                    "status": "failed",
                    "error": error_text,
                    "response_excerpt": short_text(response_text, 1600),
                    "debug_path": str(debug_path),
                    **diagnosis_validation_details,
                }
            )

    def build_errata_review_context(self, docs: list[Document]) -> dict[str, Any]:
        """Build bounded node metadata and graph paths without transcript bodies."""
        max_excerpt = int(getattr(self.config, "errata_excerpt_max_chars", 360))
        nodes = []
        for doc in docs[:500]:
            metadata = dict(doc.metadata or {})
            nodes.append(
                {
                    "node_id": metadata.get("node_id"),
                    "node_type": metadata.get("node_type"),
                    "level": metadata.get("level"),
                    "parent_id": metadata.get("parent_id"),
                    "child_ids": list(metadata.get("child_ids") or [])[:100],
                    "speaker": metadata.get("speaker"),
                    "excerpt": clip_text(doc.page_content, max_excerpt),
                }
            )
        return {"nodes": nodes, "node_count": len(docs)}

    def _record_fallback(self) -> None:
        """Count a fallback both for the run and for the active file."""
        self.fallback_count += 1
        self._file_fallback_count = getattr(self, "_file_fallback_count", 0) + 1

    def invoke_llm(self, chain, text: str, label: str) -> str:
        if runtime.STOP_REQUESTED:
            raise PipelineInterrupted("Stop requested before starting another model request.")
        diagnostic_stage = "position_extraction" if "position extraction" in label else "summarization"
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
            if self.active_errata is not None:
                self.active_errata.add_debug_artifact(debug_path)
            self.record_diagnostic(
                "context_overflow_preflight",
                "error",
                diagnostic_stage,
                f"{label} exceeded the configured prompt budget before invocation.",
                details={"prompt_tokens": prompt_tokens, "prompt_budget": self.config.prompt_token_budget},
                evidence_paths=[str(debug_path)],
            )
            raise ValueError(f"{label} estimated prompt tokens exceed configured prompt budget.")

        start = time.time()
        token_usage: dict[str, int] = {}
        attempt_text = text
        try:
            def run_and_validate():
                nonlocal token_usage
                raw_candidate = chain.invoke({"text": attempt_text})
                token_usage = extract_token_usage(raw_candidate)
                candidate = extract_llm_text(raw_candidate)
                if not has_substantive_text(candidate, min_chars=1):
                    debug_path = self.write_llm_debug_event(
                        label=label,
                        event="empty_response",
                        prompt_text=attempt_text,
                        response_text=candidate,
                        error="Model returned empty assistant message content.",
                        raw_response=raw_candidate,
                    )
                    print(f"  debug saved: {debug_path}")
                    if self.active_errata is not None:
                        self.active_errata.add_debug_artifact(debug_path)
                    self.record_diagnostic(
                        "llm_empty_response",
                        "warning",
                        diagnostic_stage,
                        f"{label} returned an empty response during a retry attempt.",
                        evidence_paths=[str(debug_path)],
                    )
                    raise EmptyLLMResponse(f"{label} returned an empty response.")
                if is_missing_context_response(candidate):
                    debug_path = self.write_llm_debug_event(
                        label=label,
                        event="missing_context_response",
                        prompt_text=attempt_text,
                        response_text=candidate,
                        error="Response looked like the model was asking for source text that was already provided.",
                        raw_response=raw_candidate,
                    )
                    print(f"  debug saved: {debug_path}")
                    if self.active_errata is not None:
                        self.active_errata.add_debug_artifact(debug_path)
                    self.record_diagnostic(
                        "llm_missing_context_response",
                        "warning",
                        diagnostic_stage,
                        f"{label} returned a missing-context response.",
                        evidence_paths=[str(debug_path)],
                    )
                    raise MissingContextResponse(f"{label} returned a missing-context response instead of a summary.")
                return candidate

            def on_retry(attempt, exc):
                nonlocal attempt_text
                corrective_prompt_applied = isinstance(exc, MissingContextResponse)
                if corrective_prompt_applied:
                    attempt_text = (
                        "CORRECTION: The source material is already present below. Episode dates are historical "
                        "metadata, not deadlines or scheduling signals. Do not request source text or refuse "
                        "because the date appears earlier or later than today's date. Return the requested "
                        "output now.\n\n"
                        f"{text}"
                    )
                self.record_diagnostic(
                    "llm_retry",
                    "warning",
                    diagnostic_stage,
                    f"{label} required a retry before succeeding or falling back.",
                    details={
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "corrective_prompt_applied": corrective_prompt_applied,
                    },
                )

            result = with_retry(
                run_and_validate,
                label,
                retries=2,
                on_retry=on_retry,
            )
        except EmptyLLMResponse:
            self.performance.record_failure()
            if "position extraction" in label:
                print(f"  {label} returned empty responses; using empty position list.")
                return '{"positions": []}'
            print(f"  {label} returned empty responses; using fallback extractive summary.")
            self._record_fallback()
            result = fallback_summary_from_text(text, label)
            debug_path = self.write_llm_debug_event(
                label=label,
                event="fallback_after_empty_response",
                prompt_text=text,
                response_text=result,
                error="All retries returned empty assistant message content.",
            )
            print(f"  debug saved: {debug_path}")
            if self.active_errata is not None:
                self.active_errata.add_debug_artifact(debug_path)
            self.record_diagnostic(
                "llm_empty_response_fallback",
                "warning",
                diagnostic_stage,
                f"{label} used a fallback after all retries returned empty responses.",
                evidence_paths=[str(debug_path)],
            )
        except MissingContextResponse:
            self.performance.record_failure()
            if "position extraction" in label:
                print(f"  {label} returned missing-context responses; using empty position list.")
                return '{"positions": []}'
            print(f"  {label} returned missing-context responses; using fallback extractive summary.")
            self._record_fallback()
            result = fallback_summary_from_text(text, label)
            debug_path = self.write_llm_debug_event(
                label=label,
                event="fallback_extractive_summary",
                prompt_text=text,
                response_text=result,
                error="All retries returned missing-context responses.",
            )
            print(f"  debug saved: {debug_path}")
            if self.active_errata is not None:
                self.active_errata.add_debug_artifact(debug_path)
            self.record_diagnostic(
                "llm_missing_context_fallback",
                "warning",
                diagnostic_stage,
                f"{label} used a fallback after all retries returned missing-context responses.",
                evidence_paths=[str(debug_path)],
            )
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
            if self.active_errata is not None:
                self.active_errata.add_debug_artifact(debug_path)
            service_failure = isinstance(exc, (ConnectionError, TimeoutError)) or any(
                token in str(exc).lower()
                for token in ("connection", "connect", "refused", "unavailable", "timed out")
            )
            self.record_diagnostic(
                "llm_service_unavailable" if service_failure else "llm_request_failed",
                "error" if service_failure else "warning",
                diagnostic_stage,
                f"{label} raised an exception after retries.",
                details={"error_type": type(exc).__name__, "error": error_text},
                evidence_paths=[str(debug_path)],
            )
            if "position extraction" in label:
                print(f"  {label} failed after retries; using empty position list. error={error_text}")
                return '{"positions": []}'
            print(f"  {label} failed after retries; using fallback extractive summary. error={error_text}")
            self._record_fallback()
            result = fallback_summary_from_text(text, label)

        self.performance.record_llm_result(label, time.time() - start, result, token_usage)
        return result

    def normalize_doc(self, doc: Document, source: str, index: int) -> Document:
        metadata = dict(doc.metadata or {})
        metadata["source"] = source
        metadata["episode_id"] = metadata.get("episode_id") or stable_episode_id(source)
        metadata["episode_title"] = metadata.get("episode_title") or episode_title_from_source(source)
        metadata["source_type"] = metadata.get("source_type") or "json_transcript"
        metadata["segment_index"] = metadata.get("segment_index", index)
        metadata["source_segment_id"] = str(
            metadata.get("source_segment_id")
            or metadata.get("segment_id")
            or f"{metadata['episode_id']}:segment:{metadata['segment_index']}"
        )
        metadata["start_time"] = safe_float(metadata.get("start_time"))
        metadata["end_time"] = safe_float(metadata.get("end_time"))
        metadata["speaker"] = metadata.get("speaker")
        metadata["episode_date"] = parse_episode_date(metadata.get("episode_date"))
        metadata["episode_date_compact"] = metadata.get("episode_date_compact") or compact_episode_date(metadata.get("episode_date"))
        metadata["episode_sort_key"] = metadata.get("episode_sort_key") or episode_sort_key(metadata.get("episode_date"))
        return Document(page_content=doc.page_content.strip(), metadata=metadata)

    def evidence_metadata(self, docs: list[Document]) -> dict[str, Any]:
        segment_ids = []
        spans = []
        seen_ids = set()
        seen_spans = set()
        for doc in docs:
            metadata = doc.metadata
            ids = metadata.get("source_segment_ids") or ([metadata.get("source_segment_id")] if metadata.get("source_segment_id") else [])
            for segment_id in ids:
                if segment_id and segment_id not in seen_ids:
                    seen_ids.add(segment_id)
                    segment_ids.append(str(segment_id))
            if metadata.get("source_spans"):
                source_spans = metadata["source_spans"]
            elif metadata.get("start_time") is not None and metadata.get("end_time") is not None:
                source_spans = [{
                    "start_time": metadata["start_time"],
                    "end_time": metadata["end_time"],
                    "segment_ids": [str(segment_id) for segment_id in ids if segment_id],
                }]
            else:
                source_spans = []
            for span in source_spans:
                key = json.dumps(span, sort_keys=True, ensure_ascii=True, default=str)
                if key not in seen_spans:
                    seen_spans.add(key)
                    spans.append(span)
        return {"source_segment_ids": segment_ids, "source_spans": spans}

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
                **self.evidence_metadata(docs),
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
                    self.record_diagnostic(
                        "clustering_reduction_fallback",
                        "warning",
                        "hierarchy",
                        "UMAP reduction was unavailable; PCA was used for clustering.",
                        details={"error_type": type(exc).__name__, "error": str(exc)},
                    )
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
            self.record_diagnostic(
                "fallback_grouping_used",
                "warning",
                "hierarchy",
                "The cluster count exceeded the configured maximum; deterministic fallback grouping was used.",
                details={"cluster_count": len(clusters), "max_clusters": self.config.max_clusters, "group_size": self.config.group_fallback_size},
            )
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
                **self.evidence_metadata(docs),
            },
        )

        if summary_doc.metadata.get("fallback_generated"):
            self.record_diagnostic(
                "fallback_summary_generated",
                "warning",
                "summarization",
                "A deterministic fallback summary was generated after model output could not be used.",
                node_ids=[node_id],
                details={"summary_generation": summary_doc.metadata.get("summary_generation")},
                excerpts=[short_text(summary, getattr(self.config, "errata_excerpt_max_chars", 360))],
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
            all_summaries = []
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
                    all_summaries.append(summary_doc)
                    continue
                seen_summary_texts.append((key, summary_doc.page_content))
                summary_doc.metadata["duplicate_summary"] = False
                unique_summaries.append(summary_doc)
                all_summaries.append(summary_doc)
            all_nodes.extend(all_summaries)
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
                **self.evidence_metadata(leaf_chunks),
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

    @staticmethod
    def normalize_position_evidence_ids(value: Any, known_node_ids: set[str]) -> list[str]:
        """Keep known position evidence IDs once, preserving model order."""
        result = []
        seen = set()
        for node_id in coerce_string_list(value):
            if node_id in known_node_ids and node_id not in seen:
                seen.add(node_id)
                result.append(node_id)
        return result

    @classmethod
    def sanitize_position_documents(
        cls, position_docs: list[Document], known_node_ids: set[str]
    ) -> list[Document]:
        """Normalize checkpointed position evidence without invoking the LLM."""
        sanitized = []
        for doc in position_docs:
            if doc.metadata.get("node_type") == "position_card":
                evidence_ids = cls.normalize_position_evidence_ids(
                    doc.metadata.get("child_ids"), known_node_ids
                )
                if not evidence_ids:
                    continue
                doc.metadata["child_ids"] = evidence_ids
            sanitized.append(doc)
        return sanitized

    def sanitize_position_documents_with_diagnostics(
        self, position_docs: list[Document], known_node_ids: set[str]
    ) -> list[Document]:
        """Apply checkpoint normalization and record what was discarded."""
        sanitized = []
        for doc in position_docs:
            if doc.metadata.get("node_type") != "position_card":
                sanitized.append(doc)
                continue
            raw_ids = coerce_string_list(doc.metadata.get("child_ids"))
            evidence_ids = self.normalize_position_evidence_ids(raw_ids, known_node_ids)
            duplicate_count = len(raw_ids) - len(set(raw_ids))
            unknown_ids = [node_id for node_id in raw_ids if node_id not in known_node_ids]
            node_id = str(doc.metadata.get("node_id") or "")
            if duplicate_count:
                self.record_diagnostic(
                    "duplicate_evidence_ids_removed",
                    "warning",
                    "checkpoint_positions",
                    "Duplicate evidence IDs were removed while loading a position checkpoint.",
                    details={"count": duplicate_count, "original_ids": raw_ids},
                    node_ids=[node_id] if node_id else None,
                )
            if unknown_ids:
                self.record_diagnostic(
                    "unknown_evidence_ids_discarded",
                    "warning",
                    "checkpoint_positions",
                    "Unknown evidence IDs were discarded while loading a position checkpoint.",
                    details={"unknown_ids": unknown_ids, "known_count": len(known_node_ids)},
                    node_ids=[node_id] if node_id else None,
                )
            if not evidence_ids:
                self.record_diagnostic(
                    "position_quarantined_orphaned_evidence",
                    "warning",
                    "checkpoint_positions",
                    "A checkpointed position card was skipped because no known evidence remained.",
                    details={"original_evidence_ids": raw_ids},
                    node_ids=[node_id] if node_id else None,
                )
                continue
            doc.metadata["child_ids"] = evidence_ids
            sanitized.append(doc)
        return sanitized

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
                self.record_diagnostic(
                    "truncated_json_recovered",
                    "warning",
                    "position_extraction",
                    f"{label} returned truncated JSON; complete position objects were recovered.",
                    details={"recovered_count": len(partial_positions), "raw_char_count": len(raw or "")},
                    excerpts=[short_text(raw, 800)],
                )
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
            if self.active_errata is not None:
                self.active_errata.add_debug_artifact(debug_path)
            self.record_diagnostic(
                "position_payload_not_list",
                "warning",
                "position_extraction",
                f"{label} returned a payload without a usable positions array.",
                details={"parsed_payload_type": type(payload).__name__},
                evidence_paths=[str(debug_path)],
                excerpts=[short_text(raw, 800)],
            )
            return []
        return [position for position in positions if isinstance(position, dict) and position.get("claim")]

    def extract_positions(self, all_nodes: list[Document], thesis_doc: Document) -> list[Document]:
        source_docs = self.build_position_source_docs(all_nodes, thesis_doc)
        batches = self.build_position_batches(source_docs)
        node_lookup = {doc.metadata.get("node_id"): doc for doc in all_nodes if doc.metadata.get("node_id")}
        known_node_ids = set(node_lookup)

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
            raw_evidence_ids = coerce_string_list(position.get("evidence_node_ids"))
            evidence_ids = self.normalize_position_evidence_ids(
                raw_evidence_ids, known_node_ids
            )
            duplicate_count = len(raw_evidence_ids) - len(set(raw_evidence_ids))
            unknown_ids = [node_id for node_id in raw_evidence_ids if node_id not in known_node_ids]
            if duplicate_count:
                self.record_diagnostic(
                    "duplicate_evidence_ids_removed",
                    "warning",
                    "position_extraction",
                    "Duplicate evidence IDs were removed from a model-generated position.",
                    details={"count": duplicate_count, "original_ids": raw_evidence_ids},
                    node_ids=evidence_ids,
                )
            if unknown_ids:
                self.record_diagnostic(
                    "unknown_evidence_ids_discarded",
                    "warning",
                    "position_extraction",
                    "Unknown evidence IDs were discarded from a model-generated position.",
                    details={"unknown_ids": unknown_ids, "known_count": len(known_node_ids)},
                    node_ids=evidence_ids,
                )
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
                self.record_diagnostic(
                    "position_quarantined_ambiguous_speaker",
                    "warning",
                    "position_extraction",
                    "A model-generated position was skipped because its speaker attribution was ambiguous.",
                    details={"speaker": position_speaker, "claim": claim},
                    node_ids=evidence_ids,
                )
                debug_path = self.write_llm_debug_event(
                    label="position extraction",
                    event="position_quarantined_ambiguous_speaker",
                    prompt_text="",
                    response_text=json.dumps(position, ensure_ascii=True, default=str),
                    error="Structured position did not contain an attributable single speaker.",
                )
                if self.active_errata is not None:
                    self.active_errata.add_debug_artifact(debug_path)
                continue
            if not claim:
                continue
            if not evidence_docs:
                self.record_diagnostic(
                    "position_quarantined_orphaned_evidence",
                    "warning",
                    "position_extraction",
                    "A model-generated position was skipped because it had no known evidence nodes.",
                    details={"original_evidence_ids": raw_evidence_ids},
                )
                debug_path = self.write_llm_debug_event(
                    label="position extraction",
                    event="position_quarantined_orphaned_evidence",
                    prompt_text="",
                    response_text=json.dumps(position, ensure_ascii=True, default=str),
                    error="Structured position referenced no known evidence node.",
                )
                if self.active_errata is not None:
                    self.active_errata.add_debug_artifact(debug_path)
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
                        **self.evidence_metadata(evidence_docs),
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
            self.record_diagnostic(
                "document_validation_failed",
                "error",
                "validation",
                "Document-level validation failed before cache creation.",
                details={"errors": bad[:100], "error_count": len(bad)},
                node_ids=[str(doc.metadata.get("node_id")) for doc in docs if doc.metadata.get("node_id")][:100],
            )
            raise ValueError(f"{label} produced invalid documents: {preview}")
        result = validate_processed_documents(docs, require_provenance=True)
        if result.errors:
            self.record_diagnostic(
                "structural_validation_failed",
                "error",
                "validation",
                "Processed documents failed structural or provenance validation.",
                details={"errors": result.errors[:100], "issues": result.issues[:100]},
                node_ids=[str(issue_node) for issue in result.issues for issue_node in issue.get("node_ids", [])][:100],
                evidence_paths=[path for issue in result.issues for path in issue.get("evidence_paths", [])][:50],
            )
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

    def save_cached_documents(self, cache_path: Path, source_path: Path, fingerprint: str, docs: list[Document], context: dict[str, Any] | None = None) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._apply_active_context(docs, context or getattr(self, "active_context", None) or {})
        self.validate_documents_before_cache(docs, f"cache write {cache_path}")
        validation = validate_processed_documents(docs)
        representation_builder = RepresentationBuilder(
            embedding_text_mode=self.config.embedding_text_mode,
            lexical_text_mode=self.config.lexical_text_mode,
            contextual_header_max_chars=self.config.contextual_header_max_chars,
        )
        active_context = context or getattr(self, "active_context", None) or {}
        cache_source_path = str(active_context.get("selected_transcript_relative_path") or source_path)
        source_artifact_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        identity_fields = {
            key: active_context.get(key)
            for key in (
                "partition_id",
                "corpus_id",
                "partition_display_name",
                "context_type",
                "workflow_profile",
                "partition_config_fingerprint",
                "handoff_id",
                "episode_id",
                "episode_uid",
                "correction_set_id",
                "selected_variant",
                "selected_transcript_relative_path",
                "selected_transcript_artifact_sha256",
                "selected_transcript_canonical_payload_sha256",
                "source_audio_fingerprint",
            )
            if active_context.get(key) not in (None, "")
        }
        representation_manifest = representation_builder.manifest()
        payload = {
            "version": 2,
            "schema_version": self.config.cache_schema_version,
            "pipeline_version": PIPELINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_path": cache_source_path,
            "source_fingerprint": fingerprint,
            "cache_key": fingerprint,
            "source_transcript_hash": source_artifact_hash,
            "source_artifact_sha256": active_context.get("selected_transcript_artifact_sha256") or source_artifact_hash,
            "canonical_payload_sha256": active_context.get("selected_transcript_canonical_payload_sha256"),
            "source_schema_version": source_schema_version(source_path),
            "stable_source_id": stable_episode_id(fingerprint),
            **identity_fields,
            "config_fingerprint": config_fingerprint(self.config),
            "generation_config_fingerprint": generation_config_fingerprint(self.config),
            "representation_config_fingerprint": representation_manifest.get("config_fingerprint"),
            "model": self.config.lm_studio_model,
            "embedding_model": self.config.embedding_model,
            "representations": representation_manifest,
            "prompt_manifest": self.prompt_manifest,
            "token_maxima": self.performance.snapshot(),
            "fallback_count": getattr(self, "_file_fallback_count", 0),
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
                "cache_key": fingerprint,
                "source_transcript_hash": source_artifact_hash,
                "source_artifact_sha256": active_context.get("selected_transcript_artifact_sha256") or source_artifact_hash,
                "canonical_payload_sha256": active_context.get("selected_transcript_canonical_payload_sha256"),
                **identity_fields,
                "model": self.config.lm_studio_model,
                "config_fingerprint": config_fingerprint(self.config),
                "generation_config_fingerprint": generation_config_fingerprint(self.config),
                "representations": representation_manifest,
            },
            "document_count": len(docs),
            "documents": document_payloads(docs, fingerprint, representation_builder),
        }
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        temp_path.replace(cache_path)

    def validate_cached_file(self, path: Path, fingerprint: str, cache_path: Path, context: dict[str, Any] | None = None) -> dict[str, Any]:
        print(f"\nValidating cached processed data: {cache_path}")
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.record_diagnostic(
                "cache_read_failed",
                "error",
                "cache_validation",
                "The processed-data cache could not be read as JSON.",
                details={"path": str(cache_path), "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        cache_validation = validate_processed_cache(payload)
        if cache_validation.errors:
            self.record_diagnostic(
                "cache_validation_structural_failure",
                "error",
                "cache_validation",
                "The processed-data cache failed envelope or document validation.",
                details={"errors": cache_validation.errors[:100], "issues": cache_validation.issues[:100]},
                node_ids=[str(node_id) for issue in cache_validation.issues for node_id in issue.get("node_ids", [])][:100],
                evidence_paths=[path for issue in cache_validation.issues for path in issue.get("evidence_paths", [])][:50],
            )
        cache_validation.raise_for_errors(f"cache {cache_path}")
        expected_context = context or getattr(self, "active_context", None) or {}
        cache_key = payload.get("cache_key") or payload.get("source_fingerprint")
        if cache_key != fingerprint:
            self.record_diagnostic(
                "cache_identity_mismatch",
                "error",
                "cache_validation",
                "The processed-data cache key does not match the selected episode input.",
                details={"expected_cache_key": fingerprint, "actual_cache_key": cache_key},
            )
            raise ValueError("processed cache key does not match the selected input")
        if expected_context:
            identity_mismatches = []
            for key in ("partition_id", "corpus_id", "episode_uid", "handoff_id", "correction_set_id", "selected_transcript_artifact_sha256", "selected_transcript_canonical_payload_sha256"):
                expected = expected_context.get(key)
                if expected not in (None, "") and payload.get(key) != expected:
                    identity_mismatches.append({"field": key, "expected": expected, "actual": payload.get(key)})
            if identity_mismatches:
                self.record_diagnostic(
                    "cache_identity_mismatch",
                    "error",
                    "cache_validation",
                    "The processed-data cache belongs to a different handoff or processing partition.",
                    details={"mismatches": identity_mismatches},
                )
                raise ValueError("processed cache identity does not match the selected handoff")
        docs = self.load_cached_documents(cache_path)
        if not docs:
            raise RuntimeError(f"Processed data cache was empty: {cache_path}")
        self.validate_documents_before_cache(docs, f"cache {cache_path}")
        print(f"  Cached processed data is valid for {path}: {len(docs)} documents")
        first_metadata = dict(docs[0].metadata or {})
        node_type_counts = {}
        for doc in docs:
            node_type = str(doc.metadata.get("node_type") or "unknown")
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        return {
            "status": "completed",
            "source": "processed_data_cache",
            "nodes": len(docs),
            "cache_path": str(cache_path),
            "episode_id": first_metadata.get("episode_id"),
            "episode_title": first_metadata.get("episode_title"),
            "episode_date": first_metadata.get("episode_date"),
            "source_type": first_metadata.get("source_type"),
            "leaf_chunks": node_type_counts.get("leaf_chunk", 0),
            "summaries": node_type_counts.get("cluster_summary", 0),
            "positions": node_type_counts.get("position_card", 0),
        }

    def load_file_checkpoint(self, source_path: Path, fingerprint: str, stage: str) -> list[Document] | None:
        if not self.config.resume_within_file:
            if self.active_errata is not None:
                self.active_errata.set_checkpoint_reuse(stage, False)
            return None
        path = checkpoint_path(self.config, self.project_dir, source_path, fingerprint, stage)
        if not path.exists():
            if self.active_errata is not None:
                self.active_errata.set_checkpoint_reuse(stage, False)
            return None
        try:
            payload = read_json_file(path)
        except Exception as exc:
            self.record_diagnostic(
                "checkpoint_corrupt",
                "warning",
                stage,
                f"The {stage} checkpoint could not be read and was ignored.",
                details={"path": str(path), "error_type": type(exc).__name__, "error": str(exc)},
            )
            if self.active_errata is not None:
                self.active_errata.set_checkpoint_reuse(stage, False)
            return None
        if not isinstance(payload, dict):
            self.record_diagnostic(
                "checkpoint_invalid_shape",
                "warning",
                stage,
                f"The {stage} checkpoint is not a JSON object and was ignored.",
                details={"path": str(path), "payload_type": type(payload).__name__},
            )
            if self.active_errata is not None:
                self.active_errata.set_checkpoint_reuse(stage, False)
            return None
        if payload.get("source_fingerprint") != fingerprint:
            self.record_diagnostic(
                "checkpoint_stale",
                "warning",
                stage,
                f"The {stage} checkpoint belongs to a different source fingerprint and was ignored.",
                details={"path": str(path), "checkpoint_fingerprint": payload.get("source_fingerprint"), "expected_fingerprint": fingerprint},
            )
            if self.active_errata is not None:
                self.active_errata.set_checkpoint_reuse(stage, False)
            return None
        active_context = getattr(self, "active_context", None) or {}
        identity_mismatches = [
            {"field": key, "expected": active_context.get(key), "actual": payload.get(key)}
            for key in ("partition_id", "corpus_id", "episode_uid", "handoff_id", "correction_set_id")
            if active_context.get(key) not in (None, "") and payload.get(key) != active_context.get(key)
        ]
        if identity_mismatches:
            self.record_diagnostic(
                "checkpoint_identity_mismatch",
                "warning",
                stage,
                f"The {stage} checkpoint belongs to another handoff or partition and was ignored.",
                details={"mismatches": identity_mismatches, "path": str(path)},
            )
            if self.active_errata is not None:
                self.active_errata.set_checkpoint_reuse(stage, False)
            return None
        documents = payload.get("documents")
        if not isinstance(documents, list) or any(not isinstance(item, dict) for item in documents):
            self.record_diagnostic(
                "checkpoint_invalid_shape",
                "warning",
                stage,
                f"The {stage} checkpoint has an invalid document array and was ignored.",
                details={"path": str(path), "document_count": len(documents) if isinstance(documents, list) else None},
            )
            if self.active_errata is not None:
                self.active_errata.set_checkpoint_reuse(stage, False)
            return None
        print(f"  checkpoint reused: {stage} ({len(documents)} document(s))")
        if self.active_errata is not None:
            self.active_errata.set_checkpoint_reuse(stage, True)
        self.record_diagnostic(
            "checkpoint_reused",
            "info",
            stage,
            f"Reused the {stage} checkpoint.",
            details={"path": str(path), "document_count": len(documents)},
        )
        return docs_from_payloads(documents)

    def save_file_checkpoint(self, source_path: Path, fingerprint: str, stage: str, docs: list[Document]) -> None:
        if not self.config.resume_within_file:
            return
        path = checkpoint_path(self.config, self.project_dir, source_path, fingerprint, stage)
        active_context = getattr(self, "active_context", None) or {}
        write_json_file(
            path,
            {
                "stage": stage,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_path": str(source_path),
                "source_fingerprint": fingerprint,
                **{
                    key: active_context[key]
                    for key in ("partition_id", "corpus_id", "handoff_id", "episode_id", "episode_uid", "correction_set_id", "selected_variant", "selected_transcript_artifact_sha256", "selected_transcript_canonical_payload_sha256", "source_audio_fingerprint")
                    if active_context.get(key) not in (None, "")
                },
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

    def process_file(
        self, path: Path, errata: ErrataRecorder | None = None, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        previous_errata = self.active_errata
        previous_context = getattr(self, "active_context", None)
        self.active_errata = errata
        self.active_context = context
        try:
            return self._process_file(path)
        except PipelineInterrupted as exc:
            self.record_diagnostic(
                "file_interrupted",
                "error",
                "file",
                "File processing was interrupted before completion.",
                details={"error": str(exc)},
            )
            raise
        except Exception as exc:
            self.record_diagnostic(
                "file_exception",
                "error",
                "file",
                "File processing raised an exception before completion.",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        finally:
            self.active_errata = previous_errata
            self.active_context = previous_context

    @staticmethod
    def _apply_active_context(docs: list[Document], context: dict[str, Any]) -> None:
        """Carry managed handoff identity onto every derived document."""
        if not context:
            return
        identity_keys = (
            "partition_id",
            "corpus_id",
            "partition_display_name",
            "context_type",
            "workflow_profile",
            "partition_config_fingerprint",
            "handoff_id",
            "episode_uid",
            "correction_set_id",
            "selected_variant",
            "source_audio_fingerprint",
        )
        for doc in docs:
            for key in identity_keys:
                if context.get(key) not in (None, ""):
                    doc.metadata[key] = context[key]
            if context.get("episode_id"):
                doc.metadata["episode_id"] = context["episode_id"]
            if context.get("episode_title"):
                doc.metadata["episode_title"] = context["episode_title"]
            if context.get("episode_date"):
                doc.metadata["episode_date"] = context["episode_date"]

    def _process_file(self, path: Path) -> dict[str, Any]:
        """Build or resume all retrieval artifacts for a single transcript file."""
        source = str(path)
        active_context = getattr(self, "active_context", None) or {}
        fingerprint = str(active_context.get("processing_key") or file_fingerprint(path))
        print(f"\nProcessing: {source}")
        self._file_fallback_count = 0
        self.performance.start_file(source)
        docs = load_transcript_json(path)
        if active_context:
            for doc in docs:
                for key in (
                    "partition_id",
                    "corpus_id",
                    "partition_display_name",
                    "context_type",
                    "workflow_profile",
                    "partition_config_fingerprint",
                    "handoff_id",
                    "episode_uid",
                    "correction_set_id",
                    "selected_variant",
                    "source_audio_fingerprint",
                ):
                    if active_context.get(key) not in (None, ""):
                        doc.metadata[key] = active_context[key]
                if active_context.get("episode_id"):
                    doc.metadata["episode_id"] = active_context["episode_id"]
                if active_context.get("episode_title"):
                    doc.metadata["episode_title"] = active_context["episode_title"]
                if active_context.get("episode_date"):
                    doc.metadata["episode_date"] = active_context["episode_date"]
        if self.active_errata is not None and docs:
            first_metadata = dict(docs[0].metadata or {})
            self.active_errata.update_source(
                episode_id=first_metadata.get("episode_id"),
                episode_title=first_metadata.get("episode_title"),
                source_type=first_metadata.get("source_type"),
                episode_date=first_metadata.get("episode_date"),
                episode_date_compact=first_metadata.get("episode_date_compact"),
                episode_sort_key=first_metadata.get("episode_sort_key"),
            )
            missing_metadata = [
                field
                for field in ("episode_id", "episode_title", "episode_date", "source_type")
                if first_metadata.get(field) in (None, "")
            ]
            if missing_metadata:
                self.record_diagnostic(
                    "missing_metadata",
                    "warning",
                    "input",
                    "The transcript metadata was incomplete; downstream attribution may be degraded.",
                    details={"fields": missing_metadata},
                )
        leaf_chunks = self.load_file_checkpoint(path, fingerprint, "leaf_chunks")
        if leaf_chunks is None:
            # Leaf chunks are deterministic, so they are the first cheap checkpoint.
            leaf_chunks = self.build_leaf_chunks(docs, source)
            self.save_file_checkpoint(path, fingerprint, "leaf_chunks", leaf_chunks)
        if self.active_errata is not None:
            self.active_errata.update_metrics(leaf_chunks=len(leaf_chunks))

        if not leaf_chunks:
            print("  No usable text found; skipping")
            self.record_diagnostic(
                "no_usable_text",
                "warning",
                "input",
                "The transcript contained no usable text after normalization.",
            )
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
        if self.active_errata is not None:
            self.active_errata.update_metrics(
                summaries=len([doc for doc in all_nodes if doc.metadata.get("node_type") == "cluster_summary"]),
                hierarchy_documents=len(all_nodes),
            )
        position_docs = self.load_file_checkpoint(path, fingerprint, "positions")
        if position_docs is None:
            position_docs = self.extract_positions(all_nodes, thesis_doc)
        position_docs = self.sanitize_position_documents_with_diagnostics(
            position_docs,
            {
                str(doc.metadata["node_id"])
                for doc in all_nodes
                if doc.metadata.get("node_id")
            },
        )
        self.save_file_checkpoint(path, fingerprint, "positions", position_docs)
        all_nodes.extend(position_docs)
        self._apply_active_context(all_nodes, active_context)
        if self.active_errata is not None:
            self.active_errata.update_metrics(positions=len(position_docs))
            self.active_errata.set_review_context(self.build_errata_review_context(all_nodes))
            self.active_errata.update_metrics(
                leaf_chunks=len(leaf_chunks),
                summaries=len([doc for doc in all_nodes if doc.metadata.get("node_type") == "cluster_summary"]),
                positions=len(position_docs),
                documents=len(all_nodes),
            )
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
            "fallbacks": getattr(self, "_file_fallback_count", 0),
            "documents": all_nodes,
        }
