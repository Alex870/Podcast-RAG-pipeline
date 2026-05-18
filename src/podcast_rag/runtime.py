from __future__ import annotations

import datetime as dt
import json
import signal
import time
from collections import deque
from pathlib import Path
from typing import Any

from podcast_rag.config import PipelineConfig, resolve_path

STOP_REQUESTED = False
RUNTIME_DEPS_LOADED = False
PIPELINE_VERSION = "0.3.0"
PROMPT_VERSION = "2026-05-12"

def load_runtime_deps() -> None:
    """Import heavyweight ML/runtime dependencies on demand for faster startup."""
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

class PipelineInterrupted(Exception):
    pass

class MissingContextResponse(Exception):
    pass

class EmptyLLMResponse(Exception):
    pass

class PerformanceTracker:
    """Track request throughput and token usage across a batch run."""

    def __init__(self, report_interval_seconds: int):
        self.report_interval_seconds = max(5, int(report_interval_seconds or 30))
        self.started_at = time.time()
        self.last_report_at = self.started_at
        self.requests = 0
        self.failures = 0
        self.total_seconds = 0.0
        self.approx_output_tokens = 0
        self.recent_results = deque(maxlen=50)
        self.current_file = ""
        self.current_file_started_at = self.started_at
        self.current_file_max_total_tokens = 0
        self.current_file_max_token_label = ""
        self.run_max_total_tokens = 0
        self.run_max_token_label = ""
        self.run_max_token_file = ""

    def start_file(self, label: str) -> None:
        self.current_file = label
        self.current_file_started_at = time.time()
        self.current_file_max_total_tokens = 0
        self.current_file_max_token_label = ""

    def finish_file(self) -> None:
        elapsed = dt.timedelta(seconds=int(time.time() - self.current_file_started_at))
        print(
            "  file telemetry: "
            f"elapsed={elapsed}, "
            f"file_max_total_tokens={self.current_file_max_total_tokens or 'unknown'}"
            f"{self._format_token_label(self.current_file_max_token_label)}, "
            f"run_max_total_tokens={self.run_max_total_tokens or 'unknown'}"
            f"{self._format_token_label(self.run_max_token_label, self.run_max_token_file)}"
        )

    def _format_token_label(self, label: str, file_label: str = "") -> str:
        if not label:
            return ""
        suffix = f" at {label}"
        if file_label:
            suffix += f" in {Path(file_label).name}"
        return suffix

    def record_token_usage(self, label: str, token_usage: dict[str, int] | None) -> None:
        if not token_usage:
            return
        total_tokens = token_usage.get("total_tokens") or 0
        if not total_tokens:
            prompt_tokens = token_usage.get("prompt_tokens") or 0
            completion_tokens = token_usage.get("completion_tokens") or 0
            total_tokens = prompt_tokens + completion_tokens
        if not total_tokens:
            return
        if total_tokens > self.current_file_max_total_tokens:
            self.current_file_max_total_tokens = total_tokens
            self.current_file_max_token_label = label
        if total_tokens > self.run_max_total_tokens:
            self.run_max_total_tokens = total_tokens
            self.run_max_token_label = label
            self.run_max_token_file = self.current_file

    def record_llm_result(self, label: str, elapsed: float, text: str, token_usage: dict[str, int] | None = None) -> None:
        self.requests += 1
        self.total_seconds += elapsed
        approx_tokens = max(1, len(text or "") // 4)
        self.approx_output_tokens += approx_tokens
        self.recent_results.append((time.time(), elapsed, approx_tokens))
        self.record_token_usage(label, token_usage)
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
        req_per_min_all = self.requests / wall * 60.0
        recent = [(ts, elapsed, tokens) for ts, elapsed, tokens in self.recent_results if now - ts <= 300]
        if len(recent) >= 2:
            recent_window = max(60.0, now - recent[0][0])
            req_per_min_recent = len(recent) / recent_window * 60.0
            recent_model_seconds = max(0.001, sum(item[1] for item in recent))
            recent_tok_per_sec = sum(item[2] for item in recent) / recent_model_seconds
        else:
            req_per_min_recent = 0.0
            recent_tok_per_sec = 0.0
        approx_tok_per_sec = self.approx_output_tokens / model_seconds
        print(
            "  perf: "
            f"requests={self.requests}, failures={self.failures}, "
            f"avg_request_seconds={self.total_seconds / max(1, self.requests):.1f}, "
            f"requests_per_min_recent={req_per_min_recent:.2f}, "
            f"requests_per_min_all={req_per_min_all:.2f}, "
            f"approx_output_tokens_per_sec_recent={recent_tok_per_sec:.1f}, "
            f"approx_output_tokens_per_sec_all={approx_tok_per_sec:.1f}, "
            f"file_max_tokens={self.current_file_max_total_tokens or 'unknown'}, "
            f"run_max_tokens={self.run_max_total_tokens or 'unknown'}, "
            f"last={label}"
        )
        self.last_report_at = now

    def final_report(self) -> None:
        elapsed = dt.timedelta(seconds=int(time.time() - self.started_at))
        print(
            "\nRun telemetry: "
            f"elapsed={elapsed}, requests={self.requests}, failures={self.failures}, "
            f"run_max_total_tokens={self.run_max_total_tokens or 'unknown'}"
            f"{self._format_token_label(self.run_max_token_label, self.run_max_token_file)}"
        )

    def snapshot(self) -> dict[str, Any]:
        wall = max(0.001, time.time() - self.started_at)
        return {
            "elapsed_seconds": round(wall, 2),
            "requests": self.requests,
            "failures": self.failures,
            "avg_request_seconds": round(self.total_seconds / max(1, self.requests), 3),
            "requests_per_minute_all": round(self.requests / wall * 60.0, 3),
            "approx_output_tokens": self.approx_output_tokens,
            "approx_output_tokens_per_second_all": round(self.approx_output_tokens / max(0.001, self.total_seconds), 3),
            "current_file": self.current_file,
            "current_file_max_total_tokens": self.current_file_max_total_tokens,
            "current_file_max_token_label": self.current_file_max_token_label,
            "run_max_total_tokens": self.run_max_total_tokens,
            "run_max_token_label": self.run_max_token_label,
            "run_max_token_file": self.run_max_token_file,
        }

class RunStats:
    """Aggregate per-file outcomes for run reports and dashboards."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self.files_total = 0
        self.files_completed = 0
        self.files_failed = 0
        self.files_skipped = 0
        self.cached_files = 0
        self.llm_files = 0
        self.fallbacks = 0
        self.documents = 0
        self.position_cards = 0
        self.failures: list[dict[str, Any]] = []
        self.files: list[dict[str, Any]] = []

    def snapshot(self, performance: PerformanceTracker | None = None) -> dict[str, Any]:
        payload = {
            "started_at": dt.datetime.fromtimestamp(self.started_at, dt.timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - self.started_at, 2),
            "files_total": self.files_total,
            "files_completed": self.files_completed,
            "files_failed": self.files_failed,
            "files_skipped": self.files_skipped,
            "cached_files": self.cached_files,
            "llm_files": self.llm_files,
            "fallbacks": self.fallbacks,
            "documents": self.documents,
            "position_cards": self.position_cards,
            "failures": self.failures,
            "files": self.files,
        }
        if performance:
            payload["performance"] = performance.snapshot()
        return payload

class RuntimeControl:
    """Read and refresh live batch concurrency settings from disk."""

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

class FakeLLMResponse:
    def __init__(self, content: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.content = content
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        }

class FakeChain:
    def invoke(self, payload: dict[str, Any]) -> FakeLLMResponse:
        text = str(payload.get("text", "") or "")
        prompt_tokens = token_estimate(text)
        if text.lstrip().startswith("{") or '"node_id"' in text:
            try:
                first = json.loads(text.splitlines()[0])
                node_id = first.get("node_id", "")
                speaker = first.get("speaker") or (first.get("speakers") or ["unknown"])[0]
                episode_date = first.get("episode_date") or ""
            except Exception:
                node_id = ""
                speaker = "unknown"
                episode_date = ""
            content = {
                "positions": [
                    {
                        "claim": "Deterministic fake model position extracted for test mode.",
                        "speaker": speaker if speaker != "unknown" else "",
                        "episode_date": episode_date,
                        "stance_category": "test",
                        "confidence": "low",
                        "rationale": "Generated by no-LM Studio fake model mode.",
                        "counterpoints": "",
                        "evidence_node_ids": [node_id] if node_id else [],
                        "evidence_timestamps": [],
                        "keywords": ["test-mode"],
                    }
                ]
            }
            output = json.dumps(content, ensure_ascii=True)
        else:
            tags = deterministic_topic_tags(text, 5)
            output = "\n".join(
                f"- Fake summary {idx + 1}: {tag}"
                for idx, tag in enumerate(tags or ["transcript"])
            )
        return FakeLLMResponse(output, prompt_tokens=prompt_tokens, completion_tokens=token_estimate(output))

def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nStop requested. No new model requests will be started; waiting for in-flight request(s) to finish.")
