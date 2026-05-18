from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

from podcast_rag.schema import PROCESSED_CACHE_SCHEMA_VERSION

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
    lm_studio_model: str = "mistral-small-3.2-24b-instruct-2506"
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
    episode_thesis_reduce_with_llm: bool = False
    episode_thesis_max_chars: int = 3200
    position_extraction_batch_char_budget: int = 8000
    position_passage_max_chars: int = 900
    context_window_tokens: int = 4096
    prompt_token_budget: int = 3500
    prompt_token_chars_per_token: float = 4.0
    resume_within_file: bool = True
    checkpoint_dir: str = "state/file_checkpoints"
    run_report_dir: str = "state/run_reports"
    run_snapshot_path: str = "state/current_run_snapshot.json"
    topic_contribution_dir: str = "state/topic_contributions"
    topic_index_path: str = "state/topic_index.json"
    topic_index_manifest_path: str = "state/topic_index_manifest.json"
    auto_refresh_topic_index: bool = True
    podcast_id: str = ""
    podcast_name: str = ""
    cache_schema_version: str = PROCESSED_CACHE_SCHEMA_VERSION
    clustering_reduction: str = "pca"
    grouping_mode: str = "semantic"
    summary_objective: str = "speaker_belief"
    enable_llm_topic_tags: bool = False
    deterministic_topic_count: int = 8
    near_duplicate_threshold: float = 0.92
    position_quote_excerpt_chars: int = 360
    fake_llm: bool = False
    model_eval_output_dir: str = "model_eval"

def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path

def load_config(config_path: Path) -> PipelineConfig:
    """Load JSON config, rejecting malformed files with actionable location info."""
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
    """Apply environment-level overrides for model and endpoint settings."""
    config.embedding_model = os.getenv("EMBEDDING_MODEL", config.embedding_model)
    config.lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", config.lm_studio_base_url)
    config.lm_studio_api_key = os.getenv("LM_STUDIO_API_KEY", config.lm_studio_api_key)
    config.lm_studio_model = os.getenv("LM_STUDIO_MODEL", config.lm_studio_model)
    return config

def config_fingerprint(config: PipelineConfig) -> str:
    """Create a stable hash of effective config values for cache provenance."""
    payload = json.dumps({field.name: getattr(config, field.name) for field in fields(config)}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
