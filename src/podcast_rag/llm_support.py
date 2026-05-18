from __future__ import annotations

from typing import Any

import podcast_rag.runtime as runtime
from podcast_rag.config import PipelineConfig

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

def extract_token_usage(response: Any) -> dict[str, int]:
    candidates = []
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        candidates.append(usage_metadata)
    response_metadata = getattr(response, "response_metadata", {}) or {}
    if isinstance(response_metadata, dict):
        candidates.append(response_metadata.get("token_usage") or {})
        candidates.append(response_metadata.get("usage") or {})
        candidates.append(response_metadata)

    usage: dict[str, int] = {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for target, keys in aliases.items():
            if target in usage:
                continue
            for key in keys:
                value = candidate.get(key)
                if isinstance(value, int):
                    usage[target] = value
                    break
    if "total_tokens" not in usage and ("prompt_tokens" in usage or "completion_tokens" in usage):
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return usage

def verify_model_available(config: PipelineConfig) -> None:
    runtime.load_runtime_deps()
    client = runtime.OpenAI(base_url=config.lm_studio_base_url, api_key=config.lm_studio_api_key)
    models = [model.id for model in client.models.list().data]
    if config.lm_studio_model not in models:
        raise RuntimeError(f"Model not found: {config.lm_studio_model}\nAvailable: {models}")
    print(f"Model '{config.lm_studio_model}' is available.")

def test_model_inference(config: PipelineConfig) -> None:
    runtime.load_runtime_deps()
    client = runtime.OpenAI(base_url=config.lm_studio_base_url, api_key=config.lm_studio_api_key)
    client.chat.completions.create(
        model=config.lm_studio_model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
    print("Inference test passed.")
