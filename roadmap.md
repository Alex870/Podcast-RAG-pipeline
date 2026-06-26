# Roadmap

This roadmap defines how `Podcast-RAG-pipeline` should evolve to support both the existing `5070 / short-context` path and a new `5090 / high-context` path without breaking existing users.

## Compatibility Principles

- Keep the current 16 GB GPU flow as the default baseline.
- Add high-context features as opt-in capabilities, not new minimum requirements.
- Preserve backward readability for old processed caches.
- Make richer metadata additive and versioned.
- Keep deterministic fallbacks for every advanced LLM-dependent feature.

## Shared Runtime Profile Model

- Add `runtime_profile` with values:
  - `baseline_16gb`
  - `enhanced_24gb`
  - `high_context_5090`
  - `custom`
- Add `backend` with values such as `lm_studio` and `vllm`.
- Resolve each profile into concrete settings:
  - model name
  - context target
  - prompt token budget
  - max parallel requests
  - structured output support
  - judge-pass support

## Config And Schema

- Keep existing config keys working unchanged.
- Add optional config keys:
  - `runtime_profile`
  - `backend`
  - `context_window_tokens`
  - `prompt_token_budget`
  - `structured_outputs_enabled`
  - `judge_model`
  - `judge_pass_enabled`
  - `high_context_mode`
- Extend processed cache metadata with:
  - `runtime_profile`
  - `backend`
  - `model_name`
  - `model_capabilities`
  - `structured_output_used`
  - `judge_pass_used`
- Version processed cache manifests so old caches still validate and load.

## High-Value Architecture Changes

- Keep the current short-context reduction path as the baseline mode.
- Add a high-context path that:
  - increases prompt budgets
  - reduces lossy rollup pressure
  - preserves more leaf evidence
  - keeps speaker/date metadata in scope longer
- Add structured JSON extraction for position cards when supported by the backend.
- Keep permissive JSON recovery as fallback when structured outputs are unavailable.
- Add a second-pass judge flow using a stronger reasoning model only for uncertain or contradictory outputs.

## Quality Improvements

- Prefer larger direct evidence windows over aggressive intermediate summarization.
- Add confidence and provenance flags:
  - model-generated
  - deterministic fallback
  - judge-reviewed
- Improve temporal synthesis:
  - preserve episode date metadata at every level
  - keep speaker attribution explicit through summary layers
- Add contradiction and ambiguity tagging for position cards.

## Performance Strategy

- Do not globally raise defaults for prompt budgets or batches.
- Enable larger batches and context only through `high_context_5090` or `custom`.
- Add vLLM-aware options:
  - structured outputs
  - prefix-caching-friendly prompt layouts
  - chunked prefill-compatible batching

## Testing

- Add profile-based tests:
  - `baseline_16gb`
  - `high_context_5090`
- Add regression fixtures for:
  - malformed JSON
  - missing-context responses
  - contradictory evidence
  - judge-pass escalation
- Add snapshot tests for manifest metadata across profile modes.

## Implementation Phases

1. Add `runtime_profile`, `backend`, and additive manifest/schema fields.
2. Build a profile resolver that preserves current defaults when unset.
3. Add high-context prompt budgeting and larger-batch planning behind profile gates.
4. Add structured-output extraction with fallback to current parsing.
5. Add optional judge-pass review for low-confidence position cards.
6. Add profile-aware telemetry and manifests for downstream compatibility.
7. Add baseline-vs-high-context regression tests and evaluation fixtures.
