# Podcast-RAG Transcription Handoff Contract

**Status:** normative redesign contract  
**Contract ID:** `podcast-rag-transcription-handoff-v1`  
**Producer:** `Podcast-Host-Transcription-Pipeline`  
**Consumer:** `Podcast-RAG-pipeline`  
**Downstream:** Chroma DB Import, PodCast Chat, and RAGScope

This document is the contract Podcast-RAG should be redesigned around. It defines the boundary between transcription and retrieval preprocessing: what the producer guarantees, what the consumer validates, what the consumer may recompute, and which identity must survive into a corpus release.

The existing [`podcast_pipeline_contract.md`](podcast_pipeline_contract.md) remains the ecosystem overview. This document is the detailed, consumer-facing contract. If the two documents disagree, this document is authoritative for the Podcast-RAG handoff after the migration described in §18.

## 1. Contract principles

1. **A handoff is a declared package, not an accidental directory scan.** The producer supplies a manifest that names the exact episode artifacts to consume.
2. **Partition identity is part of data identity.** A transcript without a partition identity is legacy input and must not silently join a managed corpus.
3. **The producer owns transcript truth.** Podcast-RAG reads producer artifacts and never edits, relocates, or deletes them.
4. **The consumer owns derived knowledge artifacts.** Chunks, summaries, position cards, representations, processed caches, and release manifests are consumer outputs.
5. **Every expensive operation is fingerprinted.** An unchanged episode and unchanged effective configuration must be reusable without LLM work, embedding work, or hierarchy reconstruction.
6. **Corrections are scoped deltas.** A correction invalidates the affected episode and its dependent derived nodes, not an unrelated corpus.
7. **Mixed corpora fail closed.** Combining processing spaces requires an explicit future aggregation contract; it is never inferred from a common parent folder.
8. **Portable artifacts contain no secrets or machine-specific absolute paths.** Local paths may exist in private run state, but not in a handoff or release intended to move between machines.

## 2. Transport and package layout

The normative transport is a filesystem package. An implementation may later expose the same contract through an API or object store, but the JSON meaning and integrity rules remain unchanged.

```text
handoff-root/
  manifest.json
  episodes/
    <episode_id>/
      transcript.json
      reviewed.json                 # optional sibling, if selected by manifest
      corrected.json                # optional sibling, if selected by manifest
  corrections/
    <correction_set_id>.json        # optional
  provenance/
    <episode_id>.json               # optional detailed producer diagnostics
```

`manifest.json` is the package entry point. Every artifact referenced by the manifest is addressed by a path relative to `handoff-root`, uses `/` separators, and is covered by a SHA-256 hash. Paths must not escape the package (`..`), be absolute, or refer to a symlink outside the package.

The package may be generated in a staging directory and atomically promoted by renaming it into the intake location. Podcast-RAG must treat a package as incomplete until `manifest.json` and every required artifact are present and hash-valid.

### 2.1 Legacy compatibility transport

During migration, Podcast-RAG continues to accept the current path-based layout containing files such as:

```text
*_speaker_transcript.json
*_cleaned_speaker_transcript.json
*_reviewed_speaker_transcript.json
*_corrected_human_speaker_transcript.json
```

The compatibility adapter must:

- synthesize an in-memory handoff manifest;
- preserve the current filename and source-file behavior;
- use producer `episode-contract-v2` when present;
- derive a synthetic scope identity such as `legacy:<root-fingerprint>` when no `partition_id` exists;
- emit a warning that the input is legacy and cannot be joined with a managed partition silently;
- never infer a managed partition from a parent directory name;
- write no new portable artifact claiming managed partition identity until the operator adopts or exports the source through the producer.

Legacy mode is for migration and read-only recovery. New production runs should require a manifest package or an explicit partition-bound adapter invocation.

## 3. Handoff manifest

The manifest must be a UTF-8 JSON object with this top-level shape:

```json
{
  "contract_version": "podcast-rag-transcription-handoff-v1",
  "handoff_id": "handoff_20260905T180000Z_01J...",
  "created_at": "2026-09-05T18:00:00Z",
  "producer": {
    "name": "podcast-host-transcription-pipeline",
    "version": "2026.09.05",
    "commit": "abc1234",
    "contract_version": "episode-contract-v2"
  },
  "partition": {
    "partition_id": "partition_podcast",
    "corpus_id": "partition_podcast",
    "display_name": "Podcast",
    "slug": "podcast",
    "context_type": "podcast",
    "workflow_profile": "podcast",
    "config_fingerprint": "sha256:..."
  },
  "selection_policy": {
    "preferred_transcript_order": [
      "corrected_human",
      "reviewed_llm",
      "cleaned",
      "raw"
    ],
    "include_failed": false,
    "require_stable_sources": true
  },
  "episodes": [],
  "correction_sets": [],
  "integrity": {
    "algorithm": "sha256",
    "manifest_hash_excludes_field": "integrity.manifest_sha256",
    "manifest_sha256": "sha256:..."
  }
}
```

### 3.1 Required manifest fields

| Field | Requirement | Meaning |
|---|---|---|
| `contract_version` | required | Exact handoff contract version. Unsupported major versions fail before processing. |
| `handoff_id` | required | Unique immutable ID for this package publication. |
| `created_at` | required | UTC ISO-8601 creation time. |
| `producer.name` | required | Must identify the transcription producer. |
| `producer.contract_version` | required | Must be `episode-contract-v2` or a supported legacy adapter. |
| `partition.partition_id` | required for managed mode | Stable internal processing-space ID. |
| `partition.corpus_id` | required for managed mode | Downstream corpus identity; defaults to `partition_id`. |
| `partition.context_type` | required | `podcast`, `meeting`, or `custom`. |
| `partition.workflow_profile` | required | Producer workflow profile, such as `podcast` or `anonymous_meeting`. |
| `episodes` | required | Non-empty array for a processable handoff. |
| `integrity` | required | Hash algorithm and manifest hash rules. |

`partition.display_name`, `slug`, and `config_fingerprint` are required for a producer-created managed package but are descriptive rather than identity-bearing. Display names may change; `partition_id` and `corpus_id` may not be reused for a different context.

### 3.2 Episode entry

Each `episodes[]` entry names exactly one selected transcript artifact:

```json
{
  "episode_id": "podcast-2026-01-03",
  "episode_uid": "partition_podcast:podcast-2026-01-03",
  "episode_title": "Podcast 20260103",
  "episode_date": "2026-01-03",
  "episode_sort_key": 20260103,
  "source_audio": {
    "name": "Podcast 20260103.mp3",
    "fingerprint": "sha256:audio-or-producer-fingerprint"
  },
  "selected_transcript": {
    "variant": "reviewed_llm",
    "path": "episodes/podcast-2026-01-03/reviewed.json",
    "artifact_sha256": "sha256:...",
    "canonical_payload_sha256": "sha256:...",
    "schema_version": "2",
    "contract_version": "episode-contract-v2"
  },
  "available_variants": ["raw", "cleaned", "reviewed_llm"],
  "review_status": "reviewed_llm",
  "correction_set_id": null,
  "speaker_label_set": ["HOST", "GUEST_01"],
  "stable": true,
  "stage_provenance": {
    "transcription": "completed",
    "alignment": "completed",
    "diarization": "completed",
    "speaker_matching": "completed",
    "review": "completed"
  }
}
```

Required episode fields are `episode_id`, `episode_uid`, `selected_transcript`, `source_audio.fingerprint`, `stable`, and the partition identity inherited from the manifest. `episode_title` is required when known; date fields are optional when genuinely unavailable, but a missing date must be represented as `null`, not guessed from an arbitrary file timestamp.

`episode_uid` is the globally safe identity: `partition_id:episode_id`. It prevents two processing spaces containing identically named recordings from colliding.

## 4. Transcript artifact contract

The selected transcript must be a JSON object conforming to `episode-contract-v2`. The artifact may retain producer-specific additive fields, but these fields are the stable minimum Podcast-RAG consumes:

```json
{
  "contract_version": "episode-contract-v2",
  "schema_version": 2,
  "pipeline": "podcast-host-transcription-pipeline",
  "pipeline_version": "2026.09.05",
  "episode_id": "podcast-2026-01-03",
  "episode_uid": "partition_podcast:podcast-2026-01-03",
  "source_file": "Podcast 20260103.mp3",
  "source_fingerprint": "sha256:...",
  "partition_id": "partition_podcast",
  "corpus_id": "partition_podcast",
  "text_version": "reviewed_llm",
  "metadata": {
    "episode_title": "Podcast 20260103",
    "episode_date": "2026-01-03",
    "episode_sort_key": 20260103,
    "partition_id": "partition_podcast",
    "corpus_id": "partition_podcast"
  },
  "segments": [
    {
      "source_span_id": "123",
      "start": 12.34,
      "end": 18.91,
      "speaker": "HOST",
      "text": "Transcript text.",
      "original_text": "Original cleaned text.",
      "llm_reviewed_text": "Transcript text."
    }
  ],
  "speech_provenance": [
    {
      "provider": "faster-whisper",
      "model_revision": "immutable-revision",
      "task": "transcribe"
    }
  ]
}
```

### 4.1 Transcript validation rules

Podcast-RAG must reject the artifact before any LLM or embedding work if any of these rules fail:

- the JSON root is an object and `segments` is a list;
- `contract_version` is `episode-contract-v2` or a supported adapter version;
- `episode_id` is non-empty and agrees with the manifest entry;
- `partition_id` and `corpus_id`, when present, agree with the manifest;
- every segment has a non-empty `source_span_id`, finite numeric `start` and `end`, `start <= end`, and non-empty `text` unless explicitly marked non-speech;
- segment source-span IDs are unique within the episode;
- segment order is non-decreasing by start time;
- no segment has an unsupported correction relationship or a `supersedes_source_span_id` without `correction_set_id`;
- `speech_provenance` is a list of records containing at least `provider` and immutable `model_revision`;
- the selected artifact is stable and its byte hash matches the manifest;
- the semantic hash used by a correction set matches the selected transcript.

The consumer may preserve producer fields such as transcription confidence, content-quality hints, review metadata, and speaker evidence. It must not discard `source_span_id`, timing, speaker, partition, correction, or provenance fields while normalizing the transcript.

## 5. Variant selection and correction semantics

The producer may publish multiple siblings. The manifest is authoritative about which one is selected. If the manifest has no explicit selection, the compatibility adapter uses this order:

```text
corrected_human > reviewed_llm > cleaned > raw
```

`reviewed_llm` means an LLM review result; it is not a claim of human approval. `corrected_human` means the selected transcript includes approved human corrections or is accompanied by an approved correction set.

### 5.1 Correction set contract

Podcast-RAG accepts `correction-manifest-v1` and `correction-manifest-v2`, normalizing both in memory to the following meaning:

```json
{
  "contract_version": "correction-manifest-v2",
  "correction_set_id": "correction_<sha256-of-identity>",
  "partition_id": "partition_podcast",
  "affected_episode_ids": ["podcast-2026-01-03"],
  "source_transcript_hash": "sha256:<canonical-transcript-payload>",
  "corrections": [
    {
      "correction_id": "corr_01",
      "source_span_id": "123",
      "field": "text",
      "before_value_guard": "Original cleaned text.",
      "after_value": "Corrected text.",
      "status": "approved",
      "kind": "text",
      "supersedes_correction_id": null
    }
  ]
}
```

Only `approved` and `accepted` corrections are applied. A correction is valid only when:

- its `partition_id` agrees with the handoff;
- its episode is in `affected_episode_ids`;
- `source_span_id` exists in the selected transcript;
- `field` is supported;
- `before_value_guard` matches the selected transcript value;
- `source_transcript_hash` matches the canonical selected transcript payload;
- superseded corrections are excluded from the active set.

The byte hash of the JSON artifact and the canonical payload hash have different purposes. `artifact_sha256` detects package/file corruption. `source_transcript_hash` detects semantic staleness and is computed over canonical JSON with sorted keys and compact separators, matching the existing correction-manifest implementation.

A failed guard is a stale correction, not permission to apply a fuzzy edit. The episode becomes `quarantined` or `needs_attention`, and unrelated episodes remain eligible.

## 6. Partition and corpus isolation

For managed input, all episodes in one handoff must have exactly one `partition_id` and one `corpus_id`. The following are hard failures:

- more than one partition ID in a package or scan result;
- more than one corpus ID in a package;
- a transcript partition ID that conflicts with the manifest;
- a correction set from another partition;
- a release or processed-cache input that mixes managed and legacy identities without an explicit migration operation.

The normal identity mapping is:

```text
partition_id == corpus_id == downstream database namespace
```

A different corpus ID is allowed only when the partition registry explicitly declares that downstream mapping and the handoff records it. The consumer must carry these fields through every processed cache, export manifest, diagnostic record, and release manifest:

```text
partition_id
corpus_id
partition_display_name
context_type
workflow_profile
partition_config_fingerprint
```

Speaker names, glossaries, corrections, review history, checkpoints, and source-span identity are partition-local by default. A future shared profile must be an explicit, versioned dependency; it must not be created by matching names across partitions.

## 7. Consumer processing stages

The redesigned consumer should expose these stages and persist their status per episode:

1. **Discover** — locate a manifest or invoke the explicit legacy adapter.
2. **Validate package** — validate paths, hashes, schema versions, partition scope, stability, and duplicate identities.
3. **Select transcript** — resolve the manifest-selected variant and correction set.
4. **Normalize transcript** — create an in-memory canonical segment model while retaining source evidence and producer metadata.
5. **Apply corrections** — verify guards and create a correction-scoped canonical view; do not modify producer files.
6. **Build leaf evidence** — chunk transcript text and attach exact source spans, times, speakers, and episode identity.
7. **Build derived knowledge** — generate hierarchy summaries, episode thesis, and position cards only when their dependency fingerprints are stale.
8. **Build representations** — generate display, dense, and lexical text according to the selected representation profile.
9. **Validate cache** — enforce document metadata, evidence closure, stable identity, and representation fingerprints.
10. **Publish** — atomically write the processed cache, run manifest, and optional corpus-release handoff.

Stages 1–5 are cheap validation/normalization work. Stages 6–8 may use an LLM. Stage 9 must finish before Chroma import is allowed.

## 8. Incremental reuse and invalidation

The consumer must calculate a per-episode processing key from:

```text
partition_id
episode_id
selected transcript artifact_sha256
selected transcript canonical_payload_sha256
source_audio fingerprint
correction_set_id
consumer pipeline version
prompt/version manifest
consumer config fingerprint
representation config fingerprint
generation config fingerprint
```

Cache directories and machine-local paths are excluded from configuration fingerprints. A cache is reusable only when the required keys and the cache schema are compatible and the cache passes validation.

| Change | Reuse allowed | Required recomputation |
|---|---|---|
| No input/config/version change | Entire validated processed cache | None |
| New package publication with identical selected artifact hashes | Entire validated processed cache | Manifest verification only |
| Transcript text/timing/speaker change in one episode | Other episodes | Affected episode from leaf evidence onward |
| Approved correction in one source span | Unaffected episodes and unrelated corpus state | Affected leaf, dependent summaries/thesis/position cards, representations, and vector records |
| Correction set identity changes with no active correction difference | Validated episode cache if normalized active set is identical | Revalidation only |
| Prompt or generation configuration changes | Unaffected episodes | LLM-derived nodes whose dependency includes the changed setting |
| Dense representation or embedding model changes | Display/lexical artifacts where fingerprints remain valid | Dense text selection and a separate vector export/index |
| Lexical representation changes only | Display/dense artifacts | Lexical representation and sparse index |
| Consumer code/schema migration with compatible output | Validated artifacts | Metadata/backfill only, with no LLM work |
| Partition configuration changes | Other partitions | Only episodes whose producer handoff fingerprint says they are affected |

The consumer must report whether it reused, backfilled, or recomputed each stage. “Processed” is not sufficient evidence of reuse.

## 9. Processed cache contract

The current processed cache schema `2.1` remains the migration target. A valid cache is an object with at least:

```json
{
  "schema_version": "2.1",
  "pipeline_version": "...",
  "prompt_version": "...",
  "partition_id": "partition_podcast",
  "corpus_id": "partition_podcast",
  "episode_id": "podcast-2026-01-03",
  "episode_uid": "partition_podcast:podcast-2026-01-03",
  "source_path": "episodes/podcast-2026-01-03/reviewed.json",
  "source_fingerprint": "sha256:...",
  "source_transcript_hash": "sha256:...",
  "config_fingerprint": "sha256:...",
  "generation_config_fingerprint": "sha256:...",
  "correction_set_id": null,
  "representations": {
    "builder_version": "...",
    "config_fingerprint": "sha256:...",
    "display_text": "page-content-v1",
    "dense_text": "page-content-v1",
    "lexical_text": "normalized-lexical-v1"
  },
  "documents": []
}
```

Each document must contain `page_content` and `metadata`. Required metadata remains:

```text
node_id
node_type
level
source
episode_id
episode_title
source_type
speaker_scope
partition_id
corpus_id
```

The following are required for evidence-bearing schema `2.1` output:

- leaf nodes have `source_segment_ids`, `source_spans`, or exact `start_time` and `end_time`;
- summaries, theses, and position cards have non-empty `child_ids` that close to leaf evidence;
- every node has `source_node_fingerprint`;
- every generated representation has a `representation_fingerprint`;
- position cards have one attributable speaker and a claim;
- no node ID is duplicated and no evidence graph cycle is present.

`page_content` is authoritative display/citation text. `embedding_text` and `lexical_text` are optional representations selected by explicit importer configuration. A vector index must record the representation version and embedding model used; vectors from different representation or embedding spaces must not be mixed.

### 9.1 Identity rules

- `episode_id` is stable within a partition.
- `episode_uid` is stable across partitions and is `partition_id:episode_id`.
- `node_id` is the stable logical identity of a generated node within an episode.
- `stable_document_id` is the current cache’s content-addressed Chroma record identity. When content changes, a replacement record may receive a new `stable_document_id`; the cache delta must identify the old record to remove and the new record to add.
- A node’s evidence links, source-span IDs, and partition identity must never be silently reassigned to a different episode.

## 10. Run and status contract

Podcast-RAG should maintain a run/episode state record separate from the producer artifacts:

```json
{
  "run_id": "ragrun_...",
  "handoff_id": "handoff_...",
  "partition_id": "partition_podcast",
  "corpus_id": "partition_podcast",
  "episode_id": "podcast-2026-01-03",
  "status": "completed",
  "current_stage": "publish",
  "input_fingerprints": {},
  "effective_config_fingerprint": "sha256:...",
  "output_cache_path": "processed/podcast-2026-01-03.processed_documents.json",
  "output_cache_sha256": "sha256:...",
  "started_at": "2026-09-05T18:00:02Z",
  "completed_at": "2026-09-05T18:12:40Z",
  "stage_results": {},
  "error": null,
  "attempt": 1
}
```

Allowed episode states are:

```text
discovered → validated → ready → processing → completed
                                      ├──────→ stale
                                      ├──────→ failed
                                      └──────→ quarantined
```

`failed` means a retry may be attempted after the error is addressed. `quarantined` means the input or correction is unsafe to process automatically and requires an explicit operator action. A failed or quarantined episode must not make the whole partition appear completed.

One partition may have at most one active processing run unless a future scheduler explicitly implements episode-level concurrency with the same lock semantics. A run must use atomic temporary files and be restartable without corrupting an existing valid cache.

## 11. Consumer command/API boundary

The exact CLI names may vary, but the redesigned consumer must provide equivalent operations:

```text
podcast-rag handoff validate --manifest <path>
podcast-rag handoff scan --manifest <path>
podcast-rag process --manifest <path> [--episode <episode-id>]
podcast-rag status --partition <partition-id>
podcast-rag export --partition <partition-id> --release <release-id>
```

Required behavior:

- validation is available without an LLM or embedding backend;
- scan reports discovered, completed, stale, failed, and quarantined episodes;
- process accepts an explicit handoff or an explicit managed partition, never an ambiguous parent directory;
- `--episode` limits work to one declared episode;
- mixed-partition override is available only as a visibly unsafe, explicit migration/aggregation operation and is not used by normal processing;
- status exposes the selected input hash, cache key, stage reuse decisions, and last error.

## 12. Error behavior

The consumer must fail before expensive work for these conditions:

| Condition | Required result |
|---|---|
| Unsupported handoff major version | `unsupported_contract`, no processing |
| Missing or invalid manifest | `invalid_manifest`, no processing |
| Missing artifact or hash mismatch | `artifact_integrity_failure`, package quarantined |
| Path traversal, absolute package path, or external symlink | `unsafe_artifact_path`, package quarantined |
| Mixed partition or corpus identity | `mixed_partition_scope`, hard failure |
| Duplicate episode UID or source-span ID | `duplicate_identity`, episode/package failure |
| Unstable source | `source_not_stable`, leave in `discovered`/`ready` as appropriate |
| Stale correction guard/hash | `stale_correction`, episode quarantined |
| Invalid transcript segment/provenance | `invalid_transcript`, episode quarantined |
| Invalid processed cache or broken evidence closure | `invalid_processed_cache`, do not publish |
| LLM/schema failure | `generation_failed`, retain diagnostics and allow retry |

Errors must include the partition, episode, stage, and safe relative artifact path. They must not print tokens, credentials, or full prompts containing private transcript text unless diagnostic retention was explicitly enabled.

## 13. Release handoff to Chroma and RAGScope

Podcast-RAG’s release manifest must preserve the handoff identity and the exact processing choices used to produce the release:

```json
{
  "release_contract_version": "podcast-rag-corpus-release-v1",
  "release_id": "release_partition_podcast_20260905_01",
  "partition_id": "partition_podcast",
  "corpus_id": "partition_podcast",
  "handoff_ids": ["handoff_..."],
  "episode_uids": ["partition_podcast:podcast-2026-01-03"],
  "cache_schema_version": "2.1",
  "representation_profile": "baseline-v1",
  "embedding_model": "BAAI/bge-large-en-v1.5",
  "embedding_dimension": 1024,
  "processed_cache_fingerprints": ["sha256:..."],
  "created_at": "2026-09-05T18:15:00Z"
}
```

Chroma DB Import must reject a release or processed-cache input with mixed partition identity unless an explicit aggregation operation is selected. PodCast Chat and RAGScope must display and filter by `partition_id`/`corpus_id`. Evaluation packs and judgments must bind to `release_id` and `corpus_id`, not merely to a directory name.

## 14. Security and privacy

Never include the following in a portable handoff or release manifest:

- Hugging Face tokens, API keys, passwords, or environment values;
- absolute paths such as `D:\...` or `C:\...`;
- machine usernames unless deliberately part of a non-portable audit record;
- raw audio bytes;
- unrestricted prompt/debug dumps.

Relative artifact paths, filenames, model names, immutable model revisions, hashes, stage timings, and safe producer version metadata are allowed. Private local run state may retain absolute paths in a clearly non-portable diagnostics area.

## 15. Versioning and evolution

- Additive optional fields do not require a major handoff-version change.
- Removing or changing the meaning of a required field requires a new major contract version.
- A new major version must ship with a validator, migration/adapter rules, fixtures, and a compatibility statement.
- Cache schema, representation profile, embedding model, and handoff contract are separate version axes and must be recorded separately.
- A consumer may read older supported handoffs, but it must write only the current contract and cache version.
- No silent downgrade from a managed package to legacy directory scanning is allowed.

## 16. Acceptance test suite

The redesigned consumer is ready only when these tests pass:

1. A valid one-episode managed package validates without an LLM or embedding backend.
2. Two partitions containing the same filename produce different `episode_uid`, cache, corpus, and release identities.
3. A package with mixed partition IDs fails before leaf construction or LLM work.
4. A missing or modified artifact fails hash validation before processing.
5. A reviewed transcript is selected over cleaned/raw according to the manifest; an approved corrected transcript is selected over reviewed output.
6. A stale correction guard quarantines only the affected episode and leaves unrelated episodes reusable.
7. Re-running an unchanged package reuses the validated processed cache and performs zero LLM calls.
8. Changing one episode’s transcript recomputes only that episode and dependent outputs.
9. Changing dense representation or embedding configuration produces a separate vector-space export and does not mutate the display/citation text.
10. Every schema `2.1` cache passes required metadata and evidence-closure validation.
11. A legacy folder can be read through the compatibility adapter, receives a warning, and cannot silently join a managed corpus.
12. The package rejects path traversal, external symlinks, duplicate IDs, invalid time ranges, and unsupported contract majors.
13. Producer files remain byte-for-byte unchanged after validation, processing, failure, retry, and release export.
14. The release manifest, Chroma metadata, Chat database identity, and RAGScope provenance all retain the same `partition_id` and `corpus_id`.
15. Interrupted publication leaves either the previous valid cache or the new complete cache, never a partially written authoritative cache.

## 17. Migration sequence

1. Keep the current transcript and correction readers and add the manifest validator as a separate boundary.
2. Make the producer export `manifest.json` from the selected processing space without moving source audio or existing outputs.
3. Make Podcast-RAG validate a package and internally normalize both package and legacy inputs to one handoff model.
4. Add partition/corpus fields to all processed-cache and release writes; keep schema `2.1` readable during the transition.
5. Move cache reuse decisions to the per-episode processing key and report stage reuse explicitly.
6. Make normal CLI/UI processing package- or partition-explicit; retain legacy scanning only behind the compatibility adapter.
7. Require Chroma Import, PodCast Chat, and RAGScope to consume release identity rather than infer corpus identity from folders.
8. After one complete two-partition smoke test, deprecate unscoped parent-directory scanning and document the removal date.

## 18. Current implementation mapping

The current repository already provides much of the intended behavior:

- `src/podcast_rag/upstream_contracts.py` validates `episode-contract-v2` and correction manifests v1/v2;
- `src/podcast_rag/transcript.py` extracts partition identity and rejects mixed scopes;
- `src/podcast_rag/schema.py` validates processed-cache schema `2.1`, provenance, and evidence closure;
- `src/podcast_rag/state.py` records config, prompt, generation, and representation fingerprints;
- the CLI has an explicit mixed-partition override for migration scenarios.

The remaining redesign work is to make the handoff manifest the normal input boundary, bind the run state to `handoff_id` and `episode_uid`, make the invalidation matrix operational rather than implicit, and expose the stage reuse decisions through status/reporting. The legacy path adapter should remain until existing corpora have been exported and verified as managed packages.


