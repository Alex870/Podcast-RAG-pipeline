# Podcast Pipeline Data Contract

This document describes the JSON and Chroma metadata contract shared by the four podcast tools:

1. `Podcast-Host-Transcription-Pipeline`
2. `Podcast-RAG-pipeline`
3. `Chroma DB Import`
4. `PodCast Chat`

The goal is to make every handoff explicit enough that a downstream tool can validate inputs before starting expensive transcription, LLM preprocessing, embedding, or chat work.

## Transcript JSON

Produced by `Podcast-Host-Transcription-Pipeline` and consumed by `Podcast-RAG-pipeline`.

Expected top-level fields:

- `source_file`: original audio filename or path.
- `episode_title`: human-readable title when known.
- `metadata`: optional object containing episode-level metadata.
- `segments`: ordered transcript segment array.

Expected segment fields:

- `start`: segment start time in seconds.
- `end`: segment end time in seconds.
- `speaker`: normalized speaker label.
- `text`: transcript text.

Recommended episode date fields:

- `episode_date`: ISO date, for example `2026-02-04`.
- `episode_date_compact`: compact date, for example `20260204`.
- `episode_sort_key`: numeric `YYYYMMDD` value.

## Processed RAG Cache

Produced by `Podcast-RAG-pipeline` and consumed by `Chroma DB Import`.

Each `*.processed_documents.json` file represents one episode. The expected shape is:

```json
{
  "schema_version": "2.1",
  "source_path": "episode_speaker_transcript.json",
  "source_fingerprint": "stable fingerprint",
  "source_transcript_hash": "sha256 of source transcript bytes",
  "representations": {
    "display_text": "page-content-v1",
    "dense_text": "page-content-v1",
    "lexical_text": "normalized-lexical-v1"
  },
  "documents": []
}
```

Each document must contain:

- `page_content`: authoritative display and citation text; also the dense-embedding fallback.
- `metadata`: Chroma-ready metadata object.

Schema `2.1` adds optional retrieval-specific text without changing the display or citation source:

- `embedding_text`: text selected for dense embedding. Importers fall back to `page_content` when absent.
- `lexical_text`: text selected for BM25 or sparse indexing. Importers fall back to `page_content` when absent.

The cache-level `representations` manifest records the method version for `display_text`, `dense_text`, and `lexical_text`. Schema `2.0` caches without this manifest remain valid and backward-readable.

`page_content` remains authoritative for display and citation. Stable document IDs continue to derive from `page_content`, so adding optional retrieval representations does not change evidence identity.

Schema `2.1` provenance requirements:

- Every leaf has `source_segment_ids`, `source_spans`, or both `start_time` and `end_time` identifying its transcript evidence.
- Every summary, thesis, and position card has non-empty `child_ids`; the validator follows those links until it reaches leaf evidence.
- `parent_id` is the structural hierarchy link. Position-card `child_ids` are evidence references and may point to nodes that also belong to the hierarchy.
- Each serialized node carries `source_node_fingerprint` and `representation_fingerprints` for deterministic audit and delta backfill.

The representation manifest also contains `builder_version` and `config_fingerprint`. A representation fingerprint includes the source cache fingerprint, source-node identity/evidence, builder version, relevant configuration, and the generated representation text. It is separate from `stable_document_id`.

Required document metadata:

- `node_id`: stable unique ID for the document.
- `node_type`: one of `leaf_chunk`, `cluster_summary`, `episode_thesis`, or `position_card`.
- `episode_id`: stable episode identifier.
- `episode_title`: human-readable episode title.
- `episode_date`: ISO episode date when known.
- `episode_sort_key`: numeric date key when known.

Speaker metadata:

- `speaker`: primary speaker for single-speaker nodes.
- `speakers`: JSON array or list of speaker names for multi-speaker nodes.
- `speaker_scope`: `single`, `multiple`, `mixed`, or empty.

Embedding metadata:

- `embedding_model`: recommended on each cache or metadata manifest when available.
- `embedding_dimension`: recommended after vectors are generated.

Importers must record whether they embedded `page_content` or `embedding_text`. Databases built from different representation versions should be treated as distinct retrieval experiments even when they use the same embedding model.

Current pipeline defaults:

- `display_text`: `page-content-v1`
- `dense_text`: `page-content-v1`
- `lexical_text`: `normalized-lexical-v1`

Set `embedding_text_mode` to `context-header-v1` only for an explicitly named experiment. The contextual header is deterministic and bounded by `contextual_header_max_chars`. Changing dense representation requires a separate database export and evaluation; it must not be mixed into an existing embedding space.

Downstream compatibility note: schema `2.1` makes these representations available, but the importer and query client must explicitly implement representation selection and lexical indexing before they affect production retrieval.

Existing caches can be upgraded without LLM work:

```powershell
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json --backfill-representations
python .\podcast_rag_pipeline.py --config .\podcast_rag_config.json --export-dense-baseline
```

The backfill reuses valid hierarchy and position nodes, rebuilds only deterministic metadata/representations, validates evidence closure, and writes a `delta` record. The export contains stable document IDs, `embedding_text`, `page_content`, metadata, and cache manifests for downstream evaluation.

## Chroma Export

Produced by `Chroma DB Import` and consumed by `PodCast Chat`.

Each export is a self-contained folder:

```text
Podcast Name/
  chroma.sqlite3
  podcast.json
  ...Chroma internal files...
```

`podcast.json` expected fields:

- `podcast_name`
- `database_id`
- `collection_name`
- `embedding_model`
- `embedding_dimension`
- `embedding_device`
- `description`
- `date_range.start`
- `date_range.end`
- `episode_count`
- `chunk_count`
- `speakers`
- `episodes`
- `generated_at`
- `generated_by`

Speaker entries:

```json
{
  "id": "speaker-slug",
  "name": "Speaker Name"
}
```

Episode entries:

- `source_file`
- `source_fingerprint`
- `episode_id`
- `episode_title`
- `episode_date`
- `document_count`
- `speakers`
- `imported_at`

## Compatibility Rules

- `Podcast-RAG-pipeline`, `Chroma DB Import`, and `PodCast Chat` must agree on the embedding model.
- The importer and query client must agree on the dense representation version as well as the embedding model.
- If an export was embedded with one model and queried with another, retrieval distances are unreliable.
- `episode_sort_key` should be preserved from transcript through Chroma metadata so date filtering works.
- Omitted speakers should be omitted from export metadata as if they were not imported.
- Episode-level thesis nodes and multi-speaker summary nodes may be preserved even when speaker-specific nodes are filtered out.

## Validation Expectations

Every stage should fail early with clear messages when required fields are missing:

- RAG should validate transcript segment shape before LLM preprocessing.
- Chroma import should validate processed documents before embedding.
- Podcast Chat should validate export metadata, Chroma collection availability, embedding model compatibility, and vector availability for speakers.
