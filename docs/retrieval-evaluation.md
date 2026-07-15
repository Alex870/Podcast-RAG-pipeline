# Retrieval Evaluation Guide

## Purpose

The retrieval evaluator scores ranked document IDs produced by another component against a versioned, human-reviewed query set. It deliberately does not open ChromaDB, call LM Studio, or generate answers. This keeps the measurements reproducible and lets the same query judgments compare dense, lexical, hybrid, reranked, hierarchy-aware, and future retrieval strategies.

## Workflow

1. Author real questions in `evaluation/query_sets/podcast-baseline-v1.jsonl`.
2. Use stable document IDs from processed caches or an imported database to grade relevant evidence.
3. Mark reviewed records as `judged`.
4. Run the same queries through a retriever such as RAGScope or PodCast Chat.
5. Export ranked results using the captured-results format.
6. Run the evaluator.
7. Compare the resulting JSON or Markdown reports.

The checked-in query set is a draft template. Its placeholder questions and empty relevance maps are not ground truth and are skipped by the evaluator.

## Query-Set Format

Query sets use JSON Lines, with one JSON object per line:

```json
{
  "query_id": "tfm-001",
  "query": "What did TFM say about multipolarity?",
  "category": "speaker_position",
  "expected_speakers": ["TFM"],
  "date_range": null,
  "relevance": {
    "stable-document-id-a": 3,
    "stable-document-id-b": 2
  },
  "acceptable_node_types": ["leaf_chunk", "position_card"],
  "answerable": true,
  "status": "judged",
  "notes": "Reviewed against direct transcript evidence."
}
```

Required fields:

- `query_id`: unique stable identifier.
- `query`: the exact test query.
- `category`: comparison category such as `direct_fact`, `speaker_position`, or `temporal_evolution`.

Judgment fields:

- `relevance`: map of stable document IDs to grades.
- `status`: `draft` or `judged`.
- `expected_speakers`: optional speaker constraint.
- `date_range`: optional object with `start` and/or `end` ISO dates.
- `acceptable_node_types`: documentation of suitable evidence forms.
- `answerable`: whether direct corpus evidence is expected to exist.

Relevance grades:

- `3`: directly answers the question with attributable evidence.
- `2`: strongly supports part of the answer.
- `1`: useful background.
- `0`: explicitly reviewed and not relevant.

A record is included in ranking metrics only when its status is `judged` and its relevance map is non-empty. Unanswerable questions require downstream abstention evaluation rather than ordinary retrieval-recall scoring.

## Captured Results Format

Captured results may be supplied as the JSON object shown below or as JSONL with one query-result object per line. Each query result may include `latency_ms` (or `latency_seconds`), `abstained`, and `abstention_reason` for operational and unanswerable-query diagnostics.

The retriever exports one ranked result list per query:

```json
{
  "run_id": "tfm-dense-baseline-20260711",
  "strategy_id": "dense-only",
  "manifest": {
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "representation": "page-content-v1",
    "database_id": "tfm-baseline"
  },
  "queries": [
    {
      "query_id": "tfm-001",
      "results": [
        {
          "document_id": "stable-document-id-a",
          "score": 0.81,
          "metadata": {
            "node_type": "leaf_chunk",
            "speaker": "TFM",
            "episode_date": "2025-04-12"
          }
        }
      ]
    }
  ]
}
```

`document_id` may also be supplied as `id` or `stable_document_id`. Include speaker, node type, and episode date metadata when constraint diagnostics are desired.

The manifest should identify every factor needed to reproduce the run, including database/export identity, embedding model and dimension, representation version, retrieval strategy, candidate counts, filters, fusion settings, and reranker when applicable.

## Running the Evaluator

```powershell
python .\podcast_rag_pipeline.py `
  --config .\podcast_rag_config.json `
  --retrieval-eval `
  --retrieval-results .\evaluation\runs\tfm-dense-baseline.json `
  --query-set .\evaluation\query_sets\podcast-baseline-v1.jsonl
```

Optional output override:

```powershell
python .\podcast_rag_pipeline.py `
  --retrieval-eval `
  --retrieval-results .\evaluation\runs\tfm-dense-baseline.json `
  --evaluation-output-dir .\evaluation\results\tfm
```

Default paths come from:

- `retrieval_evaluation_query_set`
- `retrieval_evaluation_output_dir`

## Metrics

- `recall@5`, `recall@10`, and `recall@20`: fraction of all positively graded documents retrieved by each cutoff.
- `precision@5`, `precision@10`, and `precision@20`: positively graded results divided by the cutoff.
- `mrr@10`: reciprocal rank of the first positively graded result.
- `ndcg@10`: graded ranking quality, rewarding highly relevant documents near the top.
- `evidence_coverage@10`: fraction of total positive relevance weight represented in the top ten.
- `speaker_constraint_precision@10`: fraction of returned top-ten documents matching an expected speaker when specified.
- `date_constraint_accuracy@10`: fraction satisfying the requested date range when specified.
- `node_constraint_accuracy@10`: fraction matching `acceptable_node_types` when specified.
- `node_type_counts@10`: diagnostic composition of leaf chunks, summaries, theses, position cards, and unknown types.
- `node_type_outcomes@10`: separable returned/relevant counts and recall for each node type.
- `source_diversity@10`, `duplicate_rate@10`, `latency_ms`, and `result_count`: operational retrieval diagnostics.

Aggregate metrics are arithmetic means over judged queries. Reports also preserve per-query metrics and returned IDs for failure analysis.

## Comparing Runs

Use the same query-set file and hash for every candidate run. Compare overall and category-level behavior rather than promoting a strategy from one aggregate number.

At minimum, examine:

- Exact-name and quotation recall.
- Speaker-position precision.
- Temporal and multi-passage coverage.
- Leaf evidence versus summary-node contribution.
- Retrieval latency and index size, recorded by the producing system.
- Queries that improved and queries that regressed.

Do not combine results from different embedding spaces in one dense collection. A new embedding model or dense representation requires a separate export and a separately identified evaluation run.

## Current Integration Boundary

`Podcast-RAG-pipeline` now emits:

- `page_content` for display, citation, and fallback retrieval.
- `embedding_text` for a selected dense representation.
- `lexical_text` for downstream BM25 or sparse indexing.
- A cache-level representation manifest.

The pipeline does not execute hybrid retrieval. Chroma DB Import must explicitly select `embedding_text` and create a lexical index, and PodCast Chat or RAGScope must capture ranked result runs before these representations affect user-facing retrieval.
