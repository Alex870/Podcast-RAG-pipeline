import json
import tempfile
import unittest
from pathlib import Path

from podcast_rag.evaluation.dataset import QueryRecord, load_query_set
from podcast_rag.evaluation.metrics import score_query
from podcast_rag.evaluation.runner import evaluate_retrieval_run


class EvaluationTests(unittest.TestCase):
    def test_query_validation_rejects_bad_grade(self):
        with self.assertRaises(ValueError):
            QueryRecord.from_dict({"query_id": "q", "query": "Q?", "category": "fact", "relevance": {"d": 4}})

    def test_metrics_use_graded_relevance_and_rank(self):
        query = QueryRecord(
            query_id="q1",
            query="What did the speaker say?",
            category="speaker_position",
            relevance={"doc-a": 3, "doc-b": 1},
            expected_speakers=("TFM",),
        )
        results = [
            {"document_id": "noise", "metadata": {"speaker": "Guest", "node_type": "leaf_chunk"}},
            {"document_id": "doc-a", "metadata": {"speaker": "TFM", "node_type": "position_card"}},
            {"document_id": "doc-b", "metadata": {"speaker": "TFM", "node_type": "leaf_chunk"}},
        ]
        metrics = score_query(query, results)
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["mrr@10"], 0.5)
        self.assertGreater(metrics["ndcg@10"], 0.5)
        self.assertEqual(metrics["evidence_coverage@10"], 1.0)

    def test_runner_writes_reproducible_reports_and_skips_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path = root / "queries.jsonl"
            results_path = root / "results.json"
            query_path.write_text(
                json.dumps({"query_id": "q1", "query": "Question", "category": "fact", "relevance": {"d1": 3}})
                + "\n"
                + json.dumps({"query_id": "draft", "query": "Draft", "category": "fact", "status": "draft", "relevance": {}})
                + "\n",
                encoding="utf-8",
            )
            results_path.write_text(
                json.dumps({"run_id": "run-1", "strategy_id": "dense", "queries": [{"query_id": "q1", "results": [{"document_id": "d1"}]}]}),
                encoding="utf-8",
            )
            report = evaluate_retrieval_run(query_path, results_path, root / "reports")
            self.assertEqual(report["judged_query_count"], 1)
            self.assertEqual(report["skipped_query_count"], 1)
            self.assertEqual(report["aggregate"]["recall@5"], 1.0)
            self.assertTrue(Path(report["json_report_path"]).exists())
            self.assertTrue(Path(report["markdown_report_path"]).exists())
            self.assertEqual(len(load_query_set(query_path)), 2)


if __name__ == "__main__":
    unittest.main()
