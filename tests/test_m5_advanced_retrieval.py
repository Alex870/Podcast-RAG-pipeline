import json
import tempfile
import unittest
from pathlib import Path
from podcast_rag.advanced_retrieval import (
    AdvancedRetrievalError,
    build_evidence_graph,
    build_late_chunk_alignment,
    validate_entry_gate,
)


def gate():
    return {
        "contract_version": "advanced-retrieval-entry-gate-1.0",
        "failing_query_slices": ["cross_episode"],
        "failure_examples": ["q1"],
        "simpler_methods_attempted": ["hybrid", "rerank"],
        "hypothesis": "linked evidence improves recall",
        "expected_gain": {"recall@20": 0.05},
        "budgets": {"latency_ms": 500, "storage_bytes": 100000},
        "accepted_baseline": "run_base",
        "minimum_reviewed_coverage": 30,
        "stop_condition": "no recall gain",
        "model_revision": "abc",
        "tokenizer_id": "tok",
    }


class AdvancedRetrievalTests(unittest.TestCase):
    def test_gate_rejects_incomplete(self):
        with self.assertRaisesRegex(AdvancedRetrievalError, "incomplete"):
            validate_entry_gate(
                {"contract_version": "advanced-retrieval-entry-gate-1.0"}
            )

    def test_graph_and_alignment_are_release_bound_and_exact(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as raw:
            root = Path(raw) / "processed"
            root.mkdir()
            payload = {
                "source_fingerprint": "source1",
                "documents": [
                    {
                        "page_content": "one two three four",
                        "metadata": {
                            "stable_document_id": "d1",
                            "node_type": "leaf_chunk",
                            "speaker": "A",
                            "episode_id": "E1",
                            "topic_tags": ["x"],
                        },
                    },
                    {
                        "page_content": "summary",
                        "metadata": {
                            "stable_document_id": "d2",
                            "node_type": "summary",
                            "parent_id": "d1",
                        },
                    },
                ],
            }
            (root / "a.processed_documents.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            temporal = Path(raw) / "temporal.json"
            temporal.write_text(
                json.dumps(
                    {
                        "artifact_id": "t1",
                        "claims": [
                            {
                                "claim_id": "c1",
                                "document_id": "d1",
                                "source_span_ids": ["s1"],
                            }
                        ],
                        "temporal_adjacency": [],
                    }
                ),
                encoding="utf-8",
            )
            graph = build_evidence_graph(
                root,
                temporal,
                Path(raw) / "graph.json",
                corpus_release_id="r1",
                entry_gate=gate(),
            )
            self.assertTrue(
                {"claim_evidence", "parent_child", "topic_document"}.issubset(
                    {e["edge_type"] for e in graph["edges"]}
                )
            )
            alignment = build_late_chunk_alignment(
                root,
                Path(raw) / "late.json",
                corpus_release_id="r1",
                entry_gate=gate(),
                model_id="m",
                model_revision="rev",
                tokenizer_id="tok",
                window_tokens=3,
                overlap_tokens=1,
            )
            row = next(
                item for item in alignment["documents"] if item["document_id"] == "d1"
            )
            self.assertEqual({"token": "one", "start": 0, "end": 3}, row["tokens"][0])
            self.assertEqual(18, row["windows"][-1]["char_end"])
            self.assertFalse(alignment["vectors_included"])


if __name__ == "__main__":
    unittest.main()
