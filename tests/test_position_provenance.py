import unittest

from podcast_rag.pipeline import PodcastRagPipeline
from podcast_rag.schema import validate_processed_documents


def node(node_id, node_type, child_ids=None, parent_id=None):
    metadata = {
        "node_id": node_id,
        "node_type": node_type,
        "level": "leaf" if node_type == "leaf_chunk" else node_type,
        "source": "episode.json",
        "episode_id": "episode-1",
        "episode_title": "Episode",
        "episode_date": "2026-01-01",
        "source_type": "json_transcript",
        "speaker_scope": "single",
        "parent_id": parent_id,
        "child_ids": list(child_ids or []),
    }
    if node_type == "leaf_chunk":
        metadata["source_segment_ids"] = [f"episode-1:segment:{node_id}"]
    if node_type == "position_card":
        metadata["speaker"] = "Speaker A"
        metadata["claim"] = "A durable position."
    return {"page_content": node_id, "metadata": metadata}


class StubDocument:
    def __init__(self, node_type, child_ids):
        self.metadata = {"node_type": node_type, "child_ids": child_ids}


class PositionProvenanceTests(unittest.TestCase):
    def test_position_card_may_cite_ancestor_and_descendant(self):
        docs = [
            node("leaf_1", "leaf_chunk", parent_id="summary_child"),
            node("summary_child", "cluster_summary", ["leaf_1"], "summary_parent"),
            node("summary_parent", "cluster_summary", ["summary_child"], "thesis_1"),
            node("thesis_1", "episode_thesis", ["summary_parent"]),
            node(
                "position_1",
                "position_card",
                ["summary_parent", "summary_child"],
                "thesis_1",
            ),
        ]

        result = validate_processed_documents(docs, require_provenance=True)

        self.assertTrue(result.valid, result.errors)

    def test_duplicate_position_refs_are_sanitized_and_valid(self):
        position = StubDocument(
            "position_card", ["summary_a", "summary_a", "unknown"]
        )

        sanitized = PodcastRagPipeline.sanitize_position_documents(
            [position], {"summary_a"}
        )

        self.assertEqual(["summary_a"], sanitized[0].metadata["child_ids"])

        docs = [
            node("leaf_1", "leaf_chunk", parent_id="summary_a"),
            node("summary_a", "cluster_summary", ["leaf_1"], "thesis_1"),
            node("thesis_1", "episode_thesis", ["summary_a"]),
            node(
                "position_1",
                "position_card",
                ["summary_a", "summary_a"],
                "thesis_1",
            ),
        ]
        result = validate_processed_documents(docs, require_provenance=True)

        self.assertTrue(result.valid, result.errors)

    def test_repeated_evidence_branches_to_one_leaf_are_not_a_cycle(self):
        docs = [
            node("leaf_1", "leaf_chunk", parent_id="thesis_1"),
            node("thesis_1", "episode_thesis", ["leaf_1", "leaf_1"]),
        ]

        result = validate_processed_documents(docs, require_provenance=True)

        self.assertTrue(result.valid, result.errors)

    def test_real_summary_back_edge_is_rejected(self):
        docs = [
            node("summary_a", "cluster_summary", ["summary_b"], "thesis_1"),
            node("summary_b", "cluster_summary", ["summary_a"], "summary_a"),
            node("thesis_1", "episode_thesis", ["summary_a"]),
            node("leaf_1", "leaf_chunk", parent_id="summary_b"),
        ]

        result = validate_processed_documents(docs, require_provenance=True)

        self.assertFalse(result.valid)
        self.assertTrue(
            any("evidence graph contains a cycle" in error for error in result.errors),
            result.errors,
        )
        cycle_issue = next(issue for issue in result.issues if issue["code"] == "evidence_graph_cycle")
        self.assertEqual(["summary_a", "summary_b", "summary_a"], cycle_issue["evidence_paths"][0])

    def test_cached_position_with_unresolved_reference_remains_invalid(self):
        docs = [
            node("leaf_1", "leaf_chunk", parent_id="summary_1"),
            node("summary_1", "cluster_summary", ["leaf_1"], "thesis_1"),
            node("thesis_1", "episode_thesis", ["summary_1"]),
            node("position_1", "position_card", ["summary_1", "missing"], "thesis_1"),
        ]

        result = validate_processed_documents(docs, require_provenance=True)

        self.assertFalse(result.valid)
        self.assertTrue(
            any("references missing child_id missing" in error for error in result.errors),
            result.errors,
        )

    def test_checkpointed_position_without_valid_evidence_is_dropped(self):
        valid = StubDocument(
            "position_card", ["summary_a", "missing", "summary_a"]
        )
        orphaned = StubDocument("position_card", ["missing"])
        summary = StubDocument("cluster_summary", ["leaf_a"])

        sanitized = PodcastRagPipeline.sanitize_position_documents(
            [valid, orphaned, summary], {"summary_a", "leaf_a"}
        )

        self.assertEqual([valid, summary], sanitized)
        self.assertEqual(["summary_a"], valid.metadata["child_ids"])


if __name__ == "__main__":
    unittest.main()
