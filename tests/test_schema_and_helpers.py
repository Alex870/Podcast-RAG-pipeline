import unittest

from podcast_rag.text_utils import deterministic_topic_tags, fallback_summary_from_text, token_estimate, token_set_similarity
from podcast_rag.schema import validate_processed_documents


class SchemaAndHelperTests(unittest.TestCase):
    def test_processed_cache_validator_rejects_missing_child_links(self):
        docs = [
            {
                "page_content": "Speaker A said a thing.",
                "metadata": {
                    "node_id": "leaf_1",
                    "node_type": "leaf_chunk",
                    "level": "leaf",
                    "source": "episode.json",
                    "episode_id": "ep",
                    "episode_title": "Episode",
                    "source_type": "json_transcript",
                    "speaker_scope": "single",
                },
            },
            {
                "page_content": "Episode thesis.",
                "metadata": {
                    "node_id": "thesis_1",
                    "node_type": "episode_thesis",
                    "level": "episode",
                    "source": "episode.json",
                    "episode_id": "ep",
                    "episode_title": "Episode",
                    "source_type": "json_transcript",
                    "speaker_scope": "single",
                    "child_ids": ["missing"],
                },
            },
        ]
        result = validate_processed_documents(docs)
        self.assertFalse(result.valid)
        self.assertTrue(any("missing child_id" in error for error in result.errors))

    def test_fallback_summary_is_bulleted_and_preserves_context(self):
        text = "[node_id=leaf_1 | speakers=Alex | time=00:01-00:05]\nAlex argues that policy changed over time. More detail follows."
        summary = fallback_summary_from_text(text, "cluster summary")
        self.assertIn("- Fallback summary", summary)
        self.assertIn("speaker=Alex", summary)

    def test_topic_tags_are_deterministic(self):
        tags = deterministic_topic_tags("Ukraine war policy Ukraine sanctions host policy", 3)
        self.assertEqual(tags[0], "ukraine")
        self.assertIn("policy", tags)

    def test_token_estimate(self):
        self.assertEqual(token_estimate("abcd", 4.0), 2)

    def test_token_set_similarity_detects_near_duplicates(self):
        score = token_set_similarity("host supports ukraine sanctions policy", "ukraine sanctions policy supported by host")
        self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()

