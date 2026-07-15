import json
import tempfile
import unittest
from pathlib import Path

from podcast_rag.text_utils import deterministic_topic_tags, fallback_summary_from_text, token_estimate, token_set_similarity
from podcast_rag.schema import validate_processed_cache, validate_processed_documents
from podcast_rag.config import PipelineConfig
from podcast_rag.state import backfill_cache_payload, export_dense_baseline


class SchemaAndHelperTests(unittest.TestCase):
    def test_schema_2_0_and_2_1_fixtures_are_both_valid(self):
        fixture_dir = Path(__file__).parent / "fixtures"
        for name in ("processed_cache_v2_0.json", "processed_cache_v2_1.json"):
            payload = json.loads((fixture_dir / name).read_text(encoding="utf-8"))
            result = validate_processed_cache(payload)
            self.assertTrue(result.valid, result.errors)

    def test_schema_2_1_requires_representation_manifest(self):
        fixture_path = Path(__file__).parent / "fixtures" / "processed_cache_v2_1.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        payload.pop("representations")
        result = validate_processed_cache(payload)
        self.assertFalse(result.valid)
        self.assertTrue(any("representations manifest" in error for error in result.errors))

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

    def test_schema_2_1_requires_leaf_evidence_and_closure(self):
        fixture_path = Path(__file__).parent / "fixtures" / "processed_cache_v2_1.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        payload["documents"][0]["metadata"].pop("source_segment_ids")
        result = validate_processed_cache(payload)
        self.assertFalse(result.valid)
        self.assertTrue(any("source segment IDs" in error for error in result.errors))

    def test_deterministic_backfill_reuses_documents_and_adds_fingerprints(self):
        fixture_path = Path(__file__).parent / "fixtures" / "processed_cache_v2_1.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        updated = backfill_cache_payload(payload, PipelineConfig())
        updated_again = backfill_cache_payload(updated, PipelineConfig())
        self.assertEqual(updated["schema_version"], "2.1")
        self.assertEqual(updated["delta"]["llm_stages_reused"], ["hierarchy", "positions"])
        self.assertEqual(
            updated["documents"][0]["metadata"]["stable_document_id"],
            updated_again["documents"][0]["metadata"]["stable_document_id"],
        )
        self.assertTrue(updated["documents"][0]["metadata"]["representation_fingerprints"]["dense_text"])

    def test_page_content_dense_baseline_export_is_downstream_ready(self):
        fixture_path = Path(__file__).parent / "fixtures" / "processed_cache_v2_1.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "processed_data"
            cache_dir.mkdir()
            cache_path = cache_dir / "episode.processed_documents.json"
            cache_path.write_text(json.dumps(backfill_cache_payload(payload, PipelineConfig())), encoding="utf-8")
            result = export_dense_baseline(cache_dir, root / "baseline.json")
            self.assertEqual(result["document_count"], 2)
            exported = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
            self.assertEqual(exported["representation_id"], "page-content-v1")
            self.assertTrue(exported["documents"][0]["document_id"])


if __name__ == "__main__":
    unittest.main()

