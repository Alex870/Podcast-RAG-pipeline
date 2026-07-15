import unittest

from podcast_rag.representations import RepresentationBuilder
from podcast_rag.representations.contextual import build_contextual_header
from podcast_rag.representations.lexical import build_lexical_text
from podcast_rag.schema import serialize_document


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "node_id": "leaf_1",
            "node_type": "leaf_chunk",
            "level": "leaf",
            "episode_title": "The Multipolar Episode",
            "episode_date": "2026-01-02",
            "speaker": "TFM",
            "topic_tags": ["multipolarity", "institutions"],
        }

    def test_context_header_is_deterministic_and_bounded(self):
        first = build_contextual_header(self.metadata, max_chars=90)
        second = build_contextual_header(dict(reversed(list(self.metadata.items()))), max_chars=90)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 90)
        self.assertTrue(first.startswith("Episode:"))

    def test_lexical_text_preserves_exact_terms(self):
        text = build_lexical_text("TFM mentions BRICS.", self.metadata)
        self.assertIn("BRICS", text)
        self.assertIn("TFM", text)
        self.assertIn("2026-01-02", text)

    def test_builder_defaults_keep_dense_text_equal_to_page_content(self):
        dense, lexical = RepresentationBuilder().build("Original text", self.metadata)
        self.assertEqual(dense, "Original text")
        self.assertIn("Original text", lexical)

    def test_optional_representations_do_not_change_stable_id(self):
        doc = FakeDocument("Stable source text", self.metadata)
        plain = serialize_document(doc, "source")
        enriched = serialize_document(doc, "source", "Context\nStable source text", "Stable source text TFM")
        self.assertEqual(plain["metadata"]["stable_document_id"], enriched["metadata"]["stable_document_id"])

    def test_representation_fingerprints_are_source_and_builder_specific(self):
        builder = RepresentationBuilder()
        first = builder.fingerprints("source-a", self.metadata, "Stable source text")
        second = builder.fingerprints("source-a", self.metadata, "Stable source text")
        changed_source = builder.fingerprints("source-b", self.metadata, "Stable source text")
        changed_mode = RepresentationBuilder(embedding_text_mode="context-header-v1").fingerprints(
            "source-a", self.metadata, "Stable source text"
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first["dense_text"], changed_source["dense_text"])
        self.assertNotEqual(first["dense_text"], changed_mode["dense_text"])


if __name__ == "__main__":
    unittest.main()
