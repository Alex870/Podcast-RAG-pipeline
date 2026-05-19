import json
import shutil
import uuid
import unittest
from pathlib import Path

from podcast_rag.config import PipelineConfig
from podcast_rag.topics import refresh_topic_index


def _write_cache(path: Path, source_path: str, source_fingerprint: str, topic_label: str, claim: str) -> None:
    payload = {
        "version": 2,
        "schema_version": "2.0",
        "source_path": source_path,
        "source_fingerprint": source_fingerprint,
        "documents": [
            {
                "page_content": claim,
                "metadata": {
                    "node_id": f"position_{source_fingerprint[:8]}",
                    "node_type": "position_card",
                    "speaker": "Host",
                    "speaker_scope": "single",
                    "episode_date": "2026-05-16",
                    "episode_title": Path(source_path).stem,
                    "source": source_path,
                    "stable_document_id": f"doc_{source_fingerprint[:8]}",
                    "claim": claim,
                    "keywords": [topic_label],
                    "topic_tags": [topic_label],
                },
            },
            {
                "page_content": f"Summary about {topic_label}",
                "metadata": {
                    "node_id": f"summary_{source_fingerprint[:8]}",
                    "node_type": "cluster_summary",
                    "speaker_scope": "single",
                    "episode_date": "2026-05-16",
                    "episode_title": Path(source_path).stem,
                    "source": source_path,
                    "stable_document_id": f"sum_{source_fingerprint[:8]}",
                    "topic_tags": [topic_label],
                },
            },
            {
                "page_content": f"Episode thesis about {topic_label}",
                "metadata": {
                    "node_id": f"thesis_{source_fingerprint[:8]}",
                    "node_type": "episode_thesis",
                    "speaker_scope": "single",
                    "episode_date": "2026-05-16",
                    "episode_title": Path(source_path).stem,
                    "source": source_path,
                    "stable_document_id": f"thesis_{source_fingerprint[:8]}",
                    "topic_tags": [topic_label],
                },
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_label_file(path: Path, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"labels": labels}, indent=2), encoding="utf-8")


def _make_temp_root() -> Path:
    root = Path(__file__).resolve().parents[1] / ".test_tmp" / f"topic-tests-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class TopicIndexTests(unittest.TestCase):
    def test_refresh_topic_index_prefers_cleaned_cache_for_same_episode(self):
        root = _make_temp_root()
        try:
            processed_data = root / "processed_data"
            processed_data.mkdir()
            _write_cache(
                processed_data / "TFM_20260516_speaker_transcript.raw.processed_documents.json",
                r"D:\Pod Cast RAG\podcast-host-transcription-pipeline\output\TFM 20260516_speaker_transcript.json",
                "rawfingerprint",
                "Inflation",
                "Raw cache claim",
            )
            _write_cache(
                processed_data / "TFM_20260516_cleaned_speaker_transcript.cleaned.processed_documents.json",
                r"D:\Pod Cast RAG\podcast-host-transcription-pipeline\output\TFM 20260516_cleaned_speaker_transcript.json",
                "cleanfingerprint",
                "Ukraine war",
                "Cleaned cache claim",
            )

            config = PipelineConfig(
                processed_data_dir="processed_data",
                topic_contribution_dir="state/topic_contributions",
                topic_index_path="state/topic_index.json",
                topic_index_manifest_path="state/topic_index_manifest.json",
                podcast_id="tfm",
                podcast_name="TFM",
            )
            summary = refresh_topic_index(config, root)
            self.assertEqual(summary["episode_count"], 1)
            self.assertEqual(summary["rebuilt_contributions"], 1)

            topic_index = json.loads((root / "state" / "topic_index.json").read_text(encoding="utf-8"))
            self.assertEqual(topic_index["podcasts"][0]["podcast_id"], "tfm")
            labels = [topic["label"] for topic in topic_index["topics"]]
            self.assertIn("Ukraine War", labels)
            self.assertNotIn("Inflation", labels)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_refresh_topic_index_reuses_unchanged_contributions(self):
        root = _make_temp_root()
        try:
            processed_data = root / "processed_data"
            processed_data.mkdir()
            _write_cache(
                processed_data / "TFM_20260516_cleaned_speaker_transcript.cleaned.processed_documents.json",
                r"D:\Pod Cast RAG\podcast-host-transcription-pipeline\output\TFM 20260516_cleaned_speaker_transcript.json",
                "cleanfingerprint",
                "Federal Reserve",
                "The Fed is boxed in.",
            )
            config = PipelineConfig(
                processed_data_dir="processed_data",
                topic_contribution_dir="state/topic_contributions",
                topic_index_path="state/topic_index.json",
                topic_index_manifest_path="state/topic_index_manifest.json",
                podcast_id="tfm",
                podcast_name="TFM",
            )

            first = refresh_topic_index(config, root)
            second = refresh_topic_index(config, root)

            self.assertEqual(first["rebuilt_contributions"], 1)
            self.assertEqual(second["rebuilt_contributions"], 0)
            self.assertEqual(second["reused_contributions"], 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_refresh_topic_index_suppresses_generic_reporting_labels(self):
        root = _make_temp_root()
        try:
            processed_data = root / "processed_data"
            processed_data.mkdir()
            _write_cache(
                processed_data / "TFM_20260516_cleaned_speaker_transcript.cleaned.processed_documents.json",
                r"D:\Pod Cast RAG\podcast-host-transcription-pipeline\output\TFM 20260516_cleaned_speaker_transcript.json",
                "cleanfingerprint",
                "Advocates",
                "The host strongly supports this.",
            )
            config = PipelineConfig(
                processed_data_dir="processed_data",
                topic_contribution_dir="state/topic_contributions",
                topic_index_path="state/topic_index.json",
                topic_index_manifest_path="state/topic_index_manifest.json",
                podcast_id="tfm",
                podcast_name="TFM",
            )

            refresh_topic_index(config, root)
            topic_index = json.loads((root / "state" / "topic_index.json").read_text(encoding="utf-8"))
            self.assertEqual(topic_index["topics"], [])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_refresh_topic_index_suppresses_function_word_and_speaker_labels(self):
        root = _make_temp_root()
        try:
            processed_data = root / "processed_data"
            processed_data.mkdir()
            _write_cache(
                processed_data / "TFM_20260516_cleaned_speaker_transcript.cleaned.processed_documents.json",
                r"D:\Pod Cast RAG\podcast-host-transcription-pipeline\output\TFM 20260516_cleaned_speaker_transcript.json",
                "cleanfingerprint",
                "Speaker 02",
                "This is not a real topic.",
            )
            _write_cache(
                processed_data / "TFM_20260517_cleaned_speaker_transcript.cleaned.processed_documents.json",
                r"D:\Pod Cast RAG\podcast-host-transcription-pipeline\output\TFM 20260517_cleaned_speaker_transcript.json",
                "anotherfinger",
                "Using",
                "This is also not a real topic.",
            )
            config = PipelineConfig(
                processed_data_dir="processed_data",
                topic_contribution_dir="state/topic_contributions",
                topic_index_path="state/topic_index.json",
                topic_index_manifest_path="state/topic_index_manifest.json",
                podcast_id="tfm",
                podcast_name="TFM",
            )

            refresh_topic_index(config, root)
            topic_index = json.loads((root / "state" / "topic_index.json").read_text(encoding="utf-8"))
            report = json.loads((root / "state" / "topic_label_curation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(topic_index["topics"], [])
            filtered_labels = {item["normalized_label"] for item in report["deterministic_filtered"]}
            self.assertIn("speaker 02", filtered_labels)
            self.assertIn("using", filtered_labels)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_refresh_topic_index_applies_persistent_blacklist(self):
        root = _make_temp_root()
        try:
            processed_data = root / "processed_data"
            processed_data.mkdir()
            _write_cache(
                processed_data / "TFM_20260516_cleaned_speaker_transcript.cleaned.processed_documents.json",
                r"D:\Pod Cast RAG\podcast-host-transcription-pipeline\output\TFM 20260516_cleaned_speaker_transcript.json",
                "cleanfingerprint",
                "Low maintenance",
                "The host keeps returning to low maintenance systems.",
            )
            _write_label_file(root / "state" / "topic_label_blacklist.json", ["low maintenance"])
            config = PipelineConfig(
                processed_data_dir="processed_data",
                topic_contribution_dir="state/topic_contributions",
                topic_index_path="state/topic_index.json",
                topic_index_manifest_path="state/topic_index_manifest.json",
                topic_blacklist_path="state/topic_label_blacklist.json",
                topic_whitelist_path="state/topic_label_whitelist.json",
                podcast_id="tfm",
                podcast_name="TFM",
            )

            refresh_topic_index(config, root)
            topic_index = json.loads((root / "state" / "topic_index.json").read_text(encoding="utf-8"))
            report = json.loads((root / "state" / "topic_label_curation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(topic_index["topics"], [])
            self.assertIn("low maintenance", report["persisted_blacklist"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_refresh_topic_index_emits_topic_kind_templates_and_query_hints(self):
        root = _make_temp_root()
        try:
            processed_data = root / "processed_data"
            processed_data.mkdir()
            _write_cache(
                processed_data / "TFM_20260516_cleaned_speaker_transcript.cleaned.processed_documents.json",
                r"D:\Pod Cast RAG\podcast-host-transcription-pipeline\output\TFM 20260516_cleaned_speaker_transcript.json",
                "cleanfingerprint",
                "Federal Reserve",
                "The Fed is boxed in.",
            )
            config = PipelineConfig(
                processed_data_dir="processed_data",
                topic_contribution_dir="state/topic_contributions",
                topic_index_path="state/topic_index.json",
                topic_index_manifest_path="state/topic_index_manifest.json",
                podcast_id="tfm",
                podcast_name="TFM",
            )

            refresh_topic_index(config, root)
            topic_index = json.loads((root / "state" / "topic_index.json").read_text(encoding="utf-8"))
            topic = topic_index["topics"][0]
            self.assertEqual(topic["topic_kind"], "policy_theme")
            self.assertTrue(topic["question_templates"])
            self.assertIn("Fed policy", topic["query_hints"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
