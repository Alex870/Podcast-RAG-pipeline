import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import podcast_rag.runtime as runtime
from podcast_rag.transcript import load_transcript_json, partition_scope


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class ContractFixtureTests(unittest.TestCase):
    def test_transcription_fixture_checksum_and_parser(self):
        runtime.RUNTIME_DEPS_LOADED = True
        runtime.Document = Document
        root = Path(__file__).parent / "fixtures" / "contracts" / "transcription" / "2"
        origin = json.loads((root / "origin.json").read_text(encoding="utf-8"))
        fixture = root / "transcript.json"
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), origin["files"][fixture.name])
        docs = load_transcript_json(fixture)
        self.assertEqual(2, len(docs))
        self.assertEqual("Host", docs[0].metadata["speaker"])
        self.assertEqual("2026-01-01", docs[0].metadata["episode_date"])

    def test_partition_metadata_is_propagated_to_documents(self):
        runtime.RUNTIME_DEPS_LOADED = True
        runtime.Document = Document
        with TemporaryDirectory() as directory:
            fixture = Path(directory) / "partition-transcript.json"
            fixture.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "partition": {
                                "partition_id": "podcast",
                                "partition_display_name": "Podcast",
                                "context_type": "podcast",
                                "workflow_profile": "podcast",
                            }
                        },
                        "segments": [{"text": "Partition-aware evidence.", "speaker": "Host"}],
                    }
                ),
                encoding="utf-8",
            )
            docs = load_transcript_json(fixture)
            self.assertEqual("podcast", docs[0].metadata["partition_id"])
            self.assertEqual("podcast", docs[0].metadata["corpus_id"])
            self.assertEqual("Podcast", docs[0].metadata["partition_display_name"])

    def test_partition_scope_rejects_mixed_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for partition_id in ("podcast", "meetings"):
                path = root / f"{partition_id}.json"
                path.write_text(
                    json.dumps(
                        {
                            "partition_id": partition_id,
                            "segments": [{"text": "Evidence", "speaker": "Host"}],
                        }
                    ),
                    encoding="utf-8",
                )
                files.append(path)

            scope = partition_scope(files)

            self.assertFalse(scope["valid"])
            self.assertEqual(["meetings", "podcast"], scope["partition_ids"])


if __name__ == "__main__":
    unittest.main()
