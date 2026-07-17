import hashlib
import json
import unittest
from pathlib import Path

import podcast_rag.runtime as runtime
from podcast_rag.transcript import load_transcript_json


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


if __name__ == "__main__":
    unittest.main()
