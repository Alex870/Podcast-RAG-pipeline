import unittest
from pathlib import Path
from podcast_rag.upstream_contracts import parse_correction_fixture
class UpstreamTests(unittest.TestCase):
 def test_parses_vendored_production_fixture_without_transcription_runtime(self):
  p=Path(__file__).parent/"fixtures/contracts/transcription/correction-manifest-v1/valid.json"; self.assertTrue(parse_correction_fixture(p)["correction_set_id"].startswith("correction_"))
if __name__=="__main__": unittest.main()
