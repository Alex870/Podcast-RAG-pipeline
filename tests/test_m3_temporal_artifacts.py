import json
import tempfile
import unittest
from pathlib import Path

from podcast_rag.temporal_artifacts import build_temporal_artifacts
from podcast_rag.config import PipelineConfig, generation_config_fingerprint


class TemporalArtifactTests(unittest.TestCase):
    def test_optional_artifact_flags_do_not_invalidate_generated_hierarchy(self):
        baseline=PipelineConfig();candidate=PipelineConfig(enable_temporal_artifacts=True,enable_temporal_trajectories=True)
        self.assertEqual(generation_config_fingerprint(baseline),generation_config_fingerprint(candidate))

    def test_claim_identity_is_evidence_bound_and_optional_outputs_are_gated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); processed = root / "processed"; processed.mkdir()
            fixture = json.loads((Path(__file__).parent / "fixtures" / "processed_cache_v2_1.json").read_text(encoding="utf-8"))
            fixture["documents"][0]["metadata"]["stable_document_id"] = "doc-1"
            fixture["documents"][0]["metadata"]["source_span_ids"] = ["span-1"]
            (processed / "one.processed_documents.json").write_text(json.dumps(fixture), encoding="utf-8")
            first = root / "first.json"; second = root / "second.json"
            build_temporal_artifacts(processed, first)
            value = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual("temporal-research-1.0", value["contract_version"])
            self.assertEqual(["span-1"], value["claims"][0]["source_span_ids"])
            self.assertFalse(value["trajectories"]); self.assertFalse(value["contradiction_candidates"])
            fixture["documents"][0]["page_content"] = "Generated wording changed."
            (processed / "one.processed_documents.json").write_text(json.dumps(fixture), encoding="utf-8")
            build_temporal_artifacts(processed, second)
            self.assertEqual(value["claims"][0]["claim_id"], json.loads(second.read_text(encoding="utf-8"))["claims"][0]["claim_id"])

    def test_trajectory_preserves_missing_interval_without_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); processed = root / "processed"; processed.mkdir()
            fixture = json.loads((Path(__file__).parent / "fixtures" / "processed_cache_v2_1.json").read_text(encoding="utf-8"))
            for index, day in enumerate(("2025-01-01", "2026-01-01")):
                item = json.loads(json.dumps(fixture)); leaf = item["documents"][0]
                leaf["metadata"].update({"stable_document_id": f"doc-{index}", "source_span_ids": [f"span-{index}"], "episode_date": day})
                (processed / f"{index}.processed_documents.json").write_text(json.dumps(item), encoding="utf-8")
            output = root / "temporal.json"
            build_temporal_artifacts(processed, output, include_trajectories=True, missing_interval_days=90)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("no_continuity_inference", value["missing_intervals"][0]["qualification"])
            self.assertEqual({"span-0", "span-1"}, set(value["trajectories"][0]["source_span_ids"]))


if __name__ == "__main__": unittest.main()
