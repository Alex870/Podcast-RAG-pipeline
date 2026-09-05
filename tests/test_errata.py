import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from podcast_rag.config import PipelineConfig
from podcast_rag.errata import (
    ErrataRecorder,
    validate_errata_payload,
    write_errata_artifacts,
)
from podcast_rag.pipeline import PodcastRagPipeline
from podcast_rag.runtime import FakeLLMResponse, PerformanceTracker
from podcast_rag import cli


class DiagnosisChain:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        return FakeLLMResponse(self.response)


def pipeline_for_diagnosis(config, chain, debug_dir):
    pipeline = PodcastRagPipeline.__new__(PodcastRagPipeline)
    pipeline.config = config
    pipeline.diagnosis_chain = chain
    pipeline.performance = PerformanceTracker(30)
    pipeline.debug_output_dir = debug_dir
    pipeline.fallback_count = 0
    pipeline.active_errata = None
    return pipeline


class ErrataTests(unittest.TestCase):
    def test_batch_boundary_writes_errata_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input" / "episode_speaker_transcript.json"
            source.parent.mkdir()
            source.write_text("transcript", encoding="utf-8")
            config = PipelineConfig(
                input_dir="input",
                processed_data_dir="processed_data",
                state_path="state/state.json",
                stop_file="state/stop.txt",
                control_file="state/control.json",
                run_snapshot_path="state/snapshot.json",
                run_report_dir="state/reports",
                errata_dir="errata",
                fake_llm=True,
                auto_refresh_topic_index=False,
            )

            class FakePerformance:
                requests = 0
                failures = 0

                def snapshot(self):
                    return {"requests": 0, "failures": 0, "run_max_total_tokens": 0}

                def final_report(self):
                    return None

            class FakePipeline:
                process_calls = 0

                def __init__(self, config, project_dir, control):
                    self.performance = FakePerformance()
                    self.fallback_count = 0
                    self.active_errata = None

                def process_file(self, path, errata=None):
                    FakePipeline.process_calls += 1
                    errata.record("test_anomaly", "warning", "test", "A test anomaly was recorded.")
                    return {"status": "completed", "source": "test", "nodes": 0, "position_cards": 0, "documents": []}

                def validate_cached_file(self, path, fingerprint, cache_path):
                    return {
                        "status": "completed",
                        "source": "processed_data_cache",
                        "nodes": 0,
                        "position_cards": 0,
                        "leaf_chunks": 0,
                        "summaries": 0,
                        "positions": 0,
                    }

                def diagnose_errata(self, recorder):
                    recorder.set_diagnosis({"status": "not_requested", "reason": "test_double"})

                def save_cached_documents(self, cache_path, source_path, fingerprint, docs):
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text("{}", encoding="utf-8")

            with patch.object(cli.runtime, "load_runtime_deps"), patch.object(cli, "PodcastRagPipeline", FakePipeline), patch.object(cli, "iter_transcript_files", return_value=[source]):
                self.assertEqual(0, cli.run_batch(config, root, one_file=True))
                self.assertEqual(0, cli.run_batch(config, root, one_file=True))

            records = list((root / "errata").glob("*.errata.json"))
            self.assertEqual(1, len(records))
            payload = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertEqual(1, FakePipeline.process_calls)
            self.assertEqual("cached_valid", payload["outcome"]["status"])
            self.assertEqual("not_requested", payload["llm_diagnosis"]["status"])
            self.assertEqual(2, len(list((root / "errata" / "history").glob("*"))))

    def test_clean_success_writes_valid_paired_artifacts_without_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Episode 01_speaker_transcript.json"
            source.write_text('{"segments": []}', encoding="utf-8")
            config = PipelineConfig()
            recorder = ErrataRecorder(source, "fingerprint-1", "run-1", config)
            recorder.set_outcome("clean")

            result = write_errata_artifacts(recorder, root / "errata")
            payload = result["payload"]

            self.assertTrue(Path(result["json_path"]).is_file())
            self.assertTrue(Path(result["markdown_path"]).is_file())
            self.assertEqual("file-errata-1.0", payload["contract_version"])
            self.assertEqual("clean", payload["outcome"]["status"])
            self.assertEqual("not_requested", payload["llm_diagnosis"]["status"])
            self.assertEqual([], validate_errata_payload(payload))

    def test_changed_anomaly_preserves_previous_pair_in_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.json"
            source.write_text("transcript", encoding="utf-8")
            config = PipelineConfig()
            first = ErrataRecorder(source, "fingerprint-1", "run-1", config)
            first.set_outcome("completed_with_warnings")
            first.record("truncated_json_recovered", "warning", "position_extraction", "Recovered objects.")
            write_errata_artifacts(first, root / "errata")

            second = ErrataRecorder(source, "fingerprint-1", "run-2", config)
            second.set_outcome("completed_with_warnings")
            second.record("unknown_evidence_ids_discarded", "warning", "position_extraction", "Discarded IDs.")
            write_errata_artifacts(second, root / "errata")

            history = list((root / "errata" / "history").glob("*"))
            self.assertEqual(2, len(history))
            history_json = next(path for path in history if path.suffix == ".json")
            self.assertEqual(str(history_json), json.loads(history_json.read_text(encoding="utf-8"))["artifacts"]["json"])
            self.assertEqual(1, len(list((root / "errata").glob("*.errata.json"))))
            self.assertEqual(1, len(list((root / "errata").glob("*.errata.md"))))

    def test_review_packet_and_excerpts_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.json"
            source.write_text("transcript", encoding="utf-8")
            config = PipelineConfig(errata_review_context_max_chars=1200, errata_excerpt_max_chars=40)
            recorder = ErrataRecorder(source, "fingerprint-1", "run-1", config)
            recorder.record(
                "large_finding",
                "warning",
                "test",
                "x" * 1000,
                details={"text": "y" * 4000},
                excerpts=["z" * 1000],
            )
            packet = recorder.review_packet()
            self.assertLessEqual(len(json.dumps(packet, separators=(",", ":"))), 1200)
            finding = recorder.findings[0]
            self.assertLessEqual(len(finding["excerpts"][0]), 40)

    def test_diagnosis_success_is_advisory_and_strictly_structured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.json"
            source.write_text("transcript", encoding="utf-8")
            config = PipelineConfig()
            chain = DiagnosisChain(json.dumps({
                "summary": "A bounded diagnosis.",
                "root_causes": [{
                    "category": "model_output",
                    "confidence": 0.9,
                    "finding_ids": ["F0001"],
                    "claim": "The model returned incomplete JSON.",
                    "explanation": "The packet records truncated output.",
                }],
                "actions": [{
                    "priority": "high",
                    "type": "code",
                    "action": "Inspect the JSON recovery boundary.",
                    "verification": "A malformed response produces a bounded finding without cache corruption.",
                }],
                "uncertainties": [],
                "human_review_questions": [],
            }))
            pipeline = pipeline_for_diagnosis(config, chain, root / "debug")
            recorder = ErrataRecorder(source, "fingerprint-1", "run-1", config)
            recorder.set_outcome("completed_with_warnings")
            recorder.record("truncated_json_recovered", "warning", "position_extraction", "Recovered objects.")

            pipeline.diagnose_errata(recorder)

            self.assertEqual("completed", recorder.diagnosis["status"])
            self.assertEqual(1, chain.calls)

    def test_malformed_diagnosis_does_not_raise_or_change_file_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.json"
            source.write_text("transcript", encoding="utf-8")
            config = PipelineConfig()
            chain = DiagnosisChain("not json")
            pipeline = pipeline_for_diagnosis(config, chain, root / "debug")
            recorder = ErrataRecorder(source, "fingerprint-1", "run-1", config)
            recorder.set_outcome("completed_with_warnings")
            recorder.record("fallback_summary", "warning", "summarization", "Fallback used.")

            pipeline.diagnose_errata(recorder)

            self.assertEqual("completed_with_warnings", recorder.outcome_status)
            self.assertEqual("failed", recorder.diagnosis["status"])
            self.assertTrue(any(item["code"] == "errata_diagnosis_failed" for item in recorder.findings))

    def test_malformed_diagnosis_shape_does_not_block_markdown_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.json"
            source.write_text("transcript", encoding="utf-8")
            recorder = ErrataRecorder(source, "fingerprint-1", "run-1", PipelineConfig())
            recorder.set_outcome("completed_with_warnings")
            recorder.diagnosis = "not an object"

            result = write_errata_artifacts(recorder, root / "errata")

            self.assertTrue(Path(result["markdown_path"]).is_file())
            self.assertIn("LLM diagnosis is advisory", Path(result["markdown_path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
