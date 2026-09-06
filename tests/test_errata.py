import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from podcast_rag.config import PipelineConfig
from podcast_rag.errata import (
    ErrataRecorder,
    validate_diagnosis_payload,
    validate_errata_payload,
    write_errata_artifacts,
)
from podcast_rag.pipeline import PodcastRagPipeline
from podcast_rag.runtime import FakeLLMResponse, PerformanceTracker, RuntimeControl
from podcast_rag import cli
import podcast_rag.pipeline as pipeline_module


class DiagnosisChain:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        return FakeLLMResponse(self.response)


class SequenceChain:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return FakeLLMResponse(self.responses[index])


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
    def test_summary_prompt_treats_episode_dates_as_historical_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PipelineConfig(fake_llm=True)

            class FakeEmbeddings:
                def __init__(self, **kwargs):
                    return None

            class FakeSplitter:
                def __init__(self, **kwargs):
                    return None

            class FakePromptTemplate:
                @classmethod
                def from_messages(cls, messages):
                    return messages

            with patch.object(pipeline_module, "_refresh_runtime_symbols"), \
                patch.object(pipeline_module, "HuggingFaceEmbeddings", FakeEmbeddings), \
                patch.object(pipeline_module, "RecursiveCharacterTextSplitter", FakeSplitter), \
                patch.object(pipeline_module, "ChatPromptTemplate", FakePromptTemplate):
                pipeline = PodcastRagPipeline(config, root, RuntimeControl(config, root))

            prompt = pipeline.prompt_manifest["summary_system"]
            self.assertIn("historical source metadata", prompt)
            self.assertIn("never refuse or defer summarization", prompt)

    def test_missing_context_retry_adds_date_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PipelineConfig(fake_llm=True)
            chain = SequenceChain([
                "Please provide the transcript before summarizing.",
                "- The supplied historical episode material is summarized here.",
            ])
            pipeline = PodcastRagPipeline.__new__(PodcastRagPipeline)
            pipeline.config = config
            pipeline.performance = PerformanceTracker(30)
            pipeline.debug_output_dir = root / "debug"
            pipeline.active_errata = None
            pipeline.fallback_count = 0
            pipeline._file_fallback_count = 0

            with patch("podcast_rag.text_utils.time.sleep"):
                result = pipeline.invoke_llm(chain, "[episode_date=2026-04-25] " + "source material " * 10, "cluster summary L1")

            self.assertIn("supplied historical episode material", result)
            self.assertEqual(2, len(chain.calls))
            self.assertIn("CORRECTION", chain.calls[1]["text"])
            self.assertIn("historical metadata", chain.calls[1]["text"])
            self.assertEqual(0, pipeline.fallback_count)

    def test_persistent_missing_context_stops_after_one_corrective_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PipelineConfig(fake_llm=True)
            chain = SequenceChain(["Please provide the transcript before summarizing."])
            pipeline = PodcastRagPipeline.__new__(PodcastRagPipeline)
            pipeline.config = config
            pipeline.performance = PerformanceTracker(30)
            pipeline.debug_output_dir = root / "debug"
            pipeline.active_errata = None
            pipeline.fallback_count = 0
            pipeline._file_fallback_count = 0

            with patch("podcast_rag.text_utils.time.sleep"):
                result = pipeline.invoke_llm(chain, "[episode_date=2026-04-25] " + "source material " * 10, "cluster summary L1")

            self.assertTrue(result.lstrip().startswith("- Fallback"))
            self.assertEqual(2, len(chain.calls))
            self.assertEqual(1, pipeline.fallback_count)

    def test_fallback_metadata_is_scoped_to_each_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_source = root / "first_speaker_transcript.json"
            second_source = root / "second_speaker_transcript.json"
            for source in (first_source, second_source):
                source.write_text('{"segments": []}', encoding="utf-8")

            class FakePerformance:
                def start_file(self, source):
                    return None

                def maybe_report(self, label, force=False):
                    return None

                def finish_file(self):
                    return None

                def snapshot(self):
                    return {"requests": 0, "failures": 0, "run_max_total_tokens": 0}

            def metadata(node_id, node_type, level):
                return {
                    "node_id": node_id,
                    "node_type": node_type,
                    "level": level,
                    "source": "test",
                    "episode_id": "episode",
                    "episode_title": "Test episode",
                    "episode_date": "2026-01-01",
                    "source_type": "json_transcript",
                    "speaker_scope": "mixed",
                    "child_ids": [],
                }

            leaf = types.SimpleNamespace(
                page_content="A leaf transcript chunk.",
                metadata=metadata("leaf", "leaf_chunk", 0),
            )
            summary = types.SimpleNamespace(
                page_content="A cluster summary.",
                metadata=metadata("summary", "cluster_summary", 1),
            )
            thesis = types.SimpleNamespace(
                page_content="An episode thesis.",
                metadata=metadata("thesis", "episode_thesis", 2),
            )
            input_doc = types.SimpleNamespace(page_content="Transcript text.", metadata={})
            pipeline = PodcastRagPipeline.__new__(PodcastRagPipeline)
            pipeline.config = PipelineConfig(resume_within_file=False)
            pipeline.performance = FakePerformance()
            pipeline.fallback_count = 0
            pipeline._file_fallback_count = 0
            pipeline.cluster_telemetry = []
            pipeline.prompt_manifest = {}
            pipeline.active_errata = None
            pipeline.active_context = None
            pipeline.load_file_checkpoint = lambda *args: None
            pipeline.save_file_checkpoint = lambda *args: None
            pipeline.clear_file_checkpoints = lambda *args: None
            pipeline.build_leaf_chunks = lambda docs, source: [leaf]
            pipeline.extract_positions = lambda docs, thesis_doc: []
            pipeline.validate_documents_before_cache = lambda docs, label: None
            process_calls = 0

            def build_hierarchy(docs, source):
                nonlocal process_calls
                process_calls += 1
                if process_calls == 1:
                    pipeline._record_fallback()
                return [leaf, summary, thesis], thesis

            pipeline.build_hierarchy = build_hierarchy

            with patch("podcast_rag.pipeline.load_transcript_json", return_value=[input_doc]):
                first_result = pipeline._process_file(first_source)
                first_cache = root / "first.cache.json"
                pipeline.save_cached_documents(first_cache, first_source, "first-fingerprint", first_result["documents"])

                second_result = pipeline._process_file(second_source)
                second_cache = root / "second.cache.json"
                pipeline.save_cached_documents(second_cache, second_source, "second-fingerprint", second_result["documents"])

            self.assertEqual(1, first_result["fallbacks"])
            self.assertEqual(0, second_result["fallbacks"])
            self.assertEqual(1, pipeline.fallback_count)
            self.assertEqual(1, json.loads(first_cache.read_text(encoding="utf-8"))["fallback_count"])
            self.assertEqual(0, json.loads(second_cache.read_text(encoding="utf-8"))["fallback_count"])

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

    def test_force_reprocess_bypasses_selected_cache_and_preserves_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode_speaker_transcript.json"
            source.write_text("transcript", encoding="utf-8")
            config = PipelineConfig(
                input_dir="input",
                processed_data_dir="processed_data",
                state_path="state/state.json",
                stop_file="state/stop.txt",
                control_file="state/control.json",
                run_snapshot_path="state/snapshot.json",
                run_report_dir="state/reports",
                fake_llm=True,
                verify_model=False,
                test_inference=False,
                errata_enabled=False,
                auto_refresh_topic_index=False,
                resume_within_file=False,
            )
            fingerprint = cli.file_fingerprint(source)
            cache_path = cli.processed_data_cache_path(root / config.processed_data_dir, fingerprint, source)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text('{"version": "before"}', encoding="utf-8")

            class FakePipeline:
                def __init__(self):
                    self.performance = PerformanceTracker(30)
                    self.fallback_count = 0
                    self.active_context = None
                    self.clear_calls = 0
                    self.process_calls = 0

                def clear_file_checkpoints(self, *args):
                    self.clear_calls += 1

                def process_file(self, path, errata=None, context=None):
                    self.process_calls += 1
                    return {
                        "status": "completed",
                        "source": "test",
                        "nodes": 0,
                        "position_cards": 0,
                        "documents": [],
                    }

                def save_cached_documents(self, cache_path, source_path, fingerprint, docs, context=None):
                    cache_path.write_text('{"version": "after"}', encoding="utf-8")

            pipeline = FakePipeline()
            with patch.object(cli.runtime, "load_runtime_deps"), patch.object(cli, "_make_pipeline", return_value=pipeline):
                result = cli.run_batch(
                    config,
                    root,
                    one_file=True,
                    input_files=[(source, None)],
                    force_reprocess=True,
                )

            self.assertEqual(0, result)
            self.assertEqual(1, pipeline.process_calls)
            self.assertEqual(1, pipeline.clear_calls)
            self.assertEqual("after", json.loads(cache_path.read_text(encoding="utf-8"))["version"])
            backups = list((root / "state" / "reprocess_backups").rglob(cache_path.name))
            self.assertEqual(1, len(backups))
            self.assertEqual("before", json.loads(backups[0].read_text(encoding="utf-8"))["version"])

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

    def test_all_diagnosis_action_types_validate(self):
        payload = {
            "summary": "A bounded diagnosis.",
            "root_causes": [],
            "actions": [
                {
                    "priority": "normal",
                    "type": action_type,
                    "action": "Take the bounded remediation action.",
                    "verification": "The remediation check passes.",
                }
                for action_type in ("code", "config", "data", "rerun", "human_review")
            ],
            "uncertainties": [],
            "human_review_questions": [],
        }

        self.assertEqual([], validate_diagnosis_payload(payload, set()))

    def test_diagnosis_retry_includes_action_type_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.json"
            source.write_text("transcript", encoding="utf-8")
            config = PipelineConfig()
            invalid = {
                "summary": "A bounded diagnosis.",
                "root_causes": [{
                    "category": "model_output",
                    "confidence": 0.9,
                    "finding_ids": ["F0001"],
                    "claim": "The model used an unsupported action type.",
                    "explanation": "The action type was outside the validator contract.",
                }],
                "actions": [{
                    "priority": "high",
                    "type": "unsupported_type",
                    "action": "Use a supported action type.",
                    "verification": "The diagnosis validates successfully.",
                }],
                "uncertainties": [],
                "human_review_questions": [],
            }
            valid = {**invalid, "actions": [{**invalid["actions"][0], "type": "human_review"}]}
            chain = SequenceChain([json.dumps(invalid), json.dumps(valid)])
            pipeline = pipeline_for_diagnosis(config, chain, root / "debug")
            recorder = ErrataRecorder(source, "fingerprint-1", "run-1", config)
            recorder.set_outcome("completed_with_warnings")
            recorder.record("fallback_summary", "warning", "summarization", "Fallback used.")

            with patch("podcast_rag.text_utils.time.sleep"):
                pipeline.diagnose_errata(recorder)

            self.assertEqual("completed", recorder.diagnosis["status"])
            self.assertEqual(2, len(chain.calls))
            self.assertIn("DIAGNOSIS VALIDATION FEEDBACK", chain.calls[1]["text"])
            self.assertIn("code, config, data, rerun, human_review", chain.calls[1]["text"])

    def test_invalid_diagnosis_action_types_are_retained_as_bounded_details(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.json"
            source.write_text("transcript", encoding="utf-8")
            config = PipelineConfig()
            response = json.dumps({
                "summary": "A bounded diagnosis.",
                "root_causes": [{
                    "category": "model_output",
                    "confidence": 0.9,
                    "finding_ids": ["F0001"],
                    "claim": "The model used unsupported action types.",
                    "explanation": "The action types were outside the validator contract.",
                }],
                "actions": [
                    {
                        "priority": "high",
                        "type": "unsupported_type_one",
                        "action": "Use a supported action type.",
                        "verification": "The diagnosis validates successfully.",
                    },
                    {
                        "priority": "normal",
                        "type": "unsupported_type_two",
                        "action": "Use human review when uncertain.",
                        "verification": "The diagnosis validates successfully.",
                    },
                ],
                "uncertainties": [],
                "human_review_questions": [],
            })
            pipeline = pipeline_for_diagnosis(config, DiagnosisChain(response), root / "debug")
            recorder = ErrataRecorder(source, "fingerprint-1", "run-1", config)
            recorder.set_outcome("completed_with_warnings")
            recorder.record("fallback_summary", "warning", "summarization", "Fallback used.")

            with patch("podcast_rag.text_utils.time.sleep"):
                pipeline.diagnose_errata(recorder)

            self.assertEqual("failed", recorder.diagnosis["status"])
            self.assertEqual(
                [
                    {"index": 0, "value": "unsupported_type_one"},
                    {"index": 1, "value": "unsupported_type_two"},
                ],
                recorder.diagnosis["invalid_action_types"],
            )
            failure = next(item for item in recorder.findings if item["code"] == "errata_diagnosis_failed")
            self.assertEqual(recorder.diagnosis["invalid_action_types"], failure["details"]["invalid_action_types"])
            self.assertIn("code, config, data, rerun, human_review", recorder.diagnosis["validation_errors"][0])

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
