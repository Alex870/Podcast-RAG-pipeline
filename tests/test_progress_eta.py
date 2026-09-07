import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from podcast_rag import cli
from podcast_rag.config import PipelineConfig


class ProgressEtaTests(unittest.TestCase):
    def test_cached_files_do_not_count_as_processing_eta_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            sources = [input_dir / f"episode_{index}_transcript.json" for index in range(3)]
            for source in sources:
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
                verify_model=False,
                test_inference=False,
                errata_enabled=False,
                auto_refresh_topic_index=False,
                resume_within_file=False,
            )
            cached_path = cli.processed_data_cache_path(
                root / config.processed_data_dir,
                cli.file_fingerprint(sources[0]),
                sources[0],
            )
            cached_path.parent.mkdir(parents=True)
            cached_path.write_text("{}", encoding="utf-8")

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
                    return {"status": "completed", "source": "test", "nodes": 0, "position_cards": 0, "documents": []}

                def validate_cached_file(self, path, fingerprint, cache_path, context=None):
                    return {
                        "status": "completed",
                        "source": "processed_data_cache",
                        "nodes": 0,
                        "position_cards": 0,
                        "leaf_chunks": 0,
                        "summaries": 0,
                        "positions": 0,
                    }

                def save_cached_documents(self, cache_path, source_path, fingerprint, docs, context=None):
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text("{}", encoding="utf-8")

            eta_calls = []

            def capture_eta(completed, total, elapsed_seconds):
                eta_calls.append((completed, total, elapsed_seconds))
                return None

            with patch.object(cli.runtime, "load_runtime_deps"), \
                    patch.object(cli, "PodcastRagPipeline", FakePipeline), \
                    patch.object(cli, "iter_transcript_files", return_value=sources), \
                    patch.object(cli, "estimate_remaining_seconds", side_effect=capture_eta):
                self.assertEqual(0, cli.run_batch(config, root, one_file=False))

            self.assertEqual(2, FakePipeline.process_calls)
            self.assertEqual(3, len(eta_calls))
            self.assertEqual([0, 0, 1], [completed for completed, _total, _elapsed in eta_calls])
            self.assertEqual([2, 2, 2], [total for _completed, total, _elapsed in eta_calls])


if __name__ == "__main__":
    unittest.main()
