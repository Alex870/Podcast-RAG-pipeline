import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from podcast_rag.config import PipelineConfig
from podcast_rag.handoff import HandoffError, canonical_payload, validate_handoff, validate_release_manifest
from podcast_rag.partitions import PartitionError, PartitionRegistry, PartitionSpec, effective_partition_config, processing_key
from podcast_rag import cli


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def make_package(root: Path, *, partition_id: str = "podcast-history", corpus_id: str = "podcast-history", episode_id: str = "episode-01") -> Path:
    package = root / "handoff-01"
    transcript_path = package / "episodes" / episode_id / "reviewed.json"
    transcript_path.parent.mkdir(parents=True)
    transcript = {
        "contract_version": "episode-contract-v2",
        "schema_version": 2,
        "episode_id": episode_id,
        "episode_uid": f"{partition_id}:{episode_id}",
        "partition_id": partition_id,
        "corpus_id": corpus_id,
        "segments": [{"source_span_id": f"{episode_id}:segment:0", "start": 0.0, "end": 2.0, "speaker": "HOST", "text": "A stable source span."}],
        "speech_provenance": [{"provider": "test", "model_revision": "immutable-test-revision"}],
    }
    transcript_bytes = json.dumps(transcript, indent=2).encode("utf-8")
    transcript_path.write_bytes(transcript_bytes)
    manifest = {
        "contract_version": "podcast-rag-transcription-handoff-v1",
        "handoff_id": "handoff-01",
        "created_at": "2026-09-05T18:00:00Z",
        "producer": {"name": "podcast-host-transcription-pipeline", "contract_version": "episode-contract-v2"},
        "partition": {"partition_id": partition_id, "corpus_id": corpus_id, "display_name": partition_id, "context_type": "podcast", "workflow_profile": "podcast"},
        "episodes": [{
            "episode_id": episode_id,
            "episode_uid": f"{partition_id}:{episode_id}",
            "episode_title": "Episode 01",
            "source_audio": {"fingerprint": "sha256:audio-fingerprint"},
            "selected_transcript": {
                "variant": "reviewed_llm",
                "path": f"episodes/{episode_id}/reviewed.json",
                "artifact_sha256": _sha256(transcript_bytes),
                "canonical_payload_sha256": _sha256(canonical_payload(transcript)),
            },
            "stable": True,
            "correction_set_id": None,
        }],
        "correction_sets": [],
    }
    manifest["integrity"] = {"algorithm": "sha256", "manifest_hash_excludes_field": "integrity.manifest_sha256"}
    clone = json.loads(json.dumps(manifest))
    clone["integrity"].pop("manifest_sha256", None)
    manifest["integrity"]["manifest_sha256"] = _sha256(canonical_payload(clone))
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return package


class PartitionAndHandoffTests(unittest.TestCase):
    def test_partition_registry_create_select_update_archive_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = PartitionRegistry(root)
            first = registry.create(PartitionSpec("podcast-history", "Podcast History", "podcast", "podcast"))
            second = registry.create(PartitionSpec("work-meetings", "Work Meetings", "meeting", "anonymous_meeting"))
            self.assertEqual("podcast-history", registry.active_partition_id)
            registry.use(second.partition_id)
            registry.update(second.partition_id, display_name="Meetings")
            with self.assertRaises(PartitionError):
                registry.archive(second.partition_id)
            registry.use(first.partition_id)
            registry.archive(second.partition_id)
            self.assertEqual("Meetings", registry.require("work-meetings").display_name)
            self.assertTrue(first.paths(root)["checkpoints"].is_dir())
            self.assertNotEqual(first.paths(root)["processed_data"], second.paths(root)["processed_data"])

    def test_handoff_validates_without_runtime_dependencies_and_selects_declared_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            package = make_package(Path(directory))
            result = validate_handoff(package / "manifest.json")
            self.assertTrue(result.valid, result.errors)
            self.assertEqual("reviewed_llm", result.episodes[0].selected_variant)
            self.assertEqual("episodes/episode-01/reviewed.json", result.episodes[0].selected_relative_path)

    def test_handoff_rejects_hash_mismatch_and_unsafe_portable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            package = make_package(Path(directory))
            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["episodes"][0]["selected_transcript"]["path"] = "../outside.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = validate_handoff(manifest_path)
            self.assertFalse(result.valid)
            self.assertIn("unsafe_artifact_path", {item["code"] for item in result.errors})

    def test_processing_key_changes_for_partition_and_handoff(self):
        base = {"partition_id": "one", "episode_id": "same", "episode_uid": "one:same", "handoff_id": "h1", "selected_transcript_artifact_sha256": "a", "selected_transcript_canonical_payload_sha256": "b", "source_audio_fingerprint": "c", "pipeline_version": "p"}
        one = processing_key(base, config_fingerprint="c", generation_fingerprint="g", representation_fingerprint="r")
        other = processing_key({**base, "partition_id": "two", "episode_uid": "two:same"}, config_fingerprint="c", generation_fingerprint="g", representation_fingerprint="r")
        self.assertNotEqual(one, other)

    def test_approved_correction_is_applied_only_to_consumer_view(self):
        with tempfile.TemporaryDirectory() as directory:
            package = make_package(Path(directory))
            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            transcript_path = package / "episodes" / "episode-01" / "reviewed.json"
            original_bytes = transcript_path.read_bytes()
            transcript = json.loads(original_bytes)
            correction = {
                "contract_version": "correction-manifest-v1",
                "correction_set_id": "correction-test-1",
                "partition_id": "podcast-history",
                "affected_episode_ids": ["episode-01"],
                "source_transcript_hash": hashlib.sha256(canonical_payload(transcript)).hexdigest(),
                "accepted_corrections": [{
                    "status": "approved",
                    "source_span_id": "episode-01:segment:0",
                    "field": "text",
                    "before_value_guard": "A stable source span.",
                    "after_value": "A corrected source span.",
                }],
            }
            correction_path = package / "corrections" / "correction-test-1.json"
            correction_path.parent.mkdir()
            correction_bytes = json.dumps(correction).encode("utf-8")
            correction_path.write_bytes(correction_bytes)
            manifest["episodes"][0]["correction_set_id"] = "correction-test-1"
            manifest["correction_sets"] = [{"correction_set_id": "correction-test-1", "path": "corrections/correction-test-1.json", "artifact_sha256": _sha256(correction_bytes)}]
            manifest["integrity"].pop("manifest_sha256", None)
            manifest["integrity"]["manifest_sha256"] = _sha256(canonical_payload(manifest))
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            result = validate_handoff(manifest_path)
            self.assertTrue(result.valid, result.errors)
            self.assertEqual("A corrected source span.", result.episodes[0].transcript_payload["segments"][0]["text"])
            self.assertEqual(original_bytes, transcript_path.read_bytes())

    def test_managed_process_requires_registered_partition_and_uses_partition_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({}), encoding="utf-8")
            registry = PartitionRegistry(root)
            spec = registry.create(PartitionSpec("podcast-history", "Podcast History", "podcast", "podcast"))
            package = make_package(spec.paths(root)["handoff_inbox"])
            with patch.object(cli, "run_batch", return_value=0) as run_batch:
                result = cli.managed_process_command(PipelineConfig(), root, ["--manifest", str(package / "manifest.json")])
            self.assertEqual(0, result)
            context_files = run_batch.call_args.kwargs["input_files"]
            self.assertEqual(1, len(context_files))
            self.assertEqual("podcast-history:episode-01", context_files[0][1]["episode_uid"])
            self.assertTrue(str(spec.paths(root)["processed_data"]) in cli.effective_partition_config(PipelineConfig(), root, spec)[0].processed_data_dir)

    def test_release_validation_rejects_mixed_identity(self):
        payload = {
            "release_contract_version": "podcast-rag-corpus-release-v1",
            "release_id": "release-1",
            "partition_id": "one",
            "corpus_id": "one",
            "handoff_ids": [],
            "episode_uids": ["one:a", "two:b"],
        }
        errors = validate_release_manifest(payload, expected_partition_id="one", expected_corpus_id="one")
        self.assertTrue(any("episode_uid" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
