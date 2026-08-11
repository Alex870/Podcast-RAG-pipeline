import json
import tempfile
import unittest
from pathlib import Path

from podcast_rag.delta_cli import main as delta_cli_main
from podcast_rag.ecosystem_delta import DeltaError, apply_delta, plan_delta, validate_delta


class DeltaTests(unittest.TestCase):
    def setUp(self):
        self.old = {"leaf": {"content": "before", "parent_id": "episode", "topic_id": "topic", "position_id": "position"}, "same": {"content": "same"}}

    def plan(self, new):
        return plan_delta(self.old, new, parent_corpus_id="corpus-1", correction_set_id="correction-1", processing_fingerprint="p1", representation_fingerprint="r1")

    def test_noop_is_empty_and_deterministic(self):
        first, second = self.plan(self.old), self.plan(self.old)
        self.assertEqual(first["delta_id"], second["delta_id"])
        self.assertEqual([], first["changed_document_ids"])

    def test_changed_leaf_closes_derived_effects(self):
        new = {**self.old, "leaf": {**self.old["leaf"], "content": "after"}}
        delta = self.plan(new)
        self.assertEqual(["leaf"], delta["changed_document_ids"])
        self.assertEqual(["episode"], delta["invalidated"]["ancestors"])
        self.assertEqual(["topic"], delta["invalidated"]["topics"])
        self.assertEqual(["position"], delta["invalidated"]["positions"])
        self.assertEqual("after", apply_delta(delta, self.old, new, approved_correction_set_id="correction-1")["leaf"]["content"])

    def test_removed_has_reason_and_is_advisory(self):
        delta = self.plan({"same": self.old["same"]})
        self.assertIn("leaf", delta["reasons"])
        self.assertTrue(delta["removals_advisory"])

    def test_apply_requires_approved_correction(self):
        delta = self.plan(self.old)
        with self.assertRaisesRegex(DeltaError, "approved"):
            apply_delta(delta, self.old, self.old, approved_correction_set_id="wrong")

    def test_identity_is_validated(self):
        delta = self.plan(self.old); delta["processing_fingerprint"] = "changed"
        with self.assertRaisesRegex(DeltaError, "identity"):
            validate_delta(delta)

    def test_notification_scope_allows_only_affected_episode_changes(self):
        old = {
            "affected": {"content": "before", "metadata": {"source": "C:/output/Episode 1_cleaned_speaker_transcript.json"}},
            "other": {"content": "same", "metadata": {"source": "C:/output/Episode 2_cleaned_speaker_transcript.json"}},
        }
        new = {**old, "affected": {**old["affected"], "content": "after"}}
        delta = plan_delta(
            old, new, parent_corpus_id="corpus-1", correction_set_id="correction-1",
            processing_fingerprint="p1", representation_fingerprint="r1",
            affected_episode_ids=["Episode 1"], affected_source_span_ids=["17"],
        )
        self.assertEqual(["affected"], delta["changed_document_ids"])
        self.assertEqual(["Episode 1"], delta["affected_episode_ids"])

    def test_notification_scope_rejects_unrelated_drift_and_missing_matches(self):
        old = {
            "affected": {"content": "before", "metadata": {"source": "Episode 1_cleaned_speaker_transcript.json"}},
            "other": {"content": "same", "metadata": {"source": "Episode 2_cleaned_speaker_transcript.json"}},
        }
        with self.assertRaisesRegex(DeltaError, "out-of-scope"):
            plan_delta(
                old, {**old, "other": {**old["other"], "content": "drift"}},
                parent_corpus_id="corpus-1", correction_set_id="correction-1",
                processing_fingerprint="p1", representation_fingerprint="r1",
                affected_episode_ids=["Episode 1"],
            )
        with self.assertRaisesRegex(DeltaError, "did not match"):
            plan_delta(
                old, old, parent_corpus_id="corpus-1", correction_set_id="correction-1",
                processing_fingerprint="p1", representation_fingerprint="r1",
                affected_episode_ids=["Missing Episode"],
            )

    def test_notification_cli_plans_from_validated_inbox(self):
        scratch = Path(__file__).parent.parent / ".test_tmp"
        scratch.mkdir(exist_ok=True)
        fixture = Path(__file__).parent / "fixtures/contracts/transcription/correction-manifest-v2/valid.json"
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            inbox = root / "state/transcription_corrections"
            inbox.mkdir(parents=True)
            (inbox / "correction.json").write_text(json.dumps({
                "contract_version": "correction-notification-v1",
                "correction_manifest_path": str(fixture),
            }), encoding="utf-8")
            old = {"leaf": {"content": "before", "metadata": {"source": "episode-synthetic-v2-001_cleaned_speaker_transcript.json"}}}
            new = {"leaf": {**old["leaf"], "content": "after"}}
            old_path, new_path, output_path = root / "old.json", root / "new.json", root / "delta.json"
            old_path.write_text(json.dumps(old), encoding="utf-8")
            new_path.write_text(json.dumps(new), encoding="utf-8")
            self.assertEqual(0, delta_cli_main([
                "plan-notification-delta", "--project-root", str(root), "--old", str(old_path),
                "--new", str(new_path), "--parent-corpus-id", "corpus-1",
                "--processing-fingerprint", "p1", "--representation-fingerprint", "r1",
                "--output", str(output_path),
            ]))
            delta = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(["leaf"], delta["changed_document_ids"])
            self.assertEqual(["episode-synthetic-v2-001"], delta["affected_episode_ids"])


if __name__ == "__main__": unittest.main()
