import json
import tempfile
import unittest
from pathlib import Path

from podcast_rag.delta_jobs import DeltaJobError, DeltaJobStore, apply_job, plan_notification_job


class MilestoneOneDeltaJobTests(unittest.TestCase):
    def setup_job(self, root: Path):
        fixture = Path(__file__).parent / "fixtures/contracts/transcription/correction-manifest-v2/valid.json"
        inbox = root / "project/state/transcription_corrections"
        inbox.mkdir(parents=True)
        (inbox / "event.json").write_text(json.dumps({"contract_version": "correction-notification-v1", "correction_manifest_path": str(fixture)}), encoding="utf-8")
        old = {"leaf": {"content": "before", "metadata": {"source": "episode-synthetic-v2-001_cleaned_speaker_transcript.json"}}, "same": {"content": "same"}}
        new = {**old, "leaf": {**old["leaf"], "content": "after"}}
        old_path, new_path = root / "old.json", root / "new.json"
        old_path.write_text(json.dumps(old), encoding="utf-8"); new_path.write_text(json.dumps(new), encoding="utf-8")
        job = plan_notification_job(
            project_root=root / "project", old_path=old_path, new_path=new_path, output_path=root / "output.json",
            job_root=root / "jobs", parent_corpus_id="corpus", processing_fingerprint="p", representation_fingerprint="r",
        )
        return job, old_path, new_path

    def test_preview_apply_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); job, _, _ = self.setup_job(root); store = DeltaJobStore(root / "jobs")
            completed = apply_job(store, job["job_id"], approved_delta_id=job["delta_id"])
            self.assertEqual("completed", completed["status"])
            self.assertEqual(completed, apply_job(store, job["job_id"], approved_delta_id=job["delta_id"]))
            self.assertEqual("same", json.loads((root / "output.json").read_text(encoding="utf-8"))["same"]["content"])

    def test_changed_input_and_wrong_approval_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); job, _, new_path = self.setup_job(root); store = DeltaJobStore(root / "jobs")
            with self.assertRaisesRegex(DeltaJobError, "approval"): apply_job(store, job["job_id"], approved_delta_id="wrong")
            new_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(DeltaJobError, "changed"): apply_job(store, job["job_id"], approved_delta_id=job["delta_id"])

    def test_cancel_before_apply_preserves_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); job, _, _ = self.setup_job(root); store = DeltaJobStore(root / "jobs")
            store.cancel(job["job_id"]); cancelled = apply_job(store, job["job_id"], approved_delta_id=job["delta_id"])
            self.assertEqual("cancelled", cancelled["status"]); self.assertFalse((root / "output.json").exists())


if __name__ == "__main__": unittest.main()
