import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from podcast_rag.upstream_contracts import (
    UpstreamContractError,
    discover_correction_notifications,
    parse_correction_fixture,
)


FIXTURES = Path(__file__).parent / "fixtures" / "contracts" / "transcription"


def _write_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _refresh_v2_identity(manifest: dict) -> None:
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"notes", "display_label", "display_labels", "ui_state", "correction_set_id"}
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["correction_set_id"] = f"correction_{hashlib.sha256(canonical).hexdigest()}"


class UpstreamContractTests(unittest.TestCase):
    def test_parses_v1_fixture(self):
        parsed = parse_correction_fixture(FIXTURES / "correction-manifest-v1" / "valid.json")
        self.assertEqual(parsed["normalized_contract_version"], "correction-manifest-v2")

    def test_parses_v2_fixture_and_excludes_rejected_correction(self):
        parsed = parse_correction_fixture(FIXTURES / "correction-manifest-v2" / "valid.json")
        self.assertEqual(len(parsed["accepted_corrections"]), 1)
        self.assertEqual(parsed["affected_source_span_ids"], ["span-001"])
        self.assertEqual(parsed["affected_episode_ids"], ["episode-synthetic-v2-001"])

    def test_rejects_stale_transcript_hash(self):
        payload = json.loads((FIXTURES / "correction-manifest-v2" / "valid.json").read_text(encoding="utf-8"))
        payload["transcript"]["segments"][0]["text"] = "Changed outside correction flow."
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stale.json"
            _write_payload(path, payload)
            with self.assertRaisesRegex(UpstreamContractError, "stale transcript hash"):
                parse_correction_fixture(path)

    def test_rejects_before_value_conflict(self):
        payload = json.loads((FIXTURES / "correction-manifest-v2" / "valid.json").read_text(encoding="utf-8"))
        correction = payload["manifest"]["corrections"][0]
        correction["before_value_guard"] = "Wrong value"
        payload["manifest"]["accepted_corrections"][0]["before_value_guard"] = "Wrong value"
        _refresh_v2_identity(payload["manifest"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conflict.json"
            _write_payload(path, payload)
            with self.assertRaisesRegex(UpstreamContractError, "before value mismatch"):
                parse_correction_fixture(path)

    def test_rejects_identity_tampering_and_unsupported_version(self):
        payload = json.loads((FIXTURES / "correction-manifest-v2" / "valid.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            payload["manifest"]["correction_set_id"] = "correction_tampered"
            _write_payload(path, payload)
            with self.assertRaisesRegex(UpstreamContractError, "identity mismatch"):
                parse_correction_fixture(path)
            payload["manifest"]["contract_version"] = "correction-manifest-v99"
            _write_payload(path, payload)
            with self.assertRaisesRegex(UpstreamContractError, "unsupported"):
                parse_correction_fixture(path)

    def test_notification_discovery_reports_ready_pending_and_invalid(self):
        fixture = FIXTURES / "correction-manifest-v2" / "valid.json"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inbox = root / "state" / "transcription_corrections"
            inbox.mkdir(parents=True)
            _write_payload(
                inbox / "ready.json",
                {
                    "contract_version": "correction-notification-v1",
                    "correction_manifest_path": str(fixture),
                },
            )
            _write_payload(
                inbox / "pending.json",
                {
                    "contract_version": "correction-notification-v1",
                    "correction_manifest_path": str(root / "missing.json"),
                },
            )
            _write_payload(inbox / "invalid.json", {"contract_version": "unknown"})
            by_name = {Path(item["notification_path"]).name: item for item in discover_correction_notifications(root)}
            self.assertEqual(by_name["ready.json"]["status"], "ready")
            self.assertEqual(by_name["pending.json"]["status"], "downstream_pending")
            self.assertEqual(by_name["invalid.json"]["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
