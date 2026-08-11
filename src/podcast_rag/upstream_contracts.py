"""Dependency-free parsers for upstream ecosystem artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_CORRECTION_CONTRACTS = {"correction-manifest-v1", "correction-manifest-v2"}
APPROVED_CORRECTION_STATES = {"approved", "accepted"}
MUTABLE_IDENTITY_KEYS = {"notes", "display_label", "display_labels", "ui_state"}


class UpstreamContractError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _correction_state(correction: Mapping[str, Any]) -> str:
    return str(correction.get("status") or correction.get("adjudication_state") or "accepted")


def _validate_v2_identity(manifest: Mapping[str, Any]) -> None:
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in MUTABLE_IDENTITY_KEYS and key != "correction_set_id"
    }
    expected = f"correction_{hashlib.sha256(_canonical(identity)).hexdigest()}"
    if manifest.get("correction_set_id") != expected:
        raise UpstreamContractError("correction-set identity mismatch")


def parse_correction_fixture(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UpstreamContractError("correction payload must be an object")
    manifest = value.get("manifest", value)
    if not isinstance(manifest, dict):
        raise UpstreamContractError("correction manifest must be an object")
    transcript = value.get("transcript")
    version = str(manifest.get("contract_version") or "")
    if version not in SUPPORTED_CORRECTION_CONTRACTS:
        raise UpstreamContractError("unsupported correction manifest")
    if version == "correction-manifest-v2":
        _validate_v2_identity(manifest)
        corrections = manifest.get("corrections", [])
    else:
        corrections = manifest.get("accepted_corrections", [])
    if not isinstance(corrections, list):
        raise UpstreamContractError("corrections must be a list")
    approved = [
        dict(item)
        for item in corrections
        if isinstance(item, dict) and _correction_state(item) in APPROVED_CORRECTION_STATES
    ]
    if transcript is not None:
        if not isinstance(transcript, dict):
            raise UpstreamContractError("transcript must be an object")
        actual = hashlib.sha256(_canonical(transcript)).hexdigest()
        if actual != manifest.get("source_transcript_hash"):
            raise UpstreamContractError("stale transcript hash")
        spans = {
            str(item.get("source_span_id", item.get("id", ""))): item
            for item in transcript.get("segments", [])
            if isinstance(item, dict)
        }
        for correction in approved:
            source_span_id = str(correction.get("source_span_id") or "")
            field = str(correction.get("field") or "")
            guard = correction.get("before_value_guard", correction.get("before"))
            if not source_span_id or not field or spans.get(source_span_id, {}).get(field) != guard:
                raise UpstreamContractError("before value mismatch")
    result = dict(manifest)
    result["accepted_corrections"] = approved
    result["normalized_contract_version"] = "correction-manifest-v2"
    result["affected_source_span_ids"] = sorted(
        {
            str(item.get("source_span_id") or "")
            for item in approved
            if item.get("source_span_id")
        }
    )
    episode_ids = {str(item) for item in manifest.get("affected_episode_ids", []) if item}
    if isinstance(transcript, dict) and transcript.get("episode_id"):
        episode_ids.add(str(transcript["episode_id"]))
    result["affected_episode_ids"] = sorted(episode_ids)
    return result


def discover_correction_notifications(project_root: str | Path) -> list[dict[str, Any]]:
    inbox = Path(project_root) / "state" / "transcription_corrections"
    notifications = []
    for path in sorted(inbox.glob("*.json")) if inbox.exists() else []:
        status = "invalid"
        error = ""
        manifest = None
        payload: dict[str, Any] = {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise UpstreamContractError("notification must be an object")
            payload = loaded
            if payload.get("contract_version") != "correction-notification-v1":
                raise UpstreamContractError("unsupported correction notification")
            manifest_path_value = str(payload.get("correction_manifest_path") or "").strip()
            if not manifest_path_value:
                raise UpstreamContractError("correction manifest path is missing")
            manifest_path = Path(manifest_path_value)
            if not manifest_path.exists():
                status = "downstream_pending"
                error = "correction manifest is unavailable"
            else:
                manifest = parse_correction_fixture(manifest_path)
                status = "ready"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
        notifications.append(
            {
                **payload,
                "notification_path": str(path),
                "status": status,
                "error": error,
                "manifest": manifest,
            }
        )
    return notifications
