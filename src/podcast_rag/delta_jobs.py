"""Persisted one-shot changed-episode delta jobs."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .ecosystem_delta import apply_delta, plan_delta, write_atomic
from .upstream_contracts import discover_correction_notifications


class DeltaJobError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"rag_delta_job_{hashlib.sha256(encoded).hexdigest()}"


class DeltaJobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str) -> Path:
        if not job_id.startswith("rag_delta_job_") or len(job_id) != 78:
            raise DeltaJobError("invalid delta job ID")
        return self.root / job_id

    def load(self, job_id: str) -> dict[str, Any]:
        path = self.path(job_id) / "job.json"
        if not path.is_file():
            raise DeltaJobError(f"delta job not found: {job_id}")
        return _read(path)

    def save(self, value: dict[str, Any]) -> dict[str, Any]:
        directory = self.path(str(value["job_id"]))
        directory.mkdir(parents=True, exist_ok=True)
        write_atomic(directory / "job.json", value)
        return value

    def cancel(self, job_id: str) -> dict[str, Any]:
        value = self.load(job_id)
        if value["status"] in {"completed", "failed"}:
            raise DeltaJobError(f"cannot cancel a {value['status']} delta job")
        value["cancel_requested"] = True
        value["status"] = "cancel_requested"
        value["updated_at_epoch_ms"] = int(time.time() * 1000)
        return self.save(value)


def plan_notification_job(
    *, project_root: Path, old_path: Path, new_path: Path, output_path: Path,
    job_root: Path, parent_corpus_id: str, processing_fingerprint: str,
    representation_fingerprint: str, correction_set_id: str | None = None,
) -> dict[str, Any]:
    ready = [item for item in discover_correction_notifications(project_root) if item["status"] == "ready"]
    if correction_set_id:
        ready = [item for item in ready if item.get("manifest", {}).get("correction_set_id") == correction_set_id]
    if len(ready) != 1:
        raise DeltaJobError(f"expected exactly one ready correction notification; found {len(ready)}")
    manifest = ready[0]["manifest"]
    old_documents, new_documents = _read(old_path), _read(new_path)
    delta = plan_delta(
        old_documents, new_documents, parent_corpus_id=parent_corpus_id,
        correction_set_id=manifest["correction_set_id"], processing_fingerprint=processing_fingerprint,
        representation_fingerprint=representation_fingerprint,
        affected_episode_ids=manifest.get("affected_episode_ids", []),
        affected_source_span_ids=manifest.get("affected_source_span_ids", []),
    )
    inputs = {
        "project_root": str(project_root.resolve()),
        "old_path": str(old_path.resolve()), "old_sha256": _sha256(old_path),
        "new_path": str(new_path.resolve()), "new_sha256": _sha256(new_path),
        "output_path": str(output_path.resolve()),
        "correction_set_id": manifest["correction_set_id"],
        "notification_path": ready[0].get("path"),
    }
    identity_payload = {"inputs": inputs, "delta_id": delta["delta_id"]}
    job_id = _job_id(identity_payload)
    store = DeltaJobStore(job_root)
    directory = store.path(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic(directory / "delta.json", delta)
    value = {
        "contract_version": "rag-delta-job-v1", "job_id": job_id,
        "status": "planned", "cancel_requested": False, "inputs": inputs,
        "delta_id": delta["delta_id"], "delta_path": str((directory / "delta.json").resolve()),
        "progress": {"completed": 1, "total": 2, "stage": "planned"},
        "diagnostics": {
            "changed": delta["changed_document_ids"], "added": delta["added_document_ids"],
            "removed_advisory": delta["removed_document_ids"], "unchanged": delta["unchanged_document_ids"],
            "invalidated": delta["invalidated"], "reasons": delta["reasons"],
        },
        "created_at_epoch_ms": int(time.time() * 1000),
        "updated_at_epoch_ms": int(time.time() * 1000),
    }
    return store.save(value)


def apply_job(store: DeltaJobStore, job_id: str, *, approved_delta_id: str) -> dict[str, Any]:
    value = store.load(job_id)
    if value.get("cancel_requested"):
        value["status"] = "cancelled"
        value["progress"] = {"completed": 1, "total": 2, "stage": "cancelled"}
        return store.save(value)
    if value["status"] == "completed":
        return value
    if approved_delta_id != value["delta_id"]:
        raise DeltaJobError("approval does not match the planned delta identity")
    inputs = value["inputs"]
    old_path, new_path = Path(inputs["old_path"]), Path(inputs["new_path"])
    if _sha256(old_path) != inputs["old_sha256"] or _sha256(new_path) != inputs["new_sha256"]:
        raise DeltaJobError("delta job inputs changed after preview")
    try:
        delta = _read(Path(value["delta_path"]))
        result = apply_delta(delta, _read(old_path), _read(new_path), approved_correction_set_id=inputs["correction_set_id"])
        write_atomic(Path(inputs["output_path"]), result)
        value.update({
            "status": "completed", "output_sha256": _sha256(Path(inputs["output_path"])),
            "progress": {"completed": 2, "total": 2, "stage": "applied"},
            "updated_at_epoch_ms": int(time.time() * 1000),
        })
    except Exception as exc:
        quarantine = store.path(job_id) / "quarantine.json"
        write_atomic(quarantine, {"job_id": job_id, "error_type": type(exc).__name__, "error": str(exc), "delta_id": value["delta_id"]})
        value.update({"status": "failed", "retryable": True, "error": str(exc), "quarantine_path": str(quarantine.resolve())})
        store.save(value)
        raise
    return store.save(value)
