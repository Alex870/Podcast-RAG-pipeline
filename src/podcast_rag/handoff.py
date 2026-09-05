"""Validation and normalization of producer-owned transcription handoffs.

This module is intentionally dependency-free.  Handoff validation and inbox
scanning must be safe to run on a machine that has no LLM or embedding model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from podcast_rag.upstream_contracts import SUPPORTED_CORRECTION_CONTRACTS, UpstreamContractError, parse_correction_fixture_payload, parse_episode_contract_payload


HANDOFF_CONTRACT_VERSION = "podcast-rag-transcription-handoff-v1"
RELEASE_CONTRACT_VERSION = "podcast-rag-corpus-release-v1"
SUPPORTED_CONTEXT_TYPES = {"podcast", "meeting", "custom"}
_HASH_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_ABSOLUTE_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2}|\\\\)")
_SECRET_KEY_RE = re.compile(r"(?:api[_-]?key|token|password|secret)", re.IGNORECASE)


class HandoffError(ValueError):
    """A handoff package is incomplete, unsafe, or contract-invalid."""

    def __init__(self, message: str, *, code: str = "invalid_manifest", path: str | None = None, episode_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.path = path
        self.episode_id = episode_id


@dataclass
class ManagedEpisode:
    episode_id: str
    episode_uid: str
    title: str | None
    selected_variant: str
    selected_path: Path
    selected_relative_path: str
    artifact_sha256: str
    canonical_payload_sha256: str
    source_audio_fingerprint: str
    correction_set_id: str | None
    stable: bool
    manifest_entry: dict[str, Any]
    transcript_payload: dict[str, Any]
    raw_transcript_payload: dict[str, Any]
    identity: dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffValidation:
    manifest_path: Path
    package_root: Path
    manifest: dict[str, Any] | None
    episodes: list[ManagedEpisode] = field(default_factory=list)
    corrections: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def handoff_id(self) -> str | None:
        return str((self.manifest or {}).get("handoff_id") or "") or None

    @property
    def partition(self) -> dict[str, Any]:
        value = (self.manifest or {}).get("partition")
        return dict(value) if isinstance(value, dict) else {}

    def raise_for_errors(self) -> None:
        if self.errors:
            preview = "; ".join(str(item.get("message")) for item in self.errors[:8])
            raise HandoffError(preview or "invalid handoff package", code=str(self.errors[0].get("code") or "invalid_manifest"))


def canonical_payload(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _hash_value(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not _HASH_RE.fullmatch(text):
        raise HandoffError(f"{label} must be a sha256 hash", code="invalid_manifest")
    return text.lower() if text.startswith("sha256:") else "sha256:" + text.lower()


def _relative_artifact(root: Path, value: Any, *, label: str, require_exists: bool = True) -> tuple[Path, str]:
    raw = str(value or "").strip()
    if not raw or _ABSOLUTE_RE.match(raw) or "\\" in raw:
        raise HandoffError(f"{label} must be a package-relative path using '/'", code="unsafe_artifact_path", path=raw or None)
    posix = PurePosixPath(raw)
    if not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise HandoffError(f"{label} is not a safe package-relative path", code="unsafe_artifact_path", path=raw)
    path = root.joinpath(*posix.parts)
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise HandoffError(f"{label} escapes the handoff package", code="unsafe_artifact_path", path=raw) from exc
    if path.is_symlink():
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise HandoffError(f"{label} refers to an external symlink", code="unsafe_artifact_path", path=raw) from exc
    if require_exists and not path.is_file():
        raise HandoffError(f"referenced artifact is missing: {raw}", code="artifact_integrity_failure", path=raw)
    return path, "/".join(posix.parts)


def _validate_portable_values(value: Any, path: str = "manifest") -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _SECRET_KEY_RE.search(str(key)):
                errors.append({"code": "portable_secret", "path": child_path, "message": f"portable manifest contains a secret-like field at {child_path}"})
            errors.extend(_validate_portable_values(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_validate_portable_values(child, f"{path}[{index}]"))
    elif isinstance(value, str) and _ABSOLUTE_RE.match(value.strip()):
        errors.append({"code": "portable_absolute_path", "path": path, "message": f"portable manifest contains an absolute path at {path}"})
    return errors


def _manifest_hash(manifest: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(manifest))
    integrity = clone.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("manifest_sha256", None)
    return sha256_bytes(canonical_payload(clone))


def _add_error(result: HandoffValidation, exc: HandoffError, *, episode_id: str | None = None) -> None:
    result.errors.append(
        {
            "code": exc.code,
            "message": str(exc),
            "path": exc.path,
            "episode_id": episode_id or exc.episode_id,
        }
    )


def _validate_segment_payload(transcript: dict[str, Any], *, episode_id: str) -> None:
    if not isinstance(transcript, dict) or not isinstance(transcript.get("segments"), list):
        raise HandoffError("transcript segments must be an array", code="invalid_transcript", episode_id=episode_id)
    previous_start = -math.inf
    spans: set[str] = set()
    for index, segment in enumerate(transcript["segments"]):
        if not isinstance(segment, dict):
            raise HandoffError(f"segment {index} must be an object", code="invalid_transcript", episode_id=episode_id)
        span_id = str(segment.get("source_span_id") or segment.get("id") or "").strip()
        if not span_id:
            raise HandoffError(f"segment {index} is missing source_span_id", code="invalid_transcript", episode_id=episode_id)
        if span_id in spans:
            raise HandoffError(f"duplicate source-span identity {span_id}", code="duplicate_identity", episode_id=episode_id)
        spans.add(span_id)
        start = segment.get("start", segment.get("start_time"))
        end = segment.get("end", segment.get("end_time"))
        if not isinstance(start, (int, float)) or isinstance(start, bool) or not math.isfinite(float(start)):
            raise HandoffError(f"segment {span_id} has invalid start time", code="invalid_transcript", episode_id=episode_id)
        if not isinstance(end, (int, float)) or isinstance(end, bool) or not math.isfinite(float(end)) or float(start) > float(end):
            raise HandoffError(f"segment {span_id} has invalid time range", code="invalid_transcript", episode_id=episode_id)
        if float(start) < previous_start:
            raise HandoffError(f"segment order decreases at {span_id}", code="invalid_transcript", episode_id=episode_id)
        previous_start = float(start)
        text = segment.get("text", segment.get("content"))
        non_speech = bool(segment.get("non_speech") or segment.get("is_non_speech"))
        if not non_speech and not str(text or "").strip():
            raise HandoffError(f"segment {span_id} has empty text", code="invalid_transcript", episode_id=episode_id)
        if segment.get("supersedes_source_span_id") and not segment.get("correction_set_id"):
            raise HandoffError(f"segment {span_id} has supersession without correction_set_id", code="invalid_transcript", episode_id=episode_id)
    provenance = transcript.get("speech_provenance", [])
    if not isinstance(provenance, list):
        raise HandoffError("speech_provenance must be an array", code="invalid_transcript", episode_id=episode_id)
    for provider in provenance:
        if not isinstance(provider, dict) or not provider.get("provider") or not provider.get("model_revision"):
            raise HandoffError("speech_provenance records require provider and model_revision", code="invalid_transcript", episode_id=episode_id)


def _validate_correction_sets(result: HandoffValidation, corrections: Any, episodes_by_id: dict[str, ManagedEpisode]) -> dict[str, dict[str, Any]]:
    if corrections in (None, []):
        return {}
    if not isinstance(corrections, list):
        result.errors.append({"code": "invalid_manifest", "message": "correction_sets must be an array", "path": "correction_sets"})
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(corrections):
        if not isinstance(entry, dict):
            result.errors.append({"code": "invalid_manifest", "message": f"correction_sets[{index}] must be an object", "path": f"correction_sets[{index}]"})
            continue
        path_value = entry.get("path") or entry.get("artifact_path")
        try:
            if path_value:
                path, rel = _relative_artifact(result.package_root, path_value, label=f"correction_sets[{index}].path")
                expected = entry.get("artifact_sha256")
                if not expected:
                    raise HandoffError("referenced correction artifact is missing artifact_sha256", code="artifact_integrity_failure", path=rel)
                if _hash_value(expected, label="correction artifact hash") != sha256_file(path):
                    raise HandoffError("correction artifact hash mismatch", code="artifact_integrity_failure", path=rel)
                raw = json.loads(path.read_text(encoding="utf-8"))
            else:
                raw = entry
            correction = raw.get("manifest", raw) if isinstance(raw, dict) else raw
            if not isinstance(correction, dict):
                raise HandoffError("correction manifest must be an object", code="stale_correction")
            contract = str(correction.get("contract_version") or "")
            if contract not in SUPPORTED_CORRECTION_CONTRACTS:
                raise HandoffError("unsupported correction manifest", code="stale_correction")
            correction_partition = str(correction.get("partition_id") or "")
            if correction_partition and correction_partition != result.partition.get("partition_id"):
                raise HandoffError("correction set belongs to another partition", code="mixed_partition_scope")
            correction_id = str(correction.get("correction_set_id") or entry.get("correction_set_id") or "")
            if not correction_id:
                raise HandoffError("correction_set_id is missing", code="stale_correction")
            normalized[correction_id] = {"manifest": correction, "path": path_value}
            affected = {str(item) for item in correction.get("affected_episode_ids", []) if item}
            for episode_id in affected:
                if episode_id not in episodes_by_id:
                    raise HandoffError(f"correction set references undeclared episode {episode_id}", code="stale_correction", episode_id=episode_id)
                transcript = episodes_by_id[episode_id].raw_transcript_payload
                try:
                    parse_correction_fixture_payload({"manifest": correction, "transcript": transcript})
                except (UpstreamContractError, json.JSONDecodeError) as exc:
                    raise HandoffError(str(exc), code="stale_correction", episode_id=episode_id) from exc
        except (OSError, json.JSONDecodeError, HandoffError) as exc:
            if isinstance(exc, HandoffError):
                _add_error(result, exc)
            else:
                _add_error(result, HandoffError(str(exc), code="artifact_integrity_failure"))
    return normalized


def apply_approved_corrections(transcript: dict[str, Any], correction: dict[str, Any]) -> dict[str, Any]:
    """Return a corrected in-memory transcript without touching producer data."""
    result = json.loads(json.dumps(transcript))
    spans = {
        str(item.get("source_span_id") or item.get("id") or ""): item
        for item in result.get("segments", [])
        if isinstance(item, dict)
    }
    supported_fields = {"text", "speaker", "start", "end", "start_time", "end_time", "original_text", "llm_reviewed_text"}
    for item in correction.get("accepted_corrections", []):
        field_name = str(item.get("field") or "")
        if field_name not in supported_fields:
            raise HandoffError(f"unsupported correction field {field_name}", code="stale_correction")
        span_id = str(item.get("source_span_id") or "")
        if span_id not in spans:
            raise HandoffError(f"correction references missing source span {span_id}", code="stale_correction")
        after = item.get("after_value", item.get("after"))
        spans[span_id][field_name] = after
    return result


def validate_handoff(
    manifest_path: str | Path,
    *,
    expected_partition_id: str | None = None,
    expected_corpus_id: str | None = None,
    expected_context_type: str | None = None,
    expected_workflow_profile: str | None = None,
    expected_partition_config_fingerprint: str | None = None,
) -> HandoffValidation:
    manifest_path = Path(manifest_path).expanduser().resolve()
    result = HandoffValidation(manifest_path=manifest_path, package_root=manifest_path.parent, manifest=None)
    try:
        if manifest_path.name != "manifest.json":
            raise HandoffError("managed handoff entry point must be named manifest.json", code="invalid_manifest", path=str(manifest_path))
        if not manifest_path.is_file():
            raise HandoffError("manifest.json is missing", code="invalid_manifest", path="manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise HandoffError("manifest must be a JSON object", code="invalid_manifest", path="manifest.json")
        result.manifest = manifest
        result.errors.extend(_validate_portable_values(manifest))
        if manifest.get("contract_version") != HANDOFF_CONTRACT_VERSION:
            raise HandoffError(f"unsupported handoff contract {manifest.get('contract_version')}", code="unsupported_contract")
        for field_name in ("handoff_id", "created_at"):
            if not str(manifest.get(field_name) or "").strip():
                raise HandoffError(f"manifest is missing {field_name}", code="invalid_manifest")
        producer = manifest.get("producer")
        if not isinstance(producer, dict) or not producer.get("name") or producer.get("contract_version") not in {"episode-contract-v2", "episode-contract-v2.0"}:
            raise HandoffError("producer contract identity is invalid", code="unsupported_contract")
        partition = manifest.get("partition")
        if not isinstance(partition, dict):
            raise HandoffError("manifest partition is missing", code="mixed_partition_scope")
        partition_id = str(partition.get("partition_id") or "").strip()
        corpus_id = str(partition.get("corpus_id") or "").strip()
        if not partition_id or not corpus_id or partition.get("context_type") not in SUPPORTED_CONTEXT_TYPES or not str(partition.get("workflow_profile") or "").strip():
            raise HandoffError("managed partition identity is incomplete", code="mixed_partition_scope")
        if expected_partition_id and partition_id != expected_partition_id:
            raise HandoffError(f"handoff partition {partition_id} does not match registered partition {expected_partition_id}", code="mixed_partition_scope")
        if expected_corpus_id and corpus_id != expected_corpus_id:
            raise HandoffError(f"handoff corpus {corpus_id} does not match registered corpus {expected_corpus_id}", code="mixed_partition_scope")
        if expected_context_type and partition.get("context_type") != expected_context_type:
            raise HandoffError("handoff context_type does not match the registered partition", code="mixed_partition_scope")
        if expected_workflow_profile and partition.get("workflow_profile") != expected_workflow_profile:
            raise HandoffError("handoff workflow_profile does not match the registered partition", code="mixed_partition_scope")
        declared_partition_config = str(partition.get("config_fingerprint") or "")
        if expected_partition_config_fingerprint and declared_partition_config and declared_partition_config != expected_partition_config_fingerprint:
            raise HandoffError("handoff partition configuration does not match the registered partition", code="mixed_partition_scope")
        integrity = manifest.get("integrity")
        if not isinstance(integrity, dict) or str(integrity.get("algorithm") or "").lower() != "sha256":
            raise HandoffError("manifest integrity must declare sha256", code="invalid_manifest")
        if integrity.get("manifest_hash_excludes_field") != "integrity.manifest_sha256":
            raise HandoffError("manifest integrity must declare the manifest hash exclusion field", code="invalid_manifest")
        declared_manifest_hash = integrity.get("manifest_sha256")
        if not declared_manifest_hash:
            raise HandoffError("manifest integrity is missing manifest_sha256", code="invalid_manifest", path="manifest.json")
        if declared_manifest_hash and _hash_value(declared_manifest_hash, label="manifest hash") != _manifest_hash(manifest):
            result.errors.append({
                "code": "artifact_integrity_failure",
                "message": "manifest_sha256 mismatch",
                "path": "manifest.json",
            })
        episodes = manifest.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise HandoffError("manifest episodes must be a non-empty array", code="invalid_manifest")
        episodes_by_id: dict[str, ManagedEpisode] = {}
        uids: set[str] = set()
        for index, entry in enumerate(episodes):
            episode_id = str(entry.get("episode_id") or "") if isinstance(entry, dict) else ""
            try:
                if not isinstance(entry, dict) or not episode_id:
                    raise HandoffError(f"episodes[{index}] is missing episode_id", code="invalid_manifest")
                episode_uid = str(entry.get("episode_uid") or "")
                expected_uid = f"{partition_id}:{episode_id}"
                if episode_uid != expected_uid:
                    raise HandoffError(f"episode_uid must be {expected_uid}", code="duplicate_identity", episode_id=episode_id)
                if episode_id in episodes_by_id or episode_uid in uids:
                    raise HandoffError(f"duplicate episode identity {episode_uid}", code="duplicate_identity", episode_id=episode_id)
                selected = entry.get("selected_transcript")
                if not isinstance(selected, dict):
                    raise HandoffError("selected_transcript is missing", code="invalid_manifest", episode_id=episode_id)
                selected_path, selected_rel = _relative_artifact(result.package_root, selected.get("path"), label=f"episodes[{index}].selected_transcript.path")
                artifact_hash = _hash_value(selected.get("artifact_sha256"), label="selected transcript artifact hash")
                actual_hash = sha256_file(selected_path)
                if artifact_hash != actual_hash:
                    raise HandoffError("selected transcript artifact hash mismatch", code="artifact_integrity_failure", path=selected_rel, episode_id=episode_id)
                transcript = json.loads(selected_path.read_text(encoding="utf-8"))
                try:
                    parsed = parse_episode_contract_payload(transcript)
                except UpstreamContractError as exc:
                    error_code = "duplicate_identity" if "duplicate source span" in str(exc).lower() else "invalid_transcript"
                    raise HandoffError(str(exc), code=error_code, path=selected_rel, episode_id=episode_id) from exc
                if str(parsed.get("episode_id") or "") != episode_id:
                    raise HandoffError("transcript episode_id disagrees with manifest", code="mixed_partition_scope", path=selected_rel, episode_id=episode_id)
                transcript_metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
                transcript_partition = str(parsed.get("partition_id") or transcript_metadata.get("partition_id") or "")
                transcript_corpus = str(parsed.get("corpus_id") or transcript_metadata.get("corpus_id") or "")
                if transcript_partition and transcript_partition != partition_id or transcript_corpus and transcript_corpus != corpus_id:
                    raise HandoffError("transcript partition/corpus disagrees with manifest", code="mixed_partition_scope", path=selected_rel, episode_id=episode_id)
                _validate_segment_payload(parsed, episode_id=episode_id)
                canonical_hash = _hash_value(selected.get("canonical_payload_sha256"), label="canonical transcript hash")
                if canonical_hash != sha256_bytes(canonical_payload(transcript)):
                    raise HandoffError("selected transcript canonical payload hash mismatch", code="artifact_integrity_failure", path=selected_rel, episode_id=episode_id)
                source_audio = entry.get("source_audio")
                source_audio_fingerprint = str(source_audio.get("fingerprint") or "") if isinstance(source_audio, dict) else ""
                if not source_audio_fingerprint:
                    raise HandoffError("source_audio fingerprint is missing", code="invalid_manifest", episode_id=episode_id)
                if entry.get("stable") is not True:
                    raise HandoffError("episode source is not stable", code="source_not_stable", episode_id=episode_id)
                managed = ManagedEpisode(
                    episode_id=episode_id,
                    episode_uid=episode_uid,
                    title=entry.get("episode_title"),
                    selected_variant=str(selected.get("variant") or ""),
                    selected_path=selected_path,
                    selected_relative_path=selected_rel,
                    artifact_sha256=artifact_hash,
                    canonical_payload_sha256=canonical_hash,
                    source_audio_fingerprint=source_audio_fingerprint,
                    correction_set_id=entry.get("correction_set_id"),
                    stable=True,
                    manifest_entry=dict(entry),
                    transcript_payload=parsed,
                    raw_transcript_payload=dict(transcript),
                    identity={
                        "partition_id": partition_id,
                        "corpus_id": corpus_id,
                        "partition_display_name": partition.get("display_name"),
                        "context_type": partition.get("context_type"),
                        "workflow_profile": partition.get("workflow_profile"),
                        "partition_config_fingerprint": partition.get("config_fingerprint"),
                        "handoff_id": manifest.get("handoff_id"),
                        "episode_id": episode_id,
                        "episode_uid": episode_uid,
                        "selected_variant": selected.get("variant"),
                        "selected_transcript_artifact_sha256": artifact_hash,
                        "selected_transcript_canonical_payload_sha256": canonical_hash,
                        "source_audio_fingerprint": source_audio_fingerprint,
                        "correction_set_id": entry.get("correction_set_id"),
                    },
                )
                episodes_by_id[episode_id] = managed
                uids.add(episode_uid)
                result.episodes.append(managed)
            except (OSError, json.JSONDecodeError, HandoffError) as exc:
                if isinstance(exc, HandoffError):
                    _add_error(result, exc, episode_id=episode_id or None)
                else:
                    _add_error(result, HandoffError(str(exc), code="artifact_integrity_failure"), episode_id=episode_id or None)
        normalized_corrections = _validate_correction_sets(result, manifest.get("correction_sets", []), episodes_by_id)
        result.corrections = normalized_corrections
        for correction_id, correction in normalized_corrections.items():
            for episode in episodes_by_id.values():
                if episode.correction_set_id == correction_id:
                    try:
                        episode.transcript_payload = apply_approved_corrections(
                            episode.raw_transcript_payload,
                            parse_correction_fixture_payload({"manifest": correction["manifest"]}),
                        )
                    except HandoffError as exc:
                        exc.episode_id = episode.episode_id
                        _add_error(result, exc, episode_id=episode.episode_id)
        declared_correction_ids = {
            str(item.correction_set_id)
            for item in episodes_by_id.values()
            if item.correction_set_id not in (None, "")
        }
        known_correction_ids = {
            str(item.get("manifest", {}).get("correction_set_id") or key)
            for key, item in normalized_corrections.items()
        }
        # A correction_set_id on an episode is meaningful only when the
        # corresponding package correction artifact was declared.
        declared_entries = manifest.get("correction_sets") or []
        known_correction_ids.update(
            str(entry.get("correction_set_id"))
            for entry in declared_entries
            if isinstance(entry, dict) and entry.get("correction_set_id")
        )
        for correction_id in sorted(declared_correction_ids - known_correction_ids):
            result.errors.append({
                "code": "stale_correction",
                "message": f"episode references undeclared correction_set_id {correction_id}",
                "episode_id": next((item.episode_id for item in episodes_by_id.values() if item.correction_set_id == correction_id), None),
            })
        provenance_entries = manifest.get("provenance", [])
        if provenance_entries not in (None, []):
            if not isinstance(provenance_entries, list):
                result.errors.append({"code": "invalid_manifest", "message": "provenance must be an array", "path": "provenance"})
            else:
                for index, entry in enumerate(provenance_entries):
                    if not isinstance(entry, dict) or not entry.get("path"):
                        result.errors.append({"code": "invalid_manifest", "message": f"provenance[{index}] must declare path", "path": f"provenance[{index}]"})
                        continue
                    try:
                        artifact, relative = _relative_artifact(result.package_root, entry.get("path"), label=f"provenance[{index}].path")
                        expected = entry.get("artifact_sha256")
                        if not expected or _hash_value(expected, label="provenance artifact hash") != sha256_file(artifact):
                            raise HandoffError("provenance artifact hash mismatch", code="artifact_integrity_failure", path=relative)
                    except (OSError, HandoffError) as exc:
                        _add_error(result, exc if isinstance(exc, HandoffError) else HandoffError(str(exc), code="artifact_integrity_failure"))
    except (OSError, json.JSONDecodeError, HandoffError) as exc:
        if isinstance(exc, HandoffError):
            _add_error(result, exc)
        else:
            _add_error(result, HandoffError(str(exc), code="invalid_manifest"))
    return result


def discover_handoffs(inbox: str | Path) -> list[HandoffValidation]:
    root = Path(inbox).expanduser().resolve()
    if not root.exists():
        return []
    manifests = sorted(root.glob("*/manifest.json"))
    results = [validate_handoff(path) for path in manifests]
    by_id: dict[str, list[HandoffValidation]] = {}
    for result in results:
        if result.handoff_id:
            by_id.setdefault(result.handoff_id, []).append(result)
    for handoff_id, duplicates in by_id.items():
        if len(duplicates) > 1:
            for result in duplicates:
                result.errors.append({
                    "code": "duplicate_identity",
                    "message": f"handoff_id {handoff_id} is published more than once in the inbox",
                    "path": str(result.manifest_path),
                })
    return results


def episode_context(validation: HandoffValidation, episode: ManagedEpisode) -> dict[str, Any]:
    context = dict(episode.identity)
    metadata = episode.transcript_payload.get("metadata") if isinstance(episode.transcript_payload.get("metadata"), dict) else {}
    context["handoff_root"] = str(validation.package_root)
    context["selected_transcript_path"] = str(episode.selected_path)
    context["selected_transcript_relative_path"] = episode.selected_relative_path
    context["episode_title"] = episode.title or metadata.get("episode_title")
    context["episode_date"] = episode.manifest_entry.get("episode_date") or metadata.get("episode_date") or episode.transcript_payload.get("episode_date")
    return context


def materialize_corrected_transcript(episode: ManagedEpisode, destination_dir: Path) -> Path:
    """Persist a consumer-owned corrected view for the processing run."""
    if not episode.correction_set_id:
        return episode.selected_path
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode.episode_id).strip("._") or "episode"
    destination = destination_dir / f"{safe_id}.{episode.correction_set_id}.normalized.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(episode.transcript_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(destination)
    return destination


def build_release_manifest(*, release_id: str, partition: Any, handoff_ids: list[str], episodes: list[ManagedEpisode], caches: list[dict[str, Any]], representation_profile: str, embedding_model: str, embedding_dimension: int | None = None) -> dict[str, Any]:
    cache_fingerprints = [str(item.get("source_fingerprint") or item.get("cache_fingerprint") or "") for item in caches]
    return {
        "release_contract_version": RELEASE_CONTRACT_VERSION,
        "release_id": release_id,
        "partition_id": partition.partition_id,
        "corpus_id": partition.corpus_id,
        "partition_display_name": partition.display_name,
        "context_type": partition.context_type,
        "workflow_profile": partition.workflow_profile,
        "handoff_ids": sorted(set(handoff_ids)),
        "episode_uids": sorted(item.episode_uid for item in episodes),
        "cache_schema_version": "2.1",
        "representation_profile": representation_profile,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "processed_cache_fingerprints": sorted(set(cache_fingerprints)),
    }


def validate_release_manifest(payload: Any, *, expected_partition_id: str | None = None, expected_corpus_id: str | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["release manifest must be an object"]
    if payload.get("release_contract_version") != RELEASE_CONTRACT_VERSION:
        errors.append("unsupported release contract")
    for field_name in ("release_id", "partition_id", "corpus_id", "handoff_ids", "episode_uids"):
        if field_name not in payload:
            errors.append(f"release is missing {field_name}")
    if expected_partition_id and payload.get("partition_id") != expected_partition_id:
        errors.append("release partition does not match requested partition")
    if expected_corpus_id and payload.get("corpus_id") != expected_corpus_id:
        errors.append("release corpus does not match requested partition")
    uids = payload.get("episode_uids")
    if not isinstance(uids, list) or len(uids) != len(set(map(str, uids))):
        errors.append("release contains duplicate episode_uids")
    prefix = str(payload.get("partition_id") or "") + ":"
    if isinstance(uids, list) and any(not str(uid).startswith(prefix) for uid in uids):
        errors.append("release contains an episode_uid from another partition")
    return errors
