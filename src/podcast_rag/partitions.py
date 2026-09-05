"""Managed processing-partition registry and storage layout.

The registry is deliberately small and machine-generated.  Operators should
use the CLI (or the PowerShell launcher) instead of editing the JSON files.
Partition identity is immutable; descriptive fields and approved processing
overrides are the only mutable parts of a managed partition.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PARTITION_REGISTRY_VERSION = "partition-registry-1.0"
PARTITION_MANIFEST_VERSION = "podcast-rag-partition-1.0"
MANAGED_STATUSES = {"active", "archived"}
CONTEXT_TYPES = {"podcast", "meeting", "custom"}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


class PartitionError(ValueError):
    """Raised when a partition operation would violate isolation rules."""


def validate_partition_id(value: str, label: str = "partition ID") -> str:
    value = str(value or "").strip()
    if not ID_RE.fullmatch(value):
        raise PartitionError(
            f"{label} must use 1-80 lowercase letters, digits, '.', '_' or '-' "
            "and must start with a letter or digit"
        )
    return value


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_relative(value: str, label: str) -> str:
    path = Path(str(value or ""))
    if path.is_absolute() or ".." in path.parts:
        raise PartitionError(f"{label} must be a relative path inside the project")
    return path.as_posix()


@dataclass
class PartitionSpec:
    partition_id: str
    display_name: str
    context_type: str
    workflow_profile: str
    corpus_id: str | None = None
    handoff_inbox: str | None = None
    status: str = "active"
    description: str = ""
    owner: str = ""
    tags: list[str] = field(default_factory=list)
    privacy: dict[str, Any] = field(default_factory=dict)
    retention: dict[str, Any] = field(default_factory=dict)
    approved_overrides: dict[str, Any] = field(default_factory=dict)
    legacy: bool = False
    legacy_source_dir: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    config_fingerprint: str | None = None

    def __post_init__(self) -> None:
        self.partition_id = validate_partition_id(self.partition_id)
        self.corpus_id = validate_partition_id(self.corpus_id or self.partition_id, "corpus ID")
        self.display_name = str(self.display_name or "").strip()
        if not self.display_name:
            raise PartitionError("display name cannot be empty")
        self.context_type = str(self.context_type or "").strip().lower()
        if self.context_type not in CONTEXT_TYPES:
            raise PartitionError(f"context_type must be one of {sorted(CONTEXT_TYPES)}")
        self.workflow_profile = str(self.workflow_profile or "").strip()
        if not self.workflow_profile:
            raise PartitionError("workflow_profile cannot be empty")
        if self.status not in MANAGED_STATUSES:
            raise PartitionError(f"status must be one of {sorted(MANAGED_STATUSES)}")
        if self.legacy and self.handoff_inbox and Path(str(self.handoff_inbox)).is_absolute():
            self.handoff_inbox = str(Path(str(self.handoff_inbox)).expanduser().resolve())
        else:
            self.handoff_inbox = _safe_relative(
                self.handoff_inbox or f"partitions/{self.partition_id}/handoff_inbox",
                "handoff_inbox",
            )
        self.tags = sorted({str(item).strip() for item in self.tags if str(item).strip()})
        self.created_at = self.created_at or _now()
        self.updated_at = self.updated_at or self.created_at
        self.config_fingerprint = self.config_fingerprint or self.compute_config_fingerprint()

    def compute_config_fingerprint(self) -> str:
        return _stable_hash(
            {
                "partition_id": self.partition_id,
                "corpus_id": self.corpus_id,
                "context_type": self.context_type,
                "workflow_profile": self.workflow_profile,
                "approved_overrides": self.approved_overrides,
                "privacy": self.privacy,
                "retention": self.retention,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "corpus_id": self.corpus_id,
            "display_name": self.display_name,
            "context_type": self.context_type,
            "workflow_profile": self.workflow_profile,
            "handoff_inbox": self.handoff_inbox,
            "status": self.status,
            "description": self.description,
            "owner": self.owner,
            "tags": list(self.tags),
            "privacy": dict(self.privacy),
            "retention": dict(self.retention),
            "approved_overrides": dict(self.approved_overrides),
            "legacy": bool(self.legacy),
            "legacy_source_dir": self.legacy_source_dir,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config_fingerprint": self.config_fingerprint or self.compute_config_fingerprint(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PartitionSpec":
        if not isinstance(payload, dict):
            raise PartitionError("partition entry must be an object")
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})

    def root(self, project_dir: Path) -> Path:
        if self.legacy and self.legacy_source_dir:
            return Path(self.legacy_source_dir).expanduser()
        return project_dir / "partitions" / self.partition_id

    def paths(self, project_dir: Path) -> dict[str, Path]:
        root = self.root(project_dir)
        return {
            "root": root,
            "partition_manifest": root / "partition.json",
            "handoff_inbox": project_dir / self.handoff_inbox if not self.legacy else root,
            "processed_data": root / "processed_data",
            "state": root / "state",
            "checkpoints": root / "state" / "file_checkpoints",
            "run_reports": root / "state" / "run_reports",
            "topic_contributions": root / "state" / "topic_contributions",
            "topic_index": root / "state" / "topic_index.json",
            "debug_output": root / "debug_output",
            "errata": root / "errata",
            "releases": root / "releases",
        }

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "contract_version": PARTITION_MANIFEST_VERSION,
            "partition": self.to_dict(),
            "generated_by": "podcast-rag-pipeline",
        }


class PartitionRegistry:
    """Read and update the generated partition registry."""

    def __init__(self, project_dir: Path, registry_path: Path | None = None):
        self.project_dir = project_dir.resolve()
        self.path = (registry_path or self.project_dir / "partitions" / "registry.json").resolve()
        self.active_partition_id: str | None = None
        self.partitions: dict[str, PartitionSpec] = {}
        self.load()

    def load(self) -> "PartitionRegistry":
        self.partitions = {}
        self.active_partition_id = None
        if not self.path.exists():
            return self
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PartitionError(f"could not read partition registry {self.path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PartitionError("partition registry must be a JSON object")
        version = str(payload.get("registry_version") or payload.get("version") or "")
        if version and version != PARTITION_REGISTRY_VERSION:
            raise PartitionError(f"unsupported partition registry version {version}")
        raw_active = payload.get("active_partition_id")
        self.active_partition_id = str(raw_active).strip() if raw_active not in (None, "") else None
        entries = payload.get("partitions") or []
        if not isinstance(entries, list):
            raise PartitionError("partition registry partitions must be an array")
        for raw in entries:
            item = PartitionSpec.from_dict(raw)
            if item.partition_id in self.partitions:
                raise PartitionError(f"duplicate partition ID {item.partition_id}")
            self.partitions[item.partition_id] = item
        self.validate()
        return self

    def validate(self) -> list[str]:
        errors: list[str] = []
        names: dict[str, str] = {}
        paths: dict[str, str] = {}
        for partition_id, item in self.partitions.items():
            try:
                normalized = PartitionSpec.from_dict(item.to_dict())
            except PartitionError as exc:
                errors.append(f"{partition_id}: {exc}")
            else:
                if item.config_fingerprint != normalized.compute_config_fingerprint():
                    errors.append(f"{partition_id}: config_fingerprint does not match immutable registry settings")
            name_key = item.display_name.casefold()
            if name_key in names and names[name_key] != partition_id:
                errors.append(f"duplicate display name {item.display_name!r} for {names[name_key]} and {partition_id}")
            names[name_key] = partition_id
            inbox = str((self.project_dir / item.handoff_inbox).resolve()) if not item.legacy else str(item.root(self.project_dir).resolve())
            if inbox in paths and paths[inbox] != partition_id:
                errors.append(f"handoff path collision between {paths[inbox]} and {partition_id}: {inbox}")
            paths[inbox] = partition_id
        if self.active_partition_id and self.active_partition_id not in self.partitions:
            errors.append(f"active partition does not exist: {self.active_partition_id}")
        if self.active_partition_id and self.partitions[self.active_partition_id].status == "archived":
            errors.append("active partition cannot be archived")
        return errors

    def require(self, partition_id: str) -> PartitionSpec:
        partition_id = validate_partition_id(partition_id)
        try:
            return self.partitions[partition_id]
        except KeyError as exc:
            raise PartitionError(f"unknown partition: {partition_id}") from exc

    def active(self) -> PartitionSpec:
        if not self.active_partition_id:
            raise PartitionError("no active partition is configured; use 'partitions use' first")
        return self.require(self.active_partition_id)

    def save(self) -> None:
        errors = self.validate()
        if errors:
            raise PartitionError("invalid partition registry: " + "; ".join(errors))
        payload = {
            "registry_version": PARTITION_REGISTRY_VERSION,
            "active_partition_id": self.active_partition_id,
            "updated_at": _now(),
            "partitions": [self.partitions[key].to_dict() for key in sorted(self.partitions)],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def write_partition_manifest(self, spec: PartitionSpec) -> Path:
        path = spec.paths(self.project_dir)["partition_manifest"]
        if spec.legacy:
            # Legacy adoption is intentionally read-only with respect to the
            # producer-owned source tree.  The registry is the only managed
            # metadata written for this compatibility entry.
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(spec.manifest_payload(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        temp.replace(path)
        return path

    def create(self, spec: PartitionSpec, *, make_active: bool = False) -> PartitionSpec:
        if spec.partition_id in self.partitions:
            raise PartitionError(f"partition ID already exists: {spec.partition_id}")
        if any(item.display_name.casefold() == spec.display_name.casefold() for item in self.partitions.values()):
            raise PartitionError(f"display name already exists: {spec.display_name}")
        if any(item.corpus_id == spec.corpus_id for item in self.partitions.values()):
            raise PartitionError(f"corpus ID already belongs to another partition: {spec.corpus_id}")
        proposed_inbox = str(spec.paths(self.project_dir)["handoff_inbox"].resolve())
        existing_inboxes = {
            str(item.paths(self.project_dir)["handoff_inbox"].resolve())
            for item in self.partitions.values()
        }
        if proposed_inbox in existing_inboxes:
            raise PartitionError(f"handoff inbox path already belongs to another partition: {proposed_inbox}")
        errors = self.validate()
        if errors:
            raise PartitionError("cannot create partition with invalid registry: " + "; ".join(errors))
        spec.updated_at = _now()
        spec.config_fingerprint = spec.compute_config_fingerprint()
        paths = spec.paths(self.project_dir)
        paths["root"].mkdir(parents=True, exist_ok=True)
        for key in ("handoff_inbox", "processed_data", "state", "checkpoints", "run_reports", "topic_contributions", "debug_output", "errata", "releases"):
            paths[key].mkdir(parents=True, exist_ok=True)
        self.partitions[spec.partition_id] = spec
        self.write_partition_manifest(spec)
        if make_active or not self.active_partition_id:
            self.active_partition_id = spec.partition_id
        self.save()
        return spec

    def update(self, partition_id: str, **changes: Any) -> PartitionSpec:
        current = self.require(partition_id)
        immutable = {"partition_id", "corpus_id", "legacy", "legacy_source_dir"}
        if immutable.intersection(changes):
            raise PartitionError("partition_id, corpus_id, and legacy identity cannot be changed after creation")
        payload = current.to_dict()
        payload.update({key: value for key, value in changes.items() if value is not None})
        payload["updated_at"] = _now()
        updated = PartitionSpec.from_dict(payload)
        updated.config_fingerprint = updated.compute_config_fingerprint()
        self.partitions[updated.partition_id] = updated
        errors = self.validate()
        if errors:
            self.partitions[current.partition_id] = current
            raise PartitionError("invalid partition update: " + "; ".join(errors))
        self.write_partition_manifest(updated)
        self.save()
        return updated

    def use(self, partition_id: str) -> PartitionSpec:
        spec = self.require(partition_id)
        if spec.status == "archived":
            raise PartitionError("an archived partition cannot be active")
        self.active_partition_id = spec.partition_id
        self.save()
        return spec

    def archive(self, partition_id: str, archived: bool = True) -> PartitionSpec:
        spec = self.require(partition_id)
        if archived and self.active_partition_id == spec.partition_id:
            raise PartitionError("select another active partition before archiving the active partition")
        return self.update(partition_id, status="archived" if archived else "active")

    def adopt_legacy(self, partition_id: str, source_dir: Path, *, display_name: str | None = None) -> PartitionSpec:
        if partition_id in self.partitions:
            raise PartitionError(f"partition ID already exists: {partition_id}")
        source_dir = source_dir.expanduser().resolve()
        if not source_dir.exists() or not source_dir.is_dir():
            raise PartitionError(f"legacy source directory does not exist: {source_dir}")
        root_fingerprint = hashlib.sha256(str(source_dir).encode("utf-8")).hexdigest()[:16]
        spec = PartitionSpec(
            partition_id=partition_id,
            corpus_id=f"legacy-{root_fingerprint}",
            display_name=display_name or partition_id,
            context_type="custom",
            workflow_profile="legacy",
            legacy=True,
            legacy_source_dir=str(source_dir),
            handoff_inbox=str(source_dir),
        )
        self.partitions[spec.partition_id] = spec
        if not self.active_partition_id:
            self.active_partition_id = spec.partition_id
        self.save()
        return spec

    def ensure_layout(self, spec: PartitionSpec) -> dict[str, Path]:
        paths = spec.paths(self.project_dir)
        for key in ("root", "handoff_inbox", "processed_data", "state", "checkpoints", "run_reports", "topic_contributions", "debug_output", "errata", "releases"):
            paths[key].mkdir(parents=True, exist_ok=True)
        return paths

    def status_rows(self) -> list[dict[str, Any]]:
        rows = []
        for spec in sorted(self.partitions.values(), key=lambda item: item.partition_id):
            paths = spec.paths(self.project_dir)
            rows.append(
                {
                    "partition_id": spec.partition_id,
                    "corpus_id": spec.corpus_id,
                    "display_name": spec.display_name,
                    "context_type": spec.context_type,
                    "workflow_profile": spec.workflow_profile,
                    "status": spec.status,
                    "legacy": spec.legacy,
                    "active": spec.partition_id == self.active_partition_id,
                    "handoff_inbox": str(paths["handoff_inbox"]),
                    "processed_data": str(paths["processed_data"]),
                }
            )
        return rows


def partition_registry(project_dir: Path, registry_path: Path | None = None) -> PartitionRegistry:
    return PartitionRegistry(project_dir, registry_path)


def effective_partition_config(config: Any, project_dir: Path, spec: PartitionSpec) -> tuple[Any, dict[str, Path]]:
    """Return a copy of PipelineConfig rooted entirely in one partition."""
    from dataclasses import replace

    paths = spec.paths(project_dir)
    overrides = dict(spec.approved_overrides or {})
    allowed = {field.name for field in config.__dataclass_fields__.values()}
    values: dict[str, Any] = {
        "input_dir": str(paths["handoff_inbox"]),
        "processed_dir": str(paths["root"] / "processed"),
        "processed_data_dir": str(paths["processed_data"]),
        "state_path": str(paths["state"] / "podcast_rag_state.json"),
        "stop_file": str(paths["state"] / "stop_after_current.txt"),
        "control_file": str(paths["state"] / "pipeline_control.json"),
        "checkpoint_dir": str(paths["checkpoints"]),
        "run_report_dir": str(paths["run_reports"]),
        "run_snapshot_path": str(paths["state"] / "current_run_snapshot.json"),
        "topic_contribution_dir": str(paths["topic_contributions"]),
        "topic_index_path": str(paths["topic_index"]),
        "debug_output_dir": str(paths["debug_output"]),
        "errata_dir": str(paths["errata"]),
        "move_processed_files": False,
    }
    forbidden = {
        "partition_registry_path",
        "input_dir",
        "processed_dir",
        "processed_data_dir",
        "state_path",
        "stop_file",
        "control_file",
        "checkpoint_dir",
        "run_report_dir",
        "run_snapshot_path",
        "topic_contribution_dir",
        "topic_index_path",
        "debug_output_dir",
        "errata_dir",
        "lm_studio_api_key",
    }
    blocked = sorted(set(overrides).intersection(forbidden))
    unknown = sorted(set(overrides).difference(allowed))
    if blocked:
        raise PartitionError("partition overrides cannot replace managed paths or credentials: " + ", ".join(blocked))
    if unknown:
        raise PartitionError("unknown partition processing override(s): " + ", ".join(unknown))
    values.update({key: value for key, value in overrides.items() if key in allowed and key not in forbidden})
    return replace(config, **values), paths


def processing_key(identity: dict[str, Any], *, config_fingerprint: str, generation_fingerprint: str, representation_fingerprint: str) -> str:
    payload = {
        "partition_id": identity.get("partition_id"),
        "episode_id": identity.get("episode_id"),
        "episode_uid": identity.get("episode_uid"),
        "handoff_id": identity.get("handoff_id"),
        "selected_transcript_artifact_sha256": identity.get("selected_transcript_artifact_sha256"),
        "selected_transcript_canonical_payload_sha256": identity.get("selected_transcript_canonical_payload_sha256"),
        "source_audio_fingerprint": identity.get("source_audio_fingerprint"),
        "correction_set_id": identity.get("correction_set_id"),
        "pipeline_version": identity.get("pipeline_version"),
        "prompt_manifest": identity.get("prompt_manifest"),
        "config_fingerprint": config_fingerprint,
        "generation_config_fingerprint": generation_fingerprint,
        "representation_config_fingerprint": representation_fingerprint,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
