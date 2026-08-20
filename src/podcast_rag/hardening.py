"""M6 processing resilience, offline diagnostics, and state recovery."""

from __future__ import annotations
import argparse, hashlib, json, os, shutil, tempfile, zipfile
from pathlib import Path
from typing import Any
from .config import load_config, resolve_path
from .m6_preflight import build_preflight, write_report
from .schema import validate_processed_cache

BACKUP_CONTRACT = "podcast-rag-state-backup-1.0"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def inspect_resilience(project: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    processed = resolve_path(project, config.processed_data_dir)
    state = resolve_path(project, config.state_path)
    checkpoints = resolve_path(project, config.checkpoint_dir)
    rows = []
    total = 0
    for path in sorted(processed.glob("*.processed_documents.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            validation = validate_processed_cache(payload)
            status = "valid" if validation.valid else "invalid"
            errors = list(validation.errors)
        except Exception as exc:
            status = "invalid"
            errors = [f"{type(exc).__name__}: {exc}"]
        size = path.stat().st_size
        total += size
        rows.append(
            {"file": path.name, "status": status, "bytes": size, "errors": errors[:5]}
        )
    state_value = (
        json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {}
    )
    report = {
        "contract_version": "podcast-rag-resilience-1.0",
        "cache": {
            "count": len(rows),
            "bytes": total,
            "valid": sum(x["status"] == "valid" for x in rows),
            "invalid": sum(x["status"] == "invalid" for x in rows),
            "files": rows,
        },
        "state": {
            "present": state.is_file(),
            "entries": len(state_value.get("files") or state_value),
        },
        "checkpoints": {
            "count": (
                sum(1 for x in checkpoints.rglob("*") if x.is_file())
                if checkpoints.exists()
                else 0
            )
        },
        "execution": {
            "max_parallel_model_requests": config.max_parallel_model_requests,
            "offline_deterministic_operations": [
                "inspect",
                "backfill-representations",
                "export-dense-baseline",
                "export-representation-corpus",
                "build-temporal-artifacts",
            ],
            "model_downloads_implicit": False,
        },
        "capacity": {
            "estimated_rebuild_bytes": max(total * 2, total + 100_000_000),
            "free_bytes": shutil.disk_usage(
                processed if processed.exists() else project
            ).free,
        },
    }
    report["blockers"] = (
        ["invalid_processed_cache"] if report["cache"]["invalid"] else []
    )
    report["report_id"] = "rag_resilience_" + _hash(report)
    return report


def create_backup(
    project: Path, config_path: Path, destination: Path
) -> dict[str, Any]:
    config = load_config(config_path)
    items = []
    candidates = [
        config_path,
        resolve_path(project, config.state_path),
        resolve_path(project, config.stop_file),
        resolve_path(project, config.run_snapshot_path),
    ]
    checkpoint = resolve_path(project, config.checkpoint_dir)
    if checkpoint.exists():
        candidates.extend(path for path in checkpoint.rglob("*") if path.is_file())
    for path in candidates:
        if path.is_file():
            resolved = path.resolve()
            name = (
                "project/" + resolved.relative_to(project.resolve()).as_posix()
                if project.resolve() in resolved.parents
                else "config/" + resolved.name
            )
            items.append((name, resolved))
    rows = [
        {
            "path": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for name, path in items
    ]
    manifest = {"contract_version": BACKUP_CONTRACT, "entries": rows}
    manifest["backup_id"] = "rag_backup_" + _hash(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        for name, path in items:
            archive.write(path, name)
    return {**manifest, "path": str(destination)}


def inspect_backup(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        errors = []
        if manifest.get("contract_version") != BACKUP_CONTRACT:
            errors.append("unsupported_contract")
        for row in manifest.get("entries") or []:
            try:
                data = archive.read(row["path"])
            except KeyError:
                errors.append("missing:" + row["path"])
                continue
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                errors.append("checksum:" + row["path"])
        return {
            "valid": not errors,
            "errors": errors,
            "backup_id": manifest.get("backup_id"),
            "entry_count": len(manifest.get("entries") or []),
        }


def restore_backup(
    path: Path, project: Path, *, approved_backup_id: str
) -> dict[str, Any]:
    check = inspect_backup(path)
    if not check["valid"]:
        raise ValueError("invalid backup: " + ", ".join(check["errors"]))
    if check["backup_id"] != approved_backup_id:
        raise PermissionError("approved backup identity does not match")
    restored = []
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for row in manifest["entries"]:
            relative = str(row["path"]).split("/", 1)[1]
            target = (project.resolve() / relative).resolve()
            if project.resolve() not in target.parents:
                raise ValueError("unsafe restore member")
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(archive.read(row["path"]))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            restored.append(relative)
    return {"backup_id": approved_backup_id, "restored": restored}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="M6 Podcast RAG resilience operations; no model downloads."
    )
    c = p.add_subparsers(dest="command", required=True)
    pre = c.add_parser("preflight")
    pre.add_argument("--workspace", required=True)
    pre.add_argument("--output")
    inspect = c.add_parser("inspect")
    inspect.add_argument("--project", required=True)
    inspect.add_argument("--config", required=True)
    backup = c.add_parser("backup")
    backup.add_argument("--project", required=True)
    backup.add_argument("--config", required=True)
    backup.add_argument("--destination", required=True)
    verify = c.add_parser("inspect-backup")
    verify.add_argument("path")
    restore = c.add_parser("restore")
    restore.add_argument("path")
    restore.add_argument("--project", required=True)
    restore.add_argument("--approve", required=True)
    a = p.parse_args(argv)
    if a.command == "preflight":
        value = build_preflight(
            "Podcast-RAG-pipeline",
            Path(a.workspace),
            optional_modules=("langchain", "sentence_transformers"),
            service_ports={"LM Studio": 1234},
        )
        write_report(Path(a.output), value) if a.output else None
    elif a.command == "inspect":
        value = inspect_resilience(Path(a.project).resolve(), Path(a.config).resolve())
    elif a.command == "backup":
        value = create_backup(
            Path(a.project).resolve(), Path(a.config).resolve(), Path(a.destination)
        )
    elif a.command == "inspect-backup":
        value = inspect_backup(Path(a.path))
    else:
        value = restore_backup(
            Path(a.path), Path(a.project), approved_backup_id=a.approve
        )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
