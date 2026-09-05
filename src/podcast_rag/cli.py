from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import signal
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import podcast_rag.runtime as runtime
from podcast_rag.config import PipelineConfig, apply_env_overrides, load_config, resolve_path
from podcast_rag.evaluation import evaluate_retrieval_run
from podcast_rag.errata import ErrataRecorder, errata_json_paths, validate_errata_payload, write_errata_artifacts
from podcast_rag.llm_support import test_model_inference, verify_model_available
from podcast_rag.pipeline import PodcastRagPipeline
from podcast_rag.runtime import PIPELINE_VERSION, PROMPT_VERSION, PipelineInterrupted, RunStats, RuntimeControl, request_stop
from podcast_rag.schema import dumps_schema_summary, validate_processed_cache, validate_processed_documents
from podcast_rag.state import (
    load_state,
    mark_state,
    maybe_move_processed,
    processed_data_cache_path,
    quarantine_invalid_cache,
    read_json_file,
    save_state,
    write_json_file,
    write_run_reports,
    write_run_snapshot,
    should_skip_file,
    backfill_cache_file,
    export_dense_baseline,
    export_representation_corpus,
)
from podcast_rag.text_utils import (
    deterministic_topic_tags,
    format_duration,
    estimate_remaining_seconds,
    file_fingerprint,
    is_missing_context_response,
    token_estimate,
)
from podcast_rag.topics import refresh_topic_index
from podcast_rag.temporal_artifacts import build_temporal_artifacts
from podcast_rag.advanced_retrieval import build_evidence_graph, build_late_chunk_alignment
from podcast_rag.transcript import iter_transcript_files, load_transcript_json, partition_scope
from podcast_rag.handoff import (
    HandoffError,
    materialize_corrected_transcript,
    HandoffValidation,
    build_release_manifest,
    discover_handoffs,
    episode_context,
    validate_handoff,
    validate_release_manifest,
)
from podcast_rag.partitions import (
    PartitionError,
    PartitionRegistry,
    PartitionSpec,
    effective_partition_config,
    processing_key,
)
from podcast_rag.representations import RepresentationBuilder
from podcast_rag.config import config_fingerprint, generation_config_fingerprint


def finalize_file_errata(
    pipeline: PodcastRagPipeline,
    recorder: ErrataRecorder | None,
    errata_dir: Path,
) -> dict[str, Any] | None:
    """Finalize advisory diagnosis and persist paired per-file artifacts."""
    if recorder is None:
        return None
    try:
        pipeline.diagnose_errata(recorder)
    except Exception as exc:
        # Diagnosis is advisory; even an unexpected diagnosis bug cannot change
        # the file outcome. Persist the failure as a deterministic finding.
        recorder.record(
            "errata_diagnosis_failed",
            "warning",
            "errata",
            "The advisory diagnosis stage raised an unexpected exception.",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        recorder.set_diagnosis({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    try:
        return write_errata_artifacts(recorder, errata_dir)
    except Exception as exc:
        print(f"  Errata artifact write failed (non-fatal): {type(exc).__name__}: {exc}")
        return None


def inspect_errata(config: PipelineConfig, project_dir: Path, selected: str | None) -> int:
    """Read-only validation and summary for one errata record or the directory."""
    root = resolve_path(project_dir, config.errata_dir)
    paths = errata_json_paths(root, selected)
    if not paths:
        print(f"No errata JSON artifacts found under {root}")
        return 1
    invalid = 0
    for path in paths:
        try:
            payload = read_json_file(path)
            errors = validate_errata_payload(payload)
        except Exception as exc:
            payload = {}
            errors = [f"{type(exc).__name__}: {exc}"]
        if errors:
            invalid += 1
            print(f"INVALID {path}: {'; '.join(errors[:8])}")
            continue
        findings = payload.get("findings") or []
        diagnosis_status = (payload.get("llm_diagnosis") or {}).get("status", "not_requested")
        print(
            f"{path}: status={payload.get('outcome', {}).get('status')} "
            f"findings={len(findings)} diagnosis={diagnosis_status} "
            f"finding_digest={payload.get('finding_digest', '')}"
        )
        for finding in sorted(findings, key=lambda item: (str(item.get("severity")), str(item.get("finding_id"))))[:5]:
            print(f"  - {finding.get('severity')}: {finding.get('code')} — {finding.get('message')}")
    print(f"Inspected {len(paths)} errata artifact(s); invalid={invalid}")
    return 1 if invalid else 0


def _registry(config: PipelineConfig, project_dir: Path) -> PartitionRegistry:
    return PartitionRegistry(project_dir, resolve_path(project_dir, config.partition_registry_path))


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=True, default=str))


def _parse_overrides(values: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise PartitionError(f"processing override must use key=value: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise PartitionError("processing override key cannot be empty")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[key] = value
    return overrides


def _make_pipeline(config: PipelineConfig, project_dir: Path, control: RuntimeControl, *, load_models: bool) -> PodcastRagPipeline:
    """Construct a full or cache-only pipeline while preserving test doubles."""
    if load_models:
        return PodcastRagPipeline(config, project_dir, control)
    try:
        return PodcastRagPipeline(config, project_dir, control, load_models=False)
    except TypeError as exc:
        # Existing integrations may provide a small compatible pipeline test
        # double with the historical three-argument constructor.
        if "load_models" not in str(exc):
            raise
        return PodcastRagPipeline(config, project_dir, control)


def partition_command(config: PipelineConfig, project_dir: Path, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="podcast-rag partitions")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list")
    create = subparsers.add_parser("create")
    create.add_argument("--id", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--context-type", required=True, choices=["podcast", "meeting", "custom"])
    create.add_argument("--workflow-profile", required=True)
    create.add_argument("--corpus-id")
    create.add_argument("--handoff-inbox")
    create.add_argument("--description", default="")
    create.add_argument("--owner", default="")
    create.add_argument("--tag", action="append", default=[])
    create.add_argument("--privacy", action="append", default=[])
    create.add_argument("--retention", action="append", default=[])
    create.add_argument("--override", action="append", default=[])
    create.add_argument("--make-active", action="store_true")
    show = subparsers.add_parser("show")
    show.add_argument("--id", required=True)
    use = subparsers.add_parser("use")
    use.add_argument("--id", required=True)
    update = subparsers.add_parser("update")
    update.add_argument("--id", required=True)
    update.add_argument("--name")
    update.add_argument("--context-type", choices=["podcast", "meeting", "custom"])
    update.add_argument("--workflow-profile")
    update.add_argument("--handoff-inbox")
    update.add_argument("--description")
    update.add_argument("--owner")
    update.add_argument("--tag", action="append")
    update.add_argument("--privacy", action="append")
    update.add_argument("--retention", action="append")
    update.add_argument("--override", action="append")
    archive = subparsers.add_parser("archive")
    archive.add_argument("--id", required=True)
    archive.add_argument("--restore", action="store_true")
    subparsers.add_parser("doctor")
    adopt = subparsers.add_parser("adopt-legacy")
    adopt.add_argument("--id", required=True)
    adopt.add_argument("--source", required=True)
    adopt.add_argument("--name")
    try:
        args = parser.parse_args(argv)
        registry = _registry(config, project_dir)
        if args.action == "list":
            _print_json({"active_partition_id": registry.active_partition_id, "partitions": registry.status_rows()})
            return 0
        if args.action == "doctor":
            errors = registry.validate()
            for spec in registry.partitions.values():
                paths = spec.paths(project_dir)
                if not paths["root"].exists():
                    errors.append(f"{spec.partition_id}: missing partition root {paths['root']}")
                if not spec.legacy and not paths["handoff_inbox"].exists():
                    errors.append(f"{spec.partition_id}: missing handoff inbox {paths['handoff_inbox']}")
                if not spec.legacy:
                    manifest_path = paths["partition_manifest"]
                    if not manifest_path.is_file():
                        errors.append(f"{spec.partition_id}: missing partition manifest {manifest_path}")
                    else:
                        try:
                            manifest = read_json_file(manifest_path)
                            manifest_partition = manifest.get("partition") if isinstance(manifest, dict) else None
                            if not isinstance(manifest_partition, dict) or manifest_partition.get("partition_id") != spec.partition_id or manifest_partition.get("corpus_id") != spec.corpus_id:
                                errors.append(f"{spec.partition_id}: partition manifest identity does not match the registry")
                        except (OSError, ValueError) as exc:
                            errors.append(f"{spec.partition_id}: unreadable partition manifest: {exc}")
            _print_json({"valid": not errors, "errors": errors, "registry": str(registry.path)})
            return 0 if not errors else 1
        if args.action == "create":
            spec = registry.create(
                PartitionSpec(
                    partition_id=args.id,
                    corpus_id=args.corpus_id,
                    display_name=args.name,
                    context_type=args.context_type,
                    workflow_profile=args.workflow_profile,
                    handoff_inbox=args.handoff_inbox,
                    description=args.description,
                    owner=args.owner,
                    tags=args.tag,
                    privacy=_parse_overrides(args.privacy),
                    retention=_parse_overrides(args.retention),
                    approved_overrides=_parse_overrides(args.override),
                ),
                make_active=args.make_active,
            )
            _print_json(spec.to_dict())
            return 0
        if args.action == "show":
            spec = registry.require(args.id)
            _print_json({**spec.to_dict(), "paths": {key: str(value) for key, value in spec.paths(project_dir).items()}})
            return 0
        if args.action == "use":
            _print_json(registry.use(args.id).to_dict())
            return 0
        if args.action == "update":
            changes = {
                "display_name": args.name,
                "context_type": args.context_type,
                "workflow_profile": args.workflow_profile,
                "handoff_inbox": args.handoff_inbox,
                "description": args.description,
                "owner": args.owner,
                "tags": args.tag,
                "privacy": _parse_overrides(args.privacy) if args.privacy is not None else None,
                "retention": _parse_overrides(args.retention) if args.retention is not None else None,
                "approved_overrides": _parse_overrides(args.override) if args.override is not None else None,
            }
            _print_json(registry.update(args.id, **changes).to_dict())
            return 0
        if args.action == "archive":
            _print_json(registry.archive(args.id, archived=not args.restore).to_dict())
            return 0
        if args.action == "adopt-legacy":
            _print_json(registry.adopt_legacy(args.id, Path(args.source), display_name=args.name).to_dict())
            return 0
    except (PartitionError, OSError, ValueError) as exc:
        print(f"Partition operation failed: {exc}")
        return 1
    return 1


def _print_handoff_validation(result: HandoffValidation) -> None:
    _print_json(
        {
            "valid": result.valid,
            "manifest": str(result.manifest_path),
            "handoff_id": result.handoff_id,
            "partition": result.partition,
            "episodes": [
                {
                    "episode_id": item.episode_id,
                    "episode_uid": item.episode_uid,
                    "selected_variant": item.selected_variant,
                    "selected_path": item.selected_relative_path,
                    "correction_set_id": item.correction_set_id,
                }
                for item in result.episodes
            ],
            "errors": result.errors,
            "warnings": result.warnings,
        }
    )


def _write_handoff_validation_errata(config: PipelineConfig, project_dir: Path, spec: Any, validation: HandoffValidation) -> None:
    """Persist deterministic quarantine records without invoking a model.

    Episode-level failures are attached to the selected transcript when that
    episode was materialized.  Package-level failures (or failures that
    prevent an episode from being materialized, such as a duplicate identity)
    are attached to the manifest so the failed handoff still has an actionable
    record.
    """
    errors_by_episode: dict[str, list[dict[str, Any]]] = {}
    package_errors: list[dict[str, Any]] = []
    for error in validation.errors:
        episode_id = error.get("episode_id")
        if episode_id:
            errors_by_episode.setdefault(str(episode_id), []).append(error)
        else:
            package_errors.append(error)
    if not errors_by_episode and not package_errors:
        return
    paths = spec.paths(project_dir)
    handoff_fingerprint = "handoff-" + str(validation.handoff_id or "unknown")
    targets: list[tuple[Path, str, dict[str, Any], list[dict[str, Any]]]] = []
    for episode in validation.episodes:
        errors = errors_by_episode.get(episode.episode_id)
        if not errors:
            continue
        targets.append(
            (
                episode.selected_path,
                episode.artifact_sha256.removeprefix("sha256:"),
                episode.identity,
                errors,
            )
        )
    represented_episode_ids = {str(item.get("episode_id") or "") for _path, _fingerprint, item, _errors in targets}
    for episode_id, errors in errors_by_episode.items():
        if episode_id in represented_episode_ids:
            continue
        targets.append(
            (
                validation.manifest_path,
                handoff_fingerprint + "-" + episode_id,
                {
                    "partition_id": validation.partition.get("partition_id"),
                    "corpus_id": validation.partition.get("corpus_id"),
                    "partition_display_name": validation.partition.get("display_name"),
                    "context_type": validation.partition.get("context_type"),
                    "workflow_profile": validation.partition.get("workflow_profile"),
                    "handoff_id": validation.handoff_id,
                    "episode_id": episode_id,
                },
                errors,
            )
        )
    if package_errors:
        targets.append(
            (
                validation.manifest_path,
                handoff_fingerprint,
                {
                    "partition_id": validation.partition.get("partition_id"),
                    "corpus_id": validation.partition.get("corpus_id"),
                    "partition_display_name": validation.partition.get("display_name"),
                    "context_type": validation.partition.get("context_type"),
                    "workflow_profile": validation.partition.get("workflow_profile"),
                    "handoff_id": validation.handoff_id,
                },
                package_errors,
            )
        )
    for source_path, source_fingerprint, identity, errors in targets:
        recorder = ErrataRecorder(
            source_path,
            source_fingerprint,
            "handoff-validation-" + str(validation.handoff_id or "unknown"),
            config,
        )
        recorder.update_source(**identity, processing_status="quarantined")
        for error in errors:
            recorder.record(
                str(error.get("code") or "invalid_manifest"),
                "error",
                "handoff_validation",
                str(error.get("message") or "Handoff validation failed."),
                details={"path": error.get("path"), "episode_id": error.get("episode_id")},
                evidence_paths=[error.get("path")] if error.get("path") else None,
            )
        recorder.set_outcome("failed", failed_stage="handoff_validation")
        write_errata_artifacts(recorder, paths["errata"])


def _registered_handoff(config: PipelineConfig, project_dir: Path, manifest: Path, *, require_registry: bool = True) -> tuple[Any, HandoffValidation] | None:
    raw = validate_handoff(manifest)
    if not raw.valid:
        _print_handoff_validation(raw)
        return None
    registry = _registry(config, project_dir)
    partition_id = str(raw.partition.get("partition_id") or "")
    try:
        spec = registry.require(partition_id)
    except PartitionError:
        if require_registry:
            raise
        _print_handoff_validation(raw)
        return None
    if spec.legacy:
        raise PartitionError(f"partition {spec.partition_id} is legacy; export a compliant handoff before managed processing")
    if spec.status == "archived":
        raise PartitionError(f"partition is archived: {partition_id}")
    checked = validate_handoff(
        manifest,
        expected_partition_id=spec.partition_id,
        expected_corpus_id=spec.corpus_id,
        expected_context_type=spec.context_type,
        expected_workflow_profile=spec.workflow_profile,
        expected_partition_config_fingerprint=spec.config_fingerprint,
    )
    _print_handoff_validation(checked)
    if checked.errors:
        _write_handoff_validation_errata(config, Path(project_dir), spec, checked)
    package_errors = [item for item in checked.errors if not item.get("episode_id")]
    if package_errors or not checked.episodes:
        return None
    return spec, checked


def handoff_command(config: PipelineConfig, project_dir: Path, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="podcast-rag handoff")
    subparsers = parser.add_subparsers(dest="action", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    scan = subparsers.add_parser("scan")
    scan.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    target = Path(args.manifest).expanduser()
    if args.action == "validate":
        result = validate_handoff(resolve_path(project_dir, str(target)))
        _print_handoff_validation(result)
        return 0 if result.valid else 1
    path = resolve_path(project_dir, str(target))
    if path.is_file():
        results = [validate_handoff(path)]
    else:
        results = discover_handoffs(path)
    for result in results:
        _print_handoff_validation(result)
    return 0 if results and all(item.valid for item in results) else 1


def managed_process_command(config: PipelineConfig, project_dir: Path, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="podcast-rag process")
    parser.add_argument("--manifest")
    parser.add_argument("--partition")
    parser.add_argument("--episode")
    parser.add_argument("--one-file", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest and args.partition:
        raise SystemExit("--manifest and --partition are mutually exclusive")
    registry = _registry(config, project_dir)
    if args.manifest:
        selected = _registered_handoff(config, project_dir, Path(args.manifest))
        if not selected:
            return 1
        spec, validation = selected
        validations = [validation]
    else:
        spec = registry.require(args.partition) if args.partition else registry.active()
        if spec.status == "archived":
            raise PartitionError(f"partition is archived: {spec.partition_id}")
        if spec.legacy:
            raise PartitionError(f"partition {spec.partition_id} is legacy; use the compatibility adapter or export a compliant handoff")
        validations = []
        for candidate in discover_handoffs(spec.paths(project_dir)["handoff_inbox"]):
            checked = validate_handoff(
                candidate.manifest_path,
                expected_partition_id=spec.partition_id,
                expected_corpus_id=spec.corpus_id,
                expected_context_type=spec.context_type,
                expected_workflow_profile=spec.workflow_profile,
                expected_partition_config_fingerprint=spec.config_fingerprint,
            )
            package_errors = [item for item in checked.errors if not item.get("episode_id")]
            if checked.errors:
                _write_handoff_validation_errata(config, project_dir, spec, checked)
            if checked.episodes and not package_errors:
                validations.append(checked)
            else:
                print(f"Skipping invalid handoff package {candidate.manifest_path}")
                _print_handoff_validation(checked)
    if not validations:
        print("No valid handoff packages are ready for processing.")
        return 1
    effective_config, paths = effective_partition_config(config, project_dir, spec)
    representation_manifest = RepresentationBuilder(
        embedding_text_mode=effective_config.embedding_text_mode,
        lexical_text_mode=effective_config.lexical_text_mode,
        contextual_header_max_chars=effective_config.contextual_header_max_chars,
    ).manifest()
    input_files: list[tuple[Path, dict[str, Any] | None]] = []
    quarantined_state = load_state(paths["state"] / "podcast_rag_state.json")
    quarantined_state_changed = False
    for validation in validations:
        quarantined_episode_ids = {
            str(item.get("episode_id"))
            for item in validation.errors
            if item.get("episode_id")
        }
        for episode in validation.episodes:
            if episode.episode_id in quarantined_episode_ids:
                print(f"Quarantined episode {episode.episode_id} from handoff {validation.handoff_id}: validation failed")
                quarantine_context = episode_context(validation, episode)
                quarantine_context["pipeline_version"] = PIPELINE_VERSION
                quarantine_context["effective_config_fingerprint"] = config_fingerprint(effective_config)
                quarantine_context["generation_config_fingerprint"] = generation_config_fingerprint(effective_config)
                quarantine_context["representation_config_fingerprint"] = str(representation_manifest.get("config_fingerprint") or "")
                quarantine_context["processing_key"] = processing_key(
                    quarantine_context,
                    config_fingerprint=quarantine_context["effective_config_fingerprint"],
                    generation_fingerprint=quarantine_context["generation_config_fingerprint"],
                    representation_fingerprint=quarantine_context["representation_config_fingerprint"],
                )
                quarantine_context["cache_key"] = quarantine_context["processing_key"]
                quarantine_context["status_history"] = ["discovered", "validated", "quarantined"]
                error_details = [item for item in validation.errors if str(item.get("episode_id")) == episode.episode_id]
                mark_state(
                    quarantined_state,
                    str(quarantine_context["processing_key"]),
                    episode.selected_path,
                    "quarantined",
                    {**quarantine_context, "current_stage": "validate", "error": error_details},
                )
                quarantined_state_changed = True
                continue
            if args.episode and episode.episode_id != args.episode:
                continue
            context = episode_context(validation, episode)
            context["pipeline_version"] = PIPELINE_VERSION
            context["prompt_manifest"] = PROMPT_VERSION
            context["effective_config_fingerprint"] = config_fingerprint(effective_config)
            context["generation_config_fingerprint"] = generation_config_fingerprint(effective_config)
            context["representation_config_fingerprint"] = str(representation_manifest.get("config_fingerprint") or "")
            context["processing_key"] = processing_key(
                context,
                config_fingerprint=context["effective_config_fingerprint"],
                generation_fingerprint=context["generation_config_fingerprint"],
                representation_fingerprint=context["representation_config_fingerprint"],
            )
            context["cache_key"] = context["processing_key"]
            context["status_history"] = ["discovered", "validated", "ready"]
            input_path = materialize_corrected_transcript(episode, paths["state"] / "normalized_inputs")
            input_files.append((input_path, context))
    if quarantined_state_changed:
        save_state(paths["state"] / "podcast_rag_state.json", quarantined_state)
    if not input_files:
        print("No declared episodes matched the requested filter.")
        return 1
    print(f"Processing {len(input_files)} declared episode(s) in partition {spec.partition_id}.")
    lock_path = paths["state"] / "active_run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"partition_id": spec.partition_id, "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}))
        try:
            return run_batch(effective_config, project_dir, args.one_file, input_files=input_files)
        finally:
            lock_path.unlink(missing_ok=True)
    except FileExistsError as exc:
        raise PartitionError(f"partition already has an active processing run: {spec.partition_id}") from exc


def status_command(config: PipelineConfig, project_dir: Path, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="podcast-rag status")
    parser.add_argument("--partition", required=True)
    args = parser.parse_args(argv)
    registry = _registry(config, project_dir)
    spec = registry.require(args.partition)
    paths = spec.paths(project_dir)
    state_path = paths["state"] / "podcast_rag_state.json"
    state = load_state(state_path)
    packages = [
        validate_handoff(
            item.manifest_path,
            expected_partition_id=spec.partition_id,
            expected_corpus_id=spec.corpus_id,
            expected_context_type=spec.context_type,
            expected_workflow_profile=spec.workflow_profile,
            expected_partition_config_fingerprint=spec.config_fingerprint,
        )
        for item in discover_handoffs(paths["handoff_inbox"])
    ]
    _print_json(
        {
            "partition": spec.to_dict(),
            "state_path": str(state_path),
            "files": state.get("files", {}),
            "handoffs": [
                {"manifest": str(item.manifest_path), "handoff_id": item.handoff_id, "valid": item.valid, "errors": item.errors, "episode_count": len(item.episodes)}
                for item in packages
            ],
        }
    )
    return 0


def export_partition_command(config: PipelineConfig, project_dir: Path, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="podcast-rag export")
    parser.add_argument("--partition", required=True)
    parser.add_argument("--release", required=True)
    args = parser.parse_args(argv)
    registry = _registry(config, project_dir)
    spec = registry.require(args.partition)
    if spec.legacy:
        raise PartitionError("legacy partitions cannot emit managed releases until an explicit handoff migration is complete")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", args.release):
        raise SystemExit("release ID must be a safe identifier")
    paths = spec.paths(project_dir)
    caches = []
    episodes_by_uid: dict[str, Any] = {}
    handoff_ids: set[str] = set()
    for cache_path in sorted(paths["processed_data"].glob("*.processed_documents.json")):
        payload = read_json_file(cache_path)
        validation = validate_processed_cache(payload)
        validation.raise_for_errors(f"cache {cache_path}")
        if payload.get("partition_id") != spec.partition_id or payload.get("corpus_id") != spec.corpus_id:
            raise SystemExit(f"mixed partition/corpus cache rejected: {cache_path}")
        uid = str(payload.get("episode_uid") or "")
        if uid:
            episodes_by_uid[uid] = {"episode_uid": uid}
        if payload.get("handoff_id"):
            handoff_ids.add(str(payload["handoff_id"]))
        caches.append({"source_fingerprint": payload.get("source_fingerprint"), "cache_path": str(cache_path)})
    release = build_release_manifest(
        release_id=args.release,
        partition=spec,
        handoff_ids=sorted(handoff_ids),
        episodes=[type("EpisodeRef", (), {"episode_uid": uid})() for uid in sorted(episodes_by_uid)],
        caches=caches,
        representation_profile="baseline-v1",
        embedding_model=config.embedding_model,
    )
    errors = validate_release_manifest(release, expected_partition_id=spec.partition_id, expected_corpus_id=spec.corpus_id)
    if errors:
        raise SystemExit("invalid release: " + "; ".join(errors))
    destination = paths["releases"] / f"{args.release}.release.json"
    write_json_file(destination, release)
    _print_json({"release_path": str(destination), **release})
    return 0

def run_batch(
    config: PipelineConfig,
    project_dir: Path,
    one_file: bool,
    *,
    allow_mixed_partitions: bool = False,
    input_files: list[tuple[Path, dict[str, Any] | None]] | None = None,
) -> int:
    """Process every pending transcript, reusing caches and checkpoints when possible."""
    runtime.load_runtime_deps()

    input_dir = resolve_path(project_dir, config.input_dir)
    processed_dir = resolve_path(project_dir, config.processed_dir)
    processed_data_dir = resolve_path(project_dir, config.processed_data_dir)
    state_path = resolve_path(project_dir, config.state_path)
    stop_file = resolve_path(project_dir, config.stop_file)
    snapshot_path = resolve_path(project_dir, config.run_snapshot_path)
    report_dir = resolve_path(project_dir, config.run_report_dir)
    errata_dir = resolve_path(project_dir, config.errata_dir)

    input_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    control = RuntimeControl(config, project_dir)
    control.initialize_file_for_run()

    state = load_state(state_path)
    file_entries = input_files if input_files is not None else [(path, None) for path in iter_transcript_files(input_dir, config.file_glob)]
    files = [path for path, _context in file_entries]
    if input_files is None:
        partition_info = partition_scope(files)
        if not partition_info["valid"] and not allow_mixed_partitions:
            raise RuntimeError(
                "Partition isolation check failed before Podcast-RAG processing: "
                + "; ".join(partition_info["errors"])
                + ". Use --allow-mixed-partitions only for an intentional aggregation."
            )
    else:
        partition_info = {"partition": input_files[0][1] or {}, "valid": True}
    if partition_info.get("partition"):
        print(
            "Processing-space input: "
            + ", ".join(f"{key}={value}" for key, value in sorted(partition_info["partition"].items()))
        )
    elif files and input_files is None:
        legacy_root = hashlib.sha256(str(input_dir.resolve()).encode("utf-8")).hexdigest()[:16]
        print(
            "Legacy input adapter active: this flat transcript directory is read for migration/recovery only; "
            f"synthetic scope=legacy:{legacy_root}. It cannot silently join a managed partition."
        )
    pending = []
    for path, context in file_entries:
        fingerprint = str((context or {}).get("processing_key") or file_fingerprint(path))
        cache_path = processed_data_cache_path(processed_data_dir, fingerprint, path)
        if cache_path.exists() or not should_skip_file(state, fingerprint):
            pending.append((path, fingerprint, context))

    print(f"Found {len(files)} matching files; {len(pending)} pending.")
    if not pending:
        if config.enable_temporal_artifacts:
            build_temporal_artifacts(processed_data_dir, resolve_path(project_dir, config.temporal_artifact_path), include_trajectories=config.enable_temporal_trajectories, include_contradiction_candidates=config.enable_contradiction_candidates, missing_interval_days=config.temporal_missing_interval_days)
        return 0

    cached_pending = [processed_data_cache_path(processed_data_dir, fingerprint, path).exists() for path, fingerprint, _context in pending]
    needs_llm_processing = not all(cached_pending)
    if needs_llm_processing:
        if config.fake_llm:
            print("Fake LLM mode enabled; skipping LM Studio model verification.")
        elif config.verify_model:
            verify_model_available(config)
        if not config.fake_llm and config.test_inference:
            test_model_inference(config)
    else:
        print("All pending files have processed data caches; skipping LM Studio model verification.")

    pipeline = _make_pipeline(config, project_dir, control, load_models=needs_llm_processing)
    batch_started_at = time.time()
    completed_files_this_run = 0
    stats = RunStats()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    stats.files_total = len(pending)
    write_run_snapshot(snapshot_path, stats, pipeline.performance)

    for idx, (path, fingerprint, context) in enumerate(pending, 1):
        if runtime.STOP_REQUESTED or stop_file.exists():
            print("Stop requested before starting next file.")
            break

        cache_path = processed_data_cache_path(processed_data_dir, fingerprint, path)
        pipeline.active_context = context
        file_eta = format_duration(estimate_remaining_seconds(completed_files_this_run, len(pending), time.time() - batch_started_at))
        print(f"\nFile {idx}/{len(pending)} eta_files={file_eta}")

        recorder = (
            ErrataRecorder(
                path,
                str((context or {}).get("selected_transcript_artifact_sha256") or fingerprint),
                run_id,
                config,
            )
            if config.errata_enabled
            else None
        )
        if recorder is not None and context:
            recorder.update_source(
                partition_id=context.get("partition_id"),
                corpus_id=context.get("corpus_id"),
                partition_display_name=context.get("partition_display_name"),
                context_type=context.get("context_type"),
                workflow_profile=context.get("workflow_profile"),
                partition_config_fingerprint=context.get("partition_config_fingerprint"),
                handoff_id=context.get("handoff_id"),
                episode_id=context.get("episode_id"),
                episode_uid=context.get("episode_uid"),
                correction_set_id=context.get("correction_set_id"),
                selected_variant=context.get("selected_variant"),
            )
        request_start = pipeline.performance.requests
        failure_start = pipeline.performance.failures
        fallback_start = pipeline.fallback_count

        def finalize_result(result: dict[str, Any], outcome: str | None = None) -> dict[str, Any]:
            if context:
                result.update(
                    {
                        key: context[key]
                        for key in ("partition_id", "corpus_id", "handoff_id", "episode_id", "episode_uid", "correction_set_id", "selected_variant", "processing_key", "cache_key", "status_history")
                        if context.get(key) not in (None, "")
                    }
                )
                result["current_stage"] = "publish" if outcome in {None, "completed", "cached_valid", "clean", "completed_with_warnings"} else "file"
                result["run_id"] = run_id
                result["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                result.setdefault(
                    "stage_results",
                    {
                        "cache": result.get("source") or "not_written",
                        "leaf_chunks": result.get("leaf_chunks", 0),
                        "summaries": result.get("summaries", 0),
                        "positions": result.get("positions", result.get("position_cards", 0)),
                    },
                )
            if recorder is None:
                return result
            if outcome:
                recorder.set_outcome(outcome)
            recorder.update_metrics(
                requests=pipeline.performance.requests - request_start,
                request_failures=pipeline.performance.failures - failure_start,
                fallbacks=pipeline.fallback_count - fallback_start,
            )
            artifacts = finalize_file_errata(pipeline, recorder, errata_dir)
            if artifacts:
                result["errata_json_path"] = artifacts["json_path"]
                result["errata_markdown_path"] = artifacts["markdown_path"]
                result["errata_status"] = artifacts["payload"].get("outcome", {}).get("status")
                stats.record_errata(artifacts["payload"])
            return result

        try:
            if cache_path.exists():
                try:
                    previous_active_errata = pipeline.active_errata
                    pipeline.active_errata = recorder
                    try:
                        result = (
                            pipeline.validate_cached_file(path, fingerprint, cache_path, context)
                            if context is not None
                            else pipeline.validate_cached_file(path, fingerprint, cache_path)
                        )
                    finally:
                        pipeline.active_errata = previous_active_errata
                    stats.cached_files += 1
                    result["output_cache_path"] = str(cache_path)
                    result["output_cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
                    if recorder is not None:
                        recorder.update_source(
                            episode_id=result.get("episode_id"),
                            episode_title=result.get("episode_title"),
                            episode_date=result.get("episode_date"),
                            source_type=result.get("source_type"),
                        )
                        recorder.update_metrics(
                            leaf_chunks=result.get("leaf_chunks", 0),
                            summaries=result.get("summaries", 0),
                            positions=result.get("positions", 0),
                            documents=result.get("nodes", 0),
                        )
                        recorder.set_checkpoint_reuse("processed_cache", True)
                        recorder.record(
                            "processed_cache_valid",
                            "info",
                            "cache_validation",
                            "The existing processed-data cache passed validation and was reused.",
                            details={"path": str(cache_path), "document_count": result.get("nodes", 0)},
                        )
                        recorder.set_outcome("cached_valid")
                except (ValueError, RuntimeError) as exc:
                    stats.cached_files = max(0, stats.cached_files - 1)
                    if recorder is not None:
                        recorder.set_checkpoint_reuse("processed_cache", False)
                        recorder.record(
                            "cache_validation_failure",
                            "error",
                            "cache_validation",
                            "The existing processed-data cache failed validation and will be rebuilt.",
                            details={"path": str(cache_path), "error_type": type(exc).__name__, "error": str(exc)},
                        )
                    quarantined_path = quarantine_invalid_cache(cache_path, str(exc))
                    if recorder is not None:
                        recorder.record(
                            "cache_quarantined",
                            "warning",
                            "cache_validation",
                            "The invalid processed-data cache was quarantined before rebuild.",
                            details={"path": str(quarantined_path)},
                        )
                    print(f"  Invalid processed data cache moved to: {quarantined_path}")
                    print("  Rebuilding processed data from transcript.")
                    if not needs_llm_processing:
                        pipeline = _make_pipeline(config, project_dir, control, load_models=True)
                        pipeline.active_context = context
                    mark_state(
                        state,
                        fingerprint,
                        path,
                        "processing",
                        {"run_id": run_id, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "current_stage": "processing", **(context or {})},
                    )
                    save_state(state_path, state)
                    result = (
                        pipeline.process_file(path, recorder, context)
                        if context is not None
                        else pipeline.process_file(path, recorder)
                    )
                    stats.llm_files += 1
                    if result["status"] != "completed":
                        result = finalize_result(result, "skipped")
                        mark_state(state, fingerprint, path, result["status"], result)
                        save_state(state_path, state)
                        stats.files_skipped += 1
                        stats.files.append({"path": str(path), **{key: value for key, value in result.items() if key != "documents"}})
                        write_run_snapshot(snapshot_path, stats, pipeline.performance)
                        continue
                    docs = result.pop("documents", [])
                    if context is not None:
                        pipeline.save_cached_documents(cache_path, path, fingerprint, docs, context)
                    else:
                        pipeline.save_cached_documents(cache_path, path, fingerprint, docs)
                    result["cache_path"] = str(cache_path)
                    result["output_cache_path"] = str(cache_path)
                    result["output_cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
                    result["quarantined_cache_path"] = str(quarantined_path)
                    print(f"  Saved processed data cache: {cache_path}")
            else:
                mark_state(
                    state,
                    fingerprint,
                    path,
                    "processing",
                    {"run_id": run_id, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "current_stage": "processing", **(context or {})},
                )
                save_state(state_path, state)
                result = (
                    pipeline.process_file(path, recorder, context)
                    if context is not None
                    else pipeline.process_file(path, recorder)
                )
                stats.llm_files += 1
                if result["status"] != "completed":
                    result = finalize_result(result, "skipped")
                    mark_state(state, fingerprint, path, result["status"], result)
                    save_state(state_path, state)
                    stats.files_skipped += 1
                    stats.files.append({"path": str(path), **{key: value for key, value in result.items() if key != "documents"}})
                    write_run_snapshot(snapshot_path, stats, pipeline.performance)
                    continue
                docs = result.pop("documents", [])
                if context is not None:
                    pipeline.save_cached_documents(cache_path, path, fingerprint, docs, context)
                else:
                    pipeline.save_cached_documents(cache_path, path, fingerprint, docs)
                result["cache_path"] = str(cache_path)
                result["output_cache_path"] = str(cache_path)
                result["output_cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
                print(f"  Saved processed data cache: {cache_path}")

            if recorder is not None and result.get("status") == "completed" and result.get("source") != "processed_data_cache":
                recorder.set_outcome("completed_with_warnings" if recorder.has_anomalies() else "clean")
            result = finalize_result(result)

            moved_to = None
            if result["status"] == "completed" and config.move_processed_files:
                moved_to = maybe_move_processed(path, processed_dir)
                print(f"  Moved to {moved_to}")
            if moved_to:
                result["moved_to"] = moved_to
            mark_state(state, fingerprint, path, result["status"], result)
            save_state(state_path, state)
            if result["status"] in {"completed", "skipped"}:
                completed_files_this_run += 1
            if result["status"] == "completed":
                stats.files_completed += 1
            elif result["status"] == "skipped":
                stats.files_skipped += 1
            stats.documents += int(result.get("nodes") or 0)
            stats.position_cards += int(result.get("position_cards") or 0)
            stats.fallbacks = pipeline.fallback_count
            stats.files.append({"path": str(path), **{key: value for key, value in result.items() if key != "documents"}})
            write_run_snapshot(snapshot_path, stats, pipeline.performance)
        except PipelineInterrupted as exc:
            if recorder is not None:
                recorder.set_outcome("interrupted", failed_stage="file", exception=exc)
            errata_result = finalize_result({}, "interrupted")
            mark_state(state, fingerprint, path, "interrupted", {"error": str(exc), **errata_result})
            save_state(state_path, state)
            stats.files_failed += 1
            stats.failures.append({"path": str(path), "error": str(exc), "type": "interrupted", **errata_result})
            write_run_snapshot(snapshot_path, stats, pipeline.performance)
            print("Stop request handled. Progress state was saved; this file will be retried on the next run.")
            break
        except Exception as exc:
            if recorder is not None:
                recorder.set_outcome("failed", failed_stage="file", exception=exc)
            errata_result = finalize_result({}, "failed")
            mark_state(state, fingerprint, path, "failed", {"error": f"{type(exc).__name__}: {exc}", **errata_result})
            save_state(state_path, state)
            stats.files_failed += 1
            stats.failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}", "type": "exception", **errata_result})
            write_run_snapshot(snapshot_path, stats, pipeline.performance)
            raise

        if one_file:
            print("Processed one file; stopping because --one-file was set.")
            break

        if runtime.STOP_REQUESTED or stop_file.exists():
            print("Stop request detected. Batch will resume with the next pending file on the next run.")
            break

    pipeline.performance.final_report()
    json_report, md_report = write_run_reports(report_dir, stats, pipeline.performance, config)
    print(f"Run reports saved: {json_report} and {md_report}")
    if config.auto_refresh_topic_index:
        topic_summary = refresh_topic_index(config, project_dir)
        print(
            "Topic index refreshed: "
            f"{topic_summary['topic_count']} topics across {topic_summary['episode_count']} episode contribution(s) "
            f"({topic_summary['reused_contributions']} reused, {topic_summary['rebuilt_contributions']} rebuilt)."
        )
        if topic_summary["llm_curated_keep"] or topic_summary["llm_curated_drop"]:
            print(
                "Topic label curation updated: "
                f"{topic_summary['llm_curated_keep']} kept, {topic_summary['llm_curated_drop']} dropped "
                f"(whitelist={topic_summary['whitelist_size']}, blacklist={topic_summary['blacklist_size']})."
            )
        print(f"Topic index path: {topic_summary['topic_index_path']}")
        print(f"Topic curation report: {topic_summary['topic_curation_report_path']}")
    if config.enable_temporal_artifacts:
        temporal_summary = build_temporal_artifacts(processed_data_dir, resolve_path(project_dir, config.temporal_artifact_path), include_trajectories=config.enable_temporal_trajectories, include_contradiction_candidates=config.enable_contradiction_candidates, missing_interval_days=config.temporal_missing_interval_days)
        print(f"Temporal research artifact refreshed: {temporal_summary['output_path']} ({temporal_summary['claim_count']} claims).")
    print("\nBatch run complete.")
    return 0

def build_topic_index(config: PipelineConfig, project_dir: Path) -> int:
    """Build or incrementally refresh the cache-only topic index from processed_data."""
    summary = refresh_topic_index(config, project_dir)
    print(
        "Topic index refreshed: "
        f"{summary['topic_count']} topics across {summary['episode_count']} episode contribution(s); "
        f"{summary['reused_contributions']} reused, {summary['rebuilt_contributions']} rebuilt, "
        f"{summary['removed_contributions']} removed."
    )
    if summary["llm_curated_keep"] or summary["llm_curated_drop"]:
        print(
            "Topic label curation updated: "
            f"{summary['llm_curated_keep']} kept, {summary['llm_curated_drop']} dropped "
            f"(whitelist={summary['whitelist_size']}, blacklist={summary['blacklist_size']})."
        )
    print(f"Topic index path: {summary['topic_index_path']}")
    print(f"Topic curation report: {summary['topic_curation_report_path']}")
    return 0

def inspect_processed_cache(config: PipelineConfig, project_dir: Path) -> int:
    """Validate cached processed-document files without running the model."""
    processed_data_dir = resolve_path(project_dir, config.processed_data_dir)
    files = sorted(processed_data_dir.glob("*.processed_documents.json"))
    totals = Counter()
    invalid = 0
    print(f"Inspecting {len(files)} processed cache file(s) in {processed_data_dir}")
    for cache_path in files:
        try:
            payload = read_json_file(cache_path)
            docs = payload.get("documents") or []
            validation = validate_processed_cache(payload)
            counts = Counter(validation.counts)
            totals.update(counts)
            status = "valid" if validation.valid else "invalid"
            if not validation.valid:
                invalid += 1
            print(
                f"- {cache_path.name}: {status}, docs={len(docs)}, "
                f"positions={counts.get('position_card', 0)}, schema={payload.get('schema_version', 'unknown')}"
            )
            for warning in validation.warnings[:3]:
                print(f"    warning: {warning}")
            for error in validation.errors[:3]:
                print(f"    error: {error}")
        except Exception as exc:
            invalid += 1
            print(f"- {cache_path.name}: unreadable ({type(exc).__name__}: {exc})")
    print(f"Totals: {dict(totals)} invalid_files={invalid}")
    return 1 if invalid else 0

def config_doctor(config: PipelineConfig, project_dir: Path) -> int:
    """Check the effective config against model, token, and cache expectations."""
    print("Config doctor")
    print(f"  pipeline_version: {PIPELINE_VERSION}")
    print(f"  prompt_version: {PROMPT_VERSION}")
    print(f"  model: {config.lm_studio_model}")
    print(f"  base_url: {config.lm_studio_base_url}")
    print(f"  context_window_tokens: {config.context_window_tokens}")
    print(f"  prompt_token_budget: {config.prompt_token_budget}")
    print(f"  llm_max_tokens: {config.llm_max_tokens}")
    print(f"  rollup_char_budget: {config.rollup_char_budget}")
    print(f"  estimated rollup prompt tokens: {token_estimate('x' * config.rollup_char_budget, config.prompt_token_chars_per_token)}")
    if config.prompt_token_budget + config.llm_max_tokens > config.context_window_tokens:
        print("  warning: prompt_token_budget + llm_max_tokens exceeds context_window_tokens")
    if config.max_parallel_model_requests > 1:
        print("  note: parallel requests can reduce wall time but may increase LM Studio context pressure")
    if not config.fake_llm and config.verify_model:
        runtime.load_runtime_deps()
        verify_model_available(config)
    if not config.fake_llm and config.test_inference:
        runtime.load_runtime_deps()
        test_model_inference(config)
    print("Processed cache schema:")
    print(dumps_schema_summary())
    return 0

def evaluate_model(config: PipelineConfig, project_dir: Path, limit: int = 3) -> int:
    """Benchmark the configured model on a small deterministic transcript sample."""
    runtime.load_runtime_deps()
    input_dir = resolve_path(project_dir, config.input_dir)
    output_dir = resolve_path(project_dir, config.model_eval_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = iter_transcript_files(input_dir, config.file_glob)[: max(1, limit)]
    control = RuntimeControl(config, project_dir)
    control.initialize_file_for_run()
    pipeline = PodcastRagPipeline(config, project_dir, control)
    results = []
    for path in files:
        docs = load_transcript_json(path)
        leaf_chunks = pipeline.build_leaf_chunks(docs, str(path))[:3]
        sample_text = "\n\n".join(pipeline.render_doc_for_rollup(doc) for doc in leaf_chunks)
        started = time.time()
        summary = pipeline.invoke_llm(pipeline.summary_chain, sample_text, f"model eval {path.name}")
        elapsed = time.time() - started
        results.append(
            {
                "path": str(path),
                "elapsed_seconds": round(elapsed, 3),
                "missing_context": is_missing_context_response(summary),
                "empty": not bool(summary.strip()),
                "compression_ratio": round(len(summary) / max(1, len(sample_text)), 4),
                "topic_tags": deterministic_topic_tags(summary, config.deterministic_topic_count),
            }
        )
    report_path = output_dir / f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.model_eval.json"
    write_json_file(
        report_path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "model": config.lm_studio_model,
            "performance": pipeline.performance.snapshot(),
            "results": results,
        },
    )
    print(f"Model evaluation report saved: {report_path}")
    return 0


def evaluate_retrieval(
    config: PipelineConfig,
    project_dir: Path,
    retrieval_results: str,
    query_set: str | None = None,
    output_dir: str | None = None,
) -> int:
    """Score captured ranked results against a versioned judged query set."""
    query_set_path = resolve_path(project_dir, query_set or config.retrieval_evaluation_query_set)
    results_path = resolve_path(project_dir, retrieval_results)
    report_dir = resolve_path(project_dir, output_dir or config.retrieval_evaluation_output_dir)
    report = evaluate_retrieval_run(query_set_path, results_path, report_dir)
    print(
        f"Retrieval evaluation complete: judged={report['judged_query_count']} "
        f"skipped={report['skipped_query_count']} strategy={report['strategy_id']}"
    )
    for key, value in report["aggregate"].items():
        print(f"  {key}: {value:.4f}")
    print(f"JSON report: {report['json_report_path']}")
    print(f"Markdown report: {report['markdown_report_path']}")
    return 0


def backfill_representations(config: PipelineConfig, project_dir: Path, cache_path: str | None = None) -> int:
    processed_data_dir = resolve_path(project_dir, config.processed_data_dir)
    paths = [resolve_path(project_dir, cache_path)] if cache_path else sorted(processed_data_dir.glob("*.processed_documents.json"))
    if not paths:
        print(f"No processed caches found in {processed_data_dir}")
        return 0
    for path in paths:
        result = backfill_cache_file(path, config)
        print(f"Backfilled {result['document_count']} document(s): {result['cache_path']}")
    return 0


def export_dense_retrieval_baseline(config: PipelineConfig, project_dir: Path, output_path: str | None) -> int:
    processed_data_dir = resolve_path(project_dir, config.processed_data_dir)
    destination = resolve_path(project_dir, output_path or "evaluation/exports/page-content-v1-baseline.json")
    result = export_dense_baseline(processed_data_dir, destination)
    print(
        f"Dense baseline exported: {result['document_count']} document(s) from "
        f"{result['cache_count']} cache(s) to {result['output_path']}"
    )
    return 0

def create_stop_file(config: PipelineConfig, project_dir: Path) -> int:
    stop_file = resolve_path(project_dir, config.stop_file)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text(f"Stop requested at {dt.datetime.now().isoformat()}\n", encoding="utf-8")
    print(f"Created stop file: {stop_file}")
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a podcast RAG knowledge base from transcript JSON files.")
    parser.add_argument("command", nargs="?", help="Managed command: partitions, handoff, process, status, or export.")
    parser.add_argument("command_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    parser.add_argument("--config", default="podcast_rag_config.json", help="Path to the JSON config file.")
    parser.add_argument("--input-dir", help="Override config input_dir.")
    parser.add_argument("--file-glob", help="Override config file_glob.")
    parser.add_argument("--model", help="Override config lm_studio_model.")
    parser.add_argument("--base-url", help="Override config lm_studio_base_url.")
    parser.add_argument("--max-parallel-model-requests", type=int, help="Override initial max_parallel_model_requests.")
    parser.add_argument("--one-file", action="store_true", help="Process only one pending file.")
    parser.add_argument(
        "--allow-mixed-partitions",
        action="store_true",
        help="Explicitly allow transforming transcripts from multiple processing spaces together.",
    )
    parser.add_argument("--create-stop-file", action="store_true", help="Create the configured stop file and exit.")
    parser.add_argument("--inspect-cache", action="store_true", help="Inspect processed_data caches without processing.")
    parser.add_argument(
        "--inspect-errata",
        nargs="?",
        const="",
        help="Validate and summarize errata records; optionally provide one JSON path or name.",
    )
    parser.add_argument("--config-doctor", action="store_true", help="Validate operational config and LM Studio settings before a batch.")
    parser.add_argument("--model-eval", action="store_true", help="Run the model-evaluation harness on transcript slices.")
    parser.add_argument("--model-eval-limit", type=int, default=3, help="Maximum transcript files to sample for --model-eval.")
    parser.add_argument("--retrieval-eval", action="store_true", help="Score captured retrieval results against a judged query set.")
    parser.add_argument("--retrieval-results", help="Path to captured ranked retrieval results JSON for --retrieval-eval.")
    parser.add_argument("--query-set", help="Override the configured retrieval evaluation query-set JSONL path.")
    parser.add_argument("--evaluation-output-dir", help="Override the retrieval evaluation report directory.")
    parser.add_argument("--backfill-representations", action="store_true", help="Backfill deterministic provenance and representations in existing caches without LLM work.")
    parser.add_argument("--cache-path", help="Backfill one processed cache instead of all caches.")
    parser.add_argument("--export-dense-baseline", action="store_true", help="Export page-content-v1 dense documents for downstream evaluation.")
    parser.add_argument("--export-output", help="Output path for --export-dense-baseline.")
    parser.add_argument("--export-representation-corpus", action="store_true", help="Export deterministic display, dense, and lexical corpus representations.")
    parser.add_argument("--build-topic-index", action="store_true", help="Build or refresh the cache-only topic index from processed_data.")
    parser.add_argument("--build-temporal-artifacts", action="store_true", help="Build optional evidence-bound temporal research artifacts.")
    parser.add_argument("--build-advanced-retrieval", action="store_true", help="Build gated M5 prototype sidecars without promoting them.")
    parser.add_argument("--m5-entry-gate", help="Path to a completed advanced-retrieval-entry-gate-1.0 JSON file.")
    parser.add_argument("--corpus-release-id", help="Immutable parent corpus release for M5 sidecars.")
    parser.add_argument("--curate-topic-labels", action="store_true", help="Run the optional LM Studio topic-label curation pass during topic-index refresh.")
    parser.add_argument("--fake-llm", action="store_true", help="Use deterministic fake LLM responses for no-LM Studio validation.")
    return parser.parse_args()

def main() -> int:
    """CLI entry point for batch processing, diagnostics, and model evaluation."""
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    args = parse_args()
    config_path = Path(args.config).expanduser()
    project_dir = config_path.resolve().parent if config_path.exists() else Path.cwd()
    config = apply_env_overrides(load_config(config_path))

    if args.command:
        managed_path_overrides = {
            name: value
            for name, value in {
                "--input-dir": args.input_dir,
                "--file-glob": args.file_glob,
            }.items()
            if value not in (None, "")
        }
        if managed_path_overrides:
            raise SystemExit(
                "managed commands do not accept path-oriented overrides: "
                + ", ".join(sorted(managed_path_overrides))
                + "; configure the partition through 'partitions update'"
            )
        command_args = list(args.command_args)
        if "--config" in command_args:
            config_index = command_args.index("--config")
            if config_index + 1 >= len(command_args):
                raise SystemExit("--config requires a path")
            config_path = Path(command_args[config_index + 1]).expanduser()
            project_dir = config_path.resolve().parent if config_path.exists() else Path.cwd()
            config = apply_env_overrides(load_config(config_path))
            del command_args[config_index : config_index + 2]
        try:
            if args.command == "partitions":
                return partition_command(config, project_dir, command_args)
            if args.command == "handoff":
                return handoff_command(config, project_dir, command_args)
            if args.command == "process":
                return managed_process_command(config, project_dir, command_args)
            if args.command == "status":
                return status_command(config, project_dir, command_args)
            if args.command == "export":
                return export_partition_command(config, project_dir, command_args)
            raise SystemExit(f"unknown command: {args.command}")
        except (PartitionError, HandoffError, OSError, ValueError) as exc:
            print(f"Managed command failed: {exc}")
            return 1

    if args.input_dir:
        config.input_dir = args.input_dir
    if args.file_glob:
        config.file_glob = args.file_glob
    if args.model:
        config.lm_studio_model = args.model
    if args.base_url:
        config.lm_studio_base_url = args.base_url
    if args.max_parallel_model_requests:
        config.max_parallel_model_requests = args.max_parallel_model_requests
    if args.fake_llm:
        config.fake_llm = True
        config.verify_model = False
        config.test_inference = False
    if args.curate_topic_labels:
        config.enable_llm_topic_label_curation = True

    if args.create_stop_file:
        return create_stop_file(config, project_dir)
    if args.inspect_cache:
        return inspect_processed_cache(config, project_dir)
    if args.inspect_errata is not None:
        return inspect_errata(config, project_dir, args.inspect_errata or None)
    if args.config_doctor:
        return config_doctor(config, project_dir)
    if args.model_eval:
        return evaluate_model(config, project_dir, args.model_eval_limit)
    if args.retrieval_eval:
        if not args.retrieval_results:
            raise SystemExit("--retrieval-eval requires --retrieval-results")
        return evaluate_retrieval(
            config,
            project_dir,
            args.retrieval_results,
            query_set=args.query_set,
            output_dir=args.evaluation_output_dir,
        )
    if args.backfill_representations:
        return backfill_representations(config, project_dir, args.cache_path)
    if args.export_dense_baseline:
        return export_dense_retrieval_baseline(config, project_dir, args.export_output)
    if args.export_representation_corpus:
        if not args.export_output:
            raise SystemExit("--export-representation-corpus requires --export-output")
        result = export_representation_corpus(resolve_path(project_dir, config.processed_data_dir), Path(args.export_output))
        print(result)
        return 0
    if args.build_topic_index:
        return build_topic_index(config, project_dir)
    if args.build_temporal_artifacts:
        result = build_temporal_artifacts(
            resolve_path(project_dir, config.processed_data_dir),
            resolve_path(project_dir, config.temporal_artifact_path),
            include_trajectories=config.enable_temporal_trajectories,
            include_contradiction_candidates=config.enable_contradiction_candidates,
            missing_interval_days=config.temporal_missing_interval_days,
        )
        print(result)
        return 0
    if args.build_advanced_retrieval:
        if not config.enable_advanced_retrieval_prototypes:
            raise SystemExit("M5 prototypes are disabled; set enable_advanced_retrieval_prototypes=true")
        if not args.m5_entry_gate or not args.corpus_release_id:
            raise SystemExit("--build-advanced-retrieval requires --m5-entry-gate and --corpus-release-id")
        import json
        gate = json.loads(resolve_path(project_dir, args.m5_entry_gate).read_text(encoding="utf-8"))
        destination = resolve_path(project_dir, config.advanced_retrieval_dir) / args.corpus_release_id
        temporal = resolve_path(project_dir, config.temporal_artifact_path)
        graph = build_evidence_graph(resolve_path(project_dir, config.processed_data_dir), temporal, destination / "evidence-graph.json", corpus_release_id=args.corpus_release_id, entry_gate=gate)
        alignment = build_late_chunk_alignment(resolve_path(project_dir, config.processed_data_dir), destination / "late-chunk-alignment.json", corpus_release_id=args.corpus_release_id, entry_gate=gate, model_id=config.embedding_model, model_revision=str(gate.get("model_revision") or "unresolved"), tokenizer_id=str(gate.get("tokenizer_id") or config.embedding_model))
        print({"graph_id": graph["graph_id"], "alignment_id": alignment["alignment_id"], "disposition": "prototype"})
        return 0

    return run_batch(
        config,
        project_dir,
        args.one_file,
        allow_mixed_partitions=args.allow_mixed_partitions,
    )
