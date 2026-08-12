from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ecosystem_delta import apply_delta, plan_delta, validate_delta, write_atomic
from .upstream_contracts import discover_correction_notifications
from .delta_jobs import DeltaJobStore, apply_job, plan_notification_job


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="podcast-rag-delta")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan-delta")
    for name in ("old", "new", "correction_set_id", "parent_corpus_id", "processing_fingerprint", "representation_fingerprint", "output"):
        plan.add_argument(f"--{name.replace('_', '-')}", required=True)
    notification_plan = commands.add_parser("plan-notification-delta")
    for name in ("project_root", "old", "new", "parent_corpus_id", "processing_fingerprint", "representation_fingerprint", "output"):
        notification_plan.add_argument(f"--{name.replace('_', '-')}", required=True)
    notification_plan.add_argument("--correction-set-id")
    apply = commands.add_parser("apply-delta")
    for name in ("delta", "old", "new", "approve_correction_set", "output"):
        apply.add_argument(f"--{name.replace('_', '-')}", required=True)
    check = commands.add_parser("validate")
    check.add_argument("delta")
    job_plan = commands.add_parser("plan-notification-job")
    for name in ("project_root", "old", "new", "output", "job_root", "parent_corpus_id", "processing_fingerprint", "representation_fingerprint"):
        job_plan.add_argument(f"--{name.replace('_', '-')}", required=True)
    job_plan.add_argument("--correction-set-id")
    job_apply = commands.add_parser("apply-job")
    job_apply.add_argument("job_id"); job_apply.add_argument("--job-root", required=True); job_apply.add_argument("--approve-delta", required=True)
    job_status = commands.add_parser("job-status")
    job_status.add_argument("job_id"); job_status.add_argument("--job-root", required=True)
    job_cancel = commands.add_parser("cancel-job")
    job_cancel.add_argument("job_id"); job_cancel.add_argument("--job-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        validate_delta(_read(args.delta)); return 0
    if args.command == "plan-notification-job":
        value = plan_notification_job(
            project_root=Path(args.project_root), old_path=Path(args.old), new_path=Path(args.new),
            output_path=Path(args.output), job_root=Path(args.job_root), parent_corpus_id=args.parent_corpus_id,
            processing_fingerprint=args.processing_fingerprint, representation_fingerprint=args.representation_fingerprint,
            correction_set_id=args.correction_set_id,
        )
        print(value["job_id"]); return 0
    if args.command in {"apply-job", "job-status", "cancel-job"}:
        store = DeltaJobStore(Path(args.job_root))
        if args.command == "apply-job": value = apply_job(store, args.job_id, approved_delta_id=args.approve_delta)
        elif args.command == "cancel-job": value = store.cancel(args.job_id)
        else: value = store.load(args.job_id)
        print(json.dumps(value, sort_keys=True)); return 0
    if args.command == "plan-delta":
        value = plan_delta(_read(args.old), _read(args.new), parent_corpus_id=args.parent_corpus_id,
            correction_set_id=args.correction_set_id, processing_fingerprint=args.processing_fingerprint,
            representation_fingerprint=args.representation_fingerprint)
    elif args.command == "plan-notification-delta":
        ready = [item for item in discover_correction_notifications(args.project_root) if item["status"] == "ready"]
        if args.correction_set_id:
            ready = [item for item in ready if item.get("manifest", {}).get("correction_set_id") == args.correction_set_id]
        if len(ready) != 1:
            qualifier = f" matching {args.correction_set_id}" if args.correction_set_id else ""
            parser.error(f"expected exactly one ready correction notification{qualifier}; found {len(ready)}")
        manifest = ready[0]["manifest"]
        value = plan_delta(
            _read(args.old), _read(args.new), parent_corpus_id=args.parent_corpus_id,
            correction_set_id=manifest["correction_set_id"],
            processing_fingerprint=args.processing_fingerprint,
            representation_fingerprint=args.representation_fingerprint,
            affected_episode_ids=manifest.get("affected_episode_ids", []),
            affected_source_span_ids=manifest.get("affected_source_span_ids", []),
        )
    else:
        value = apply_delta(_read(args.delta), _read(args.old), _read(args.new),
            approved_correction_set_id=args.approve_correction_set)
    write_atomic(Path(args.output), value)
    print(value.get("delta_id", "applied"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
