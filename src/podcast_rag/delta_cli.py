from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ecosystem_delta import apply_delta, plan_delta, validate_delta, write_atomic


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="podcast-rag-delta")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan-delta")
    for name in ("old", "new", "correction_set_id", "parent_corpus_id", "processing_fingerprint", "representation_fingerprint", "output"):
        plan.add_argument(f"--{name.replace('_', '-')}", required=True)
    apply = commands.add_parser("apply-delta")
    for name in ("delta", "old", "new", "approve_correction_set", "output"):
        apply.add_argument(f"--{name.replace('_', '-')}", required=True)
    check = commands.add_parser("validate")
    check.add_argument("delta")
    args = parser.parse_args(argv)
    if args.command == "validate":
        validate_delta(_read(args.delta)); return 0
    if args.command == "plan-delta":
        value = plan_delta(_read(args.old), _read(args.new), parent_corpus_id=args.parent_corpus_id,
            correction_set_id=args.correction_set_id, processing_fingerprint=args.processing_fingerprint,
            representation_fingerprint=args.representation_fingerprint)
    else:
        value = apply_delta(_read(args.delta), _read(args.old), _read(args.new),
            approved_correction_set_id=args.approve_correction_set)
    write_atomic(Path(args.output), value)
    print(value.get("delta_id", "applied"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
