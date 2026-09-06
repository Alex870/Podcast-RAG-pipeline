"""Per-file processing diagnostics and bounded remediation reports."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from podcast_rag.config import config_fingerprint
from podcast_rag.runtime import PIPELINE_VERSION, PROMPT_VERSION

ERRATA_CONTRACT_VERSION = "file-errata-1.0"
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
_ANOMALY_SEVERITIES = {"warning", "error"}
_DIAGNOSIS_CATEGORIES = {
    "model_output",
    "pipeline_logic",
    "configuration",
    "input_data",
    "environment",
    "unknown",
}
DIAGNOSIS_ACTION_TYPES = ("code", "config", "data", "rerun", "human_review")
_ACTION_TYPES = set(DIAGNOSIS_ACTION_TYPES)
_OUTCOME_STATUSES = {"clean", "completed_with_warnings", "cached_valid", "skipped", "interrupted", "failed"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "transcript"


def _truncate(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _bounded(value: Any, max_chars: int, depth: int = 0) -> Any:
    if depth > 4:
        return _truncate(value, max_chars)
    if isinstance(value, str):
        return _truncate(value, max_chars)
    if isinstance(value, dict):
        return {
            str(key): _bounded(item, max_chars, depth + 1)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded(item, max_chars, depth + 1) for item in list(value)[:100]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate(value, max_chars)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fit_json(value: Any, max_chars: int) -> Any:
    """Keep a JSON-compatible value within a total character budget."""
    max_chars = max(256, int(max_chars or 12000))
    if len(json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)) <= max_chars:
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key = str(key)
            if isinstance(item, (list, tuple)):
                kept = []
                for entry in item:
                    candidate = dict(result)
                    candidate[key] = kept + [entry]
                    if len(json.dumps(candidate, ensure_ascii=True, separators=(",", ":"), default=str)) > max_chars:
                        break
                    kept.append(entry)
                candidate = dict(result)
                candidate[key] = kept
            else:
                candidate = dict(result)
                candidate[key] = item
            if len(json.dumps(candidate, ensure_ascii=True, separators=(",", ":"), default=str)) <= max_chars:
                result = candidate
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            candidate = result + [item]
            if len(json.dumps(candidate, ensure_ascii=True, separators=(",", ":"), default=str)) > max_chars:
                break
            result = candidate
        return result
    return _truncate(value, max_chars)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _remediation_category(code: str) -> str:
    if code.startswith("llm_") or "json" in code or "position" in code:
        return "model_output"
    if "fallback" in code or "reduction" in code:
        return "pipeline_logic"
    if "config" in code or "context" in code:
        return "config"
    if "input" in code or "transcript" in code:
        return "data"
    if "checkpoint" in code or "cache" in code or "validation" in code:
        return "code"
    return "unknown"


@dataclass
class ErrataRecorder:
    """Collect bounded, deterministic findings for one source file."""

    source_path: Path
    source_fingerprint: str
    run_id: str
    config: Any
    started_at: str = field(default_factory=_now)
    findings: list[dict[str, Any]] = field(default_factory=list)
    related_debug_artifacts: list[str] = field(default_factory=list)
    checkpoint_reuse: dict[str, bool] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    review_context: dict[str, Any] = field(default_factory=dict)
    outcome_status: str = "failed"
    failed_stage: str | None = None
    exception: dict[str, str] | None = None
    diagnosis: dict[str, Any] = field(default_factory=lambda: {"status": "not_requested"})
    source_transcript_hash: str | None = None
    diagnosis_input_digest: str | None = None
    diagnosis_requested_explicitly: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            self.source_transcript_hash = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        except OSError:
            self.source_transcript_hash = None

    def record(
        self,
        code: str,
        severity: str,
        stage: str,
        message: str,
        *,
        details: Any = None,
        node_ids: list[str] | None = None,
        evidence_paths: list[Any] | None = None,
        excerpts: list[str] | None = None,
    ) -> str:
        severity = severity if severity in _SEVERITY_ORDER else "warning"
        finding_id = f"F{len(self.findings) + 1:04d}"
        finding = {
            "finding_id": finding_id,
            "created_at": _now(),
            "severity": severity,
            "code": str(code),
            "stage": str(stage),
            "message": _truncate(message, 1000),
            "deterministic": True,
            "remediation_category": _remediation_category(str(code)),
            "suggested_remediation_category": _remediation_category(str(code)),
            "details": _bounded(details or {}, 1200),
            "node_ids": [str(item) for item in (node_ids or [])[:100]],
            "evidence_paths": _bounded(evidence_paths or [], 1000),
            "excerpts": [_truncate(item, int(getattr(self.config, "errata_excerpt_max_chars", 360))) for item in (excerpts or [])[:20]],
        }
        with self._lock:
            finding_id = f"F{len(self.findings) + 1:04d}"
            finding["finding_id"] = finding_id
            self.findings.append(finding)
        return finding_id

    def add_debug_artifact(self, path: Path | str) -> None:
        value = str(path)
        with self._lock:
            if value not in self.related_debug_artifacts and len(self.related_debug_artifacts) < 100:
                self.related_debug_artifacts.append(value)

    def set_checkpoint_reuse(self, stage: str, reused: bool) -> None:
        self.checkpoint_reuse[str(stage)] = bool(reused)

    def update_source(self, **metadata: Any) -> None:
        self.source_metadata.update(
            {key: _bounded(value, 500) for key, value in metadata.items() if value not in (None, "")}
        )

    def update_metrics(self, **metrics: Any) -> None:
        self.metrics.update({key: _bounded(value, 500) for key, value in metrics.items()})

    def set_review_context(self, context: dict[str, Any]) -> None:
        max_chars = int(getattr(self.config, "errata_review_context_max_chars", 12000))
        self.review_context = _fit_json(
            _bounded(context, int(getattr(self.config, "errata_excerpt_max_chars", 360))),
            max_chars,
        )

    def set_outcome(
        self,
        status: str,
        *,
        failed_stage: str | None = None,
        exception: BaseException | None = None,
    ) -> None:
        self.outcome_status = status
        self.failed_stage = failed_stage
        if exception is not None:
            self.exception = {
                "type": type(exception).__name__,
                "message": _truncate(str(exception), 1600),
            }

    def set_diagnosis(self, diagnosis: dict[str, Any]) -> None:
        self.diagnosis = _bounded(diagnosis, int(getattr(self.config, "errata_review_context_max_chars", 12000)))

    def request_diagnosis(self) -> None:
        """Request review even when deterministic findings are informational only."""
        self.diagnosis_requested_explicitly = True

    def has_anomalies(self) -> bool:
        return any(item.get("severity") in _ANOMALY_SEVERITIES for item in self.findings)

    def has_model_service_failure(self) -> bool:
        return any(item.get("code") in {"llm_service_unavailable", "model_unavailable"} for item in self.findings)

    def finding_digest(self) -> str:
        stable = [
            {
                key: value
                for key, value in finding.items()
                if key not in {"finding_id", "created_at"}
            }
            for finding in sorted(self.findings, key=lambda item: (str(item.get("code")), str(item.get("stage")), str(item.get("message"))))
        ]
        return _digest(stable)

    def review_packet(self) -> dict[str, Any]:
        packet = {
            "contract_version": ERRATA_CONTRACT_VERSION,
            "source": {
                "path": str(self.source_path),
                "source_fingerprint": self.source_fingerprint,
                "transcript_hash": self.source_transcript_hash,
                **self.source_metadata,
            },
            "outcome": {
                "status": self.outcome_status,
                "failed_stage": self.failed_stage,
                "exception": self.exception,
            },
            "metrics": self.metrics,
            "checkpoint_reuse": self.checkpoint_reuse,
            "diagnosis_requested_explicitly": self.diagnosis_requested_explicitly,
            "findings": self.findings,
            "review_context": self.review_context,
        }
        max_chars = int(getattr(self.config, "errata_review_context_max_chars", 12000))
        encoded = json.dumps(packet, ensure_ascii=True, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return packet
        minimal_findings = [
            {
                "finding_id": finding.get("finding_id"),
                "severity": finding.get("severity"),
                "code": finding.get("code"),
                "stage": finding.get("stage"),
                "message": finding.get("message"),
                "node_ids": finding.get("node_ids") or [],
                "evidence_paths": finding.get("evidence_paths") or [],
            }
            for finding in self.findings
        ]
        packet["review_context"] = _fit_json(
            {"truncated": True, "context": self.review_context},
            max(256, max_chars // 2),
        )
        for count in range(min(50, len(minimal_findings)), -1, -1):
            packet["findings"] = minimal_findings[:count]
            encoded = json.dumps(packet, ensure_ascii=True, separators=(",", ":"))
            if len(encoded) <= max_chars:
                return packet
        # This only applies to an unusually tiny configured bound. Keep the
        # packet valid JSON and bounded even if fixed envelope fields dominate.
        packet = {"contract_version": ERRATA_CONTRACT_VERSION, "truncated": True, "findings": []}
        return packet

    def to_payload(self, artifact_paths: dict[str, str] | None = None) -> dict[str, Any]:
        diagnosis_input = self.review_packet()
        self.metrics.setdefault("elapsed_seconds", round((dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(self.started_at)).total_seconds(), 2))
        self.metrics.setdefault(
            "retries",
            sum(1 for finding in self.findings if str(finding.get("code")) in {"llm_retry", "errata_diagnosis_retry"}),
        )
        payload = {
            "contract_version": ERRATA_CONTRACT_VERSION,
            "artifact_id": "errata_" + _digest(
                {"source_fingerprint": self.source_fingerprint, "run_id": self.run_id}
            )[:24],
            "created_at": _now(),
            "source": {
                "path": str(self.source_path),
                "source_fingerprint": self.source_fingerprint,
                "transcript_hash": self.source_transcript_hash,
                **self.source_metadata,
            },
            "run": {
                "run_id": self.run_id,
                "pipeline_version": PIPELINE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "config_fingerprint": config_fingerprint(self.config),
                "model": str(getattr(self.config, "lm_studio_model", "")),
            },
            "outcome": {
                "status": self.outcome_status,
                "failed_stage": self.failed_stage,
                "exception": self.exception,
            },
            "metrics": self.metrics,
            "checkpoint_reuse": self.checkpoint_reuse,
            "findings": self.findings,
            "llm_diagnosis": self.diagnosis,
            "related_debug_artifacts": self.related_debug_artifacts,
            "review_context": self.review_context,
            "finding_digest": self.finding_digest(),
            "diagnosis_input_digest": self.diagnosis_input_digest or _digest(diagnosis_input),
            "artifacts": artifact_paths or {},
        }
        payload["material_digest"] = _digest(
            {
                "status": self.outcome_status,
                "failed_stage": self.failed_stage,
                "finding_digest": payload["finding_digest"],
            }
        )
        return payload


def _is_anomalous_status(status: str) -> bool:
    return status in {"completed_with_warnings", "interrupted", "failed"}


def render_markdown(payload: dict[str, Any]) -> str:
    outcome = payload.get("outcome") or {}
    status = str(outcome.get("status") or "unknown")
    findings = sorted(
        payload.get("findings") or [],
        key=lambda item: (_SEVERITY_ORDER.get(str(item.get("severity")), 9), str(item.get("finding_id"))),
    )
    diagnosis = payload.get("llm_diagnosis") or {}
    if not isinstance(diagnosis, dict):
        # A malformed advisory result must never prevent the deterministic
        # Markdown record from being rendered.  The machine validator reports
        # the malformed shape separately; rendering falls back to the
        # deterministic findings here.
        diagnosis = {"status": "failed", "error": "llm_diagnosis is not an object"}
    diagnosis_summary = str(diagnosis.get("summary") or "").strip()
    if not diagnosis_summary:
        if status == "clean":
            diagnosis_summary = "No actionable anomalies were detected."
        elif status == "cached_valid":
            diagnosis_summary = "The existing processed cache passed validation."
        else:
            diagnosis_summary = f"File ended with status `{status}` and {len(findings)} recorded finding(s)."

    lines = [
        "# File Errata Report",
        "",
        f"- Status: **{status}**",
        f"- Source: `{payload.get('source', {}).get('path', 'unknown')}`",
        f"- Source fingerprint: `{payload.get('source', {}).get('source_fingerprint', '')}`",
        f"- Artifact contract: `{payload.get('contract_version', '')}`",
        "",
        "## Condensed summary",
        "",
        diagnosis_summary,
        "",
        "## Outcome and impact",
        "",
        f"The file outcome is **{status}**. "
        f"{('Cache creation or reuse was not completed.' if status in {'failed', 'interrupted'} else 'The processed result was retained according to the outcome above.')}",
        f"Failed stage: `{outcome.get('failed_stage') or 'none'}`.",
        "",
        "## Findings",
        "",
    ]
    if not findings:
        lines.append("No findings.")
    else:
        for finding in findings:
            lines.append(
                f"- **{str(finding.get('severity', 'warning')).upper()}** "
                f"`{finding.get('code')}` ({finding.get('stage')}): {finding.get('message')}"
            )

    if diagnosis.get("root_causes"):
        lines.extend(["", "## Likely causes", ""])
        for cause in diagnosis["root_causes"][:10]:
            lines.append(
                f"- **{cause.get('category', 'unknown')}** "
                f"(confidence {cause.get('confidence', 'unknown')}): {cause.get('explanation') or cause.get('claim', '')}"
            )
    elif findings:
        lines.extend(["", "## Likely causes", ""])
        for finding in findings[:10]:
            lines.append(
                f"- `{finding.get('code')}` is categorized as `{finding.get('remediation_category', 'unknown')}`; "
                "the deterministic record is the source of truth until reviewed."
            )
    if diagnosis.get("actions"):
        lines.extend(["", "## Recommended actions", ""])
        for action in diagnosis["actions"][:10]:
            lines.append(
                f"- **{action.get('priority', 'normal')}** `{action.get('type', 'human_review')}`: "
                f"{action.get('action', '')} Verification: {action.get('verification', 'n/a')}"
            )
    elif findings:
        lines.extend(["", "## Recommended actions", ""])
        for finding in findings[:10]:
            priority = "high" if finding.get("severity") == "error" else "normal"
            lines.append(
                f"- **{priority}** `human_review`: inspect `{finding.get('code')}` and its bounded details. "
                "Verification: confirm the finding is understood and rerun or repair validation as appropriate."
            )
    if diagnosis.get("uncertainties"):
        lines.extend(["", "## Uncertainties", ""])
        lines.extend(f"- {item}" for item in diagnosis["uncertainties"][:10])
    if diagnosis.get("human_review_questions"):
        lines.extend(["", "## Human review questions", ""])
        lines.extend(f"- {item}" for item in diagnosis["human_review_questions"][:10])

    lines.extend(
        [
            "",
            "## Checkpoint and diagnostic context",
            "",
            f"- Checkpoint reuse: `{json.dumps(payload.get('checkpoint_reuse') or {}, sort_keys=True)}`",
            f"- Metrics: `{json.dumps(payload.get('metrics') or {}, sort_keys=True)}`",
            f"- Machine record: `{payload.get('artifacts', {}).get('json', '')}`",
            f"- Markdown record: `{payload.get('artifacts', {}).get('markdown', '')}`",
            f"- Related debug artifacts: `{json.dumps(payload.get('related_debug_artifacts') or [])}`",
            "",
            "LLM diagnosis is advisory and must be verified against the machine findings before remediation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_errata_artifacts(
    recorder: ErrataRecorder,
    output_dir: Path,
) -> dict[str, Any]:
    """Write current artifacts and preserve changed anomalous history."""
    safe_stem = _safe_stem(recorder.source_path)
    safe_fingerprint = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(recorder.source_fingerprint)).strip("._") or "unknown"
    base = output_dir / f"{safe_stem}.{safe_fingerprint}.errata"
    json_path = base.with_suffix(".errata.json")
    markdown_path = base.with_suffix(".errata.md")
    artifact_paths = {"json": str(json_path), "markdown": str(markdown_path)}
    payload = recorder.to_payload(artifact_paths)

    if json_path.is_file():
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if (
            isinstance(previous, dict)
            and previous.get("material_digest") != payload.get("material_digest")
            and (
                _is_anomalous_status(str(previous.get("outcome", {}).get("status", "")))
                or _is_anomalous_status(str(payload.get("outcome", {}).get("status", "")))
            )
        ):
            history_dir = output_dir / "history"
            history_base = history_dir / f"{safe_stem}.{safe_fingerprint}.{recorder.run_id}.errata"
            history_json = history_base.with_suffix(".errata.json")
            history_markdown = history_base.with_suffix(".errata.md")
            history_dir.mkdir(parents=True, exist_ok=True)
            previous["artifacts"] = {
                "json": str(history_json),
                "markdown": str(history_markdown),
            }
            _atomic_write(history_json, json.dumps(previous, indent=2, ensure_ascii=True) + "\n")
            if markdown_path.is_file():
                _atomic_write(history_markdown, render_markdown(previous))

    _atomic_write(json_path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    _atomic_write(markdown_path, render_markdown(payload))
    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "payload": payload,
    }


def validate_errata_payload(payload: Any) -> list[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["errata payload must be an object"]
    if payload.get("contract_version") != ERRATA_CONTRACT_VERSION:
        errors.append(f"unsupported contract_version={payload.get('contract_version')}")
    for section in ("source", "run", "outcome", "findings", "llm_diagnosis"):
        if section not in payload:
            errors.append(f"missing section {section}")
    source = payload.get("source") or {}
    if not isinstance(source, dict):
        errors.append("source must be an object")
        source = {}
    else:
        if not str(source.get("path") or "").strip():
            errors.append("source is missing path")
        if not str(source.get("source_fingerprint") or "").strip():
            errors.append("source is missing source_fingerprint")
    outcome = payload.get("outcome") or {}
    if not isinstance(outcome, dict):
        errors.append("outcome must be an object")
        outcome = {}
    if outcome.get("status") not in _OUTCOME_STATUSES:
        errors.append(f"invalid outcome status={outcome.get('status')}")
    if not isinstance(payload.get("findings"), list):
        errors.append("findings must be an array")
    for index, finding in enumerate(payload.get("findings") or []):
        if not isinstance(finding, dict):
            errors.append(f"finding {index} must be an object")
            continue
        for field in ("finding_id", "severity", "code", "stage", "message"):
            if not str(finding.get(field) or "").strip():
                errors.append(f"finding {index} missing {field}")
    diagnosis = payload.get("llm_diagnosis")
    if not isinstance(diagnosis, dict):
        errors.append("llm_diagnosis must be an object")
        diagnosis = {}
    if diagnosis.get("status") == "completed":
        errors.extend(
            validate_diagnosis_payload(
                diagnosis,
                {str(item.get("finding_id")) for item in payload.get("findings") or []},
            )
        )
    return errors


def validate_diagnosis_payload(payload: Any, finding_ids: set[str]) -> list[str]:
    """Validate the strict advisory diagnosis shape and its grounding references."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["diagnosis must be an object"]
    for field in ("summary", "root_causes", "actions", "uncertainties", "human_review_questions"):
        if field not in payload:
            errors.append(f"completed diagnosis missing {field}")
    for field in ("summary",):
        if field in payload and not isinstance(payload[field], str):
            errors.append(f"diagnosis {field} must be a string")
    for field in ("root_causes", "actions", "uncertainties", "human_review_questions"):
        if field in payload and not isinstance(payload[field], list):
            errors.append(f"diagnosis {field} must be an array")
    for index, cause in enumerate(payload.get("root_causes") or []):
        if not isinstance(cause, dict):
            errors.append(f"diagnosis root cause {index} must be an object")
            continue
        if cause.get("category") not in _DIAGNOSIS_CATEGORIES:
            errors.append(f"diagnosis root cause {index} has invalid category")
        if not isinstance(cause.get("confidence"), (int, float, str)):
            errors.append(f"diagnosis root cause {index} has invalid confidence")
        if not isinstance(cause.get("finding_ids"), list) or not set(map(str, cause.get("finding_ids") or [])).issubset(finding_ids):
            errors.append(f"diagnosis root cause {index} references an unknown finding")
        for field in ("claim", "explanation"):
            if not isinstance(cause.get(field), str) or not cause.get(field).strip():
                errors.append(f"diagnosis root cause {index} missing {field}")
    for index, action in enumerate(payload.get("actions") or []):
        if not isinstance(action, dict):
            errors.append(f"diagnosis action {index} must be an object")
            continue
        if action.get("type") not in _ACTION_TYPES:
            allowed = ", ".join(DIAGNOSIS_ACTION_TYPES)
            errors.append(f"diagnosis action {index} has invalid type; expected one of: {allowed}")
        for field in ("priority", "action", "verification"):
            if not isinstance(action.get(field), str) or not action.get(field).strip():
                errors.append(f"diagnosis action {index} missing {field}")
    return errors


def errata_json_paths(root: Path, selected: str | None = None) -> list[Path]:
    if selected:
        path = Path(selected).expanduser()
        if not path.is_absolute():
            path = root / path
        return [path] if path.is_file() else []
    return sorted(root.rglob("*.errata.json")) if root.is_dir() else []
