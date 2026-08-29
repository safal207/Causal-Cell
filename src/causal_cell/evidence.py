"""Unique, manifest-bound evidence bundles and independent local verification."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, digest_bytes, digest_json
from .models import EvidenceVerification

BUNDLE_RE = re.compile(r"^cc-[a-f0-9]{32}$")
RECORD_FILES = {
    "intent": "intent.json",
    "action": "proposal.json",
    "authorization": "authorization.json",
    "result": "observation.json",
    "response_integrity": "response-integrity.json",
    "causal_audit": "causal-audit.json",
    "continuity": "ltp-continuity-input.json",
    "replay": "replay-trace.json",
    "verification": "runtime-verification.json",
}
RECORD_ORDER = tuple(RECORD_FILES)
REQUIRED_EVIDENCE_ROLES = frozenset((*RECORD_ORDER, "ledger"))


class EvidenceError(RuntimeError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json_strict(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_pairs)


def _loads_json_strict(value: str) -> Any:
    return json.loads(value, object_pairs_hook=_reject_duplicate_pairs)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _artifact(path: Path, role: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "evidence_role": role,
        "size": len(payload),
        "sha256": digest_bytes(payload),
    }


def _ledger_events(artifacts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for sequence, role in enumerate(RECORD_ORDER, start=1):
        artifact = artifacts[role]
        event = {
            "schema_version": 1,
            "profile": "org.causalcell.hash-linked-ledger-event.v0.1",
            "sequence": sequence,
            "record_kind": role,
            "record_ref": f"{role}:{artifact['sha256']}",
            "payload_digest": artifact["sha256"],
            "previous_event_hash": previous_hash,
        }
        event["event_hash"] = digest_json(event)
        events.append(event)
        previous_hash = event["event_hash"]
    return events


def write_bundle(
    evidence_root: str | Path,
    records: Mapping[str, Mapping[str, Any]],
    *,
    created_at: str,
) -> tuple[Path, EvidenceVerification]:
    """Write one new evidence bundle and immediately reopen it for verification."""

    if set(records) != set(RECORD_ORDER):
        missing = set(RECORD_ORDER) - set(records)
        extra = set(records) - set(RECORD_ORDER)
        raise EvidenceError(
            f"record roles mismatch; missing={sorted(missing)} extra={sorted(extra)}"
        )

    root = Path(evidence_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise EvidenceError("evidence root must be a real directory")
    root = root.resolve(strict=True)
    bundle_id = f"cc-{uuid.uuid4().hex}"
    bundle = root / bundle_id
    bundle.mkdir(mode=0o700)

    artifacts_by_role: dict[str, dict[str, Any]] = {}
    for role in RECORD_ORDER:
        destination = bundle / RECORD_FILES[role]
        _write_exclusive(destination, canonical_bytes(records[role]) + b"\n")
        artifacts_by_role[role] = _artifact(destination, role)

    ledger_path = bundle / "ledger.jsonl"
    ledger_payload = b"".join(
        canonical_bytes(event) + b"\n" for event in _ledger_events(artifacts_by_role)
    )
    _write_exclusive(ledger_path, ledger_payload)
    ledger_artifact = _artifact(ledger_path, "ledger")

    artifacts = [*artifacts_by_role.values(), ledger_artifact]
    artifacts.sort(key=lambda item: item["path"])
    manifest = {
        "schema_version": 1,
        "profile": "org.causalcell.evidence-manifest.v0.1",
        "bundle_id": bundle_id,
        "created_at": created_at,
        "required_evidence_roles": sorted(REQUIRED_EVIDENCE_ROLES),
        "artifacts": artifacts,
        "manifest_scope": "Artifacts are byte-bound; manifest.json excludes itself.",
        "claim_boundary": (
            "Local byte integrity and hash-chain continuity only; no authenticity, "
            "external finality, sandbox, or exactly-once guarantee."
        ),
    }
    _write_exclusive(bundle / "manifest.json", canonical_bytes(manifest) + b"\n")

    verification = verify_bundle(bundle)
    if not verification.valid:
        raise EvidenceError(f"new evidence bundle failed verification: {verification.errors}")
    return bundle, verification


def _safe_artifact_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value not in {"", ".", "..", "manifest.json"}
        and "/" not in value
        and "\\" not in value
        and Path(value).name == value
    )


def _verify_ledger(
    path: Path, artifact_by_role: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return ["LEDGER_UNREADABLE"]
    if len(lines) != len(RECORD_ORDER):
        errors.append("LEDGER_EVENT_COUNT_MISMATCH")
        return errors
    previous_hash: str | None = None
    for index, (line, expected_role) in enumerate(
        zip(lines, RECORD_ORDER, strict=True), start=1
    ):
        try:
            event = _loads_json_strict(line)
        except (json.JSONDecodeError, ValueError):
            errors.append("LEDGER_JSON_INVALID")
            continue
        if not isinstance(event, Mapping):
            errors.append("LEDGER_EVENT_INVALID")
            continue
        if event.get("sequence") != index or event.get("record_kind") != expected_role:
            errors.append("LEDGER_SEQUENCE_MISMATCH")
        if event.get("profile") != "org.causalcell.hash-linked-ledger-event.v0.1":
            errors.append("LEDGER_PROFILE_INVALID")
        if event.get("previous_event_hash") != previous_hash:
            errors.append("LEDGER_PREVIOUS_HASH_MISMATCH")
        artifact = artifact_by_role.get(expected_role)
        if not artifact or event.get("payload_digest") != artifact.get("sha256"):
            errors.append("LEDGER_PAYLOAD_MISMATCH")
        if artifact and event.get("record_ref") != (
            f"{expected_role}:{artifact.get('sha256')}"
        ):
            errors.append("LEDGER_RECORD_REF_MISMATCH")
        digestible = dict(event)
        supplied_hash = digestible.pop("event_hash", None)
        expected_hash = digest_json(digestible)
        if supplied_hash != expected_hash:
            errors.append("LEDGER_EVENT_HASH_MISMATCH")
        previous_hash = supplied_hash if isinstance(supplied_hash, str) else None
    return errors


def verify_bundle(bundle_path: str | Path) -> EvidenceVerification:
    """Independently reopen and verify one local bundle."""

    bundle = Path(bundle_path)
    errors: list[str] = []
    if bundle.is_symlink() or not bundle.is_dir() or not BUNDLE_RE.fullmatch(bundle.name):
        return EvidenceVerification(False, ("BUNDLE_PATH_INVALID",), None, None, 0)
    bundle = bundle.resolve(strict=True)
    manifest_path = bundle / "manifest.json"
    try:
        manifest = load_json_strict(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return EvidenceVerification(False, ("MANIFEST_INVALID",), bundle.name, None, 0)
    manifest_digest = digest_bytes(manifest_path.read_bytes())
    if not isinstance(manifest, Mapping):
        return EvidenceVerification(
            False, ("MANIFEST_INVALID",), bundle.name, manifest_digest, 0
        )
    if manifest.get("profile") != "org.causalcell.evidence-manifest.v0.1":
        errors.append("MANIFEST_PROFILE_INVALID")
    if manifest.get("bundle_id") != bundle.name:
        errors.append("BUNDLE_ID_MISMATCH")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return EvidenceVerification(
            False,
            tuple((*errors, "MANIFEST_ARTIFACTS_INVALID")),
            bundle.name,
            manifest_digest,
            0,
        )

    artifact_by_role: dict[str, Mapping[str, Any]] = {}
    expected_names: set[str] = {"manifest.json"}
    seen_names: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            errors.append("MANIFEST_ARTIFACT_INVALID")
            continue
        name = artifact.get("path")
        role = artifact.get("evidence_role")
        if not _safe_artifact_name(name):
            errors.append("ARTIFACT_PATH_UNSAFE")
            continue
        if name in seen_names or not isinstance(role, str) or role in artifact_by_role:
            errors.append("ARTIFACT_DUPLICATE")
            continue
        seen_names.add(name)
        expected_names.add(name)
        artifact_by_role[role] = artifact
        path = bundle / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != bundle:
            errors.append("ARTIFACT_PATH_INVALID")
            continue
        payload = path.read_bytes()
        if artifact.get("size") != len(payload):
            errors.append("ARTIFACT_SIZE_MISMATCH")
        if artifact.get("sha256") != digest_bytes(payload):
            errors.append("ARTIFACT_DIGEST_MISMATCH")

    try:
        entries = list(bundle.iterdir())
    except OSError:
        entries = []
        errors.append("BUNDLE_UNREADABLE")
    actual_names: set[str] = set()
    for entry in entries:
        actual_names.add(entry.name)
        if entry.is_symlink() or not entry.is_file():
            errors.append("UNSAFE_OR_NONFILE_ENTRY")
    if actual_names != expected_names:
        errors.append("BUNDLE_INVENTORY_MISMATCH")
    if set(artifact_by_role) != REQUIRED_EVIDENCE_ROLES:
        errors.append("EVIDENCE_ROLE_SET_MISMATCH")
    if set(manifest.get("required_evidence_roles", [])) != REQUIRED_EVIDENCE_ROLES:
        errors.append("MANIFEST_ROLE_SET_MISMATCH")

    ledger_artifact = artifact_by_role.get("ledger")
    if ledger_artifact and _safe_artifact_name(ledger_artifact.get("path")):
        errors.extend(_verify_ledger(bundle / str(ledger_artifact["path"]), artifact_by_role))
    else:
        errors.append("LEDGER_MISSING")
    return EvidenceVerification(
        valid=not errors,
        errors=tuple(dict.fromkeys(errors)),
        bundle_id=bundle.name,
        manifest_digest=manifest_digest,
        artifact_count=len(artifacts),
    )
