"""Public result models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class DecisionStatus(StrEnum):
    ACCEPT = "ACCEPT"
    HOLD = "HOLD"
    BLOCK = "BLOCK"


@dataclass(frozen=True, slots=True)
class Decision:
    status: DecisionStatus
    reasons: tuple[str, ...]
    findings: tuple[str, ...]
    proposal_digest: str | None
    policy_version: str | None
    decided_at: str

    @property
    def executor_permitted(self) -> bool:
        return self.status is DecisionStatus.ACCEPT

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "org.causalcell.authorization-record.v0.1",
            "decision": self.status.value,
            "reasons": list(self.reasons),
            "findings": list(self.findings),
            "proposal_digest": self.proposal_digest,
            "policy_version": self.policy_version,
            "decided_at": self.decided_at,
            "executor_permitted": self.executor_permitted,
            "side_effect_executed": False,
        }


@dataclass(frozen=True, slots=True)
class EvidenceVerification:
    valid: bool
    errors: tuple[str, ...]
    bundle_id: str | None
    manifest_digest: str | None
    artifact_count: int

    def to_record(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "bundle_id": self.bundle_id,
            "manifest_digest": self.manifest_digest,
            "artifact_count": self.artifact_count,
        }


@dataclass(frozen=True, slots=True)
class CellRun:
    decision: Decision
    observation: dict[str, Any]
    continuity: dict[str, Any]
    bundle_path: Path
    verification: EvidenceVerification
