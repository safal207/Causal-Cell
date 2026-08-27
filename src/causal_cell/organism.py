"""Fixed linear multi-model organism built from guarded Causal Cells.

Organism v0.1 is deliberately narrow:

    guarded observer model -> guarded analyst model -> strict action draft
    -> trusted proposal compiler -> existing CausalCell action guard

Models provide data and a capability draft only. The trusted host supplies every
authority-critical action field.
"""

from __future__ import annotations

import copy
import math
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from .adapters import (
    AdapterIdentity,
    AdapterRegistry,
    ModelCall,
    ModelResult,
    ModelResultStatus,
    snapshot_model_result,
    validate_model_result,
)
from .canonical import (
    bind_proposal,
    digest_json,
    format_timestamp,
    parse_timestamp,
    snapshot_json,
)
from .guard import normalize_https_origin, validate_policy_document
from .models import CellRun, DecisionStatus
from .runtime import CausalCell, InMemoryNonceStore

MANIFEST_PROFILE = "org.causalcell.organism-manifest.v0.1"
ACTION_DRAFT_PROFILE = "org.causalcell.action-draft.v0.1"
DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
MODEL_STAGES = ("observer", "analyst")
PIPELINE_STAGES = ("observer", "analyst", "executor")
BUDGET_FIELDS = {"max_steps", "max_seconds", "max_cost", "max_fan_out", "max_retries"}
DATA_CLASSIFICATION_RANK = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
    "unknown": 4,
}
MAX_UTC_DATETIME = datetime.max.replace(tzinfo=UTC)


class OrganismStatus(StrEnum):
    COMPLETED = "COMPLETED"
    NO_ACTION = "NO_ACTION"
    HOLD = "HOLD"
    BLOCK = "BLOCK"
    FAILED = "FAILED"
    EFFECT_UNCERTAIN = "EFFECT_UNCERTAIN"


@dataclass(frozen=True, slots=True)
class OrganismDecision:
    status: DecisionStatus
    reasons: tuple[str, ...]
    manifest_digest: str | None
    decided_at: str


@dataclass(frozen=True, slots=True)
class OrganismPolicy:
    policy_version: str
    allowed_adapter_identity_digests: frozenset[str]
    allowed_capability_ids: frozenset[str]
    max_steps: int = 3
    max_seconds: int = 60
    max_model_calls: int = 2
    max_output_tokens_per_call: int = 4_000
    max_total_tokens: int = 8_000
    max_cost_microunits_per_call: int = 1_000_000
    max_total_cost_microunits: int = 2_000_000
    max_retries: int = 0
    max_fan_out: int = 1

    def __post_init__(self) -> None:
        allowlists_valid = (
            type(self.allowed_adapter_identity_digests) is frozenset
            and bool(self.allowed_adapter_identity_digests)
            and all(
                type(value) is str and bool(DIGEST_RE.fullmatch(value))
                for value in self.allowed_adapter_identity_digests
            )
            and type(self.allowed_capability_ids) is frozenset
            and bool(self.allowed_capability_ids)
            and all(
                type(value) is str and _nonempty(value)
                for value in self.allowed_capability_ids
            )
        )
        integers = (
            self.max_steps,
            self.max_seconds,
            self.max_model_calls,
            self.max_output_tokens_per_call,
            self.max_total_tokens,
            self.max_cost_microunits_per_call,
            self.max_total_cost_microunits,
            self.max_retries,
            self.max_fan_out,
        )
        if (
            not _nonempty(self.policy_version)
            or not allowlists_valid
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in integers
            )
            or self.max_steps < 3
            or self.max_seconds < 1
            or self.max_model_calls < 2
            or self.max_output_tokens_per_call < 1
            or self.max_total_tokens < 2 * self.max_output_tokens_per_call
            or self.max_cost_microunits_per_call < 0
            or self.max_total_cost_microunits
            < 2 * self.max_cost_microunits_per_call
            or self.max_retries < 0
            or self.max_fan_out < 1
        ):
            raise ValueError("invalid trusted organism policy")


@dataclass(frozen=True, slots=True)
class ManifestActivation:
    organism_id: str
    manifest_digest: str
    subject: str
    policy_version: str
    expires_at: str


class StaticActivationRegistry:
    """Application-owned manifest activations. Model output cannot add entries."""

    def __init__(self, activations: list[ManifestActivation]) -> None:
        if type(activations) is not list:
            raise ValueError("activations must be an exact list")
        for item in activations:
            if (
                type(item) is not ManifestActivation
                or not _nonempty(item.organism_id)
                or type(item.manifest_digest) is not str
                or not DIGEST_RE.fullmatch(item.manifest_digest)
                or not _nonempty(item.subject)
                or not _nonempty(item.policy_version)
                or not _nonempty(item.expires_at)
            ):
                raise ValueError("invalid manifest activation")
            try:
                parse_timestamp(item.expires_at)
            except (TypeError, ValueError):
                raise ValueError("invalid manifest activation") from None
        self._activations = tuple(activations)

    def _exact(
        self,
        *,
        organism_id: str,
        manifest_digest: str,
        subject: str,
        policy_version: str,
    ) -> tuple[ManifestActivation, ...]:
        return tuple(
            item
            for item in self._activations
            if item.organism_id == organism_id
            and item.manifest_digest == manifest_digest
            and item.subject == subject
            and item.policy_version == policy_version
        )

    def active_until(
        self,
        *,
        organism_id: str,
        manifest_digest: str,
        subject: str,
        policy_version: str,
        now: datetime,
    ) -> datetime | None:
        active: list[datetime] = []
        for item in self._exact(
            organism_id=organism_id,
            manifest_digest=manifest_digest,
            subject=subject,
            policy_version=policy_version,
        ):
            try:
                expires_at = parse_timestamp(item.expires_at)
            except (TypeError, ValueError):
                continue
            if expires_at > now:
                active.append(expires_at)
        return max(active) if active else None

    def decision(
        self,
        *,
        organism_id: str,
        manifest_digest: str,
        subject: str,
        policy_version: str,
        now: datetime,
    ) -> tuple[DecisionStatus, tuple[str, ...]]:
        same_organism = [
            item for item in self._activations if item.organism_id == organism_id
        ]
        if not same_organism:
            return DecisionStatus.HOLD, ("MANIFEST_ACTIVATION_REQUIRED",)
        exact = self._exact(
            organism_id=organism_id,
            manifest_digest=manifest_digest,
            subject=subject,
            policy_version=policy_version,
        )
        if not exact:
            return DecisionStatus.BLOCK, ("MANIFEST_ACTIVATION_MISMATCH",)
        if (
            self.active_until(
                organism_id=organism_id,
                manifest_digest=manifest_digest,
                subject=subject,
                policy_version=policy_version,
                now=now,
            )
            is not None
        ):
            return DecisionStatus.ACCEPT, ()
        reasons: list[str] = []
        parsed_any = False
        for item in exact:
            try:
                parse_timestamp(item.expires_at)
                parsed_any = True
            except (TypeError, ValueError):
                reasons.append("MANIFEST_ACTIVATION_INVALID")
        if parsed_any:
            reasons.insert(0, "MANIFEST_ACTIVATION_EXPIRED")
        return DecisionStatus.BLOCK, tuple(dict.fromkeys(reasons))


@dataclass(frozen=True, slots=True)
class RunContext:
    subject: str
    intent_id: str
    parent_cause: str
    auth_context_digest: str
    issued_at: str
    expires_at: str
    contains_secret: bool = False
    data_classification: str = "public"


@dataclass(frozen=True, slots=True)
class StaticCapability:
    capability_id: str
    action: str
    scope: str
    executor_id: str
    tool_origin: str
    tool_version: str
    tool_schema_digest: str
    target_state_digest: str
    reversibility: str
    risk_tier: str
    allowed_target_prefixes: tuple[str, ...]
    target_validator: Callable[[str], bool]
    allowed_argument_keys: frozenset[str]
    required_argument_keys: frozenset[str]
    argument_validator: Callable[[Mapping[str, Any]], bool]
    resource_budget: Mapping[str, int | float]
    contains_secret: bool = False
    destination: str | None = None
    data_classification: str = "public"


@dataclass(frozen=True, slots=True)
class PreparedAction:
    proposal: Mapping[str, Any]
    executor_id: str


@dataclass(frozen=True, slots=True)
class OrganismRun:
    status: OrganismStatus
    terminal_stage: str
    decision_reasons: tuple[str, ...]
    manifest_decision: OrganismDecision
    run_id: str
    trace_id: str
    model_results: tuple[ModelResult, ...]
    cell_runs: tuple[CellRun, ...]
    action_run: CellRun | None
    action_effect_boundary_started: bool
    model_calls: int
    reported_tokens: int
    reported_cost_microunits: int

    @property
    def executor_invoked(self) -> bool:
        return self.action_effect_boundary_started


class ProposalCompilationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class InMemoryOrganismStore:
    """Process-local semantic-run replay guard; not durable or distributed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_keys: set[str] = set()

    def consume_run(self, semantic_run_key: str) -> bool:
        with self._lock:
            if semantic_run_key in self._run_keys:
                return False
            self._run_keys.add(semantic_run_key)
            return True


def organism_manifest_digest(manifest: Mapping[str, Any]) -> str:
    digestible = snapshot_json(manifest)
    if not isinstance(digestible, dict):
        raise TypeError("organism manifest must be a JSON object")
    digestible.pop("manifest_digest", None)
    return digest_json(digestible)


def bind_organism_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    bound = snapshot_json(manifest)
    if not isinstance(bound, dict):
        raise TypeError("organism manifest must be a JSON object")
    bound["manifest_digest"] = organism_manifest_digest(bound)
    return bound


def _nonempty(value: Any) -> bool:
    if type(value) is not str:
        return False
    try:
        snapshot_json(value)
    except (TypeError, ValueError):
        return False
    return bool(value.strip()) and value == value.strip()


def _valid_integer(value: Any, *, minimum: int = 0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= minimum
    )


def _valid_resource_budget(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != BUDGET_FIELDS:
        return False
    for field in ("max_steps", "max_seconds"):
        if not _valid_integer(value.get(field), minimum=1):
            return False
    for field in ("max_fan_out", "max_retries"):
        if not _valid_integer(value.get(field), minimum=0):
            return False
    cost = value.get("max_cost")
    return (
        not isinstance(cost, bool)
        and isinstance(cost, (int, float))
        and math.isfinite(cost)
        and cost >= 0
    )


def _join_data_classification(left: str, right: str) -> str:
    if left not in DATA_CLASSIFICATION_RANK or right not in DATA_CLASSIFICATION_RANK:
        return "unknown"
    return max((left, right), key=DATA_CLASSIFICATION_RANK.__getitem__)


def _deadline_after(start: datetime, seconds: int) -> datetime:
    """Saturate valid near-maximum timestamps instead of escaping fail-closed paths."""

    try:
        return start + timedelta(seconds=seconds)
    except OverflowError:
        return MAX_UTC_DATETIME


def _target_matches_prefix(target: str, prefix: str) -> bool:
    if not _nonempty(prefix):
        return False
    if prefix.endswith(("/", ":", "#", "?", "=")):
        return target.startswith(prefix)
    if target == prefix:
        return True
    separators = ("/", "#", "?") if "://" in prefix else ("/", ":", "#", "?")
    return any(target.startswith(prefix + separator) for separator in separators)


def _looks_like_network_locator(target: str) -> bool:
    lowered = target.lower()
    return (
        target.startswith("//")
        or "://" in target
        or lowered.startswith(("http:", "https:"))
    )


def _canonical_https_target(target: str) -> str | None:
    origin = normalize_https_origin(target)
    if origin is None:
        return None
    try:
        parsed = urlsplit(target)
    except ValueError:
        return None
    if parsed.fragment:
        return None
    canonical = origin + (parsed.path or "/")
    if parsed.query:
        canonical += "?" + parsed.query
    return canonical


def _policy_statically_allows(
    policy: Mapping[str, Any],
    *,
    subject: str,
    agent: str,
    workload: str,
    action: str,
    scope: str,
    tool_origin: str,
    tool_version: str,
    tool_schema_digest: str,
    resource_budget: Mapping[str, int | float],
    destination: str | None,
    contains_secret: bool,
    data_classification: str,
) -> bool:
    if (
        subject not in policy["allowed_subjects"]
        or agent not in policy["allowed_agents"]
        or workload not in policy["allowed_workloads"]
        or action not in policy["allowed_actions"]
        or scope not in policy["allowed_scopes"]
        or scope not in policy["allowed_action_scopes"].get(action, [])
    ):
        return False
    if not any(
        isinstance(tool, Mapping)
        and tool.get("origin") == tool_origin
        and tool.get("version") == tool_version
        and tool.get("schema_digest") == tool_schema_digest
        for tool in policy["trusted_tools"]
    ):
        return False
    try:
        if any(
            resource_budget[field] > policy["max_resource_budget"][field]
            for field in BUDGET_FIELDS
        ):
            return False
    except (KeyError, TypeError):
        return False
    scope_is_network = scope.startswith("network.") or scope in policy[
        "network_scopes"
    ]
    if not scope_is_network:
        return destination is None
    canonical_destination = normalize_https_origin(destination)
    allowed_destinations = {
        normalized
        for item in policy["allowed_destinations"]
        if (normalized := normalize_https_origin(item)) is not None
    }
    if (
        canonical_destination is None
        or canonical_destination not in allowed_destinations
    ):
        return False
    sensitive = contains_secret or data_classification in {
        "confidential",
        "restricted",
        "unknown",
    }
    if not sensitive:
        return True
    secret_destinations = {
        normalized
        for item in policy["allowed_secret_destinations"]
        if (normalized := normalize_https_origin(item)) is not None
    }
    return canonical_destination in secret_destinations


def _manifest_decision(
    status: DecisionStatus,
    reasons: list[str] | tuple[str, ...],
    manifest: Mapping[str, Any],
    now: datetime,
) -> OrganismDecision:
    supplied = manifest.get("manifest_digest")
    return OrganismDecision(
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
        manifest_digest=supplied if isinstance(supplied, str) else None,
        decided_at=format_timestamp(now),
    )


def evaluate_organism_manifest(
    manifest: Mapping[str, Any],
    policy: OrganismPolicy,
    adapters: AdapterRegistry,
    activations: StaticActivationRegistry,
    *,
    subject: str,
    now: datetime | None = None,
) -> OrganismDecision:
    """Validate an exact fixed-topology manifest and its host activation."""

    now = (now or datetime.now(UTC)).astimezone(UTC)
    try:
        detached_manifest = snapshot_json(manifest)
        if not isinstance(detached_manifest, dict):
            raise TypeError("organism manifest must be a JSON object")
    except (TypeError, ValueError):
        return _manifest_decision(
            DecisionStatus.BLOCK,
            ("MANIFEST_MALFORMED",),
            {},
            now,
        )
    manifest = detached_manifest
    required = {
        "schema_version",
        "profile",
        "organism_id",
        "manifest_version",
        "manifest_digest",
        "policy_version",
        "pipeline",
        "limits",
    }
    reasons: list[str] = []
    if set(manifest) != required:
        reasons.append("MANIFEST_MALFORMED")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("profile") != MANIFEST_PROFILE
    ):
        reasons.append("MANIFEST_MALFORMED")
    for field in ("organism_id", "manifest_version", "policy_version"):
        if not _nonempty(manifest.get(field)):
            reasons.append("MANIFEST_MALFORMED")
    supplied_digest = manifest.get("manifest_digest")
    if not isinstance(supplied_digest, str) or not DIGEST_RE.fullmatch(supplied_digest):
        reasons.append("MANIFEST_MALFORMED")
    else:
        try:
            if supplied_digest != organism_manifest_digest(manifest):
                reasons.append("MANIFEST_DIGEST_MISMATCH")
        except (TypeError, ValueError):
            reasons.append("MANIFEST_MALFORMED")
    if manifest.get("policy_version") != policy.policy_version:
        reasons.append("MANIFEST_POLICY_MISMATCH")

    pipeline = manifest.get("pipeline")
    if not isinstance(pipeline, Mapping) or set(pipeline) != set(PIPELINE_STAGES):
        reasons.append("MANIFEST_TOPOLOGY_INVALID")
        pipeline = {}
    seen_cells: set[str] = set()
    seen_agents: set[str] = set()
    for stage in MODEL_STAGES:
        cell = pipeline.get(stage)
        expected_fields = {
            "cell_id",
            "agent_id",
            "adapter_id",
            "adapter_identity_digest",
            "invocation_action",
            "invocation_scope",
            "resource_budget",
        }
        if not isinstance(cell, Mapping) or set(cell) != expected_fields:
            reasons.append("MANIFEST_CELL_INVALID")
            continue
        scalar_fields = (
            "cell_id",
            "agent_id",
            "adapter_id",
            "adapter_identity_digest",
            "invocation_action",
            "invocation_scope",
        )
        scalars_valid = all(_nonempty(cell.get(field)) for field in scalar_fields)
        if (
            not scalars_valid
            or not DIGEST_RE.fullmatch(cell["adapter_identity_digest"])
        ):
            reasons.append("MANIFEST_CELL_INVALID")
        else:
            cell_id = cell["cell_id"]
            agent_id = cell["agent_id"]
            if cell_id in seen_cells or agent_id in seen_agents:
                reasons.append("MANIFEST_IDENTITY_CONFLICT")
            seen_cells.add(cell_id)
            seen_agents.add(agent_id)
        if not _valid_resource_budget(cell.get("resource_budget")):
            reasons.append("MANIFEST_BUDGET_INVALID")
        adapter = adapters.get(cell["adapter_id"]) if scalars_valid else None
        if scalars_valid and adapter is None:
            reasons.append("ADAPTER_UNKNOWN")
        elif adapter is not None:
            try:
                identity = adapter.identity
                if type(identity) is not AdapterIdentity:
                    raise TypeError("adapter identity must be exact")
                identity_digest = identity.identity_digest
            except Exception:
                reasons.append("ADAPTER_PROVENANCE_DENIED")
                continue
            if (
                cell.get("adapter_identity_digest") != identity_digest
                or identity_digest not in policy.allowed_adapter_identity_digests
            ):
                reasons.append("ADAPTER_PROVENANCE_DENIED")

    executor = pipeline.get("executor")
    expected_executor_fields = {"cell_id", "agent_id", "allowed_capability_ids"}
    if not isinstance(executor, Mapping) or set(executor) != expected_executor_fields:
        reasons.append("MANIFEST_CELL_INVALID")
    else:
        executor_ids_valid = _nonempty(executor.get("cell_id")) and _nonempty(
            executor.get("agent_id")
        )
        if not executor_ids_valid:
            reasons.append("MANIFEST_CELL_INVALID")
        else:
            if (
                executor["cell_id"] in seen_cells
                or executor["agent_id"] in seen_agents
            ):
                reasons.append("MANIFEST_IDENTITY_CONFLICT")
        capabilities = executor.get("allowed_capability_ids")
        if (
            not isinstance(capabilities, list)
            or not capabilities
            or any(not _nonempty(item) for item in capabilities)
            or len(set(capabilities)) != len(capabilities)
            or not set(capabilities).issubset(policy.allowed_capability_ids)
        ):
            reasons.append("CAPABILITY_DENIED")

    limits = manifest.get("limits")
    limit_fields = {
        "max_steps",
        "max_seconds",
        "max_model_calls",
        "max_output_tokens_per_call",
        "max_total_tokens",
        "max_cost_microunits_per_call",
        "max_total_cost_microunits",
        "max_retries",
        "max_fan_out",
    }
    if not isinstance(limits, Mapping) or set(limits) != limit_fields:
        reasons.append("MANIFEST_LIMITS_INVALID")
    else:
        if any(not _valid_integer(limits.get(field)) for field in limit_fields):
            reasons.append("MANIFEST_LIMITS_INVALID")
        elif (
            limits["max_steps"] != 3
            or limits["max_model_calls"] != 2
            or limits["max_retries"] != 0
            or limits["max_fan_out"] != 1
            or limits["max_seconds"] < 1
            or limits["max_output_tokens_per_call"] < 1
            or limits["max_total_tokens"]
            < 2 * limits["max_output_tokens_per_call"]
            or limits["max_total_cost_microunits"]
            < 2 * limits["max_cost_microunits_per_call"]
            or limits["max_steps"] > policy.max_steps
            or limits["max_seconds"] > policy.max_seconds
            or limits["max_model_calls"] > policy.max_model_calls
            or limits["max_output_tokens_per_call"]
            > policy.max_output_tokens_per_call
            or limits["max_total_tokens"] > policy.max_total_tokens
            or limits["max_cost_microunits_per_call"]
            > policy.max_cost_microunits_per_call
            or limits["max_total_cost_microunits"]
            > policy.max_total_cost_microunits
            or limits["max_retries"] > policy.max_retries
            or limits["max_fan_out"] > policy.max_fan_out
        ):
            reasons.append("MANIFEST_LIMITS_EXCEEDED")

    if reasons:
        return _manifest_decision(DecisionStatus.BLOCK, reasons, manifest, now)
    activation_status, activation_reasons = activations.decision(
        organism_id=manifest["organism_id"],
        manifest_digest=manifest["manifest_digest"],
        subject=subject,
        policy_version=manifest["policy_version"],
        now=now,
    )
    return _manifest_decision(activation_status, activation_reasons, manifest, now)


def validate_action_draft(draft: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        detached_draft = snapshot_json(draft)
    except (TypeError, ValueError):
        return ("ACTION_DRAFT_INVALID",)
    if not isinstance(detached_draft, dict):
        return ("ACTION_DRAFT_INVALID",)
    draft = detached_draft
    reasons: list[str] = []
    common = {"schema_version", "profile", "kind"}
    if (
        type(draft.get("schema_version")) is not int
        or draft.get("schema_version") != 1
        or draft.get("profile") != ACTION_DRAFT_PROFILE
    ):
        reasons.append("ACTION_DRAFT_INVALID")
    kind = draft.get("kind")
    if kind == "ACTION":
        if set(draft) != common | {"capability_id", "target", "arguments"}:
            reasons.append("ACTION_DRAFT_AUTHORITY_CONFLICT")
        if not _nonempty(draft.get("capability_id")) or not _nonempty(draft.get("target")):
            reasons.append("ACTION_DRAFT_INVALID")
        if not isinstance(draft.get("arguments"), dict):
            reasons.append("ACTION_DRAFT_INVALID")
        else:
            try:
                digest_json(draft["arguments"])
            except (TypeError, ValueError):
                reasons.append("ACTION_DRAFT_INVALID")
    elif kind == "NO_ACTION":
        if set(draft) != common | {"reason_codes"}:
            reasons.append("ACTION_DRAFT_AUTHORITY_CONFLICT")
        codes = draft.get("reason_codes")
        if (
            not isinstance(codes, list)
            or not codes
            or any(not _nonempty(code) for code in codes)
            or len(set(codes)) != len(codes)
        ):
            reasons.append("ACTION_DRAFT_INVALID")
    else:
        reasons.append("ACTION_DRAFT_INVALID")
    return tuple(dict.fromkeys(reasons))


class StaticProposalFactory:
    """Trusted host compiler. It never merges model dictionaries into authority."""

    def __init__(
        self,
        *,
        invocation_policy_version: str,
        action_policy_version: str,
        capabilities: list[StaticCapability],
    ) -> None:
        if (
            not _nonempty(invocation_policy_version)
            or not _nonempty(action_policy_version)
        ):
            raise ValueError("invalid proposal-factory policy version")
        if type(capabilities) is not list:
            raise ValueError("capabilities must be an exact list")
        self._invocation_policy_version = invocation_policy_version
        self._action_policy_version = action_policy_version
        normalized_capabilities: list[StaticCapability] = []
        for capability in capabilities:
            if type(capability) is not StaticCapability:
                raise ValueError("capability must be an exact StaticCapability")
            scalar_fields = (
                capability.capability_id,
                capability.action,
                capability.scope,
                capability.executor_id,
                capability.tool_origin,
                capability.tool_version,
                capability.tool_schema_digest,
                capability.target_state_digest,
                capability.reversibility,
                capability.risk_tier,
                capability.data_classification,
            )
            if (
                not all(_nonempty(value) for value in scalar_fields)
                or not DIGEST_RE.fullmatch(capability.tool_schema_digest)
                or not DIGEST_RE.fullmatch(capability.target_state_digest)
                or (
                    capability.destination is not None
                    and not _nonempty(capability.destination)
                )
            ):
                raise ValueError("invalid capability authority fields")
            if (
                type(capability.allowed_target_prefixes) is not tuple
                or type(capability.allowed_argument_keys) is not frozenset
                or type(capability.required_argument_keys) is not frozenset
                or any(
                    not _nonempty(item)
                    for item in capability.allowed_target_prefixes
                )
                or any(
                    not _nonempty(item)
                    for item in capability.allowed_argument_keys
                    | capability.required_argument_keys
                )
            ):
                raise ValueError("invalid capability collections")
            try:
                budget = snapshot_json(capability.resource_budget)
                if not isinstance(budget, dict):
                    raise TypeError("capability budget must be a JSON object")
                normalized = replace(
                    capability,
                    allowed_target_prefixes=tuple(
                        capability.allowed_target_prefixes
                    ),
                    allowed_argument_keys=frozenset(
                        capability.allowed_argument_keys
                    ),
                    required_argument_keys=frozenset(
                        capability.required_argument_keys
                    ),
                    resource_budget=MappingProxyType(budget),
                )
            except (TypeError, ValueError):
                raise ValueError("invalid capability definition") from None
            if normalized.destination is not None:
                canonical_destination = normalize_https_origin(
                    normalized.destination
                )
                try:
                    parsed_destination = urlsplit(normalized.destination)
                except ValueError:
                    parsed_destination = None
                if (
                    canonical_destination is None
                    or parsed_destination is None
                    or parsed_destination.path not in {"", "/"}
                    or bool(parsed_destination.query)
                    or bool(parsed_destination.fragment)
                ):
                    raise ValueError("invalid capability destination")
                normalized = replace(
                    normalized,
                    destination=canonical_destination,
                )
            if (
                normalized.reversibility != "reversible"
                or normalized.risk_tier not in {"low", "medium"}
                or not normalized.allowed_target_prefixes
                or any(
                    not _nonempty(item)
                    for item in normalized.allowed_target_prefixes
                )
                or not callable(normalized.target_validator)
                or not normalized.required_argument_keys.issubset(
                    normalized.allowed_argument_keys
                )
                or not callable(normalized.argument_validator)
                or not _valid_resource_budget(normalized.resource_budget)
                or not isinstance(normalized.contains_secret, bool)
                or normalized.data_classification
                not in DATA_CLASSIFICATION_RANK
                or (
                    normalized.scope.startswith("network.")
                    and normalized.destination is None
                )
            ):
                raise ValueError("invalid or approval-requiring capability")
            normalized_capabilities.append(normalized)
        self._capabilities = {
            item.capability_id: item
            for item in normalized_capabilities
        }
        if len(self._capabilities) != len(normalized_capabilities):
            raise ValueError("duplicate capability_id")

    @property
    def invocation_policy_version(self) -> str:
        return self._invocation_policy_version

    @property
    def action_policy_version(self) -> str:
        return self._action_policy_version

    def registry_reasons(
        self,
        capability_ids: list[str],
        executors: Mapping[str, Executor],
    ) -> tuple[str, ...]:
        for capability_id in capability_ids:
            capability = self._capabilities.get(capability_id)
            if capability is None:
                return ("CAPABILITY_UNKNOWN",)
            if not callable(executors.get(capability.executor_id)):
                return ("EXECUTOR_UNKNOWN",)
        return ()

    def action_policy_reasons(
        self,
        manifest: Mapping[str, Any],
        policy: Mapping[str, Any],
        context: RunContext,
    ) -> tuple[str, ...]:
        executor_cell = manifest["pipeline"]["executor"]
        for capability_id in executor_cell["allowed_capability_ids"]:
            capability = self._capabilities[capability_id]
            classification = _join_data_classification(
                context.data_classification,
                capability.data_classification,
            )
            if not _policy_statically_allows(
                policy,
                subject=context.subject,
                agent=executor_cell["agent_id"],
                workload=manifest["organism_id"],
                action=capability.action,
                scope=capability.scope,
                tool_origin=capability.tool_origin,
                tool_version=capability.tool_version,
                tool_schema_digest=capability.tool_schema_digest,
                resource_budget=capability.resource_budget,
                destination=capability.destination,
                contains_secret=(
                    context.contains_secret
                    or capability.contains_secret
                    or classification == "unknown"
                ),
                data_classification=classification,
            ):
                return ("ACTION_POLICY_INCOMPATIBLE",)
        return ()

    def model_invocation(
        self,
        *,
        manifest: Mapping[str, Any],
        stage: str,
        call: ModelCall,
        adapter_identity: AdapterIdentity,
        context: RunContext,
        parent_request_id: str | None,
    ) -> dict[str, Any]:
        cell = manifest["pipeline"][stage]
        proposal = {
            "schema_version": 1,
            "profile": "org.causalcell.action-proposal.v0.1",
            "trace_id": call.trace_id,
            "span_id": call.span_id,
            "parent_span_id": call.parent_span_id,
            "request_id": f"{call.run_id}:request:{stage}",
            "attempt_id": f"{call.run_id}:attempt:{stage}",
            "parent_request_id": parent_request_id,
            "retry_of_attempt_id": None,
            "agent": cell["agent_id"],
            "workload": manifest["organism_id"],
            "subject": context.subject,
            "method": "model_adapter",
            "intent_id": context.intent_id,
            "parent_cause": call.parent_cause,
            "action": cell["invocation_action"],
            "scope": cell["invocation_scope"],
            "target": f"model:{adapter_identity.provider}:{adapter_identity.model}",
            "target_state_digest": call.payload_digest,
            "reversibility": "reversible",
            "approval_ref": None,
            "nonce": f"nonce:{call.run_id}:{stage}",
            "risk_tier": "low" if stage == "observer" else "medium",
            "policy_version": self._invocation_policy_version,
            "arguments": {
                "model_call": call.to_record(include_payload=False),
                "adapter_identity_digest": adapter_identity.identity_digest,
            },
            "idempotency_key": f"{call.run_id}:{stage}",
            "issued_at": context.issued_at,
            "expires_at": context.expires_at,
            "tool_origin": adapter_identity.origin,
            "tool_version": adapter_identity.version,
            "tool_schema_digest": adapter_identity.schema_digest,
            "auth_context_digest": context.auth_context_digest,
            "contains_secret": context.contains_secret,
            "destination": adapter_identity.destination,
            "data_classification": context.data_classification,
            "delegation_chain": [],
            "resource_budget": copy.deepcopy(cell["resource_budget"]),
            "metadata": {
                "organism": {
                    "organism_id": manifest["organism_id"],
                    "manifest_digest": manifest["manifest_digest"],
                    "run_id": call.run_id,
                    "stage": stage,
                    "payload_digest": call.payload_digest,
                }
            },
            "untrusted_context": (
                []
                if stage == "observer"
                else [{"source": "inter_agent", "content_digest": call.payload_digest}]
            ),
        }
        return bind_proposal(proposal)

    def action_from_draft(
        self,
        *,
        draft: Mapping[str, Any],
        manifest: Mapping[str, Any],
        context: RunContext,
        run_id: str,
        trace_id: str,
        parent_span_id: str,
        parent_request_id: str,
        upstream_result_id: str,
        upstream_result_digest: str,
    ) -> PreparedAction:
        try:
            detached_draft = snapshot_json(draft)
            if not isinstance(detached_draft, dict):
                raise TypeError("action draft must be a JSON object")
        except (TypeError, ValueError):
            raise ProposalCompilationError("ACTION_DRAFT_INVALID") from None
        draft = detached_draft
        reasons = validate_action_draft(draft)
        if reasons:
            raise ProposalCompilationError(reasons[0])
        if draft["kind"] != "ACTION":
            raise ProposalCompilationError("NO_ACTION")
        capability_id = draft["capability_id"]
        capability = self._capabilities.get(capability_id)
        if capability is None:
            raise ProposalCompilationError("CAPABILITY_UNKNOWN")
        allowed = manifest["pipeline"]["executor"]["allowed_capability_ids"]
        if capability_id not in allowed:
            raise ProposalCompilationError("CAPABILITY_DENIED")
        target = draft["target"]
        if not any(
            _target_matches_prefix(target, prefix)
            for prefix in capability.allowed_target_prefixes
        ):
            raise ProposalCompilationError("TARGET_DENIED")
        try:
            target_valid = capability.target_validator(target)
        except Exception:
            target_valid = False
        if target_valid is not True:
            raise ProposalCompilationError("TARGET_DENIED")
        compiled_target = target
        if _looks_like_network_locator(target):
            compiled_target = _canonical_https_target(target)
            if (
                compiled_target is None
                or capability.destination is None
                or normalize_https_origin(compiled_target)
                != capability.destination
            ):
                raise ProposalCompilationError(
                    "TARGET_DESTINATION_MISMATCH"
                )
            try:
                canonical_target_valid = capability.target_validator(
                    compiled_target
                )
            except Exception:
                canonical_target_valid = False
            if canonical_target_valid is not True:
                raise ProposalCompilationError("TARGET_DENIED")
        arguments = snapshot_json(draft["arguments"])
        keys = set(arguments)
        if (
            not capability.required_argument_keys.issubset(keys)
            or not keys.issubset(capability.allowed_argument_keys)
        ):
            raise ProposalCompilationError("ARGUMENTS_DENIED")
        try:
            arguments_valid = capability.argument_validator(
                snapshot_json(arguments)
            )
        except Exception:
            arguments_valid = False
        if arguments_valid is not True:
            raise ProposalCompilationError("ARGUMENTS_DENIED")
        arguments_digest = digest_json(arguments)
        semantic_action_digest = digest_json(
            {
                "subject": context.subject,
                "intent_id": context.intent_id,
                "parent_cause": context.parent_cause,
                "action": capability.action,
                "scope": capability.scope,
                "target": compiled_target,
                "destination": capability.destination,
                "arguments_digest": arguments_digest,
            }
        )
        proposal = {
            "schema_version": 1,
            "profile": "org.causalcell.action-proposal.v0.1",
            "trace_id": trace_id,
            "span_id": f"{run_id}:span:action",
            "parent_span_id": parent_span_id,
            "request_id": f"{run_id}:request:action",
            "attempt_id": f"{run_id}:attempt:action",
            "parent_request_id": parent_request_id,
            "retry_of_attempt_id": None,
            "agent": manifest["pipeline"]["executor"]["agent_id"],
            "workload": manifest["organism_id"],
            "subject": context.subject,
            "method": "tool_call",
            "intent_id": context.intent_id,
            "parent_cause": upstream_result_id,
            "action": capability.action,
            "scope": capability.scope,
            "target": compiled_target,
            "target_state_digest": capability.target_state_digest,
            "reversibility": capability.reversibility,
            "approval_ref": None,
            "nonce": f"nonce:{run_id}:action",
            "risk_tier": capability.risk_tier,
            "policy_version": self._action_policy_version,
            "arguments": arguments,
            "idempotency_key": (
                "organism-action:"
                + semantic_action_digest.removeprefix("sha256:")
            ),
            "issued_at": context.issued_at,
            "expires_at": context.expires_at,
            "tool_origin": capability.tool_origin,
            "tool_version": capability.tool_version,
            "tool_schema_digest": capability.tool_schema_digest,
            "auth_context_digest": context.auth_context_digest,
            "contains_secret": (
                context.contains_secret
                or capability.contains_secret
                or _join_data_classification(
                    context.data_classification,
                    capability.data_classification,
                )
                == "unknown"
            ),
            "destination": capability.destination,
            "data_classification": _join_data_classification(
                context.data_classification,
                capability.data_classification,
            ),
            "delegation_chain": [],
            "resource_budget": copy.deepcopy(dict(capability.resource_budget)),
            "metadata": {
                "organism": {
                    "organism_id": manifest["organism_id"],
                    "manifest_digest": manifest["manifest_digest"],
                    "run_id": run_id,
                    "stage": "action",
                    "capability_id": capability_id,
                    "draft_digest": digest_json(draft),
                    "semantic_action_digest": semantic_action_digest,
                    "upstream_result_id": upstream_result_id,
                    "upstream_result_digest": upstream_result_digest,
                }
            },
            "untrusted_context": [
                {"source": "inter_agent", "content_digest": digest_json(draft)}
            ],
        }
        return PreparedAction(bind_proposal(proposal), capability.executor_id)


Executor = Callable[[Mapping[str, Any]], Any]


class OrganismRunner:
    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        organism_policy: OrganismPolicy,
        invocation_policy: Mapping[str, Any],
        action_policy: Mapping[str, Any],
        activations: StaticActivationRegistry,
        adapters: AdapterRegistry,
        proposal_factory: StaticProposalFactory,
        executors: Mapping[str, Executor],
        evidence_root: str | Path,
        clock: Callable[[], datetime] | None = None,
        nonce_store: InMemoryNonceStore | None = None,
        organism_store: InMemoryOrganismStore | None = None,
    ) -> None:
        if type(organism_policy) is not OrganismPolicy:
            raise ValueError("organism_policy must be an exact OrganismPolicy")
        if type(proposal_factory) is not StaticProposalFactory:
            raise ValueError(
                "proposal_factory must be an exact StaticProposalFactory"
            )
        if type(activations) is not StaticActivationRegistry:
            raise ValueError(
                "activations must be an exact StaticActivationRegistry"
            )
        if type(adapters) is not AdapterRegistry:
            raise ValueError("adapters must be an exact AdapterRegistry")
        if type(executors) is not dict or any(
            not _nonempty(executor_id) or not callable(executor)
            for executor_id, executor in executors.items()
        ):
            raise ValueError("invalid executor registry")
        detached_policies: list[dict[str, Any]] = []
        for policy in (invocation_policy, action_policy):
            try:
                detached_policy = snapshot_json(policy)
                if not isinstance(detached_policy, dict):
                    raise TypeError("policy must be a JSON object")
            except (TypeError, ValueError):
                raise ValueError("invalid organism policy document") from None
            if not validate_policy_document(detached_policy):
                raise ValueError("invalid organism policy document")
            if (
                detached_policy.get("require_approval_for_irreversible")
                is not False
                or detached_policy.get("approval_required_risk_tiers") != []
                or detached_policy.get("approvals") != {}
            ):
                raise ValueError(
                    "Organism v0.1 excludes approval-required policies; "
                    "use a future plan/resume protocol"
                )
            detached_policies.append(detached_policy)
        detached_manifest = snapshot_json(manifest)
        if not isinstance(detached_manifest, dict):
            raise ValueError("organism manifest must be a JSON object")
        self._manifest = detached_manifest
        self._organism_policy = organism_policy
        self._invocation_policy, self._action_policy = detached_policies
        self._activations = activations
        self._adapters = adapters
        self._proposal_factory = proposal_factory
        self._executors = dict(executors)
        self._configuration_reasons: tuple[str, ...] = ()
        if (
            self._invocation_policy.get("policy_version")
            != proposal_factory.invocation_policy_version
            or self._action_policy.get("policy_version")
            != proposal_factory.action_policy_version
        ):
            self._configuration_reasons = (
                "PROPOSAL_FACTORY_POLICY_MISMATCH",
            )
        self._evidence_root = Path(evidence_root)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._nonces = nonce_store or InMemoryNonceStore()
        self._organisms = organism_store or InMemoryOrganismStore()

    def _result(
        self,
        *,
        status: OrganismStatus,
        terminal_stage: str,
        reasons: tuple[str, ...],
        manifest_decision: OrganismDecision,
        run_id: str,
        trace_id: str,
        model_results: list[ModelResult],
        cell_runs: list[CellRun],
        action_run: CellRun | None,
        attempted_model_calls: int | None = None,
        action_effect_boundary_started: bool | None = None,
        reported_tokens: int | None = None,
        reported_cost_microunits: int | None = None,
    ) -> OrganismRun:
        return OrganismRun(
            status=status,
            terminal_stage=terminal_stage,
            decision_reasons=reasons,
            manifest_decision=manifest_decision,
            run_id=run_id,
            trace_id=trace_id,
            model_results=tuple(model_results),
            cell_runs=tuple(cell_runs),
            action_run=action_run,
            action_effect_boundary_started=(
                action_effect_boundary_started
                if action_effect_boundary_started is not None
                else bool(
                    action_run
                    and action_run.observation.get(
                        "executor_invoked",
                        False,
                    )
                )
            ),
            model_calls=(
                attempted_model_calls
                if attempted_model_calls is not None
                else sum(
                    int(item.observation.get("executor_invoked", False))
                    for item in cell_runs
                )
            ),
            reported_tokens=(
                reported_tokens
                if reported_tokens is not None
                else sum(
                    value
                    for item in model_results
                    for value in (item.input_tokens, item.output_tokens)
                    if _valid_integer(value)
                )
            ),
            reported_cost_microunits=(
                reported_cost_microunits
                if reported_cost_microunits is not None
                else sum(
                    item.cost_microunits
                    for item in model_results
                    if _valid_integer(item.cost_microunits)
                )
            ),
        )

    def run(
        self,
        root_input: Mapping[str, Any],
        context: RunContext,
    ) -> OrganismRun:
        run_id = f"orun-{uuid.uuid4().hex}"
        trace_id = f"otrace-{uuid.uuid4().hex}"
        now = self._clock().astimezone(UTC)
        empty_decision = _manifest_decision(
            DecisionStatus.BLOCK, ("CONTEXT_INVALID",), self._manifest, now
        )
        if (
            not isinstance(root_input, Mapping)
            or type(context) is not RunContext
            or not all(
                _nonempty(value)
                for value in (
                    context.subject,
                    context.intent_id,
                    context.parent_cause,
                    context.auth_context_digest,
                    context.issued_at,
                    context.expires_at,
                )
            )
            or not DIGEST_RE.fullmatch(context.auth_context_digest)
            or type(context.contains_secret) is not bool
            or type(context.data_classification) is not str
            or context.data_classification not in DATA_CLASSIFICATION_RANK
        ):
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="context",
                reasons=("CONTEXT_INVALID",),
                manifest_decision=empty_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=[],
                cell_runs=[],
                action_run=None,
            )
        try:
            root_snapshot = snapshot_json(root_input)
            if not isinstance(root_snapshot, dict):
                raise TypeError("root input must be a JSON object")
            root_digest = digest_json(root_snapshot)
            issued_at = parse_timestamp(context.issued_at)
            expires_at = parse_timestamp(context.expires_at)
        except (TypeError, ValueError):
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="context",
                reasons=("CONTEXT_INVALID",),
                manifest_decision=empty_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=[],
                cell_runs=[],
                action_run=None,
            )
        if issued_at > now or expires_at <= now or expires_at <= issued_at:
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="context",
                reasons=("CONTEXT_TIME_INVALID",),
                manifest_decision=empty_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=[],
                cell_runs=[],
                action_run=None,
            )

        manifest_decision = evaluate_organism_manifest(
            self._manifest,
            self._organism_policy,
            self._adapters,
            self._activations,
            subject=context.subject,
            now=now,
        )
        if manifest_decision.status is not DecisionStatus.ACCEPT:
            return self._result(
                status=(
                    OrganismStatus.HOLD
                    if manifest_decision.status is DecisionStatus.HOLD
                    else OrganismStatus.BLOCK
                ),
                terminal_stage="manifest",
                reasons=manifest_decision.reasons,
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=[],
                cell_runs=[],
                action_run=None,
            )

        executor_cell = self._manifest["pipeline"]["executor"]
        registry_reasons = self._proposal_factory.registry_reasons(
            executor_cell["allowed_capability_ids"],
            self._executors,
        )
        configuration_reasons = list(
            self._configuration_reasons + registry_reasons
        )
        if not configuration_reasons:
            for stage in MODEL_STAGES:
                cell = self._manifest["pipeline"][stage]
                adapter = self._adapters.get(cell["adapter_id"])
                try:
                    if adapter is None:
                        raise TypeError("adapter missing")
                    identity = adapter.identity
                    if type(identity) is not AdapterIdentity:
                        raise TypeError("adapter identity must be exact")
                except Exception:
                    configuration_reasons.append(
                        "ADAPTER_PROVENANCE_DENIED"
                    )
                    break
                if not _policy_statically_allows(
                    self._invocation_policy,
                    subject=context.subject,
                    agent=cell["agent_id"],
                    workload=self._manifest["organism_id"],
                    action=cell["invocation_action"],
                    scope=cell["invocation_scope"],
                    tool_origin=identity.origin,
                    tool_version=identity.version,
                    tool_schema_digest=identity.schema_digest,
                    resource_budget=cell["resource_budget"],
                    destination=identity.destination,
                    contains_secret=context.contains_secret,
                    data_classification=context.data_classification,
                ):
                    configuration_reasons.append(
                        "INVOCATION_POLICY_INCOMPATIBLE"
                    )
                    break
        if not configuration_reasons:
            configuration_reasons.extend(
                self._proposal_factory.action_policy_reasons(
                    self._manifest,
                    self._action_policy,
                    context,
                )
            )
        if configuration_reasons:
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="manifest",
                reasons=tuple(dict.fromkeys(configuration_reasons)),
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=[],
                cell_runs=[],
                action_run=None,
            )

        activation_expires_at = self._activations.active_until(
            organism_id=self._manifest["organism_id"],
            manifest_digest=self._manifest["manifest_digest"],
            subject=context.subject,
            policy_version=self._manifest["policy_version"],
            now=now,
        )
        if activation_expires_at is None:
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="manifest",
                reasons=("MANIFEST_ACTIVATION_INVALID",),
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=[],
                cell_runs=[],
                action_run=None,
            )

        semantic_run_key = digest_json(
            {
                "organism_id": self._manifest["organism_id"],
                "manifest_digest": self._manifest["manifest_digest"],
                "subject": context.subject,
                "intent_id": context.intent_id,
                "parent_cause": context.parent_cause,
                "root_input_digest": root_digest,
            }
        )
        if not self._organisms.consume_run(semantic_run_key):
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="manifest",
                reasons=("ORGANISM_REPLAYED",),
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=[],
                cell_runs=[],
                action_run=None,
            )

        limits = self._manifest["limits"]
        started_at = now
        effective_deadline = min(
            expires_at,
            activation_expires_at,
            _deadline_after(started_at, limits["max_seconds"]),
        )
        effective_context = RunContext(
            subject=context.subject,
            intent_id=context.intent_id,
            parent_cause=context.parent_cause,
            auth_context_digest=context.auth_context_digest,
            issued_at=context.issued_at,
            expires_at=format_timestamp(effective_deadline),
            contains_secret=context.contains_secret,
            data_classification=context.data_classification,
        )
        def revalidate_manifest(at: datetime) -> OrganismDecision:
            return evaluate_organism_manifest(
                self._manifest,
                self._organism_policy,
                self._adapters,
                self._activations,
                subject=context.subject,
                now=at,
            )

        model_results: list[ModelResult] = []
        cell_runs: list[CellRun] = []
        total_tokens = 0
        total_cost = 0
        parent_span_id: str | None = None
        parent_request_id: str | None = None
        parent_cause = context.parent_cause
        payload: Mapping[str, Any] = root_snapshot
        upstream_result_id: str | None = None
        upstream_result_digest: str | None = None

        for stage in MODEL_STAGES:
            current = self._clock().astimezone(UTC)
            runtime_manifest_decision = revalidate_manifest(current)
            if runtime_manifest_decision.status is not DecisionStatus.ACCEPT:
                provenance_uncertain = (
                    bool(cell_runs)
                    and "ADAPTER_PROVENANCE_DENIED"
                    in runtime_manifest_decision.reasons
                )
                return self._result(
                    status=(
                        OrganismStatus.EFFECT_UNCERTAIN
                        if provenance_uncertain
                        else (
                            OrganismStatus.HOLD
                            if runtime_manifest_decision.status
                            is DecisionStatus.HOLD
                            else OrganismStatus.BLOCK
                        )
                    ),
                    terminal_stage=stage,
                    reasons=(
                        ("ADAPTER_PROVENANCE_CHANGED_EFFECT_UNCERTAIN",)
                        if provenance_uncertain
                        else runtime_manifest_decision.reasons
                    ),
                    manifest_decision=runtime_manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            if current >= effective_deadline:
                return self._result(
                    status=OrganismStatus.BLOCK,
                    terminal_stage=stage,
                    reasons=("ORGANISM_DEADLINE_EXCEEDED",),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            if (
                len(model_results) >= limits["max_model_calls"]
                or total_tokens + limits["max_output_tokens_per_call"]
                > limits["max_total_tokens"]
                or total_cost + limits["max_cost_microunits_per_call"]
                > limits["max_total_cost_microunits"]
            ):
                return self._result(
                    status=OrganismStatus.BLOCK,
                    terminal_stage=stage,
                    reasons=("BUDGET_RESERVATION_FAILED",),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )

            cell = self._manifest["pipeline"][stage]
            cell_deadline = min(
                effective_deadline,
                _deadline_after(
                    current,
                    cell["resource_budget"]["max_seconds"],
                ),
            )
            stage_context = RunContext(
                subject=effective_context.subject,
                intent_id=effective_context.intent_id,
                parent_cause=effective_context.parent_cause,
                auth_context_digest=effective_context.auth_context_digest,
                issued_at=effective_context.issued_at,
                expires_at=format_timestamp(cell_deadline),
                contains_secret=effective_context.contains_secret,
                data_classification=effective_context.data_classification,
            )
            adapter = self._adapters.get(cell["adapter_id"])
            if adapter is None:
                return self._result(
                    status=OrganismStatus.BLOCK,
                    terminal_stage=stage,
                    reasons=("ADAPTER_UNKNOWN",),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            try:
                adapter_identity = adapter.identity
                if type(adapter_identity) is not AdapterIdentity:
                    raise TypeError("adapter identity must be exact")
                adapter_identity_digest = adapter_identity.identity_digest
            except Exception:
                return self._result(
                    status=OrganismStatus.BLOCK,
                    terminal_stage=stage,
                    reasons=("ADAPTER_PROVENANCE_DENIED",),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            if adapter_identity_digest != cell["adapter_identity_digest"]:
                return self._result(
                    status=OrganismStatus.BLOCK,
                    terminal_stage=stage,
                    reasons=("ADAPTER_PROVENANCE_DENIED",),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            span_id = f"{run_id}:span:{stage}"
            call = ModelCall(
                run_id=run_id,
                organism_id=self._manifest["organism_id"],
                stage=stage,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                intent_id=context.intent_id,
                parent_cause=parent_cause,
                payload=copy.deepcopy(dict(payload)),
                payload_digest=digest_json(payload),
                deadline_at=format_timestamp(cell_deadline),
                max_output_tokens=limits["max_output_tokens_per_call"],
                data_classification=effective_context.data_classification,
            )
            proposal = self._proposal_factory.model_invocation(
                manifest=self._manifest,
                stage=stage,
                call=call,
                adapter_identity=adapter_identity,
                context=stage_context,
                parent_request_id=parent_request_id,
            )
            result_box: list[ModelResult] = []
            result_reasons_box: list[tuple[str, ...]] = []
            observed_usage_box: list[tuple[int, int]] = []
            expected_identity_digest = cell["adapter_identity_digest"]
            adapter_invocation_started = False
            provenance_changed_after_invoke = False

            def invoke_adapter(
                _: Mapping[str, Any],
                *,
                adapter_for_call: Any = adapter,
                call_for_adapter: ModelCall = call,
                expected_digest: str = expected_identity_digest,
                identity_for_call: AdapterIdentity = adapter_identity,
                results: list[ModelResult] = result_box,
                invalid_reasons: list[tuple[str, ...]] = result_reasons_box,
                observed_usage: list[tuple[int, int]] = observed_usage_box,
            ) -> dict[str, Any]:
                nonlocal adapter_invocation_started
                nonlocal provenance_changed_after_invoke
                if adapter_for_call.identity.identity_digest != expected_digest:
                    raise RuntimeError("adapter provenance changed before invoke")
                adapter_invocation_started = True
                raw_result = adapter_for_call.invoke(call_for_adapter)
                if type(raw_result) is ModelResult:
                    raw_usage = (
                        raw_result.input_tokens,
                        raw_result.output_tokens,
                        raw_result.cost_microunits,
                    )
                    if all(
                        type(value) is int
                        and value >= 0
                        and value.bit_length() <= 4_096
                        for value in raw_usage
                    ):
                        observed_usage.append(
                            (
                                raw_result.input_tokens
                                + raw_result.output_tokens,
                                raw_result.cost_microunits,
                            )
                        )
                try:
                    post_identity_digest = (
                        adapter_for_call.identity.identity_digest
                    )
                except Exception:
                    provenance_changed_after_invoke = True
                    raise
                if post_identity_digest != expected_digest:
                    provenance_changed_after_invoke = True
                    raise RuntimeError("adapter provenance changed after invoke")
                try:
                    result = snapshot_model_result(raw_result)
                except Exception:
                    reason = (
                        "MODEL_RESULT_INVALID"
                        if type(raw_result) is not ModelResult
                        else "MODEL_OUTPUT_INVALID"
                    )
                    invalid_reasons.append((reason,))
                    return {
                        "schema_version": 1,
                        "profile": "org.causalcell.invalid-model-result.v0.1",
                        "status": "INVALID",
                        "reason_codes": [reason],
                    }
                result_reasons = validate_model_result(
                    result,
                    identity_for_call,
                    call_for_adapter,
                )
                if result_reasons:
                    invalid_reasons.append(result_reasons)
                    return {
                        "schema_version": 1,
                        "profile": "org.causalcell.invalid-model-result.v0.1",
                        "status": "INVALID",
                        "reason_codes": list(result_reasons),
                    }
                results.append(result)
                return result.to_record()

            try:
                cell_run = CausalCell(
                    self._invocation_policy,
                    self._evidence_root,
                    nonce_store=self._nonces,
                    clock=self._clock,
                ).execute(proposal, invoke_adapter)
            except Exception:
                uncertain = adapter_invocation_started
                observed_tokens = sum(item[0] for item in observed_usage_box)
                observed_cost = sum(item[1] for item in observed_usage_box)
                return self._result(
                    status=(
                        OrganismStatus.EFFECT_UNCERTAIN
                        if uncertain
                        else OrganismStatus.FAILED
                    ),
                    terminal_stage=stage,
                    reasons=(
                        (
                            "ADAPTER_PROVENANCE_CHANGED_EFFECT_UNCERTAIN"
                            if provenance_changed_after_invoke
                            else (
                                "MODEL_CELL_RUNTIME_ERROR_EFFECT_UNCERTAIN"
                                if uncertain
                                else "MODEL_CELL_RUNTIME_ERROR"
                            )
                        ),
                    ),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                    attempted_model_calls=len(cell_runs) + int(uncertain),
                    reported_tokens=total_tokens + observed_tokens,
                    reported_cost_microunits=total_cost + observed_cost,
                )
            cell_runs.append(cell_run)
            if cell_run.decision.status is not DecisionStatus.ACCEPT:
                return self._result(
                    status=(
                        OrganismStatus.HOLD
                        if cell_run.decision.status is DecisionStatus.HOLD
                        else OrganismStatus.BLOCK
                    ),
                    terminal_stage=stage,
                    reasons=cell_run.decision.reasons,
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            if (
                cell_run.observation["status"] != "EXECUTOR_RETURNED"
                or (not result_box and not result_reasons_box)
            ):
                uncertain = adapter_invocation_started
                observed_tokens = sum(item[0] for item in observed_usage_box)
                observed_cost = sum(item[1] for item in observed_usage_box)
                return self._result(
                    status=(
                        OrganismStatus.EFFECT_UNCERTAIN
                        if uncertain
                        else OrganismStatus.BLOCK
                    ),
                    terminal_stage=stage,
                    reasons=(
                        (
                            "ADAPTER_PROVENANCE_CHANGED_EFFECT_UNCERTAIN"
                            if provenance_changed_after_invoke
                            else (
                                "MODEL_EFFECT_UNCERTAIN"
                                if uncertain
                                else "ADAPTER_PROVENANCE_DENIED"
                            )
                        ),
                    ),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                    attempted_model_calls=(
                        sum(
                            int(item.observation.get("executor_invoked", False))
                            for item in cell_runs[:-1]
                        )
                        + int(uncertain)
                    ),
                    reported_tokens=total_tokens + observed_tokens,
                    reported_cost_microunits=total_cost + observed_cost,
                )
            if result_reasons_box:
                observed_tokens = sum(item[0] for item in observed_usage_box)
                observed_cost = sum(item[1] for item in observed_usage_box)
                return self._result(
                    status=OrganismStatus.BLOCK,
                    terminal_stage=stage,
                    reasons=result_reasons_box[0],
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                    reported_tokens=total_tokens + observed_tokens,
                    reported_cost_microunits=total_cost + observed_cost,
                )
            result = result_box[0]
            model_results.append(result)
            actual_tokens = result.input_tokens + result.output_tokens
            total_tokens += actual_tokens
            total_cost += result.cost_microunits
            if (
                result.output_tokens > limits["max_output_tokens_per_call"]
                or result.cost_microunits > limits["max_cost_microunits_per_call"]
                or total_tokens > limits["max_total_tokens"]
                or total_cost > limits["max_total_cost_microunits"]
            ):
                return self._result(
                    status=OrganismStatus.BLOCK,
                    terminal_stage=stage,
                    reasons=("BUDGET_USAGE_EXCEEDED",),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            if result.status is not ModelResultStatus.RETURNED:
                return self._result(
                    status=OrganismStatus.FAILED,
                    terminal_stage=stage,
                    reasons=(
                        (
                            "MODEL_REFUSED"
                            if result.status is ModelResultStatus.REFUSED
                            else "MODEL_ERROR"
                        ),
                    ),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            joined_classification = _join_data_classification(
                effective_context.data_classification,
                result.data_classification,
            )
            effective_context = replace(
                effective_context,
                contains_secret=(
                    effective_context.contains_secret
                    or result.contains_secret
                    or joined_classification == "unknown"
                ),
                data_classification=joined_classification,
            )
            current = self._clock().astimezone(UTC)
            runtime_manifest_decision = revalidate_manifest(current)
            if runtime_manifest_decision.status is not DecisionStatus.ACCEPT:
                provenance_uncertain = (
                    "ADAPTER_PROVENANCE_DENIED"
                    in runtime_manifest_decision.reasons
                )
                return self._result(
                    status=(
                        OrganismStatus.EFFECT_UNCERTAIN
                        if provenance_uncertain
                        else (
                            OrganismStatus.HOLD
                            if runtime_manifest_decision.status
                            is DecisionStatus.HOLD
                            else OrganismStatus.BLOCK
                        )
                    ),
                    terminal_stage=stage,
                    reasons=(
                        ("ADAPTER_PROVENANCE_CHANGED_EFFECT_UNCERTAIN",)
                        if provenance_uncertain
                        else runtime_manifest_decision.reasons
                    ),
                    manifest_decision=runtime_manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            if current >= cell_deadline:
                return self._result(
                    status=OrganismStatus.BLOCK,
                    terminal_stage=stage,
                    reasons=(
                        (
                            "ORGANISM_DEADLINE_EXCEEDED"
                            if cell_deadline == effective_deadline
                            else "MODEL_CALL_DEADLINE_EXCEEDED"
                        ),
                    ),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            if result.output is None or result.output_digest is None:
                return self._result(
                    status=OrganismStatus.BLOCK,
                    terminal_stage=stage,
                    reasons=("MODEL_OUTPUT_INVALID",),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            upstream_result_digest = result.output_digest
            upstream_result_id = (
                f"model-result:{run_id}:{stage}:{result.output_digest.removeprefix('sha256:')}"
            )
            payload = {
                "upstream_result": snapshot_json(result.output),
                "upstream_result_id": upstream_result_id,
                "upstream_result_digest": upstream_result_digest,
            }
            parent_span_id = span_id
            parent_request_id = proposal["request_id"]
            parent_cause = upstream_result_id

        analyst_result = model_results[-1]
        assert analyst_result.output is not None
        draft = analyst_result.output
        draft_reasons = validate_action_draft(draft)
        if draft_reasons:
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="action_compile",
                reasons=draft_reasons,
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=model_results,
                cell_runs=cell_runs,
                action_run=None,
            )
        if draft["kind"] == "NO_ACTION":
            return self._result(
                status=OrganismStatus.NO_ACTION,
                terminal_stage="analyst",
                reasons=tuple(draft["reason_codes"]),
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=model_results,
                cell_runs=cell_runs,
                action_run=None,
            )

        assert upstream_result_id is not None
        assert upstream_result_digest is not None
        assert parent_span_id is not None
        assert parent_request_id is not None
        try:
            prepared = self._proposal_factory.action_from_draft(
                draft=draft,
                manifest=self._manifest,
                context=effective_context,
                run_id=run_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                parent_request_id=parent_request_id,
                upstream_result_id=upstream_result_id,
                upstream_result_digest=upstream_result_digest,
            )
        except ProposalCompilationError as exc:
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="action_compile",
                reasons=(exc.code,),
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=model_results,
                cell_runs=cell_runs,
                action_run=None,
            )
        action_proposal = prepared.proposal
        executor_cell = self._manifest["pipeline"]["executor"]
        if (
            action_proposal.get("parent_cause") != upstream_result_id
            or action_proposal.get("intent_id") != context.intent_id
            or action_proposal.get("subject") != context.subject
            or action_proposal.get("workload") != self._manifest["organism_id"]
            or action_proposal.get("agent") != executor_cell["agent_id"]
        ):
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="action_guard",
                reasons=("ACTION_CAUSAL_BINDING_MISMATCH",),
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=model_results,
                cell_runs=cell_runs,
                action_run=None,
            )
        if (
            action_proposal.get("action") == "create_organism"
            or action_proposal.get("scope") == "organism.spawn"
        ):
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="action_guard",
                reasons=("NESTED_SPAWN_DENIED",),
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=model_results,
                cell_runs=cell_runs,
                action_run=None,
            )
        current = self._clock().astimezone(UTC)
        runtime_manifest_decision = revalidate_manifest(current)
        if runtime_manifest_decision.status is not DecisionStatus.ACCEPT:
            provenance_uncertain = (
                "ADAPTER_PROVENANCE_DENIED"
                in runtime_manifest_decision.reasons
            )
            return self._result(
                status=(
                    OrganismStatus.EFFECT_UNCERTAIN
                    if provenance_uncertain
                    else (
                        OrganismStatus.HOLD
                        if runtime_manifest_decision.status
                        is DecisionStatus.HOLD
                        else OrganismStatus.BLOCK
                    )
                ),
                terminal_stage="action_guard",
                reasons=(
                    ("ADAPTER_PROVENANCE_CHANGED_EFFECT_UNCERTAIN",)
                    if provenance_uncertain
                    else runtime_manifest_decision.reasons
                ),
                manifest_decision=runtime_manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=model_results,
                cell_runs=cell_runs,
                action_run=None,
            )
        if current >= effective_deadline:
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="action_guard",
                reasons=("ORGANISM_DEADLINE_EXCEEDED",),
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=model_results,
                cell_runs=cell_runs,
                action_run=None,
            )
        executor = self._executors.get(prepared.executor_id)
        if executor is None:
            return self._result(
                status=OrganismStatus.BLOCK,
                terminal_stage="action_guard",
                reasons=("EXECUTOR_UNKNOWN",),
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=model_results,
                cell_runs=cell_runs,
                action_run=None,
            )
        action_executor_started = False

        def invoke_action(proposal: Mapping[str, Any]) -> Any:
            nonlocal action_executor_started
            action_executor_started = True
            return executor(proposal)

        try:
            action_run = CausalCell(
                self._action_policy,
                self._evidence_root,
                nonce_store=self._nonces,
                clock=self._clock,
            ).execute(action_proposal, invoke_action)
        except Exception:
            return self._result(
                status=(
                    OrganismStatus.EFFECT_UNCERTAIN
                    if action_executor_started
                    else OrganismStatus.FAILED
                ),
                terminal_stage="action",
                reasons=(
                    (
                        "ACTION_CELL_RUNTIME_ERROR_EFFECT_UNCERTAIN"
                        if action_executor_started
                        else "ACTION_CELL_RUNTIME_ERROR"
                    ),
                ),
                manifest_decision=manifest_decision,
                run_id=run_id,
                trace_id=trace_id,
                model_results=model_results,
                cell_runs=cell_runs,
                action_run=None,
                action_effect_boundary_started=action_executor_started,
            )
        if action_run.decision.status is DecisionStatus.HOLD:
            status = OrganismStatus.HOLD
        elif action_run.decision.status is DecisionStatus.BLOCK:
            status = OrganismStatus.BLOCK
        elif action_run.observation["status"] == "EXECUTOR_ERROR":
            status = OrganismStatus.EFFECT_UNCERTAIN
        else:
            status = OrganismStatus.COMPLETED
        action_reasons = action_run.decision.reasons
        if (
            status is OrganismStatus.EFFECT_UNCERTAIN
            and not action_reasons
        ):
            action_reasons = ("ACTION_EFFECT_UNCERTAIN",)
        return self._result(
            status=status,
            terminal_stage="action",
            reasons=action_reasons,
            manifest_decision=manifest_decision,
            run_id=run_id,
            trace_id=trace_id,
            model_results=model_results,
            cell_runs=cell_runs,
            action_run=action_run,
        )
