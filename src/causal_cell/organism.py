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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping

from .adapters import (
    AdapterIdentity,
    AdapterRegistry,
    ModelCall,
    ModelResult,
    ModelResultStatus,
    validate_model_result,
)
from .canonical import bind_proposal, digest_json, format_timestamp, parse_timestamp
from .models import CellRun, DecisionStatus
from .runtime import CausalCell, InMemoryNonceStore


MANIFEST_PROFILE = "org.causalcell.organism-manifest.v0.1"
ACTION_DRAFT_PROFILE = "org.causalcell.action-draft.v0.1"
DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
MODEL_STAGES = ("observer", "analyst")
PIPELINE_STAGES = ("observer", "analyst", "executor")
BUDGET_FIELDS = {"max_steps", "max_seconds", "max_cost", "max_fan_out", "max_retries"}


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
            not self.policy_version
            or any(isinstance(value, bool) or not isinstance(value, int) for value in integers)
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
        self._activations = tuple(activations)

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
        for item in same_organism:
            if (
                item.manifest_digest == manifest_digest
                and item.subject == subject
                and item.policy_version == policy_version
            ):
                try:
                    if parse_timestamp(item.expires_at) <= now:
                        return DecisionStatus.BLOCK, ("MANIFEST_ACTIVATION_EXPIRED",)
                except (TypeError, ValueError):
                    return DecisionStatus.BLOCK, ("MANIFEST_ACTIVATION_INVALID",)
                return DecisionStatus.ACCEPT, ()
        return DecisionStatus.BLOCK, ("MANIFEST_ACTIVATION_MISMATCH",)


@dataclass(frozen=True, slots=True)
class RunContext:
    subject: str
    intent_id: str
    parent_cause: str
    auth_context_digest: str
    issued_at: str
    expires_at: str
    action_approval_ref: str | None = None
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
    allowed_argument_keys: frozenset[str]
    required_argument_keys: frozenset[str]
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
    model_calls: int
    reported_tokens: int
    reported_cost_microunits: int

    @property
    def executor_invoked(self) -> bool:
        return bool(self.action_run and self.action_run.observation["executor_invoked"])


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
    digestible = copy.deepcopy(dict(manifest))
    digestible.pop("manifest_digest", None)
    return digest_json(digestible)


def bind_organism_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    bound = copy.deepcopy(dict(manifest))
    bound["manifest_digest"] = organism_manifest_digest(bound)
    return bound


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


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
    if manifest.get("schema_version") != 1 or manifest.get("profile") != MANIFEST_PROFILE:
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
        if not all(
            _nonempty(cell.get(field))
            for field in (
                "cell_id",
                "agent_id",
                "adapter_id",
                "adapter_identity_digest",
                "invocation_action",
                "invocation_scope",
            )
        ):
            reasons.append("MANIFEST_CELL_INVALID")
        if cell.get("cell_id") in seen_cells or cell.get("agent_id") in seen_agents:
            reasons.append("MANIFEST_IDENTITY_CONFLICT")
        seen_cells.add(str(cell.get("cell_id")))
        seen_agents.add(str(cell.get("agent_id")))
        if not _valid_resource_budget(cell.get("resource_budget")):
            reasons.append("MANIFEST_BUDGET_INVALID")
        adapter = adapters.get(str(cell.get("adapter_id")))
        if adapter is None:
            reasons.append("ADAPTER_UNKNOWN")
        else:
            identity_digest = adapter.identity.identity_digest
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
        if not _nonempty(executor.get("cell_id")) or not _nonempty(executor.get("agent_id")):
            reasons.append("MANIFEST_CELL_INVALID")
        if executor.get("cell_id") in seen_cells or executor.get("agent_id") in seen_agents:
            reasons.append("MANIFEST_IDENTITY_CONFLICT")
        capabilities = executor.get("allowed_capability_ids")
        if (
            not isinstance(capabilities, list)
            or not capabilities
            or any(not _nonempty(item) for item in capabilities)
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
    if not isinstance(draft, Mapping):
        return ("ACTION_DRAFT_INVALID",)
    reasons: list[str] = []
    common = {"schema_version", "profile", "kind"}
    if draft.get("schema_version") != 1 or draft.get("profile") != ACTION_DRAFT_PROFILE:
        reasons.append("ACTION_DRAFT_INVALID")
    kind = draft.get("kind")
    if kind == "ACTION":
        if set(draft) != common | {"capability_id", "target", "arguments"}:
            reasons.append("ACTION_DRAFT_AUTHORITY_CONFLICT")
        if not _nonempty(draft.get("capability_id")) or not _nonempty(draft.get("target")):
            reasons.append("ACTION_DRAFT_INVALID")
        if not isinstance(draft.get("arguments"), Mapping):
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
        self._invocation_policy_version = invocation_policy_version
        self._action_policy_version = action_policy_version
        self._capabilities = {item.capability_id: item for item in capabilities}
        if len(self._capabilities) != len(capabilities):
            raise ValueError("duplicate capability_id")

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
        if not any(target.startswith(prefix) for prefix in capability.allowed_target_prefixes):
            raise ProposalCompilationError("TARGET_DENIED")
        arguments = copy.deepcopy(dict(draft["arguments"]))
        keys = set(arguments)
        if (
            not capability.required_argument_keys.issubset(keys)
            or not keys.issubset(capability.allowed_argument_keys)
        ):
            raise ProposalCompilationError("ARGUMENTS_DENIED")
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
            "target": target,
            "target_state_digest": capability.target_state_digest,
            "reversibility": capability.reversibility,
            "approval_ref": context.action_approval_ref,
            "nonce": f"nonce:{run_id}:action",
            "risk_tier": capability.risk_tier,
            "policy_version": self._action_policy_version,
            "arguments": arguments,
            "idempotency_key": f"{run_id}:action:{capability_id}",
            "issued_at": context.issued_at,
            "expires_at": context.expires_at,
            "tool_origin": capability.tool_origin,
            "tool_version": capability.tool_version,
            "tool_schema_digest": capability.tool_schema_digest,
            "auth_context_digest": context.auth_context_digest,
            "contains_secret": capability.contains_secret,
            "destination": capability.destination,
            "data_classification": capability.data_classification,
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
        self._manifest = copy.deepcopy(dict(manifest))
        self._organism_policy = organism_policy
        self._invocation_policy = copy.deepcopy(dict(invocation_policy))
        self._action_policy = copy.deepcopy(dict(action_policy))
        self._activations = activations
        self._adapters = adapters
        self._proposal_factory = proposal_factory
        self._executors = dict(executors)
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
            model_calls=sum(
                int(item.observation.get("executor_invoked", False))
                for item in cell_runs
            ),
            reported_tokens=sum(
                item.input_tokens + item.output_tokens for item in model_results
            ),
            reported_cost_microunits=sum(
                item.cost_microunits for item in model_results
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
            or not isinstance(context.contains_secret, bool)
            or context.data_classification
            not in {"public", "internal", "confidential", "restricted", "unknown"}
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
            root_digest = digest_json(root_input)
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
            started_at + timedelta(seconds=limits["max_seconds"]),
        )
        effective_context = RunContext(
            subject=context.subject,
            intent_id=context.intent_id,
            parent_cause=context.parent_cause,
            auth_context_digest=context.auth_context_digest,
            issued_at=context.issued_at,
            expires_at=format_timestamp(effective_deadline),
            action_approval_ref=context.action_approval_ref,
            contains_secret=context.contains_secret,
            data_classification=context.data_classification,
        )
        model_results: list[ModelResult] = []
        cell_runs: list[CellRun] = []
        total_tokens = 0
        total_cost = 0
        parent_span_id: str | None = None
        parent_request_id: str | None = None
        parent_cause = context.parent_cause
        payload: Mapping[str, Any] = copy.deepcopy(dict(root_input))
        upstream_result_id: str | None = None
        upstream_result_digest: str | None = None

        for stage in MODEL_STAGES:
            current = self._clock().astimezone(UTC)
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
            adapter_identity = adapter.identity
            if adapter_identity.identity_digest != cell["adapter_identity_digest"]:
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
                deadline_at=format_timestamp(effective_deadline),
                max_output_tokens=limits["max_output_tokens_per_call"],
                data_classification=context.data_classification,
            )
            proposal = self._proposal_factory.model_invocation(
                manifest=self._manifest,
                stage=stage,
                call=call,
                adapter_identity=adapter_identity,
                context=effective_context,
                parent_request_id=parent_request_id,
            )
            result_box: list[ModelResult] = []
            expected_identity_digest = cell["adapter_identity_digest"]

            def invoke_adapter(_: Mapping[str, Any]) -> dict[str, Any]:
                if adapter.identity.identity_digest != expected_identity_digest:
                    raise RuntimeError("adapter provenance changed")
                result = adapter.invoke(call)
                if adapter.identity.identity_digest != expected_identity_digest:
                    raise RuntimeError("adapter provenance changed")
                result_box.append(result)
                return result.to_record()

            cell_run = CausalCell(
                self._invocation_policy,
                self._evidence_root,
                nonce_store=self._nonces,
                clock=self._clock,
            ).execute(proposal, invoke_adapter)
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
            if cell_run.observation["status"] != "EXECUTOR_RETURNED" or not result_box:
                return self._result(
                    status=OrganismStatus.FAILED,
                    terminal_stage=stage,
                    reasons=("MODEL_ADAPTER_ERROR",),
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
            result = result_box[0]
            result_reasons = validate_model_result(result, adapter_identity, call)
            model_results.append(result)
            if result_reasons:
                return self._result(
                    status=OrganismStatus.BLOCK,
                    terminal_stage=stage,
                    reasons=result_reasons,
                    manifest_decision=manifest_decision,
                    run_id=run_id,
                    trace_id=trace_id,
                    model_results=model_results,
                    cell_runs=cell_runs,
                    action_run=None,
                )
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
            if self._clock().astimezone(UTC) >= effective_deadline:
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
                "upstream_result": copy.deepcopy(dict(result.output)),
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
        if self._clock().astimezone(UTC) >= effective_deadline:
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
        action_run = CausalCell(
            self._action_policy,
            self._evidence_root,
            nonce_store=self._nonces,
            clock=self._clock,
        ).execute(action_proposal, executor)
        if action_run.decision.status is DecisionStatus.HOLD:
            status = OrganismStatus.HOLD
        elif action_run.decision.status is DecisionStatus.BLOCK:
            status = OrganismStatus.BLOCK
        elif action_run.observation["status"] == "EXECUTOR_ERROR":
            status = OrganismStatus.EFFECT_UNCERTAIN
        else:
            status = OrganismStatus.COMPLETED
        return self._result(
            status=status,
            terminal_stage="action",
            reasons=action_run.decision.reasons,
            manifest_decision=manifest_decision,
            run_id=run_id,
            trace_id=trace_id,
            model_results=model_results,
            cell_runs=cell_runs,
            action_run=action_run,
        )
