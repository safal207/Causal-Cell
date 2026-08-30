"""Pure pre-execution guard for canonical action proposals."""

from __future__ import annotations

import ipaddress
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from .canonical import (
    digest_json,
    format_timestamp,
    parse_timestamp,
    proposal_digest,
    require_aware_utc,
)
from .models import Decision, DecisionStatus

PROPOSAL_PROFILE = "org.causalcell.action-proposal.v0.1"
POLICY_PROFILE = "org.causalcell.policy.v0.1"
DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")

REQUIRED_FIELDS = {
    "schema_version",
    "profile",
    "trace_id",
    "span_id",
    "parent_span_id",
    "request_id",
    "attempt_id",
    "parent_request_id",
    "retry_of_attempt_id",
    "agent",
    "workload",
    "subject",
    "method",
    "intent_id",
    "parent_cause",
    "action",
    "scope",
    "target",
    "target_state_digest",
    "reversibility",
    "approval_ref",
    "nonce",
    "risk_tier",
    "policy_version",
    "proposal_digest",
    "arguments",
    "arguments_digest",
    "idempotency_key",
    "issued_at",
    "expires_at",
    "tool_origin",
    "tool_version",
    "tool_schema_digest",
    "auth_context_digest",
    "contains_secret",
    "destination",
    "data_classification",
    "delegation_chain",
    "resource_budget",
}

OPTIONAL_FIELDS = {"metadata", "untrusted_context"}
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
NONEMPTY_FIELDS = {
    "trace_id",
    "span_id",
    "request_id",
    "attempt_id",
    "agent",
    "workload",
    "subject",
    "method",
    "action",
    "scope",
    "target",
    "idempotency_key",
    "tool_origin",
    "tool_version",
}
RISK_LEVELS = {"low", "medium", "high", "critical"}
REVERSIBILITY_LEVELS = {"reversible", "irreversible", "unknown"}
DATA_CLASSES = {"public", "internal", "confidential", "restricted", "unknown"}
BUDGET_FIELDS = {"max_steps", "max_seconds", "max_cost", "max_fan_out", "max_retries"}
UNTRUSTED_SOURCE_FINDINGS = {
    "web": "UNTRUSTED_WEB_INSTRUCTION_IGNORED",
    "memory": "UNTRUSTED_MEMORY_INSTRUCTION_IGNORED",
    "tool_output": "UNTRUSTED_TOOL_OUTPUT_INSTRUCTION_IGNORED",
    "inter_agent": "UNTRUSTED_INTER_AGENT_INSTRUCTION_IGNORED",
    "retrieval": "UNTRUSTED_RETRIEVAL_INSTRUCTION_IGNORED",
}


def _nonempty(value: Any) -> bool:
    if type(value) is not str:
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return bool(value.strip()) and value == value.strip()


def _valid_cost(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _decision(
    status: DecisionStatus,
    reasons: list[str],
    findings: list[str],
    proposal: Mapping[str, Any],
    now: datetime,
) -> Decision:
    proposal_value = proposal.get("proposal_digest")
    policy_value = proposal.get("policy_version")
    return Decision(
        status=status,
        reasons=_unique(reasons),
        findings=_unique(findings),
        proposal_digest=proposal_value if isinstance(proposal_value, str) else None,
        policy_version=policy_value if isinstance(policy_value, str) else None,
        decided_at=format_timestamp(now),
    )


def normalize_https_origin(value: Any) -> str | None:
    """Return the canonical HTTPS origin used for egress authorization."""

    if type(value) is not str or not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None
    host = parsed.hostname.lower().rstrip(".")
    if not host:
        return None
    if "%" in host:
        return None
    try:
        host = ipaddress.ip_address(host).compressed.lower()
    except ValueError:
        pass
    try:
        port = parsed.port
    except ValueError:
        return None
    rendered_host = f"[{host}]" if ":" in host else host
    authority = (
        rendered_host
        if port in (None, 443)
        else f"{rendered_host}:{port}"
    )
    return f"https://{authority}"


def normalize_local_model_origin(value: Any) -> str | None:
    """Return an HTTP origin only when it is a literal loopback endpoint.

    Local model servers such as Ollama commonly expose an HTTP-only API.  This
    deliberately excludes hostnames (including ``localhost``), private-network
    addresses, userinfo, and scoped IPv6 literals so the exception cannot turn
    into a general clear-text egress path.
    """

    if type(value) is not str or not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme.lower() != "http" or not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None
    host = parsed.hostname.lower().rstrip(".")
    if not host or "%" in host:
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    if not address.is_loopback:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    rendered_host = f"[{address.compressed.lower()}]" if address.version == 6 else str(address)
    authority = rendered_host if port in (None, 80) else f"{rendered_host}:{port}"
    return f"http://{authority}"


def normalize_egress_origin(value: Any, scope: Any) -> str | None:
    """Canonicalize a destination under the transport rules for its scope."""

    https_origin = normalize_https_origin(value)
    if https_origin is not None:
        return https_origin
    if scope == "network.local_model":
        return normalize_local_model_origin(value)
    return None


def _untrusted_findings(proposal: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    context = proposal.get("untrusted_context", [])
    if not isinstance(context, list):
        return findings
    for item in context:
        if not isinstance(item, Mapping):
            continue
        source = item.get("source")
        findings.append(
            UNTRUSTED_SOURCE_FINDINGS.get(
                str(source), "UNTRUSTED_CONTEXT_INSTRUCTION_IGNORED"
            )
        )
    return findings


def _structural_reasons(proposal: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not _nonempty(proposal.get("intent_id")):
        reasons.append("MISSING_INTENT")
    if not _nonempty(proposal.get("parent_cause")):
        reasons.append("MISSING_CAUSAL_PARENT")
    if not _nonempty(proposal.get("nonce")):
        reasons.append("MISSING_NONCE")

    missing = REQUIRED_FIELDS - proposal.keys()
    if missing - {"intent_id", "parent_cause", "nonce"}:
        reasons.append("MALFORMED_PROPOSAL")
    if proposal.keys() - ALLOWED_FIELDS:
        reasons.append("UNKNOWN_PROPOSAL_FIELD")
    if (
        type(proposal.get("schema_version")) is not int
        or proposal.get("schema_version") != 1
        or proposal.get("profile") != PROPOSAL_PROFILE
    ):
        reasons.append("MALFORMED_PROPOSAL")
    if any(not _nonempty(proposal.get(field)) for field in NONEMPTY_FIELDS):
        reasons.append("MALFORMED_PROPOSAL")
    for nullable_identifier in (
        "parent_span_id",
        "parent_request_id",
        "retry_of_attempt_id",
        "approval_ref",
    ):
        value = proposal.get(nullable_identifier)
        if value is not None and not _nonempty(value):
            reasons.append("MALFORMED_PROPOSAL")
    reversibility = proposal.get("reversibility")
    if type(reversibility) is not str or reversibility not in REVERSIBILITY_LEVELS:
        reasons.append("MALFORMED_PROPOSAL")
    risk_tier = proposal.get("risk_tier")
    if type(risk_tier) is not str or risk_tier not in RISK_LEVELS:
        reasons.append("MALFORMED_PROPOSAL")
    data_classification = proposal.get("data_classification")
    if (
        type(data_classification) is not str
        or data_classification not in DATA_CLASSES
    ):
        reasons.append("MALFORMED_PROPOSAL")
    if not isinstance(proposal.get("arguments"), Mapping):
        reasons.append("MALFORMED_PROPOSAL")
    if not isinstance(proposal.get("contains_secret"), bool):
        reasons.append("MALFORMED_PROPOSAL")
    destination = proposal.get("destination")
    if destination is not None and not _nonempty(destination):
        reasons.append("MALFORMED_PROPOSAL")
    if not isinstance(proposal.get("delegation_chain"), list) or any(
        not _nonempty(item) for item in proposal.get("delegation_chain", [])
    ):
        reasons.append("MALFORMED_PROPOSAL")
    for digest_field in (
        "proposal_digest",
        "arguments_digest",
        "target_state_digest",
        "tool_schema_digest",
        "auth_context_digest",
    ):
        value = proposal.get(digest_field)
        if not isinstance(value, str) or not DIGEST_RE.fullmatch(value):
            reasons.append("MALFORMED_PROPOSAL")

    budget = proposal.get("resource_budget")
    if not isinstance(budget, Mapping) or set(budget) != BUDGET_FIELDS:
        reasons.append("MALFORMED_PROPOSAL")
    else:
        positive_integer_fields = ("max_steps", "max_seconds")
        nonnegative_integer_fields = ("max_fan_out", "max_retries")
        if any(
            isinstance(budget.get(field), bool)
            or not isinstance(budget.get(field), int)
            or budget[field] < 1
            for field in positive_integer_fields
        ) or any(
            isinstance(budget.get(field), bool)
            or not isinstance(budget.get(field), int)
            or budget[field] < 0
            for field in nonnegative_integer_fields
        ):
            reasons.append("MALFORMED_PROPOSAL")
        cost = budget.get("max_cost")
        if not _valid_cost(cost):
            reasons.append("MALFORMED_PROPOSAL")
    metadata = proposal.get("metadata", {})
    if not isinstance(metadata, Mapping):
        reasons.append("MALFORMED_PROPOSAL")
    context = proposal.get("untrusted_context", [])
    if not isinstance(context, list) or any(
        not isinstance(item, Mapping) for item in context
    ):
        reasons.append("MALFORMED_PROPOSAL")
    return reasons


def _identifier_array_valid(value: Any) -> bool:
    return (
        type(value) is list
        and all(_nonempty(item) for item in value)
        and len(value) == len(set(value))
    )


def _approval_valid(approval: Any) -> bool:
    required = {
        "status",
        "proposal_digest",
        "arguments_digest",
        "target",
        "target_state_digest",
        "policy_version",
        "subject",
        "auth_context_digest",
        "expires_at",
    }
    if type(approval) is not dict or set(approval) != required:
        return False
    if approval["status"] not in {"ACTIVE", "REVOKED"}:
        return False
    if any(
        type(approval[field]) is not str
        or not DIGEST_RE.fullmatch(approval[field])
        for field in (
            "proposal_digest",
            "arguments_digest",
            "target_state_digest",
            "auth_context_digest",
        )
    ):
        return False
    if any(
        not _nonempty(approval[field])
        for field in ("target", "policy_version", "subject", "expires_at")
    ):
        return False
    try:
        parse_timestamp(approval["expires_at"])
    except (TypeError, ValueError):
        return False
    return True


def _policy_valid_unchecked(policy: Mapping[str, Any]) -> bool:
    required = {
        "schema_version",
        "profile",
        "policy_version",
        "allowed_subjects",
        "allowed_agents",
        "allowed_workloads",
        "allowed_actions",
        "allowed_action_scopes",
        "allowed_scopes",
        "network_scopes",
        "allowed_destinations",
        "allowed_secret_destinations",
        "trusted_tools",
        "allowed_delegation_chains",
        "approval_required_risk_tiers",
        "require_approval_for_irreversible",
        "max_resource_budget",
        "approvals",
    }
    if (
        type(policy.get("schema_version")) is not int
        or policy.get("schema_version") != 1
        or type(policy.get("profile")) is not str
        or policy.get("profile") != POLICY_PROFILE
        or not _nonempty(policy.get("policy_version"))
    ):
        return False
    if set(policy) != required:
        return False
    identifier_array_fields = {
        "allowed_subjects",
        "allowed_agents",
        "allowed_workloads",
        "allowed_actions",
        "allowed_scopes",
        "network_scopes",
        "allowed_destinations",
        "allowed_secret_destinations",
    }
    if any(
        not _identifier_array_valid(policy.get(field))
        for field in identifier_array_fields
    ):
        return False
    action_scopes = policy.get("allowed_action_scopes")
    if type(action_scopes) is not dict or not action_scopes:
        return False
    allowed_actions = policy["allowed_actions"]
    policy_scopes = policy["allowed_scopes"]
    allowed_scopes = set(policy_scopes)
    if set(action_scopes) != set(allowed_actions) or any(
        not _nonempty(action)
        or not _identifier_array_valid(scopes)
        or not scopes
        or any(scope not in allowed_scopes for scope in scopes)
        for action, scopes in action_scopes.items()
    ):
        return False
    trusted_tools = policy.get("trusted_tools")
    if type(trusted_tools) is not list or any(
        type(tool) is not dict
        or set(tool) != {"origin", "version", "schema_digest"}
        or not _nonempty(tool["origin"])
        or not _nonempty(tool["version"])
        or type(tool["schema_digest"]) is not str
        or not DIGEST_RE.fullmatch(tool["schema_digest"])
        for tool in trusted_tools
    ):
        return False
    delegation_chains = policy.get("allowed_delegation_chains")
    if type(delegation_chains) is not list or any(
        not _identifier_array_valid(chain) for chain in delegation_chains
    ):
        return False
    risk_tiers = policy.get("approval_required_risk_tiers")
    if (
        type(risk_tiers) is not list
        or len(risk_tiers) != len(set(risk_tiers))
        or any(item not in RISK_LEVELS for item in risk_tiers)
    ):
        return False
    if type(policy.get("max_resource_budget")) is not dict:
        return False
    budget = policy["max_resource_budget"]
    if set(budget) != BUDGET_FIELDS:
        return False
    if any(
        isinstance(budget.get(field), bool)
        or not isinstance(budget.get(field), int)
        or budget[field] < 1
        for field in ("max_steps", "max_seconds")
    ) or any(
        isinstance(budget.get(field), bool)
        or not isinstance(budget.get(field), int)
        or budget[field] < 0
        for field in ("max_fan_out", "max_retries")
    ):
        return False
    cost = budget.get("max_cost")
    if not _valid_cost(cost):
        return False
    approvals = policy.get("approvals")
    if type(approvals) is not dict or any(
        not _nonempty(approval_ref) or not _approval_valid(approval)
        for approval_ref, approval in approvals.items()
    ):
        return False
    return type(
        policy.get("require_approval_for_irreversible")
    ) is bool


def validate_policy_document(policy: Mapping[str, Any]) -> bool:
    """Validate a policy shape without letting hostile values escape."""

    try:
        return _policy_valid_unchecked(policy)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _approval_reasons(
    proposal: Mapping[str, Any], policy: Mapping[str, Any], now: datetime, required: bool
) -> tuple[DecisionStatus | None, list[str]]:
    approval_ref = proposal.get("approval_ref")
    if approval_ref is None:
        if required:
            return DecisionStatus.HOLD, ["APPROVAL_REQUIRED"]
        return None, []
    approval = policy["approvals"].get(approval_ref)
    if not isinstance(approval, Mapping):
        return DecisionStatus.BLOCK, ["APPROVAL_UNKNOWN"]
    if approval.get("status") == "REVOKED":
        return DecisionStatus.BLOCK, ["APPROVAL_REVOKED"]
    if approval.get("status") != "ACTIVE":
        return DecisionStatus.BLOCK, ["APPROVAL_INVALID"]
    try:
        if parse_timestamp(str(approval.get("expires_at"))) <= now:
            return DecisionStatus.BLOCK, ["APPROVAL_EXPIRED"]
    except (TypeError, ValueError):
        return DecisionStatus.BLOCK, ["APPROVAL_INVALID"]
    bindings = {
        "proposal_digest": proposal.get("proposal_digest"),
        "arguments_digest": proposal.get("arguments_digest"),
        "target": proposal.get("target"),
        "target_state_digest": proposal.get("target_state_digest"),
        "policy_version": proposal.get("policy_version"),
        "subject": proposal.get("subject"),
        "auth_context_digest": proposal.get("auth_context_digest"),
    }
    if any(approval.get(key) != value for key, value in bindings.items()):
        return DecisionStatus.BLOCK, ["APPROVAL_BINDING_MISMATCH"]
    return None, []


def evaluate_proposal(
    proposal: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> Decision:
    """Evaluate one exact proposal without performing a side effect."""

    now = require_aware_utc(datetime.now(UTC) if now is None else now)
    findings = _untrusted_findings(proposal)
    structural = _structural_reasons(proposal)
    if structural:
        return _decision(DecisionStatus.BLOCK, structural, findings, proposal, now)
    if not validate_policy_document(policy):
        return _decision(DecisionStatus.BLOCK, ["POLICY_INVALID"], findings, proposal, now)

    reasons: list[str] = []
    if proposal["policy_version"] != policy["policy_version"]:
        reasons.append("POLICY_VERSION_MISMATCH")
    if proposal["arguments_digest"] != digest_json(proposal["arguments"]):
        reasons.append("ARGUMENTS_DIGEST_MISMATCH")
    if proposal["proposal_digest"] != proposal_digest(proposal):
        reasons.append("PROPOSAL_DIGEST_MISMATCH")
    try:
        issued_at = parse_timestamp(proposal["issued_at"])
        expires_at = parse_timestamp(proposal["expires_at"])
        if expires_at <= issued_at:
            reasons.append("INVALID_TIME_WINDOW")
        if issued_at > now:
            reasons.append("PROPOSAL_NOT_YET_VALID")
        if expires_at <= now:
            reasons.append("PROPOSAL_EXPIRED")
    except (TypeError, ValueError):
        reasons.append("MALFORMED_PROPOSAL")

    separate_identities = (
        ("subject", "allowed_subjects"),
        ("agent", "allowed_agents"),
        ("workload", "allowed_workloads"),
    )
    if any(proposal[field] not in policy[allowed] for field, allowed in separate_identities):
        reasons.append("IDENTITY_DENIED")
    if proposal["action"] not in policy["allowed_actions"]:
        reasons.append("ACTION_DENIED")
    if proposal["scope"] not in policy["allowed_scopes"]:
        reasons.append("SCOPE_DENIED")
    if (
        proposal["action"] in policy["allowed_actions"]
        and proposal["scope"] in policy["allowed_scopes"]
        and proposal["scope"]
        not in policy["allowed_action_scopes"].get(proposal["action"], [])
    ):
        reasons.append("ACTION_SCOPE_DENIED")

    trusted_tool = any(
        isinstance(tool, Mapping)
        and tool.get("origin") == proposal["tool_origin"]
        and tool.get("version") == proposal["tool_version"]
        and tool.get("schema_digest") == proposal["tool_schema_digest"]
        for tool in policy["trusted_tools"]
    )
    if not trusted_tool:
        reasons.append("TOOL_ORIGIN_DENIED")

    chain = proposal["delegation_chain"]
    if chain and chain not in policy["allowed_delegation_chains"]:
        reasons.append("DELEGATION_DENIED")

    policy_budget = policy["max_resource_budget"]
    proposal_budget = proposal["resource_budget"]
    try:
        if any(proposal_budget[field] > policy_budget[field] for field in BUDGET_FIELDS):
            reasons.append("RESOURCE_BUDGET_EXCEEDED")
    except (KeyError, TypeError):
        reasons.append("POLICY_INVALID")

    scope_is_network = proposal["scope"].startswith("network.") or proposal["scope"] in policy[
        "network_scopes"
    ]
    destination = normalize_egress_origin(proposal["destination"], proposal["scope"])
    allowed_destinations = {
        value
        for value in (
            normalize_egress_origin(item, proposal["scope"])
            for item in policy["allowed_destinations"]
        )
        if value is not None
    }
    secret_destinations = {
        value
        for value in (
            normalize_egress_origin(item, proposal["scope"])
            for item in policy["allowed_secret_destinations"]
        )
        if value is not None
    }
    if scope_is_network and (destination is None or destination not in allowed_destinations):
        reasons.append("DESTINATION_DENIED")
    if not scope_is_network and proposal["destination"] is not None:
        reasons.append("DESTINATION_SCOPE_MISMATCH")
    sensitive = proposal["contains_secret"] or proposal["data_classification"] in {
        "confidential",
        "restricted",
        "unknown",
    }
    if scope_is_network and sensitive and (
        destination is None or destination not in secret_destinations
    ):
        reasons.append("SECRET_DESTINATION_DENIED")

    if reasons:
        return _decision(DecisionStatus.BLOCK, reasons, findings, proposal, now)

    approval_required = (
        policy["require_approval_for_irreversible"]
        and proposal["reversibility"] != "reversible"
    ) or proposal["risk_tier"] in policy["approval_required_risk_tiers"]
    approval_status, approval_reasons = _approval_reasons(
        proposal, policy, now, approval_required
    )
    if approval_status is not None:
        return _decision(approval_status, approval_reasons, findings, proposal, now)
    return _decision(DecisionStatus.ACCEPT, [], findings, proposal, now)
