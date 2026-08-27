from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

from causal_cell import bind_proposal


NOW = datetime(2026, 8, 27, 21, 0, tzinfo=UTC)
TOOL_SCHEMA_DIGEST = "sha256:" + "a" * 64
TARGET_STATE_DIGEST = "sha256:" + "c" * 64
AUTH_CONTEXT_DIGEST = "sha256:" + "d" * 64


def base_policy() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": "org.causalcell.policy.v0.1",
        "policy_version": "policy-2026-08-27",
        "allowed_subjects": ["user:alice"],
        "allowed_agents": ["agent:contract-qa"],
        "allowed_workloads": ["workload:local-reference"],
        "allowed_actions": ["inspect_contract", "release_payment", "send_payload"],
        "allowed_scopes": ["contract.read", "contract.write", "network.egress"],
        "network_scopes": ["network.egress"],
        "allowed_destinations": ["https://evidence.example.test"],
        "allowed_secret_destinations": ["https://evidence.example.test"],
        "trusted_tools": [
            {
                "origin": "registry.example.test/contract-reader",
                "version": "1.0.0",
                "schema_digest": TOOL_SCHEMA_DIGEST,
            }
        ],
        "allowed_delegation_chains": [["user:alice", "agent:trusted-delegate"]],
        "approval_required_risk_tiers": ["high", "critical"],
        "require_approval_for_irreversible": True,
        "max_resource_budget": {
            "max_steps": 10,
            "max_seconds": 60,
            "max_cost": 10.0,
            "max_fan_out": 2,
            "max_retries": 2,
        },
        "approvals": {},
    }


def base_proposal() -> dict[str, Any]:
    return bind_proposal(
        {
            "schema_version": 1,
            "profile": "org.causalcell.action-proposal.v0.1",
            "trace_id": "trace-001",
            "span_id": "span-001",
            "parent_span_id": "span-parent",
            "request_id": "request-001",
            "attempt_id": "attempt-001",
            "parent_request_id": None,
            "retry_of_attempt_id": None,
            "agent": "agent:contract-qa",
            "workload": "workload:local-reference",
            "subject": "user:alice",
            "method": "tool_call",
            "intent_id": "intent-inspect-001",
            "parent_cause": "user-request-001",
            "action": "inspect_contract",
            "scope": "contract.read",
            "target": "evm:8453:0x1111111111111111111111111111111111111111",
            "target_state_digest": TARGET_STATE_DIGEST,
            "reversibility": "reversible",
            "approval_ref": None,
            "nonce": "nonce-001",
            "risk_tier": "low",
            "policy_version": "policy-2026-08-27",
            "arguments": {"function": "owner", "block": 123456},
            "idempotency_key": "inspect-001",
            "issued_at": "2026-08-27T20:00:00Z",
            "expires_at": "2026-08-27T22:00:00Z",
            "tool_origin": "registry.example.test/contract-reader",
            "tool_version": "1.0.0",
            "tool_schema_digest": TOOL_SCHEMA_DIGEST,
            "auth_context_digest": AUTH_CONTEXT_DIGEST,
            "contains_secret": False,
            "destination": None,
            "data_classification": "public",
            "delegation_chain": [],
            "resource_budget": {
                "max_steps": 2,
                "max_seconds": 10,
                "max_cost": 1.0,
                "max_fan_out": 0,
                "max_retries": 0,
            },
            "metadata": {"source": "synthetic-test"},
            "untrusted_context": [],
        }
    )


def rebound(proposal: dict[str, Any], **changes: Any) -> dict[str, Any]:
    changed = copy.deepcopy(proposal)
    changed.update(changes)
    return bind_proposal(changed)


def approved_irreversible() -> tuple[dict[str, Any], dict[str, Any]]:
    policy = base_policy()
    proposal = rebound(
        base_proposal(),
        action="release_payment",
        scope="contract.write",
        intent_id="intent-release-001",
        request_id="request-release-001",
        nonce="nonce-release-001",
        idempotency_key="release-001",
        reversibility="irreversible",
        risk_tier="high",
        approval_ref="approval-release-001",
        arguments={"function": "releasePayment", "escrow_id": 42},
    )
    policy["approvals"]["approval-release-001"] = {
        "status": "ACTIVE",
        "proposal_digest": proposal["proposal_digest"],
        "arguments_digest": proposal["arguments_digest"],
        "target": proposal["target"],
        "target_state_digest": proposal["target_state_digest"],
        "policy_version": proposal["policy_version"],
        "subject": proposal["subject"],
        "auth_context_digest": proposal["auth_context_digest"],
        "expires_at": "2026-08-27T21:30:00Z",
    }
    return proposal, policy
