"""Synthetic approval-to-dispatch flow; no RPC call or real value is used."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from causal_cell import CausalCell, InMemoryNonceStore, bind_proposal


NOW = datetime(2026, 8, 27, 21, 0, tzinfo=UTC)
TOOL_SCHEMA_DIGEST = "sha256:" + "a" * 64
TARGET_STATE_DIGEST = "sha256:" + "c" * 64
AUTH_CONTEXT_DIGEST = "sha256:" + "d" * 64


def policy() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile": "org.causalcell.policy.v0.1",
        "policy_version": "demo-policy-v0.1",
        "allowed_subjects": ["user:demo"],
        "allowed_agents": ["agent:contract-qa"],
        "allowed_workloads": ["workload:synthetic-demo"],
        "allowed_actions": ["release_payment"],
        "allowed_action_scopes": {"release_payment": ["contract.write"]},
        "allowed_scopes": ["contract.write"],
        "network_scopes": [],
        "allowed_destinations": [],
        "allowed_secret_destinations": [],
        "trusted_tools": [
            {
                "origin": "registry.example.test/synthetic-evm-adapter",
                "version": "1.0.0",
                "schema_digest": TOOL_SCHEMA_DIGEST,
            }
        ],
        "allowed_delegation_chains": [],
        "approval_required_risk_tiers": ["high", "critical"],
        "require_approval_for_irreversible": True,
        "max_resource_budget": {
            "max_steps": 3,
            "max_seconds": 15,
            "max_cost": 0.0,
            "max_fan_out": 0,
            "max_retries": 0,
        },
        "approvals": {},
    }


def proposal() -> dict[str, object]:
    return bind_proposal(
        {
            "schema_version": 1,
            "profile": "org.causalcell.action-proposal.v0.1",
            "trace_id": "escrow-42-release",
            "span_id": "release-span-1",
            "parent_span_id": "delivery-span-1",
            "request_id": "release-escrow-42",
            "attempt_id": "release-attempt-1",
            "parent_request_id": None,
            "retry_of_attempt_id": None,
            "agent": "agent:contract-qa",
            "workload": "workload:synthetic-demo",
            "subject": "user:demo",
            "method": "tool_call",
            "intent_id": "intent-release-escrow-42",
            "parent_cause": "delivery-accepted-42",
            "action": "release_payment",
            "scope": "contract.write",
            "target": "evm:8453:0x1111111111111111111111111111111111111111",
            "target_state_digest": TARGET_STATE_DIGEST,
            "reversibility": "irreversible",
            "approval_ref": None,
            "nonce": "release-nonce-42",
            "risk_tier": "high",
            "policy_version": "demo-policy-v0.1",
            "arguments": {"function": "releasePayment", "escrow_id": 42},
            "idempotency_key": "release-escrow-42",
            "issued_at": "2026-08-27T20:00:00Z",
            "expires_at": "2026-08-27T22:00:00Z",
            "tool_origin": "registry.example.test/synthetic-evm-adapter",
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
                "max_cost": 0.0,
                "max_fan_out": 0,
                "max_retries": 0,
            },
            "metadata": {"fixture": "synthetic-only"},
            "untrusted_context": [],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", default="evidence/demo")
    args = parser.parse_args()
    root = Path(args.evidence_root)
    nonces = InMemoryNonceStore()
    base_policy = policy()
    original = proposal()
    calls = 0

    def synthetic_executor(_: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"synthetic_tx_hash": "0x" + "b" * 64, "rpc_used": False}

    held = CausalCell(
        base_policy, root, nonce_store=nonces, clock=lambda: NOW
    ).execute(original, synthetic_executor)

    approved = copy.deepcopy(original)
    approved["approval_ref"] = "approval-release-42"
    approved = bind_proposal(approved)
    approved_policy = copy.deepcopy(base_policy)
    approved_policy["approvals"]["approval-release-42"] = {
        "status": "ACTIVE",
        "proposal_digest": approved["proposal_digest"],
        "arguments_digest": approved["arguments_digest"],
        "target": approved["target"],
        "target_state_digest": approved["target_state_digest"],
        "policy_version": approved["policy_version"],
        "subject": approved["subject"],
        "auth_context_digest": approved["auth_context_digest"],
        "expires_at": "2026-08-27T21:30:00Z",
    }
    accepted = CausalCell(
        approved_policy, root, nonce_store=nonces, clock=lambda: NOW
    ).execute(approved, synthetic_executor)
    print(
        json.dumps(
            {
                "first_decision": held.decision.status.value,
                "second_decision": accepted.decision.status.value,
                "executor_calls": calls,
                "held_bundle": str(held.bundle_path),
                "accepted_bundle": str(accepted.bundle_path),
                "both_bundles_verified": (
                    held.verification.valid and accepted.verification.valid
                ),
                "external_effect_verified": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
