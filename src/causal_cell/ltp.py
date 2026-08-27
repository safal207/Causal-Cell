"""Exact-shape LTP request/outcome envelope export."""

from __future__ import annotations

from typing import Any, Mapping

from .models import Decision, DecisionStatus


def export_ltp_continuity_input(
    proposal: Mapping[str, Any], decision: Decision, observation: Mapping[str, Any]
) -> dict[str, Any]:
    if decision.status is DecisionStatus.HOLD:
        request_state = "DEFERRED"
        continuation_id: str | None = f"approval:{proposal['proposal_digest']}"
    elif decision.status is DecisionStatus.ACCEPT:
        request_state = "ACCEPTED"
        continuation_id = None
    else:
        request_state = "CREATED"
        continuation_id = None
    request = {
        "schema_version": 1,
        "profile": "org.ltp.request-envelope.v0.1",
        "record_type": "REQUEST",
        "request_id": proposal["request_id"],
        "trace_id": proposal["trace_id"],
        "attempt_id": proposal["attempt_id"],
        "occurred_at": proposal["issued_at"],
        "state": request_state,
        "deadline_at": proposal["expires_at"],
        "parent_request_id": proposal["parent_request_id"],
        "retry_of_attempt_id": proposal["retry_of_attempt_id"],
        "continuation_id": continuation_id,
        "payload_digest": proposal["proposal_digest"],
        "metadata": {
            "producer": "Causal-Cell",
            "source_kind": "ACTION_PROPOSAL",
            "authorization_decision": decision.status.value,
        },
    }
    outcomes: list[dict[str, Any]] = []
    if decision.status is not DecisionStatus.HOLD:
        if decision.status is DecisionStatus.BLOCK:
            terminal_status = "REJECTED"
        elif observation["status"] == "EXECUTOR_RETURNED":
            terminal_status = "COMPLETED"
        else:
            terminal_status = "FAILED"
        outcomes.append(
            {
                "schema_version": 1,
                "profile": "org.ltp.outcome-envelope.v0.1",
                "record_type": "OUTCOME",
                "outcome_id": observation["observation_id"],
                "request_id": proposal["request_id"],
                "trace_id": proposal["trace_id"],
                "attempt_id": proposal["attempt_id"],
                "occurred_at": observation["captured_at"],
                "terminal_status": terminal_status,
                "result_digest": observation.get("reported_result_digest"),
                "replay_of_outcome_id": None,
                "metadata": {
                    "producer": "Causal-Cell",
                    "source_kind": "RUNTIME_OBSERVATION",
                    "executor_invoked": observation["executor_invoked"],
                    "external_effect_verified": False,
                },
            }
        )
    return {
        "as_of": observation["captured_at"],
        "requests": [request],
        "outcomes": outcomes,
        "verifier": {"id": "causal-cell-exporter", "version": "0.1.0"},
    }
