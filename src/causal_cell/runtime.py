"""Guarded execution boundary and evidence orchestration."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .canonical import digest_json, format_timestamp, require_aware_utc
from .evidence import write_bundle
from .guard import evaluate_proposal
from .ltp import export_ltp_continuity_input
from .models import CellRun, Decision, DecisionStatus

Executor = Callable[[Mapping[str, Any]], Any]
Clock = Callable[[], datetime]


class InMemoryNonceStore:
    """Process-local atomic replay store for the v0.1 reference runtime.

    It is not a distributed transaction or an exactly-once guarantee.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consumed: dict[str, str] = {}
        self._idempotency: dict[str, str] = {}

    def consume(
        self, nonce: str, idempotency_key: str, bound_proposal_digest: str
    ) -> str | None:
        with self._lock:
            if nonce in self._consumed:
                return "INTENT_REPLAYED"
            if idempotency_key in self._idempotency:
                return "IDEMPOTENCY_REPLAYED"
            self._consumed[nonce] = bound_proposal_digest
            self._idempotency[idempotency_key] = bound_proposal_digest
            return None

    def consumed_by(self, nonce: str) -> str | None:
        with self._lock:
            return self._consumed.get(nonce)

    def idempotency_consumed_by(self, idempotency_key: str) -> str | None:
        with self._lock:
            return self._idempotency.get(idempotency_key)


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _replay_block(decision: Decision, reason: str) -> Decision:
    return Decision(
        status=DecisionStatus.BLOCK,
        reasons=(reason,),
        findings=decision.findings,
        proposal_digest=decision.proposal_digest,
        policy_version=decision.policy_version,
        decided_at=decision.decided_at,
    )


class CausalCell:
    """Small reference kernel that never invokes an executor before ACCEPT."""

    def __init__(
        self,
        policy: Mapping[str, Any],
        evidence_root: str | Path,
        *,
        nonce_store: InMemoryNonceStore | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._policy = copy.deepcopy(dict(policy))
        self._evidence_root = Path(evidence_root)
        self._nonces = (
            nonce_store if nonce_store is not None else InMemoryNonceStore()
        )
        self._clock = clock if clock is not None else _default_clock

    def execute(self, proposal: Mapping[str, Any], executor: Executor) -> CellRun:
        """Evaluate, conditionally dispatch, and preserve separate evidence layers.

        The callback is in-process for the reference runtime. Production callers
        must make it a thin adapter to an externally contained executor.
        """

        exact_proposal = copy.deepcopy(dict(proposal))
        now = require_aware_utc(self._clock())
        decision = evaluate_proposal(exact_proposal, self._policy, now=now)
        evidence_time = now
        nonce_consumed = False
        executor_invoked = False
        result_digest: str | None = None
        error_type: str | None = None

        if decision.status is DecisionStatus.ACCEPT:
            evidence_time = require_aware_utc(self._clock())
            decision = evaluate_proposal(
                exact_proposal,
                self._policy,
                now=evidence_time,
            )
        if decision.status is DecisionStatus.ACCEPT:
            replay_reason = self._nonces.consume(
                exact_proposal["nonce"],
                exact_proposal["idempotency_key"],
                exact_proposal["proposal_digest"],
            )
            nonce_consumed = replay_reason is None
            if replay_reason is not None:
                decision = _replay_block(decision, replay_reason)
            else:
                executor_invoked = True
                try:
                    result = executor(copy.deepcopy(exact_proposal))
                    result_digest = digest_json(result)
                    observation_status = "EXECUTOR_RETURNED"
                except Exception as exc:
                    error_type = type(exc).__name__
                    observation_status = "EXECUTOR_ERROR"
        if decision.status is not DecisionStatus.ACCEPT:
            observation_status = "NOT_INVOKED"

        captured_at = format_timestamp(evidence_time)
        observation_id = (
            f"obs:{exact_proposal.get('trace_id', 'unknown')}:"
            f"{exact_proposal.get('attempt_id', 'unknown')}"
        )
        observation = {
            "schema_version": 1,
            "profile": "org.causalcell.observation-record.v0.1",
            "observation_id": observation_id,
            "trace_id": exact_proposal.get("trace_id"),
            "request_id": exact_proposal.get("request_id"),
            "attempt_id": exact_proposal.get("attempt_id"),
            "captured_at": captured_at,
            "status": observation_status,
            "executor_invoked": executor_invoked,
            "nonce_consumed": nonce_consumed,
            "side_effect_executed": False if not executor_invoked else None,
            "reported_result_digest": result_digest,
            "error_type": error_type,
            "claim_boundary": (
                "Executor return/error was observed; an external side effect was not "
                "independently verified."
            ),
        }
        response_integrity = {
            "schema_version": 1,
            "profile": "org.causalcell.response-integrity-record.v0.1",
            "status": "NOT_EVALUATED",
            "reported_result_digest": result_digest,
            "reason": "No independent response verifier is configured in v0.1.",
        }
        causal_validity = (
            "INVALID"
            if {"MISSING_INTENT", "MISSING_CAUSAL_PARENT"}.intersection(decision.reasons)
            else "VALID"
        )
        causal_audit = {
            "schema_version": 1,
            "profile": "org.causalcell.causal-audit-record.v0.1",
            "causal_validity": causal_validity,
            "authorization_decision": decision.status.value,
            "findings": list(dict.fromkeys((*decision.reasons, *decision.findings))),
            "parent_cause": exact_proposal.get("parent_cause"),
            "intent_id": exact_proposal.get("intent_id"),
        }
        continuity = export_ltp_continuity_input(exact_proposal, decision, observation)
        intent = {
            "schema_version": 1,
            "profile": "org.causalcell.intent-record.v0.1",
            "intent_id": exact_proposal.get("intent_id"),
            "request_id": exact_proposal.get("request_id"),
            "parent_cause": exact_proposal.get("parent_cause"),
            "subject": exact_proposal.get("subject"),
            "issued_at": exact_proposal.get("issued_at"),
            "expires_at": exact_proposal.get("expires_at"),
            "proposal_digest": exact_proposal.get("proposal_digest"),
        }
        replay_trace = {
            "schema_version": 1,
            "profile": "org.causalcell.replay-trace.v0.1",
            "trace_id": exact_proposal.get("trace_id"),
            "attempt_id": exact_proposal.get("attempt_id"),
            "events": [
                {"event": "PROPOSAL_RECEIVED"},
                {"event": "POLICY_DECIDED", "decision": decision.status.value},
                {"event": "NONCE_CONSUMPTION", "consumed": nonce_consumed},
                {"event": "EXECUTOR_INVOCATION", "invoked": executor_invoked},
                {"event": "OBSERVATION_RECORDED", "status": observation_status},
            ],
            "claim_boundary": "Replayable decision path, not replay of an external side effect.",
        }
        runtime_verification = {
            "schema_version": 1,
            "profile": "org.causalcell.runtime-verification-record.v0.1",
            "status": "RUNTIME_RECORDS_COMPLETE",
            "checks": {
                "authorization_separate_from_observation": True,
                "executor_called_only_after_accept": (
                    not executor_invoked or decision.status is DecisionStatus.ACCEPT
                ),
                "hold_or_block_has_no_side_effect": (
                    decision.status is DecisionStatus.ACCEPT
                    or (not executor_invoked and observation["side_effect_executed"] is False)
                ),
            },
            "independent_bundle_verification": "PENDING_AT_RECORD_CREATION",
        }
        records = {
            "intent": intent,
            "action": exact_proposal,
            "authorization": decision.to_record(),
            "result": observation,
            "response_integrity": response_integrity,
            "causal_audit": causal_audit,
            "continuity": continuity,
            "replay": replay_trace,
            "verification": runtime_verification,
        }
        bundle_path, verification = write_bundle(
            self._evidence_root, records, created_at=captured_at
        )
        return CellRun(decision, observation, continuity, bundle_path, verification)
