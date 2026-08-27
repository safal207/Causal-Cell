# Causal Cell v0.1 specification

## Purpose

Causal Cell implements the smallest executable chain:

```text
canonical proposal
  -> trusted policy evaluation
  -> ACCEPT / HOLD / BLOCK
  -> callback only after ACCEPT
  -> independent records
  -> local evidence verification
```

It is an orchestration boundary, not a new model, blockchain verifier,
continuity protocol, or database.

## Inputs and authority

The proposal binds trace/request identities, subject/agent/workload/tool
identities, intent and causal parent, action, scope, target, expected target
state, arguments, reversibility, risk, policy, approval, nonce, idempotency,
time window, authentication context, destination, data classification,
delegation, and resource budget.

`bind_proposal()` calculates only `arguments_digest` and `proposal_digest`. It
does not invent authority-critical values. Policy and approval records must
come from the application/authority layer. Web, retrieval, memory, tool output,
inter-agent messages, model rationale, and proposal metadata are untrusted.

## Decision order

The guard fails closed across structure, causal identity, policy version,
digests, time, separate identities, exact action-to-scope binding, tool
provenance, delegation,
resource budgets, network destination, sensitive egress, and approval binding.

A missing required approval produces `HOLD`. A forged, unknown, expired,
revoked, or differently bound approval produces `BLOCK`. Approval is bound to
the exact proposal, arguments, target, expected target state, subject,
authentication context, policy version, and expiry.

Immediately before replay-key consumption, the runtime evaluates the proposal
again using its current clock.

## Dispatch and replay boundary

The reference store atomically checks and consumes both `nonce` and
`idempotency_key`. The first eligible proposal wins. Reusing the nonce returns
`INTENT_REPLAYED`; changing the nonce while reusing the idempotency key returns
`IDEMPOTENCY_REPLAYED`.

This store is process-local and volatile. It is not a database transaction or
an exactly-once guarantee. Production must replace it with a durable shared
atomic store.

## Observation and continuity

The executor is called only after `ACCEPT` and replay-key consumption. The
reference callback is in-process for testability; production execution must be
externally contained.

Executor return/error is observed, but external side-effect truth remains
unknown without an independent observer.

Structurally complete runs export:

- `org.ltp.request-envelope.v0.1`;
- `org.ltp.outcome-envelope.v0.1` when a terminal outcome exists.

`HOLD` maps to `DEFERRED` with no outcome. `BLOCK` maps to `REJECTED`.
Executor return/error maps to `COMPLETED`/`FAILED`.

The export covers supplied records only. LTP remains the normative continuity
judge.

## Evidence integrity

Every run writes a generated `cc-<uuid>` directory. Untrusted identifiers never
become paths. The manifest binds exact size and SHA-256; a semantic JSONL ledger
binds ordered record ancestry.

Reopen verification checks exact inventory, no symlinks, safe unique paths and
roles, byte digests, ledger order, previous hashes, event hashes, record
references, and payload digests.

This proves local byte integrity and reopenability only. It does not prove
authorship, time, external finality, complete history, or exactly-once effects.
