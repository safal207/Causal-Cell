# Organism v0.1 threat model

This document extends the base Causal Cell threat model for a fixed multi-model
organism.

## Assets and trust boundaries

Trusted:

- host-supplied `RunContext`;
- `OrganismPolicy` and both Causal Cell policies;
- exact manifest activation;
- adapter registry and adapter identity allowlist;
- static capability definitions and executor registry;
- canonical hashing, guards, replay stores, and evidence writer.

Untrusted:

- root input;
- every model input and output;
- provider request IDs, token counts, cost reports, and error strings;
- inter-cell content;
- capability target and arguments selected in an `ActionDraft`.

The reference adapter and executor callbacks run in-process. This is explicitly
not a sandbox.

## Threat matrix

| Threat | v0.1 control | Residual risk |
|---|---|---|
| Model invocation bypasses authorization | Model adapter is called only inside an accepted `CausalCell.execute` callback | Embedding application could call its adapter elsewhere |
| Provider/adapter substitution | Exact `AdapterIdentity` digest in manifest, policy allowlist, and pre/post-dispatch rechecks | Identity is host metadata, not remote attestation |
| Prompt-injected authority fields | Strict ActionDraft shape; unknown fields block; trusted compiler constructs authority | Semantic manipulation can still choose among allowed data/capabilities |
| Arbitrary tool selection | Static capability ID allowlist and executor registry | Capability implementation may itself be unsafe |
| Causal-chain forgery | Host-generated IDs; exact upstream result ID/digest binding | Base parent-cause validity is syntactic outside the organism |
| Replay or duplicated run | Atomic process-local semantic-run store plus existing nonce store | Not distributed or durable across processes |
| Token/cost amplification | Two calls, no retries, fan-out one, per-call and aggregate caps | Usage is provider-reported; input-token reservation is approximate |
| Recursive self-replication | Hard block for `create_organism` and `organism.spawn` | A privileged host can still activate a new manifest |
| Adapter exception/refusal | Stop downstream; terminal `FAILED`; no retry | Remote side may have consumed cost before failure |
| Executor exception | `EFFECT_UNCERTAIN`, never success | External effect may have happened before exception |
| Secret leakage | Classification and destination flow into the base guard; payload omitted from proposal evidence | In-process adapters can access payload; no isolation in reference code |
| Evidence conflation | Separate per-cell authorization, observation, causal, replay, and verification records | No signed aggregate organism record |

## Required negative cases

The test suite proves fail-closed behavior for:

- missing manifest activation;
- denied model destination before adapter dispatch;
- adapter exception, provider terminal error, and mid-call identity change;
- no-action termination;
- authority-field smuggling;
- unknown capability;
- aggregate budget overrun and return after the effective deadline;
- semantic replay;
- nested spawn;
- executor error/effect uncertainty;
- adapter provenance mismatch; and
- topology mutation.

All fixtures are synthetic. Tests use no credentials, network, subprocess, dynamic
evaluation, or real side effects.

## Non-claims

Passing these checks does not prove provider truthfulness, model correctness,
external side-effect success, distributed exactly-once execution, sandbox
containment, or production readiness.
