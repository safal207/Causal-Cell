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
- provider request IDs, token counts, cost reports, result labels, and error
  strings;
- inter-cell content;
- capability target and arguments selected in an `ActionDraft`.

The reference adapter and executor callbacks run in-process. This is explicitly
not a sandbox.

## Threat matrix

| Threat | v0.1 control | Residual risk |
|---|---|---|
| Model invocation bypasses authorization | Model adapter is called only inside an accepted `CausalCell.execute` callback | Embedding application could call its adapter elsewhere; SDK-side tools/functions/handoffs must be disabled or separately guarded |
| Provider/adapter substitution | Exact registry key/`adapter_id` match, `AdapterIdentity` digest in manifest, policy allowlist, and pre/post-dispatch rechecks | Identity is host metadata, not remote attestation |
| Deterministically incompatible trusted setup | Full policy-shape and static manifest/adapter/capability/executor compatibility preflight before semantic-run consumption or model dispatch | Result-derived taint and model-selected data still require dynamic cell checks |
| Prompt-injected authority fields | Strict ActionDraft shape; unknown fields block; trusted compiler constructs authority | Semantic manipulation can still choose among allowed data/capabilities |
| Arbitrary tool selection | Static capability ID allowlist and executor registry | Capability implementation may itself be unsafe |
| Causal-chain forgery | Host-generated IDs; exact upstream result ID/digest binding | Base parent-cause validity is syntactic outside the organism |
| Replay or duplicated run/effect | Atomic semantic-run store plus stable semantic-effect idempotency over canonical targets/destinations in the action cell; injected falsey stores are preserved | Defaults are per runner; protection across runners requires injected shared stores, which remain non-durable |
| Token/cost amplification | Two guarded adapter dispatches, no runner retry/fan-out, per-call output-token/cost and aggregate token/cost checks | SDK/provider-internal retries and fan-out are unobserved; usage is adapter-reported and known usage is post-call |
| Recursive self-replication | Preflight hard block for `create_organism` and `organism.spawn` in every stage/capability | A privileged host can still activate a new manifest |
| Adapter exception/provenance drift | Guard before call; once invocation starts, exceptions or post-call provenance change are `EFFECT_UNCERTAIN`; no retry | Remote side may have consumed cost or another effect |
| Provider terminal result | Valid `REFUSED` or `ERROR` stops downstream as `FAILED` | Result, usage, and remote effect claims are adapter-reported |
| Activation expires during run | Expiry is refreshed before semantic-run consumption, clamped when a shorter activation is observed, and revalidated at every dispatch boundary | Synchronous callback cannot be preempted; control-plane replacement remains subject to a narrow check/use race |
| Clock/control-store failure | Only exact timezone-aware clock values are accepted; infrastructure exceptions become terminal `FAILED` runs instead of escaping; semantic-store failure occurs before dispatch | The local reference result is not a durable incident record; injected replay stores must make `consume_run` atomic and non-blocking because v0.1 has no reservation rollback |
| Executor exception or post-effect evidence failure | `EFFECT_UNCERTAIN`, never success | External effect may have happened without a complete bundle |
| Secret leakage | Host/capability labels plus required adapter result labels join monotonically; unknown/confidential/restricted public egress is denied | No independent content classifier; an under-labeling trusted adapter or in-process access can still leak |
| Mutable/untrusted JSON | Strict detached plain-JSON snapshots before hashing/reuse; model-call and retained-result access returns defensive copies | Application-owned callbacks remain trusted code and must transmit the exact authorized snapshot or bind transformations independently |
| Target/argument/state confusion | Segment-aware canonical prefixes, host-owned validators, per-target trusted state resolver, canonical HTTPS/IP destination origin, same-origin binding for URL-like targets, and rejection of destinations on non-network scopes | The resolver digest is a trusted host claim, not independently verified state; opaque target/arguments must remain data identifiers; executor-derived endpoints and every redirect hop must use the same trusted resolution and egress checks |
| Approval confusion | v0.1 rejects approval-required policies and irreversible capabilities before any call | Approval plan/resume is deferred |
| Evidence conflation | Separate per-cell authorization, observation, causal, replay, and verification records | No signed aggregate organism record |

## Required negative cases

The test suite proves fail-closed behavior for:

- missing manifest activation;
- denied model destination before adapter dispatch;
- adapter exception, provider terminal error, malformed/hostile result scalars,
  oversized metadata, mutable output, and mid-call identity change;
- activation rotation, expiry during a run, global deadlines, and per-cell
  deadlines, including expiry during preflight without replay-key poisoning;
- no-action termination and authority-field smuggling;
- unknown/duplicate/missing capability or executor, malformed/incompatible
  policies, strict trusted setup,
  segment-bound target, traversal/userinfo/egress-origin target confusion, and
  argument value checks;
- per-call and aggregate budget overrun, including retained known usage on a
  later evidence failure;
- exact-run replay under concurrent access and duplicate semantic-effect replay
  across changed roots, manifest/provider revisions, URL spellings, and IPv6
  aliases;
- host, observer-result, and analyst-result taint blocking unapproved egress,
  including conservative `unknown`;
- nested spawn in model/final cells and approval-required configuration;
- falsey shared stores, semantic-store failure, and early/late clock failure;
- executor, model-evidence, and action-evidence effect uncertainty;
- invalid context/root snapshots, timestamp overflow, UTF-8/depth/string bounds,
  malformed manifest
  identity scalars, adapter provenance lookup failure/mismatch, and topology
  mutation.

All fixtures are synthetic. Tests use no credentials, network, subprocess,
dynamic evaluation, or external/operational side effects. They do write, rename,
and verify evidence only inside temporary directories.

## Non-claims

Passing these checks does not prove provider truthfulness, model correctness,
external side-effect success, distributed exactly-once execution, sandbox
containment, or production readiness.
