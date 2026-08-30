# Causal Cell Organism v0.1

Organism v0.1 lets different LLM providers participate in one bounded workflow
without treating any model response as authority.

The reference topology is intentionally fixed:

```text
guarded observer model -> guarded analyst model -> trusted ActionDraft compiler
                       -> guarded capability executor
```

The observer and analyst can use different providers, models, and adapter
implementations. The executor is not an LLM-selected arbitrary tool: it is a
host-owned capability declared before the run.

## Security invariant

> Model output is a proposal, never authorization.

This applies twice:

1. Calling a model adapter is itself a network/cost side effect, so an existing
   `CausalCell` evaluates the model-call proposal before the adapter is invoked.
2. The analyst can return only a strict `ActionDraft`. A trusted
   `StaticProposalFactory` constructs the full canonical action proposal and an
   existing `CausalCell` evaluates it before the capability executor is invoked.

A model cannot supply or override the subject, agent identity, workload, scope,
policy version, nonce, idempotency key, delegation chain, tool provenance,
target-state digest, risk tier, reversibility, or resource budget. Organism
v0.1 has no approval resume protocol: it rejects approval-required policies and
capabilities during trusted setup, and generated proposals bind
`approval_ref=None`.

## Can another LLM connect?

Yes. Implement the small synchronous adapter contract:

```python
class ModelAdapter(Protocol):
    @property
    def identity(self) -> AdapterIdentity: ...

    def invoke(self, call: ModelCall) -> ModelResult: ...
```

The application registers the adapter under a stable `adapter_id`. The exact
`AdapterIdentity` digest (adapter ID, provider, model, origin, version, schema
digest, and destination) must be:

- present in the organism manifest;
- allowed by the trusted organism policy; and
- backed by a host activation for the exact manifest digest.

Provider credentials stay inside the adapter implementation. They are not fields
of `ModelCall`, the organism manifest, or the evidence bundle. Provider request
IDs and usage are returned only as adapter-reported observations; they grant no
authority. A `ModelCall` carries both `contains_secret` and
`data_classification`; its payload is a private detached snapshot exposed only
through defensive copies and checked against `payload_digest`. `OrganismRun`
likewise exposes defensive copies of retained model output, while the
per-cell bundle evidence-binds its record digest rather than persisting the raw
ID/usage fields. Every `ModelResult` must also carry adapter-assigned
`contains_secret` and `data_classification` labels. The runner can only join
those labels monotonically with earlier labels; the adapter must classify
conservatively because the kernel does not inspect response content.

`ModelCall` and `ModelResult` use frozen dataclasses as an implementation detail,
with canonical bytes in private fields. Consumers must use `payload`, `output`,
and `to_record()`; `dataclasses.replace()`, `asdict()`, and field reflection are
not supported transport APIs.

The core v0.1 contract includes `CallbackModelAdapter`, a reference wrapper with
no provider SDK or transport of its own. Its application-supplied callback may
still perform I/O; the bundled example callbacks are fully offline. v0.2 also
ships a narrow `OllamaModelAdapter` for the loopback-only repository pilot.
A production remote-provider connector belongs in the embedding application and
must preserve the same contract. It must disable
provider-side tools, functions, and automatic agent handoffs, or route every such
effect back through its own guarded `CausalCell`; otherwise an SDK can bypass the
`ActionDraft` boundary. Redirects must be disabled or every hop must be checked
against the same destination and secret-egress policy. The adapter must transmit
the exact authorized payload snapshot or independently bind any transformation
to its versioned adapter schema; the in-process kernel cannot attest remote
request bytes.

## Can an LLM create its own organism?

Not autonomously in v0.1.

An LLM may produce data that a separate control plane uses to draft a candidate
manifest, but a trusted host must validate, bind, and explicitly activate the
exact manifest digest. The runtime hard-blocks `create_organism` and
`organism.spawn` in observer/analyst invocation cells and final capabilities,
before a model call or semantic-run reservation. There is no recursive spawn and
no model-controlled activation.

This keeps creation as a governance transition rather than a side effect hidden
inside model text.

## Fixed manifest

`schemas/organism-manifest.v0.1.schema.json` describes the wire shape. Runtime
validation additionally enforces policy caps and cross-field budget invariants.

The only supported pipeline is:

| Stage | Role | Side effect boundary |
|---|---|---|
| `observer` | Convert root input into bounded facts | Guard before model adapter |
| `analyst` | Convert facts into `ACTION` or `NO_ACTION` | Guard before model adapter |
| `executor` | Compile and execute one predeclared capability | Guard before executor |

Required global limits are three stages, at most two guarded adapter
invocations, zero runner retries, and runner fan-out of one. Token, cost, and
deadline caps are bound in the manifest and capped by trusted policy. Each
model call also receives its stage-local `max_seconds` deadline. The effective
run deadline is the earliest of context expiry, global manifest limit, and exact
activation expiry. These counters do not attest or constrain hidden SDK/provider
retries or fan-out inside an in-process adapter.

The runner revalidates activation and time before every model dispatch, after
each valid returned result before downstream dispatch, and before final action
dispatch. Malformed or terminal results already stop downstream. A callback is
synchronous and cannot be preempted, but a late result cannot authorize a
downstream effect.

Before consuming the semantic-run key or calling either model, the runner also
validates both complete policy documents and their statically known compatibility
with the fixed manifest, adapter identities, capabilities, executors, tool
provenance, budgets, and current context egress labels. A corrected retry is
therefore not poisoned by a deterministic trusted-configuration failure. Taint
introduced by a model result remains dynamic and is checked again at the normal
cell boundary. Time and activation are refreshed after preflight and immediately
before semantic-run consumption, so an expiry during preflight does not poison a
later valid retry.

A manifest digest covers the whole manifest except the digest field itself.
`StaticActivationRegistry` accepts any unexpired exact match on organism ID,
digest, subject, and policy version, allowing safe activation rotation. Missing
activation is `HOLD`; an expired exact activation or mismatch is `BLOCK`.

## Strict ActionDraft

The analyst returns one of two shapes from
`schemas/action-draft.v0.1.schema.json`.

Action:

```json
{
  "schema_version": 1,
  "profile": "org.causalcell.action-draft.v0.1",
  "kind": "ACTION",
  "capability_id": "cap.record",
  "target": "resource:demo/result",
  "arguments": {
    "result": "safe value"
  }
}
```

No action:

```json
{
  "schema_version": 1,
  "profile": "org.causalcell.action-draft.v0.1",
  "kind": "NO_ACTION",
  "reason_codes": [
    "INSUFFICIENT_EVIDENCE"
  ]
}
```

Unknown fields fail closed. Inputs and model outputs are detached into strict
plain-JSON snapshots before digest or reuse; mutable mapping/list subclasses,
non-string keys, non-finite numbers, depth over 128, and more than 100,000
JSON nodes are rejected. Individual strings are capped at 1,000,000 UTF-8 bytes,
aggregate string/key bytes at 4,000,000, and integers at 4,096 bits. Invalid
Unicode such as lone surrogates is rejected before hashing or dispatch.

The trusted capability definition restricts target prefixes, allowed/required
argument keys, a host-owned target validator/canonicalizer contract, a host-owned
argument value validator, and a trusted `target_state_resolver`. The resolver is
called only after target canonicalization and argument validation, and its exact
SHA-256 result is bound into the proposal. The resolver is a trusted host claim:
the runtime validates its shape but cannot prove that it describes real target
state. Implementations must resolve the selected target's current expected state,
not reuse a constant or substitute a request/argument digest. A target prefix must match exactly or at a
segment boundary; `tenant-a` cannot authorize `tenant-attacker`. The target
validator and state resolver must use the same interpretation as the executor.

For a network capability, `destination` is a trusted HTTPS origin, not a path or
payload field. The factory canonicalizes host case, IPv4/IPv6 literals, IPv6
brackets, and default port 443, and rejects path/query/fragment destinations and
scoped IPv6 zone identifiers. URL-like target prefixes and targets are
canonicalized before matching when they identify a complete URL; deliberately
broad prefixes such as `https://` remain broad but are still constrained by the
exact static destination check. Canonical URL prefixes that can reach only a
different origin are rejected during setup when no opaque, broad, or same-origin
prefix remains. If the selected target is URL-like, its canonical HTTPS origin
must equal the destination and the executor receives the canonical target; a
different, invalid, or fragmented URL is blocked. A non-network scope must use
`destination=None`. Opaque targets
and arguments are resource/data identifiers only:
the trusted executor must not reinterpret either as a second network endpoint.
If it derives an endpoint at all, the target/argument validator must bind that
derivation exactly, including redirects.

Capabilities must be reversible and use only `low` or `medium` risk in v0.1.
Their definitions are separate trusted runtime configuration: activation binds
the allowed capability IDs, not the validator code or full capability fields.
The factory snapshots each capability budget at setup to prevent later mutation.
Factory policy versions and every manifest capability/executor registration are
checked before the first model call.

The compiler copies only validated data fields and never merges the model
dictionary into an action proposal. Host context, capability labels, and
adapter-assigned `ModelResult` labels are joined monotonically: downstream
proposals can only retain or increase sensitivity. Adapter claims can raise
taint but cannot lower it. v0.1 has no independent content classifier, so an
under-labeling adapter remains a trusted-integration failure.

## Causal binding

The host generates run, trace, span, request, attempt, and nonce IDs. The final
action idempotency key is instead stable: it is derived from the semantic effect
(trusted subject and intent, root cause, action/scope, compiled target, canonical
destination origin, and validated argument digest), not the random attempt ID or
manifest topology. Executor-equivalent canonicalization beyond the kernel's HTTPS
rules remains part of the trusted validator/executor contract.

Each downstream call binds to the exact previous model-result ID and digest. The
final action must bind to the analyst result and to the original trusted subject,
intent, workload, and executor identity. Any mismatch blocks before dispatch.

A runner's semantic-run store atomically binds:

- organism and manifest digest;
- subject and intent;
- root parent cause;
- one detached root-input snapshot and its digest.

A second equivalent run through the same injected store is blocked as
`ORGANISM_REPLAYED`. A different root or manifest/adapter revision can run the
models again, but a shared action-cell nonce store still blocks the same
semantic executor effect. Default stores are created per `OrganismRunner`;
cross-runner protection requires callers to inject and share stores.

## Terminal states

| Status | Meaning |
|---|---|
| `COMPLETED` | Both model calls and the final guarded executor returned |
| `NO_ACTION` | Analyst returned a valid no-action draft |
| `HOLD` | Exact manifest activation is absent; nothing dispatched |
| `BLOCK` | Policy, provenance, shape, causality, replay, or budget check failed |
| `FAILED` | A provider returned a terminal result, or orchestration/control-plane infrastructure (including clock, replay store, or target-state resolver) failed without an unobserved invoked effect; earlier completed calls may be present |
| `EFFECT_UNCERTAIN` | An invoked adapter/executor raised, changed provenance, or evidence persistence failed after invocation began |

No automatic retry or approval resume occurs in v0.1. `COMPLETED` means the
callback returned; it is not independent proof that an external system committed
the effect. `OrganismRun.action_effect_boundary_started` (and its
`executor_invoked` compatibility property) remains true when the final
callback started but post-effect evidence persistence failed.

## Evidence

Every model call and the final capability action uses the existing `CausalCell`
evidence bundle. Authorization, observation, causal audit, continuity, replay,
and verification remain separate. Model payloads are kept in memory; the model
call proposal records their digest rather than raw content.

The organism result aggregates references to the individual `CellRun` values.
Retained model-call payloads and results remain digest-consistent under caller
mutation because access returns detached copies. v0.1 does not yet produce a
separately signed aggregate organism bundle.

## Example

Run the fully offline mixed-provider example:

```bash
python examples/multi_model_organism.py
```

It uses synthetic “OpenAI” and “Anthropic” identities to demonstrate
adapter-contract and provider-identity interchangeability without claiming real
provider SDK interoperability.

## Production limitations

Organism v0.1 is a reference kernel, not a production isolation boundary:

- adapters and executors are synchronous, in-process callbacks; an over-deadline
  callback cannot be preempted, though downstream dispatch is blocked;
- default replay, nonce, and semantic-run stores are in-memory and per runner;
  v0.2 also ships an explicitly injected, single-filesystem SQLite store;
- provider identity, usage, and result labels are adapter-reported, not
  independently attested;
- model output content is not independently classified; callers must label
  sensitive context before the run and adapters must conservatively label every
  result;
- token/cost checks use reported post-call usage, so they cannot prevent already
  consumed provider cost;
- adapter/SDK-internal retries, fan-out, and provider request counts are not
  observable or enforced by the runner;
- provider-side tools/functions/handoffs must be disabled or separately guarded,
  and every outbound redirect hop must be reauthorized;
- budgets are per run and not durable distributed reservations;
- the v0.2 Ollama pilot includes a zero-key loopback HTTP transport, but no
  remote-provider SDK, credential store, or general sandbox ships;
- executor callback return/error does not independently prove an external effect;
- no approval-required organism actions or plan/resume flow;
- no recursive organisms, dynamic topology, retries, or parallel fan-out;
- no durable aggregate organism evidence bundle exists yet.

Production deployments need isolated adapters/executors, durable atomic stores,
credential isolation, outbound network and redirect enforcement, distributed
budgets, independent effect verification, and an activation control plane.
