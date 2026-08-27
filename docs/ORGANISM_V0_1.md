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
policy version, approval, nonce, idempotency key, delegation chain, tool
provenance, target-state digest, risk tier, reversibility, or resource budget.

## Can another LLM connect?

Yes. Implement the small synchronous adapter contract:

```python
class ModelAdapter(Protocol):
    @property
    def identity(self) -> AdapterIdentity: ...

    def invoke(self, call: ModelCall) -> ModelResult: ...
```

The application registers the adapter under a stable `adapter_id`. The exact
`AdapterIdentity` digest (provider, model, origin, version, schema digest, and
destination) must be:

- present in the organism manifest;
- allowed by the trusted organism policy; and
- backed by a host activation for the exact manifest digest.

Provider credentials stay inside the adapter implementation. They are not fields
of `ModelCall`, the organism manifest, or the evidence bundle. Provider request
IDs and usage are recorded only as observations; they grant no authority.

The repository ships only `CallbackModelAdapter`, an offline/reference adapter.
It performs no network call. A production OpenAI, Anthropic, Gemini, local-model,
or other connector belongs in the embedding application and must preserve the
same contract.

## Can an LLM create its own organism?

Not autonomously in v0.1.

An LLM may produce data that a separate control plane uses to draft a candidate
manifest, but a trusted host must validate, bind, and explicitly activate the
exact manifest digest. The runtime hard-blocks `create_organism` and
`organism.spawn` actions. There is no recursive spawn and no model-controlled
activation.

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

Required global limits are three stages, two model calls, zero retries, and fan-out
of one. Token, cost, and deadline caps are also bound in the manifest and capped
by trusted policy.

A manifest digest covers the whole manifest except the digest field itself.
`StaticActivationRegistry` requires an exact match on organism ID, digest,
subject, policy version, and expiry. Missing activation is `HOLD`; a mismatch or
expired activation is `BLOCK`.

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

Unknown fields fail closed. The trusted capability definition restricts target
prefixes and allowed/required argument keys. The compiler copies only those data
fields; it never merges the model dictionary into an action proposal.

## Causal binding

The host generates run, trace, span, request, attempt, nonce, and idempotency IDs.

Each downstream call binds to the exact previous model-result ID and digest. The
final action must bind to the analyst result and to the original trusted subject,
intent, workload, and executor identity. Any mismatch blocks before dispatch.

A process-local atomic semantic-run key binds:

- organism and manifest digest;
- subject and intent;
- root parent cause;
- root input digest.

A second equivalent run is blocked as `ORGANISM_REPLAYED`.

## Terminal states

| Status | Meaning |
|---|---|
| `COMPLETED` | Both model calls and the final guarded executor returned |
| `NO_ACTION` | Analyst returned a valid no-action draft |
| `HOLD` | Required activation or approval is absent |
| `BLOCK` | Policy, provenance, shape, causality, replay, or budget check failed |
| `FAILED` | A model adapter refused, errored, or raised |
| `EFFECT_UNCERTAIN` | Final executor raised; no success claim is made |

No automatic retry occurs in v0.1.

## Evidence

Every model call and the final capability action uses the existing `CausalCell`
evidence bundle. Authorization, observation, causal audit, continuity, replay,
and verification remain separate. Model payloads are kept in memory; the model
call proposal records their digest rather than raw content.

The organism result aggregates references to the individual `CellRun` values.
v0.1 does not yet produce a separately signed aggregate organism bundle.

## Example

Run the fully offline mixed-provider example:

```bash
python examples/multi_model_organism.py
```

It uses synthetic “OpenAI” and “Anthropic” identities to prove provider
interchangeability without credentials or network access.

## Production limitations

Organism v0.1 is a reference kernel, not a production isolation boundary:

- adapters and executors are synchronous, in-process callbacks;
- replay and semantic-run stores are process-local;
- provider identity and usage are adapter-reported, not independently attested;
- budgets are per run and not distributed reservations;
- no real provider SDK, credential store, sandbox, or network transport ships;
- executor return/error does not independently prove the external side effect;
- no recursive organisms, dynamic topology, retries, or parallel fan-out;
- no durable aggregate organism evidence bundle exists yet.

Production deployments need isolated adapters/executors, durable atomic stores,
credential isolation, outbound network enforcement, distributed budgets,
independent effect verification, and an activation control plane.
