# Causal Cell

**The smallest verifiable unit of an agent action.**

Causal Cell is a provider-neutral reference kernel that treats every model
output as a proposal, evaluates trusted authority and policy before dispatch,
and preserves authorization, observation, continuity, and evidence as separate
records.

Status: `v0.2.0` alpha reference implementation; Organism protocol `v0.1`.

```mermaid
flowchart TD
    P["Agent proposal"] --> G["Pre-execution guard"]
    G -->|ACCEPT| E["Executor adapter"]
    G -->|HOLD or BLOCK| N["No dispatch"]
    E --> O["Observation and LTP export"]
    N --> O
    O --> B["Hash-linked evidence bundle"]
```

## Core invariant

```text
proposal ≠ authorization ≠ execution ≠ observation ≠ verification
```

- `ACCEPT` permits the executor adapter to be called.
- `HOLD` waits for bound approval or revalidation and does not dispatch.
- `BLOCK` rejects malformed, expired, replayed, out-of-scope, action/scope-
  confused, or denied work and does not dispatch.
- A process-local atomic store consumes both `nonce` and `idempotency_key`
  before the executor callback.
- Every attempt receives a unique evidence directory; a replay cannot overwrite
  the original attempt.

## Multi-model organisms

Organism v0.1 composes two provider-neutral model adapters and one capability
executor without giving any LLM authority:

```mermaid
flowchart TD
    O["Observer adapter"] --> A["Analyst adapter"]
    A --> D["Strict ActionDraft"]
    D --> C["Trusted compiler"]
    C --> X["Guarded executor"]
```

Each model adapter invocation is itself guarded because it can spend money and
cross a network boundary. The analyst may return only a capability ID, target,
and data arguments. The trusted host supplies identity, scope, policy, approval,
tool provenance, causal IDs, replay keys, and budgets.

OpenAI, Anthropic, Gemini, a local model, or another provider can occupy either
model role through the same `ModelAdapter` contract. Exact adapter identity is
digest-bound in the manifest and explicitly activated by the host. Recursive
`create_organism` / `organism.spawn` actions are hard-blocked.

See [Organism v0.1](docs/ORGANISM_V0_1.md), its
[threat model](docs/ORGANISM_THREAT_MODEL.md), and the
[offline mixed-provider example](examples/multi_model_organism.py).

## Quick start

```bash
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python -m benchmarks.run_safety_matrix
python examples/multi_model_organism.py
```

Evaluate a proposal without executing anything:

```bash
python -m causal_cell evaluate \
  --proposal examples/fixtures/safe-proposal.json \
  --policy examples/fixtures/policy.json \
  --now 2026-08-27T21:00:00Z
```

Run the synthetic approval-to-dispatch example:

```bash
python examples/smart_contract_release.py --evidence-root evidence/demo
```

The example first returns `HOLD`, then accepts a separately bound approval,
invokes one synthetic executor, and locally verifies both evidence bundles. It
does not use RPC, credentials, a wallet, customer data, or real value.

## What one bundle contains

| Layer | Artifact | Meaning |
|---|---|---|
| Intent | `intent.json` | Application-supplied intent and causal parent |
| Action | `proposal.json` | Exact digest-bound proposal |
| Authorization | `authorization.json` | `ACCEPT`, `HOLD`, or `BLOCK` before dispatch |
| Result | `observation.json` | What the runtime observed at the callback boundary |
| Response integrity | `response-integrity.json` | Kept separate; `NOT_EVALUATED` in v0.1 |
| Causal audit | `causal-audit.json` | Causal validity and compact findings |
| Continuity | `ltp-continuity-input.json` | LTP v0.1 request/outcome envelope snapshot |
| Replay | `replay-trace.json` | Decision-path events, not external-effect replay |
| Integrity | `ledger.jsonl`, `manifest.json` | Hash chain plus exact size/SHA-256 inventory |

The verifier rejects changed bytes, broken ledger ancestry, unsafe paths,
duplicate roles, missing files, and unlisted-file injection.

## Relationship to the existing stack

Causal Cell is deliberately thin:

- **ProofPath** remains the authority/guard responsibility.
- **ContractGraph-QA** remains the source of smart-contract receipts, events,
  state, and technical evidence.
- **LTP** remains the normative request/outcome continuity verifier. Causal Cell
  exports its versioned envelope shapes; it does not reinterpret an LTP verdict.
- **LiminalDB** remains the durable transition-ledger destination. The local
  JSONL ledger here is only a handoff/reference format.

Exact inspected revisions and mapping details are in
[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).

## Measured local baseline

The synthetic matrix covers 16 decision cases plus a runtime replay exercise:

- 16/16 expected decisions matched;
- 100% detection across fixtures expected to be held or blocked;
- 0 false positives and 0 false negatives in that bounded matrix;
- 100% bundle completeness for the two runtime attempts;
- one executor call across original + replay.

These are synthetic fixture results, not production-security or universal
detection claims.

## Honest boundaries

Version `v0.2` does **not** provide:

- an OS/container/VM sandbox — adapter and executor callbacks are in-process;
- persistent or distributed nonce, semantic-run, or budget transactions;
- cryptographic approval signatures, key management, or provider attestation;
- independent truth verification of a model or executor response;
- blockchain finality, evidence authenticity, or complete history;
- external exactly-once side effects;
- dynamic organism topology, retries, parallel fan-out, or recursive spawn; or
- a separately signed aggregate organism evidence bundle.

Production callers must use durable atomic stores and thin adapters to externally
contained model/tool executors. See [the base specification](docs/SPEC_V0_1.md),
[base threat model](docs/THREAT_MODEL.md), and
[organism limitations](docs/ORGANISM_V0_1.md#production-limitations).

## License

Apache-2.0.
