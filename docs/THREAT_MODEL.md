# Threat model

## Protected assets

- external side-effect boundary;
- user intent and causal ancestry;
- subject, agent, workload, delegation, and tool authority;
- secrets and restricted data crossing a network boundary;
- replay keys and resource budgets;
- authorization, observation, continuity, and evidence integrity.

## Trust boundaries

Trusted inputs are application-supplied policy, approvals, clock, and the
production replay store. The model proposal and web, retrieval, memory, tool,
metadata, and inter-agent content are untrusted.

The Python executor callback is not a sandbox. External executors, blockchain
nodes, APIs, indexers, databases, and their responses remain outside v0.1.

## Covered threats

| Threat | v0.1 control |
|---|---|
| Model output treated as authority | Proposal and authorization remain separate |
| Missing intent or causal parent | Fail-closed guard |
| Action/scope confusion or unknown-action escalation | Exact action-to-scope bindings plus allow-lists |
| Confused identity/delegation | Separate identity allow-lists and exact chains |
| Tool supply-chain change | Origin, version, and schema digest binding |
| Approval copied to another action | Exact proposal/target/context binding |
| Secret egress | Network scope plus canonical HTTPS destination allow-list |
| Context hijack | Untrusted instructions ignored and recorded |
| Retry/fan-out amplification | Declared budget ceilings |
| Nonce/idempotency replay | Process-local atomic first-winner consumption |
| Evidence overwrite/path traversal | Runtime-generated unique directory |
| Evidence modification/injection | Exact manifest and hash-linked ledger |
| False-success collapse | Authorization, observation, and response integrity stay separate |

## Residual risks

- replay state disappears on restart and is not shared across processes;
- expected target state and authentication context are supplied, not fetched;
- approvals are not cryptographically signed;
- policy revocation after runtime construction is not observed;
- the in-process executor can affect the host process;
- declared budgets are not externally enforced by v0.1;
- executor return does not prove an external effect or response truth;
- local SHA-256 does not prove authorship, time, or external finality;
- local JSON canonicalization is not claimed as RFC 8785;
- LTP envelopes cover supplied observations, not omitted history.
