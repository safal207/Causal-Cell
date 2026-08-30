# Durable Ollama repository pilot

This v0.2 pilot runs both Organism model stages through a local Ollama server,
stores replay claims in SQLite, observes a repository without executing its
code, and permits only one reversible local-report capability.

## Why this free path

Checked against official sources on 2026-08-30:

- [Ollama Docker](https://docs.ollama.com/docker) runs locally without an API
  credential or per-request charge.
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
  accept a JSON schema, which fits the strict `ModelResult` / `ActionDraft`
  boundary.
- [`qwen3:4b`](https://ollama.com/library/qwen3:4b) is a 2.5 GB Apache-2.0 model
  and is the default here.
- [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing)
  still has a free tier, but requires an account/key and says free-tier content
  can be used to improve Google products.
- [Groq rate limits](https://console.groq.com/docs/rate-limits) list a free plan,
  but it also requires an account/key and has daily/model limits.
- [GitHub Models](https://docs.github.com/en/github-models) was fully retired on
  2026-07-30, so it is not a fallback.

The repository therefore defaults to Ollama. Gemini and Groq are possible future
remote adapters, not hidden fallbacks.

## Run

Prerequisites: Docker Desktop is running. Allow at least 6 GB of free disk space
for the default model, images, and build layers; 8 GB of system RAM is a
practical minimum for Docker plus the 4B model, although host overhead varies.

```bash
docker compose up -d ollama
docker compose run --rm model-init
docker compose build pilot
docker compose run --rm --no-deps pilot
docker compose stop ollama
```

The first run downloads the model. No API key or billing account is used.
Model initialization is intentionally a separate step, so its normal one-shot
exit cannot stop the pilot before the pilot starts. If a download fails, rerun
the same `model-init` command; Ollama resumes retained layers. The final `stop`
preserves both named volumes. `docker compose down` also preserves them;
`docker compose down -v` is a destructive reset of models and pilot state.

To select a smaller model on a constrained machine:

```bash
OLLAMA_MODEL=qwen3:1.7b docker compose run --rm model-init
OLLAMA_MODEL=qwen3:1.7b docker compose run --rm --no-deps pilot
```

On PowerShell, set `$env:OLLAMA_MODEL = "qwen3:1.7b"` once, then run the same
two Compose commands without the leading assignment.

## Containment and authority

- the checkout is mounted at `/workspace` read-only;
- model and pilot share a network namespace, so the adapter reaches Ollama only
  at literal loopback `127.0.0.1:11434`;
- Causal Cell policy accepts clear-text HTTP only for scope
  `network.local_model` and literal loopback IPs; hostnames, private LAN
  addresses, redirects, environment proxies, and other adapter destinations are
  denied. The shared container network itself still has outbound connectivity
  for Ollama model downloads and is not an OS firewall;
- repository files are bounded, digest-labeled untrusted data; excluded trees,
  symlinks, exact common credential filenames, and `.env.*` files are skipped;
  excerpts are included only for an allowlist of text-oriented suffixes;
- model tools, streaming, redirects, credentials, environment proxies, and
  model-side retries are not enabled;
- the analyst can select only `cap.record_repository_observation` or `NO_ACTION`;
- reports, evidence, nonce claims, idempotency claims, and semantic-run claims
  live in the `causal-cell-state` volume;
- rerunning the exact same snapshot is blocked durably as `ORGANISM_REPLAYED`.

The Python and Ollama container images are digest-pinned. The default
`qwen3:4b` model tag can still resolve to different weights in a future Ollama
registry update; v0.2 records the requested tag and adapter configuration but
does not attest the downloaded model artifact digest.

SQLite is durable and process-safe on one filesystem. It is not a distributed
transaction, a remote consensus store, or proof of an external side effect.
Freshness is rechecked at the atomic replay-claim boundary after any SQLite lock
wait. An external executor that requires a hard start-time deadline must also
enforce that deadline itself because a database commit and an external side
effect cannot be one atomic transaction.
