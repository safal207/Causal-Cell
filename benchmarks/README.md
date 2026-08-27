# Safety matrix v0.1

Run:

```bash
python -m benchmarks.run_safety_matrix
```

The matrix evaluates 16 deterministic proposal/policy fixtures and performs an
original-plus-replay runtime exercise.

Metrics cover exact decision matches, bounded detection rate, false positives,
false negatives, evidence completeness, executor call count, and replay
stability.

The matrix contains no real secret, network request, exploit, customer data, or
external side effect. Results do not generalize beyond these fixtures.
