# Synthetic smart-contract release

Run:

```bash
python examples/smart_contract_release.py --evidence-root evidence/demo
```

Expected: first decision `HOLD`, second decision `ACCEPT`, one synthetic
executor call, and two locally verified bundles.

The example does not use RPC, wallets, credentials, customer data, or real
value. Its returned transaction hash is fixture data, not a blockchain fact.
