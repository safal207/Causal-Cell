# Integration map and inspected revisions

Causal Cell was designed after inventorying these repositories. Pins make the
design input reproducible; they are not dependency or endorsement claims.

| Component | Inspected revision | Responsibility retained |
|---|---|---|
| [ContractGraph-QA](https://github.com/safal207/ContractGraph-QA) | `a0c654fe49799b15315da5b8d31f5efb0db08457` | EVM/RPC receipt, event, state, and evidence capture |
| [LTP](https://github.com/safal207/L-THREAD-Liminal-Thread-Secure-Protocol-LTP-) | `5647a3047c12ebc5c3d39b2c9da98a8d9ce49cb6` | Normative request/outcome continuity judgment |
| [ProofPath](https://github.com/safal207/ProofPath) | `17b0b3a46bd9dc5d61e2782098208dedd82faa80` | Pre-execution authority and guard semantics |
| [LiminalDB](https://github.com/safal207/LiminalDB) | `61b02fc81e0cb5cf1f1ed4658ecff58f683cb728` | Durable transition storage and replay |

Causal Cell exports LTP v0.1 envelope shapes but does not copy its verdict
algorithm or turn continuity into authorization. It does not duplicate
ContractGraph-QA receipt/state checks. Its ACCEPT/HOLD/BLOCK separation aligns
with inspected ProofPath patterns without claiming an official integration.
Its local JSONL ledger is not LiminalDB WAL, snapshot, crash recovery, or
durable multi-process replay.
