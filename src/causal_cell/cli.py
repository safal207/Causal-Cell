"""Small read/evaluate/verify CLI; it performs no external action."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime

from .canonical import parse_timestamp
from .evidence import load_json_strict, verify_bundle
from .guard import evaluate_proposal
from .models import DecisionStatus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causal-cell")
    subcommands = parser.add_subparsers(dest="command", required=True)
    evaluate = subcommands.add_parser("evaluate", help="evaluate without executing")
    evaluate.add_argument("--proposal", required=True)
    evaluate.add_argument("--policy", required=True)
    evaluate.add_argument("--now", help="explicit timestamp for deterministic replay")
    verify = subcommands.add_parser("verify", help="verify one evidence bundle")
    verify.add_argument("--bundle", required=True)
    return parser


def _emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        proposal = load_json_strict(args.proposal)
        policy = load_json_strict(args.policy)
        now: datetime | None = parse_timestamp(args.now) if args.now else None
        decision = evaluate_proposal(proposal, policy, now=now)
        _emit(decision.to_record())
        return 0 if decision.status is DecisionStatus.ACCEPT else 2
    verification = verify_bundle(args.bundle)
    _emit(verification.to_record())
    return 0 if verification.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
