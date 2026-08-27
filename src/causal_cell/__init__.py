"""Causal Cell v0.1 reference kernel."""

from .canonical import bind_proposal, digest_json, proposal_digest
from .evidence import EvidenceVerification, verify_bundle
from .guard import evaluate_proposal
from .models import CellRun, Decision, DecisionStatus
from .normalize import normalize_proposal
from .runtime import CausalCell, InMemoryNonceStore

__all__ = [
    "CausalCell",
    "CellRun",
    "Decision",
    "DecisionStatus",
    "EvidenceVerification",
    "InMemoryNonceStore",
    "bind_proposal",
    "digest_json",
    "evaluate_proposal",
    "normalize_proposal",
    "proposal_digest",
    "verify_bundle",
]

__version__ = "0.1.0"
