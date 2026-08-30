"""Causal Cell v0.2 reference kernel and fixed-topology Organism runtime."""

from .adapters import (
    AdapterIdentity,
    AdapterRegistry,
    CallbackModelAdapter,
    ModelAdapter,
    ModelCall,
    ModelResult,
    ModelResultStatus,
    snapshot_model_result,
    validate_model_result,
)
from .canonical import bind_proposal, digest_json, proposal_digest, snapshot_json
from .evidence import EvidenceVerification, verify_bundle
from .guard import evaluate_proposal
from .models import CellRun, Decision, DecisionStatus
from .normalize import normalize_proposal
from .ollama import OLLAMA_ADAPTER_SCHEMA_DIGEST, OllamaModelAdapter
from .organism import (
    ACTION_DRAFT_PROFILE,
    MANIFEST_PROFILE,
    InMemoryOrganismStore,
    ManifestActivation,
    OrganismDecision,
    OrganismPolicy,
    OrganismRun,
    OrganismStore,
    OrganismRunner,
    OrganismStatus,
    PreparedAction,
    ProposalCompilationError,
    ProposalInfrastructureError,
    RunContext,
    StaticActivationRegistry,
    StaticCapability,
    StaticProposalFactory,
    bind_organism_manifest,
    evaluate_organism_manifest,
    organism_manifest_digest,
    validate_action_draft,
)
from .runtime import CausalCell, InMemoryNonceStore, NonceStore
from .sqlite_store import SQLiteReplayStore

__all__ = [
    "ACTION_DRAFT_PROFILE",
    "MANIFEST_PROFILE",
    "AdapterIdentity",
    "AdapterRegistry",
    "CallbackModelAdapter",
    "CausalCell",
    "CellRun",
    "Decision",
    "DecisionStatus",
    "EvidenceVerification",
    "InMemoryNonceStore",
    "InMemoryOrganismStore",
    "ManifestActivation",
    "ModelAdapter",
    "ModelCall",
    "ModelResult",
    "ModelResultStatus",
    "OLLAMA_ADAPTER_SCHEMA_DIGEST",
    "OllamaModelAdapter",
    "OrganismDecision",
    "OrganismPolicy",
    "OrganismRun",
    "OrganismStore",
    "OrganismRunner",
    "OrganismStatus",
    "PreparedAction",
    "ProposalCompilationError",
    "ProposalInfrastructureError",
    "RunContext",
    "NonceStore",
    "SQLiteReplayStore",
    "StaticActivationRegistry",
    "StaticCapability",
    "StaticProposalFactory",
    "bind_organism_manifest",
    "bind_proposal",
    "digest_json",
    "evaluate_organism_manifest",
    "evaluate_proposal",
    "normalize_proposal",
    "organism_manifest_digest",
    "proposal_digest",
    "snapshot_json",
    "snapshot_model_result",
    "validate_action_draft",
    "validate_model_result",
    "verify_bundle",
]

__version__ = "0.2.0"
