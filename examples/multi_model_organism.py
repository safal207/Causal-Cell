"""Offline mixed-provider Organism v0.1 example.

Both adapters are synthetic callbacks. For real providers, preserve the guarded
adapter/compiler pattern while replacing the capability validators, budgets,
provenance, and executor with application-owned production configuration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from causal_cell import (
    ACTION_DRAFT_PROFILE,
    MANIFEST_PROFILE,
    AdapterIdentity,
    AdapterRegistry,
    CallbackModelAdapter,
    ManifestActivation,
    ModelCall,
    ModelResult,
    OrganismPolicy,
    OrganismRunner,
    RunContext,
    StaticActivationRegistry,
    StaticCapability,
    StaticProposalFactory,
    bind_organism_manifest,
)

NOW = datetime(2026, 8, 27, 21, 0, tzinfo=UTC)
AUTH_DIGEST = "sha256:" + "d" * 64
TARGET_DIGEST = "sha256:" + "c" * 64
RECORDER_SCHEMA_DIGEST = "sha256:" + "e" * 64


def resource_budget() -> dict[str, int | float]:
    return {
        "max_steps": 1,
        "max_seconds": 15,
        "max_cost": 1.0,
        "max_fan_out": 0,
        "max_retries": 0,
    }


def policy(
    *,
    version: str,
    agents: list[str],
    actions: list[str],
    action_scopes: dict[str, list[str]],
    scopes: list[str],
    network_scopes: list[str],
    destinations: list[str],
    tools: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": "org.causalcell.policy.v0.1",
        "policy_version": version,
        "allowed_subjects": ["user:alice"],
        "allowed_agents": agents,
        "allowed_workloads": ["organism:mixed-provider-demo"],
        "allowed_actions": actions,
        "allowed_action_scopes": action_scopes,
        "allowed_scopes": scopes,
        "network_scopes": network_scopes,
        "allowed_destinations": destinations,
        "allowed_secret_destinations": [],
        "trusted_tools": tools,
        "allowed_delegation_chains": [],
        "approval_required_risk_tiers": [],
        "require_approval_for_irreversible": False,
        "max_resource_budget": {
            "max_steps": 3,
            "max_seconds": 30,
            "max_cost": 2.0,
            "max_fan_out": 1,
            "max_retries": 0,
        },
        "approvals": {},
    }


def build_runner(evidence_root: Path) -> tuple[OrganismRunner, list[dict[str, Any]]]:
    observer_identity = AdapterIdentity(
        adapter_id="adapter:observer:openai",
        provider="openai",
        model="synthetic-observer",
        origin="registry.example.test/openai-adapter",
        version="1.0.0",
        schema_digest="sha256:" + "1" * 64,
        destination="https://api.openai.example.test",
    )
    analyst_identity = AdapterIdentity(
        adapter_id="adapter:analyst:anthropic",
        provider="anthropic",
        model="synthetic-analyst",
        origin="registry.example.test/anthropic-adapter",
        version="1.0.0",
        schema_digest="sha256:" + "2" * 64,
        destination="https://api.anthropic.example.test",
    )

    manifest = bind_organism_manifest(
        {
            "schema_version": 1,
            "profile": MANIFEST_PROFILE,
            "organism_id": "organism:mixed-provider-demo",
            "manifest_version": "demo-v1",
            "policy_version": "organism-policy-v1",
            "pipeline": {
                "observer": {
                    "cell_id": "cell:observer",
                    "agent_id": "agent:observer",
                    "adapter_id": observer_identity.adapter_id,
                    "adapter_identity_digest": observer_identity.identity_digest,
                    "invocation_action": "invoke_observer_model",
                    "invocation_scope": "network.model",
                    "resource_budget": resource_budget(),
                },
                "analyst": {
                    "cell_id": "cell:analyst",
                    "agent_id": "agent:analyst",
                    "adapter_id": analyst_identity.adapter_id,
                    "adapter_identity_digest": analyst_identity.identity_digest,
                    "invocation_action": "invoke_analyst_model",
                    "invocation_scope": "network.model",
                    "resource_budget": resource_budget(),
                },
                "executor": {
                    "cell_id": "cell:executor",
                    "agent_id": "agent:executor",
                    "allowed_capability_ids": ["cap.record"],
                },
            },
            "limits": {
                "max_steps": 3,
                "max_seconds": 30,
                "max_model_calls": 2,
                "max_output_tokens_per_call": 64,
                "max_total_tokens": 128,
                "max_cost_microunits_per_call": 100_000,
                "max_total_cost_microunits": 200_000,
                "max_retries": 0,
                "max_fan_out": 1,
            },
        }
    )

    def observe(call: ModelCall) -> ModelResult:
        return ModelResult.returned(
            observer_identity,
            {
                "facts": ["input is a synthetic demo"],
                "input_digest": call.payload_digest,
            },
            provider_request_id="synthetic-openai-request",
            input_tokens=4,
            output_tokens=5,
            cost_microunits=100,
            contains_secret=False,
            data_classification="public",
        )

    def analyse(call: ModelCall) -> ModelResult:
        # The model can select only a predeclared capability plus data arguments.
        return ModelResult.returned(
            analyst_identity,
            {
                "schema_version": 1,
                "profile": ACTION_DRAFT_PROFILE,
                "kind": "ACTION",
                "capability_id": "cap.record",
                "target": "resource:demo/result",
                "arguments": {"result": "safe synthetic result"},
            },
            provider_request_id="synthetic-anthropic-request",
            input_tokens=6,
            output_tokens=7,
            cost_microunits=200,
            contains_secret=False,
            data_classification="public",
        )

    adapters = AdapterRegistry(
        [
            CallbackModelAdapter(observer_identity, observe),
            CallbackModelAdapter(analyst_identity, analyse),
        ]
    )
    capability = StaticCapability(
        capability_id="cap.record",
        action="record_result",
        scope="synthetic.write",
        executor_id="synthetic-recorder",
        tool_origin="registry.example.test/synthetic-recorder",
        tool_version="1.0.0",
        tool_schema_digest=RECORDER_SCHEMA_DIGEST,
        target_state_digest=TARGET_DIGEST,
        reversibility="reversible",
        risk_tier="low",
        allowed_target_prefixes=("resource:",),
        target_validator=lambda target: all(
            segment not in {".", ".."}
            for segment in target.split("/")
        ),
        allowed_argument_keys=frozenset({"result"}),
        required_argument_keys=frozenset({"result"}),
        argument_validator=lambda arguments: (
            isinstance(arguments.get("result"), str)
            and 0 < len(arguments["result"]) <= 256
        ),
        resource_budget=resource_budget(),
    )

    invocation_policy = policy(
        version="invocation-policy-v1",
        agents=["agent:observer", "agent:analyst"],
        actions=["invoke_observer_model", "invoke_analyst_model"],
        action_scopes={
            "invoke_observer_model": ["network.model"],
            "invoke_analyst_model": ["network.model"],
        },
        scopes=["network.model"],
        network_scopes=["network.model"],
        destinations=[
            observer_identity.destination,
            analyst_identity.destination,
        ],
        tools=[
            {
                "origin": identity.origin,
                "version": identity.version,
                "schema_digest": identity.schema_digest,
            }
            for identity in (observer_identity, analyst_identity)
        ],
    )
    action_policy = policy(
        version="action-policy-v1",
        agents=["agent:executor"],
        actions=["record_result"],
        action_scopes={"record_result": ["synthetic.write"]},
        scopes=["synthetic.write"],
        network_scopes=[],
        destinations=[],
        tools=[
            {
                "origin": capability.tool_origin,
                "version": capability.tool_version,
                "schema_digest": capability.tool_schema_digest,
            }
        ],
    )
    organism_policy = OrganismPolicy(
        policy_version="organism-policy-v1",
        allowed_adapter_identity_digests=frozenset(
            {observer_identity.identity_digest, analyst_identity.identity_digest}
        ),
        allowed_capability_ids=frozenset({"cap.record"}),
        max_seconds=30,
        max_output_tokens_per_call=64,
        max_total_tokens=128,
        max_cost_microunits_per_call=100_000,
        max_total_cost_microunits=200_000,
    )
    activations = StaticActivationRegistry(
        [
            ManifestActivation(
                organism_id=manifest["organism_id"],
                manifest_digest=manifest["manifest_digest"],
                subject="user:alice",
                policy_version=manifest["policy_version"],
                expires_at="2026-08-27T21:45:00Z",
            )
        ]
    )
    recorded: list[dict[str, Any]] = []

    def record(proposal: Mapping[str, Any]) -> dict[str, Any]:
        recorded.append(dict(proposal))
        return {"recorded": True, "target": proposal["target"]}

    runner = OrganismRunner(
        manifest=manifest,
        organism_policy=organism_policy,
        invocation_policy=invocation_policy,
        action_policy=action_policy,
        activations=activations,
        adapters=adapters,
        proposal_factory=StaticProposalFactory(
            invocation_policy_version="invocation-policy-v1",
            action_policy_version="action-policy-v1",
            capabilities=[capability],
        ),
        executors={"synthetic-recorder": record},
        evidence_root=evidence_root,
        clock=lambda: NOW,
    )
    return runner, recorded


def main() -> None:
    with TemporaryDirectory() as temp:
        runner, recorded = build_runner(Path(temp))
        run = runner.run(
            {"request": "show that two providers can form one guarded organism"},
            RunContext(
                subject="user:alice",
                intent_id="intent:mixed-provider-demo",
                parent_cause="user-request:mixed-provider-demo",
                auth_context_digest=AUTH_DIGEST,
                issued_at="2026-08-27T20:00:00Z",
                expires_at="2026-08-27T22:00:00Z",
            ),
        )
        print(
            json.dumps(
                {
                    "status": run.status.value,
                    "model_providers": [
                        item.provider for item in run.model_results
                    ],
                    "guarded_model_calls": len(run.cell_runs),
                    "action_executor_invoked": run.executor_invoked,
                    "recorded_actions": len(recorded),
                    "all_evidence_valid": all(
                        item.verification.valid
                        for item in (
                            *run.cell_runs,
                            *([run.action_run] if run.action_run else []),
                        )
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
