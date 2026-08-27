from __future__ import annotations

import copy
import json
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from causal_cell.adapters import (
    AdapterIdentity,
    AdapterRegistry,
    CallbackModelAdapter,
    ModelCall,
    ModelResult,
    ModelResultStatus,
)
from causal_cell.models import DecisionStatus
from causal_cell.organism import (
    ACTION_DRAFT_PROFILE,
    MANIFEST_PROFILE,
    ManifestActivation,
    OrganismPolicy,
    OrganismRunner,
    OrganismStatus,
    RunContext,
    StaticActivationRegistry,
    StaticCapability,
    StaticProposalFactory,
    bind_organism_manifest,
    validate_action_draft,
)


NOW = datetime(2026, 8, 27, 21, 0, tzinfo=UTC)
AUTH_DIGEST = "sha256:" + "d" * 64
TARGET_DIGEST = "sha256:" + "c" * 64
RECORDER_SCHEMA_DIGEST = "sha256:" + "e" * 64


def _budget() -> dict[str, int | float]:
    return {
        "max_steps": 1,
        "max_seconds": 15,
        "max_cost": 1.0,
        "max_fan_out": 0,
        "max_retries": 0,
    }


def _identity(role: str, provider: str) -> AdapterIdentity:
    marker = "1" if role == "observer" else "2"
    return AdapterIdentity(
        adapter_id=f"adapter:{role}",
        provider=provider,
        model=f"{provider}-{role}-synthetic",
        origin=f"registry.example.test/{provider}-{role}-adapter",
        version="1.0.0",
        schema_digest="sha256:" + marker * 64,
        destination=f"https://api.{provider}.example.test",
    )


def _manifest(
    observer: AdapterIdentity,
    analyst: AdapterIdentity,
    capability_ids: list[str],
) -> dict[str, Any]:
    return bind_organism_manifest(
        {
            "schema_version": 1,
            "profile": MANIFEST_PROFILE,
            "organism_id": "organism:demo",
            "manifest_version": "organism-demo-v1",
            "policy_version": "organism-policy-v1",
            "pipeline": {
                "observer": {
                    "cell_id": "cell:observer",
                    "agent_id": "agent:observer",
                    "adapter_id": observer.adapter_id,
                    "adapter_identity_digest": observer.identity_digest,
                    "invocation_action": "invoke_observer_model",
                    "invocation_scope": "network.model",
                    "resource_budget": _budget(),
                },
                "analyst": {
                    "cell_id": "cell:analyst",
                    "agent_id": "agent:analyst",
                    "adapter_id": analyst.adapter_id,
                    "adapter_identity_digest": analyst.identity_digest,
                    "invocation_action": "invoke_analyst_model",
                    "invocation_scope": "network.model",
                    "resource_budget": _budget(),
                },
                "executor": {
                    "cell_id": "cell:executor",
                    "agent_id": "agent:executor",
                    "allowed_capability_ids": capability_ids,
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


def _invocation_policy(
    observer: AdapterIdentity,
    analyst: AdapterIdentity,
    *,
    deny_destinations: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": "org.causalcell.policy.v0.1",
        "policy_version": "invocation-policy-v1",
        "allowed_subjects": ["user:alice"],
        "allowed_agents": ["agent:observer", "agent:analyst"],
        "allowed_workloads": ["organism:demo"],
        "allowed_actions": ["invoke_observer_model", "invoke_analyst_model"],
        "allowed_action_scopes": {
            "invoke_observer_model": ["network.model"],
            "invoke_analyst_model": ["network.model"],
        },
        "allowed_scopes": ["network.model"],
        "network_scopes": ["network.model"],
        "allowed_destinations": (
            [] if deny_destinations else [observer.destination, analyst.destination]
        ),
        "allowed_secret_destinations": [],
        "trusted_tools": [
            {
                "origin": identity.origin,
                "version": identity.version,
                "schema_digest": identity.schema_digest,
            }
            for identity in (observer, analyst)
        ],
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


def _action_policy(*, allow_spawn: bool = False) -> dict[str, Any]:
    actions = ["record_result"]
    scopes = ["synthetic.write"]
    mapping = {"record_result": ["synthetic.write"]}
    if allow_spawn:
        actions.append("create_organism")
        scopes.append("organism.spawn")
        mapping["create_organism"] = ["organism.spawn"]
    return {
        "schema_version": 1,
        "profile": "org.causalcell.policy.v0.1",
        "policy_version": "action-policy-v1",
        "allowed_subjects": ["user:alice"],
        "allowed_agents": ["agent:executor"],
        "allowed_workloads": ["organism:demo"],
        "allowed_actions": actions,
        "allowed_action_scopes": mapping,
        "allowed_scopes": scopes,
        "network_scopes": [],
        "allowed_destinations": [],
        "allowed_secret_destinations": [],
        "trusted_tools": [
            {
                "origin": "registry.example.test/synthetic-recorder",
                "version": "1.0.0",
                "schema_digest": RECORDER_SCHEMA_DIGEST,
            }
        ],
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


def _capability(*, spawn: bool = False) -> StaticCapability:
    return StaticCapability(
        capability_id="cap.spawn" if spawn else "cap.record",
        action="create_organism" if spawn else "record_result",
        scope="organism.spawn" if spawn else "synthetic.write",
        executor_id="synthetic-recorder",
        tool_origin="registry.example.test/synthetic-recorder",
        tool_version="1.0.0",
        tool_schema_digest=RECORDER_SCHEMA_DIGEST,
        target_state_digest=TARGET_DIGEST,
        reversibility="reversible",
        risk_tier="low",
        allowed_target_prefixes=("resource:",),
        allowed_argument_keys=frozenset({"result"}),
        required_argument_keys=frozenset({"result"}),
        resource_budget=_budget(),
    )


def _action_draft(capability_id: str = "cap.record") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": ACTION_DRAFT_PROFILE,
        "kind": "ACTION",
        "capability_id": capability_id,
        "target": "resource:result/demo",
        "arguments": {"result": "synthetic-ok"},
    }


def _no_action() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": ACTION_DRAFT_PROFILE,
        "kind": "NO_ACTION",
        "reason_codes": ["INSUFFICIENT_EVIDENCE"],
    }


def build_fixture(
    *,
    observer_provider: str = "openai",
    analyst_provider: str = "anthropic",
    observer_mode: str = "return",
    analyst_output: dict[str, Any] | None = None,
    activation: bool = True,
    deny_invocation: bool = False,
    spawn: bool = False,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
    executor_raises: bool = False,
) -> dict[str, Any]:
    observer_identity = _identity("observer", observer_provider)
    analyst_identity = _identity("analyst", analyst_provider)
    capabilities = [_capability(spawn=spawn)]
    capability_ids = [item.capability_id for item in capabilities]
    manifest = _manifest(observer_identity, analyst_identity, capability_ids)
    if mutate_manifest is not None:
        raw = copy.deepcopy(manifest)
        raw.pop("manifest_digest")
        mutate_manifest(raw)
        manifest = bind_organism_manifest(raw)

    calls: list[tuple[str, ModelCall]] = []
    executed: list[dict[str, Any]] = []
    draft = copy.deepcopy(
        analyst_output
        if analyst_output is not None
        else _action_draft("cap.spawn" if spawn else "cap.record")
    )

    def observe(call: ModelCall) -> ModelResult:
        calls.append(("observer", call))
        if observer_mode == "raise":
            raise RuntimeError("synthetic observer failure")
        if observer_mode == "terminal":
            return ModelResult.terminal(
                observer_identity,
                ModelResultStatus.ERROR,
                "SYNTHETIC_PROVIDER_ERROR",
                provider_request_id="provider-observer-error",
            )
        if observer_mode == "over_budget":
            return ModelResult.returned(
                observer_identity,
                {"facts": ["synthetic fact"], "root_digest": call.payload_digest},
                provider_request_id="provider-observer-budget",
                input_tokens=200,
                output_tokens=1,
                cost_microunits=100,
            )
        return ModelResult.returned(
            observer_identity,
            {"facts": ["synthetic fact"], "root_digest": call.payload_digest},
            provider_request_id="provider-observer-ok",
            input_tokens=4,
            output_tokens=5,
            cost_microunits=100,
        )

    def analyse(call: ModelCall) -> ModelResult:
        calls.append(("analyst", call))
        return ModelResult.returned(
            analyst_identity,
            draft,
            provider_request_id="provider-analyst-ok",
            input_tokens=6,
            output_tokens=7,
            cost_microunits=200,
        )

    adapters = AdapterRegistry(
        [
            CallbackModelAdapter(observer_identity, observe),
            CallbackModelAdapter(analyst_identity, analyse),
        ]
    )
    organism_policy = OrganismPolicy(
        policy_version="organism-policy-v1",
        allowed_adapter_identity_digests=frozenset(
            {observer_identity.identity_digest, analyst_identity.identity_digest}
        ),
        allowed_capability_ids=frozenset(capability_ids),
        max_steps=3,
        max_seconds=30,
        max_model_calls=2,
        max_output_tokens_per_call=64,
        max_total_tokens=128,
        max_cost_microunits_per_call=100_000,
        max_total_cost_microunits=200_000,
        max_retries=0,
        max_fan_out=1,
    )
    activations = []
    if activation:
        activations.append(
            ManifestActivation(
                organism_id="organism:demo",
                manifest_digest=manifest["manifest_digest"],
                subject="user:alice",
                policy_version="organism-policy-v1",
                expires_at="2026-08-27T21:45:00Z",
            )
        )

    def execute(proposal: dict[str, Any]) -> dict[str, Any]:
        executed.append(copy.deepcopy(proposal))
        if executor_raises:
            raise RuntimeError("synthetic executor failure")
        return {"recorded": True, "target": proposal["target"]}

    temp = TemporaryDirectory()
    runner = OrganismRunner(
        manifest=manifest,
        organism_policy=organism_policy,
        invocation_policy=_invocation_policy(
            observer_identity,
            analyst_identity,
            deny_destinations=deny_invocation,
        ),
        action_policy=_action_policy(allow_spawn=spawn),
        activations=StaticActivationRegistry(activations),
        adapters=adapters,
        proposal_factory=StaticProposalFactory(
            invocation_policy_version="invocation-policy-v1",
            action_policy_version="action-policy-v1",
            capabilities=capabilities,
        ),
        executors={"synthetic-recorder": execute},
        evidence_root=temp.name,
        clock=lambda: NOW,
    )
    return {
        "runner": runner,
        "root": {"task": "record this synthetic result"},
        "context": RunContext(
            subject="user:alice",
            intent_id="intent:demo:001",
            parent_cause="user-request:demo:001",
            auth_context_digest=AUTH_DIGEST,
            issued_at="2026-08-27T20:00:00Z",
            expires_at="2026-08-27T22:00:00Z",
        ),
        "calls": calls,
        "executed": executed,
        "temp": temp,
        "manifest": manifest,
        "observer_identity": observer_identity,
        "analyst_identity": analyst_identity,
    }


class OrganismTests(unittest.TestCase):
    def test_mixed_provider_happy_path_is_guarded_end_to_end(self) -> None:
        fixture = build_fixture()
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.COMPLETED, run.status)
        self.assertEqual(["observer", "analyst"], [item[0] for item in fixture["calls"]])
        self.assertEqual(2, run.model_calls)
        self.assertEqual(2, len(run.cell_runs))
        self.assertEqual(1, len(fixture["executed"]))
        self.assertTrue(all(item.verification.valid for item in run.cell_runs))
        self.assertIsNotNone(run.action_run)
        assert run.action_run is not None
        self.assertEqual(DecisionStatus.ACCEPT, run.action_run.decision.status)
        self.assertTrue(run.action_run.verification.valid)

        proposal = fixture["executed"][0]
        self.assertEqual("agent:executor", proposal["agent"])
        self.assertEqual("synthetic.write", proposal["scope"])
        self.assertEqual("action-policy-v1", proposal["policy_version"])
        self.assertTrue(proposal["parent_cause"].startswith("model-result:"))
        self.assertEqual(
            proposal["parent_cause"],
            proposal["metadata"]["organism"]["upstream_result_id"],
        )

    def test_providers_can_be_swapped_without_runner_changes(self) -> None:
        fixture = build_fixture(
            observer_provider="anthropic",
            analyst_provider="openai",
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.COMPLETED, run.status)
        self.assertEqual(
            ["anthropic", "openai"],
            [item.provider for item in run.model_results],
        )

    def test_missing_manifest_activation_holds_before_any_model_call(self) -> None:
        fixture = build_fixture(activation=False)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.HOLD, run.status)
        self.assertEqual(("MANIFEST_ACTIVATION_REQUIRED",), run.decision_reasons)
        self.assertEqual([], fixture["calls"])
        self.assertEqual([], fixture["executed"])

    def test_model_adapter_is_not_called_when_cell_guard_denies_destination(self) -> None:
        fixture = build_fixture(deny_invocation=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertIn("DESTINATION_DENIED", run.decision_reasons)
        self.assertEqual([], fixture["calls"])
        self.assertEqual([], fixture["executed"])
        self.assertEqual("NOT_INVOKED", run.cell_runs[0].observation["status"])

    def test_adapter_exception_stops_downstream_cells(self) -> None:
        fixture = build_fixture(observer_mode="raise")
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.FAILED, run.status)
        self.assertEqual(("MODEL_ADAPTER_ERROR",), run.decision_reasons)
        self.assertEqual(["observer"], [item[0] for item in fixture["calls"]])
        self.assertEqual([], fixture["executed"])
        self.assertEqual("EXECUTOR_ERROR", run.cell_runs[0].observation["status"])

    def test_provider_terminal_error_stops_downstream_cells(self) -> None:
        fixture = build_fixture(observer_mode="terminal")
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.FAILED, run.status)
        self.assertEqual(("MODEL_ERROR",), run.decision_reasons)
        self.assertEqual(1, run.model_calls)
        self.assertEqual([], fixture["executed"])

    def test_no_action_is_terminal_and_does_not_invoke_action_executor(self) -> None:
        fixture = build_fixture(analyst_output=_no_action())
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.NO_ACTION, run.status)
        self.assertEqual(("INSUFFICIENT_EVIDENCE",), run.decision_reasons)
        self.assertEqual(2, run.model_calls)
        self.assertIsNone(run.action_run)
        self.assertEqual([], fixture["executed"])

    def test_model_cannot_smuggle_authority_fields_into_action(self) -> None:
        malicious = _action_draft()
        malicious["scope"] = "root.everything"
        malicious["policy_version"] = "model-chosen-policy"
        fixture = build_fixture(analyst_output=malicious)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertIn("ACTION_DRAFT_AUTHORITY_CONFLICT", run.decision_reasons)
        self.assertEqual([], fixture["executed"])
        self.assertIsNone(run.action_run)

    def test_unknown_capability_is_blocked_by_trusted_compiler(self) -> None:
        fixture = build_fixture(analyst_output=_action_draft("cap.root"))
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("CAPABILITY_UNKNOWN",), run.decision_reasons)
        self.assertEqual([], fixture["executed"])

    def test_reported_usage_over_aggregate_budget_blocks_downstream(self) -> None:
        fixture = build_fixture(observer_mode="over_budget")
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("BUDGET_USAGE_EXCEEDED",), run.decision_reasons)
        self.assertEqual(["observer"], [item[0] for item in fixture["calls"]])
        self.assertEqual([], fixture["executed"])

    def test_semantic_run_replay_is_atomic_and_fail_closed(self) -> None:
        fixture = build_fixture()
        first = fixture["runner"].run(fixture["root"], fixture["context"])
        call_count = len(fixture["calls"])
        execution_count = len(fixture["executed"])
        second = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.COMPLETED, first.status)
        self.assertEqual(OrganismStatus.BLOCK, second.status)
        self.assertEqual(("ORGANISM_REPLAYED",), second.decision_reasons)
        self.assertEqual(call_count, len(fixture["calls"]))
        self.assertEqual(execution_count, len(fixture["executed"]))
        self.assertEqual(0, second.model_calls)

    def test_nested_organism_spawn_is_hard_blocked(self) -> None:
        fixture = build_fixture(spawn=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("NESTED_SPAWN_DENIED",), run.decision_reasons)
        self.assertEqual([], fixture["executed"])
        self.assertIsNone(run.action_run)

    def test_executor_error_is_effect_uncertain_not_success(self) -> None:
        fixture = build_fixture(executor_raises=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.EFFECT_UNCERTAIN, run.status)
        self.assertEqual(1, len(fixture["executed"]))
        self.assertIsNotNone(run.action_run)
        assert run.action_run is not None
        self.assertEqual("EXECUTOR_ERROR", run.action_run.observation["status"])
        self.assertTrue(run.action_run.verification.valid)

    def test_adapter_provenance_mismatch_blocks_before_calls(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["pipeline"]["observer"]["adapter_identity_digest"] = (
                "sha256:" + "9" * 64
            )

        fixture = build_fixture(mutate_manifest=mutate)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertIn("ADAPTER_PROVENANCE_DENIED", run.decision_reasons)
        self.assertEqual([], fixture["calls"])

    def test_topology_change_is_not_an_organism_v01_manifest(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["pipeline"]["rogue"] = {
                "cell_id": "cell:rogue",
                "agent_id": "agent:rogue",
            }

        fixture = build_fixture(mutate_manifest=mutate)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertIn("MANIFEST_TOPOLOGY_INVALID", run.decision_reasons)
        self.assertEqual([], fixture["calls"])

    def test_action_draft_runtime_validator_is_strict(self) -> None:
        self.assertEqual((), validate_action_draft(_action_draft()))
        duplicate_no_action = _no_action()
        duplicate_no_action["reason_codes"].append("INSUFFICIENT_EVIDENCE")
        self.assertEqual(
            ("ACTION_DRAFT_INVALID",),
            validate_action_draft(duplicate_no_action),
        )
        self.assertEqual(
            ("ACTION_DRAFT_AUTHORITY_CONFLICT",),
            validate_action_draft({**_action_draft(), "nonce": "model-nonce"}),
        )

    def test_published_json_schemas_are_strict_and_parseable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        action_schema = json.loads(
            (root / "schemas" / "action-draft.v0.1.schema.json").read_text()
        )
        manifest_schema = json.loads(
            (root / "schemas" / "organism-manifest.v0.1.schema.json").read_text()
        )

        self.assertEqual(2, len(action_schema["oneOf"]))
        self.assertTrue(
            all(branch["additionalProperties"] is False for branch in action_schema["oneOf"])
        )
        self.assertEqual(
            "org.causalcell.organism-manifest.v0.1",
            manifest_schema["properties"]["profile"]["const"],
        )
        self.assertFalse(
            manifest_schema["properties"]["pipeline"]["additionalProperties"]
        )


if __name__ == "__main__":
    unittest.main()
