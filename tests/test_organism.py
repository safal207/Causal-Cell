from __future__ import annotations

import copy
import json
import re
import unittest
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from causal_cell import InMemoryNonceStore, digest_json
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
    InMemoryOrganismStore,
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
_TEMP_DIRECTORIES: list[TemporaryDirectory] = []


def _matches_json_schema(instance: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    """Validate the JSON Schema keywords used by the action-proposal contract."""

    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or not reference.startswith("#/"):
            return False
        resolved: Any = root
        for part in reference[2:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(resolved, dict) or key not in resolved:
                return False
            resolved = resolved[key]
        return isinstance(resolved, dict) and _matches_json_schema(
            instance,
            resolved,
            root,
        )
    if "oneOf" in schema:
        return sum(
            _matches_json_schema(instance, branch, root)
            for branch in schema["oneOf"]
        ) == 1
    if "anyOf" in schema:
        return any(
            _matches_json_schema(instance, branch, root)
            for branch in schema["anyOf"]
        )
    if "const" in schema and instance != schema["const"]:
        return False
    if "enum" in schema and instance not in schema["enum"]:
        return False

    expected_type = schema.get("type")
    if expected_type == "object" and type(instance) is not dict:
        return False
    if expected_type == "array" and type(instance) is not list:
        return False
    if expected_type == "string" and type(instance) is not str:
        return False
    if expected_type == "integer" and type(instance) is not int:
        return False
    if expected_type == "number" and type(instance) not in {int, float}:
        return False
    if expected_type == "boolean" and type(instance) is not bool:
        return False
    if expected_type == "null" and instance is not None:
        return False

    if type(instance) is dict:
        required = schema.get("required", [])
        if not set(required).issubset(instance):
            return False
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and not set(
            instance
        ).issubset(properties):
            return False
        if len(instance) < schema.get("minProperties", 0):
            return False
        if any(
            not _matches_json_schema(value, properties[key], root)
            for key, value in instance.items()
            if key in properties
        ):
            return False
    if type(instance) is list:
        item_schema = schema.get("items")
        if isinstance(item_schema, dict) and any(
            not _matches_json_schema(item, item_schema, root)
            for item in instance
        ):
            return False
    if type(instance) is str:
        if len(instance) < schema.get("minLength", 0):
            return False
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, instance) is None:
            return False
    if type(instance) in {int, float} and "minimum" in schema:
        if instance < schema["minimum"]:
            return False
    return True


def _budget(max_seconds: int = 30) -> dict[str, int | float]:
    return {
        "max_steps": 1,
        "max_seconds": max_seconds,
        "max_cost": 1.0,
        "max_fan_out": 0,
        "max_retries": 0,
    }


class MutatingIdentityAdapter:
    def __init__(
        self,
        identity: AdapterIdentity,
        callback: Callable[[ModelCall], ModelResult],
    ) -> None:
        self._identity = identity
        self._callback = callback

    @property
    def identity(self) -> AdapterIdentity:
        return self._identity

    def invoke(self, call: ModelCall) -> ModelResult:
        result = self._callback(call)
        self._identity = AdapterIdentity(
            adapter_id=self._identity.adapter_id,
            provider="untrusted-provider",
            model="changed-after-authorization",
            origin="registry.example.test/untrusted-adapter",
            version="9.9.9",
            schema_digest="sha256:" + "9" * 64,
            destination="https://untrusted.example.test",
        )
        return result


class FailingIdentityAfterInvokeAdapter:
    def __init__(
        self,
        identity: AdapterIdentity,
        callback: Callable[[ModelCall], ModelResult],
        *,
        allowed_post_invoke_reads: int = 1,
    ) -> None:
        self._identity = identity
        self._callback = callback
        self._invoked = False
        self._post_invoke_reads = 0
        self._allowed_post_invoke_reads = allowed_post_invoke_reads

    @property
    def identity(self) -> AdapterIdentity:
        if self._invoked:
            self._post_invoke_reads += 1
            if self._post_invoke_reads > self._allowed_post_invoke_reads:
                raise RuntimeError("synthetic identity lookup failure")
        return self._identity

    def invoke(self, call: ModelCall) -> ModelResult:
        result = self._callback(call)
        self._invoked = True
        return result


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
    *,
    analyst_budget_seconds: int = 30,
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
                    "resource_budget": _budget(analyst_budget_seconds),
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
    allow_secret_destinations: bool = False,
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
            []
            if deny_destinations
            else list(dict.fromkeys([observer.destination, analyst.destination]))
        ),
        "allowed_secret_destinations": (
            list(dict.fromkeys([observer.destination, analyst.destination]))
            if allow_secret_destinations and not deny_destinations
            else []
        ),
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


def _action_policy(
    *,
    allow_spawn: bool = False,
    network: bool = False,
    network_destination: str = "https://public-sink.example.test",
) -> dict[str, Any]:
    if network:
        actions = ["send_result"]
        scopes = ["network.egress"]
        mapping = {"send_result": ["network.egress"]}
        network_scopes = ["network.egress"]
        destinations = [network_destination]
    else:
        actions = ["record_result"]
        scopes = ["synthetic.write"]
        mapping = {"record_result": ["synthetic.write"]}
        network_scopes = []
        destinations = []
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
        "network_scopes": network_scopes,
        "allowed_destinations": destinations,
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


def _capability(
    *,
    spawn: bool = False,
    network: bool = False,
    target_prefixes: tuple[str, ...] = ("resource:",),
    network_destination: str = "https://public-sink.example.test",
) -> StaticCapability:
    if spawn:
        capability_id = "cap.spawn"
        action = "create_organism"
        scope = "organism.spawn"
    elif network:
        capability_id = "cap.egress"
        action = "send_result"
        scope = "network.egress"
    else:
        capability_id = "cap.record"
        action = "record_result"
        scope = "synthetic.write"
    return StaticCapability(
        capability_id=capability_id,
        action=action,
        scope=scope,
        executor_id="synthetic-recorder",
        tool_origin="registry.example.test/synthetic-recorder",
        tool_version="1.0.0",
        tool_schema_digest=RECORDER_SCHEMA_DIGEST,
        target_state_digest=TARGET_DIGEST,
        reversibility="reversible",
        risk_tier="low",
        allowed_target_prefixes=target_prefixes,
        target_validator=lambda target: (
            all(
                segment not in {".", ".."}
                for segment in target.split("/")
            )
            and "@" not in target.partition("://")[2].split("/", 1)[0]
        ),
        allowed_argument_keys=frozenset({"result"}),
        required_argument_keys=frozenset({"result"}),
        argument_validator=lambda arguments: (
            isinstance(arguments.get("result"), str)
            and 0 < len(arguments["result"]) <= 256
        ),
        resource_budget=_budget(),
        contains_secret=False,
        destination=(
            network_destination if network else None
        ),
        data_classification="public",
    )


def _action_draft(
    capability_id: str = "cap.record",
    *,
    target: str = "resource:result/demo",
    result: Any = "synthetic-ok",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": ACTION_DRAFT_PROFILE,
        "kind": "ACTION",
        "capability_id": capability_id,
        "target": target,
        "arguments": {"result": result},
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
    activation_expires_at: str = "2026-08-27T21:45:00Z",
    expired_activation_before_valid: bool = False,
    deny_invocation: bool = False,
    spawn: bool = False,
    network_capability: bool = False,
    target_prefixes: tuple[str, ...] = ("resource:",),
    analyst_budget_seconds: int = 30,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
    executor_raises: bool = False,
    mutate_observer_identity: bool = False,
    fail_identity_after_invoke: bool = False,
    fail_identity_on_first_post_read: bool = False,
    advance_during_observer_seconds: int | None = None,
    advance_during_analyst_seconds: int | None = None,
    sensitive_context: bool = False,
    approval_required_config: bool = False,
    fail_model_evidence: bool = False,
    fail_action_evidence: bool = False,
    observer_contains_secret: bool = False,
    observer_data_classification: str = "public",
    analyst_contains_secret: bool = False,
    analyst_data_classification: str = "public",
    nonce_store: InMemoryNonceStore | None = None,
    omit_factory_capability: bool = False,
    omit_executor: bool = False,
    network_destination: str = "https://public-sink.example.test",
    factory_action_policy_version: str = "action-policy-v1",
    action_policy_schema_version: Any = 1,
    deny_analyst_invocation_preflight: bool = False,
    omit_action_policy_capability: bool = False,
    organism_store: InMemoryOrganismStore | None = None,
    clock_now: datetime = NOW,
    context_issued_at: str = "2026-08-27T20:00:00Z",
    context_expires_at: str = "2026-08-27T22:00:00Z",
) -> dict[str, Any]:
    observer_identity = _identity("observer", observer_provider)
    analyst_identity = _identity("analyst", analyst_provider)
    capabilities = [
        _capability(
            spawn=spawn,
            network=network_capability,
            target_prefixes=target_prefixes,
            network_destination=network_destination,
        )
    ]
    capability_ids = [item.capability_id for item in capabilities]
    manifest = _manifest(
        observer_identity,
        analyst_identity,
        capability_ids,
        analyst_budget_seconds=analyst_budget_seconds,
    )
    if mutate_manifest is not None:
        raw = copy.deepcopy(manifest)
        raw.pop("manifest_digest")
        mutate_manifest(raw)
        manifest = bind_organism_manifest(raw)

    calls: list[tuple[str, ModelCall]] = []
    executed: list[dict[str, Any]] = []
    clock_state = {"now": clock_now}
    default_capability_id = (
        "cap.spawn"
        if spawn
        else ("cap.egress" if network_capability else "cap.record")
    )
    draft = copy.deepcopy(
        analyst_output
        if analyst_output is not None
        else _action_draft(default_capability_id)
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
                contains_secret=observer_contains_secret,
                data_classification=observer_data_classification,
            )
        output: dict[str, Any] = {
            "facts": ["synthetic fact"],
            "root_digest": call.payload_digest,
        }
        if observer_mode == "evil_usage_scalar":
            class EvilInt(int):
                def __lt__(self, _: object) -> bool:
                    raise RuntimeError("untrusted integer comparison")

            result = ModelResult(
                status=ModelResultStatus.RETURNED,
                output=output,
                output_digest=digest_json(output),
                provider_request_id="provider-observer-evil-usage",
                provider=observer_identity.provider,
                model=observer_identity.model,
                input_tokens=EvilInt(1),
                output_tokens=1,
                cost_microunits=100,
                contains_secret=observer_contains_secret,
                data_classification=observer_data_classification,
                error_code=None,
            )
        elif observer_mode == "evil_identity_scalar":
            class EqBomb(str):
                def __ne__(self, _: object) -> bool:
                    raise RuntimeError("untrusted string comparison")

            result = ModelResult(
                status=ModelResultStatus.RETURNED,
                output=output,
                output_digest=digest_json(output),
                provider_request_id="provider-observer-evil-identity",
                provider=EqBomb(observer_identity.provider),
                model=observer_identity.model,
                input_tokens=1,
                output_tokens=1,
                cost_microunits=100,
                contains_secret=observer_contains_secret,
                data_classification=observer_data_classification,
                error_code=None,
            )
        elif observer_mode == "oversized_metadata":
            result = ModelResult.returned(
                observer_identity,
                output,
                provider_request_id="x" * 1_000_001,
                input_tokens=1,
                output_tokens=1,
                cost_microunits=100,
                contains_secret=observer_contains_secret,
                data_classification=observer_data_classification,
            )
        elif observer_mode == "over_budget":
            result = ModelResult.returned(
                observer_identity,
                output,
                provider_request_id="provider-observer-budget",
                input_tokens=200,
                output_tokens=1,
                cost_microunits=100,
                contains_secret=observer_contains_secret,
                data_classification=observer_data_classification,
            )
        elif observer_mode == "output_over_budget":
            result = ModelResult.returned(
                observer_identity,
                output,
                provider_request_id="provider-observer-output-budget",
                input_tokens=4,
                output_tokens=65,
                cost_microunits=100,
                contains_secret=observer_contains_secret,
                data_classification=observer_data_classification,
            )
        elif observer_mode == "invalid_usage":
            result = ModelResult(
                status=ModelResultStatus.RETURNED,
                output=output,
                output_digest=digest_json(output),
                provider_request_id="provider-observer-invalid-usage",
                provider=observer_identity.provider,
                model=observer_identity.model,
                input_tokens="not-an-integer",  # type: ignore[arg-type]
                output_tokens=1,
                cost_microunits=100,
                contains_secret=observer_contains_secret,
                data_classification=observer_data_classification,
                error_code=None,
            )
        elif observer_mode == "mutable_output":
            class OutputSubclass(dict[str, Any]):
                pass

            unsafe_output = OutputSubclass(output)
            result = ModelResult(
                status=ModelResultStatus.RETURNED,
                output=unsafe_output,
                output_digest=digest_json(output),
                provider_request_id="provider-observer-mutable-output",
                provider=observer_identity.provider,
                model=observer_identity.model,
                input_tokens=4,
                output_tokens=5,
                cost_microunits=100,
                contains_secret=observer_contains_secret,
                data_classification=observer_data_classification,
                error_code=None,
            )
        else:
            result = ModelResult.returned(
                observer_identity,
                output,
                provider_request_id="provider-observer-ok",
                input_tokens=4,
                output_tokens=5,
                cost_microunits=100,
                contains_secret=observer_contains_secret,
                data_classification=observer_data_classification,
            )
        if advance_during_observer_seconds is not None:
            clock_state["now"] = clock_now + timedelta(
                seconds=advance_during_observer_seconds
            )
        return result

    def analyse(call: ModelCall) -> ModelResult:
        calls.append(("analyst", call))
        result = ModelResult.returned(
            analyst_identity,
            draft,
            provider_request_id="provider-analyst-ok",
            input_tokens=6,
            output_tokens=7,
            cost_microunits=200,
            contains_secret=analyst_contains_secret,
            data_classification=analyst_data_classification,
        )
        if advance_during_analyst_seconds is not None:
            clock_state["now"] = clock_now + timedelta(
                seconds=advance_during_analyst_seconds
            )
        return result

    if mutate_observer_identity:
        observer_adapter = MutatingIdentityAdapter(observer_identity, observe)
    elif fail_identity_after_invoke or fail_identity_on_first_post_read:
        observer_adapter = FailingIdentityAfterInvokeAdapter(
            observer_identity,
            observe,
            allowed_post_invoke_reads=(
                0 if fail_identity_on_first_post_read else 1
            ),
        )
    else:
        observer_adapter = CallbackModelAdapter(observer_identity, observe)
    adapters = AdapterRegistry(
        [
            observer_adapter,
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
    activations: list[ManifestActivation] = []
    if activation:
        if expired_activation_before_valid:
            activations.append(
                ManifestActivation(
                    organism_id="organism:demo",
                    manifest_digest=manifest["manifest_digest"],
                    subject="user:alice",
                    policy_version="organism-policy-v1",
                    expires_at="2026-08-27T20:30:00Z",
                )
            )
        activations.append(
            ManifestActivation(
                organism_id="organism:demo",
                manifest_digest=manifest["manifest_digest"],
                subject="user:alice",
                policy_version="organism-policy-v1",
                expires_at=activation_expires_at,
            )
        )

    temp = TemporaryDirectory()
    _TEMP_DIRECTORIES.append(temp)
    evidence_root = Path(temp.name) / "evidence"
    if fail_model_evidence:
        evidence_root.write_text("synthetic blocked evidence root", encoding="utf-8")

    def execute(proposal: dict[str, Any]) -> dict[str, Any]:
        executed.append(copy.deepcopy(proposal))
        if fail_action_evidence:
            saved = evidence_root.with_name("saved-evidence")
            evidence_root.rename(saved)
            evidence_root.write_text(
                "synthetic blocked evidence root",
                encoding="utf-8",
            )
        if executor_raises:
            raise RuntimeError("synthetic executor failure")
        return {"recorded": True, "target": proposal["target"]}

    invocation_policy = _invocation_policy(
        observer_identity,
        analyst_identity,
        deny_destinations=deny_invocation,
        allow_secret_destinations=sensitive_context,
    )
    if deny_analyst_invocation_preflight:
        invocation_policy["allowed_destinations"] = [
            observer_identity.destination
        ]
    action_policy = _action_policy(
        allow_spawn=spawn,
        network=network_capability,
        network_destination=network_destination,
    )
    action_policy["schema_version"] = action_policy_schema_version
    if omit_action_policy_capability:
        action_policy["allowed_actions"] = ["unrelated_action"]
        action_policy["allowed_action_scopes"] = {
            "unrelated_action": ["synthetic.write"]
        }
    if approval_required_config:
        action_policy["approval_required_risk_tiers"] = ["low"]

    runner = OrganismRunner(
        manifest=manifest,
        organism_policy=organism_policy,
        invocation_policy=invocation_policy,
        action_policy=action_policy,
        activations=StaticActivationRegistry(activations),
        adapters=adapters,
        proposal_factory=StaticProposalFactory(
            invocation_policy_version="invocation-policy-v1",
            action_policy_version=factory_action_policy_version,
            capabilities=[] if omit_factory_capability else capabilities,
        ),
        executors={} if omit_executor else {"synthetic-recorder": execute},
        evidence_root=evidence_root,
        clock=lambda: clock_state["now"],
        nonce_store=nonce_store,
        organism_store=organism_store,
    )
    return {
        "runner": runner,
        "root": {"task": "record this synthetic result"},
        "context": RunContext(
            subject="user:alice",
            intent_id="intent:demo:001",
            parent_cause="user-request:demo:001",
            auth_context_digest=AUTH_DIGEST,
            issued_at=context_issued_at,
            expires_at=context_expires_at,
            contains_secret=sensitive_context,
            data_classification=(
                "confidential" if sensitive_context else "public"
            ),
        ),
        "calls": calls,
        "executed": executed,
        "temp": temp,
        "evidence_root": evidence_root,
        "manifest": manifest,
        "observer_identity": observer_identity,
        "analyst_identity": analyst_identity,
    }


class OrganismTests(unittest.TestCase):
    def tearDown(self) -> None:
        while _TEMP_DIRECTORIES:
            _TEMP_DIRECTORIES.pop().cleanup()

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

    def test_emitted_proposals_match_the_published_json_schema(self) -> None:
        fixture = build_fixture()
        run = fixture["runner"].run(fixture["root"], fixture["context"])
        self.assertEqual(OrganismStatus.COMPLETED, run.status)
        self.assertIsNotNone(run.action_run)
        assert run.action_run is not None
        schema = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "schemas"
                / "action-proposal.v0.1.schema.json"
            ).read_text(encoding="utf-8")
        )
        bundle_paths = [
            *(cell.bundle_path for cell in run.cell_runs),
            run.action_run.bundle_path,
        ]
        for bundle_path in bundle_paths:
            with self.subTest(bundle_path=bundle_path):
                proposal = json.loads(
                    (bundle_path / "proposal.json").read_text(encoding="utf-8")
                )
                self.assertTrue(
                    _matches_json_schema(proposal, schema, schema),
                    f"proposal does not match published schema: {proposal!r}",
                )

    def test_organism_policy_allowlists_are_exact_and_validated(self) -> None:
        valid_digest = "sha256:" + "1" * 64
        invalid_cases: list[dict[str, Any]] = [
            {
                "allowed_adapter_identity_digests": None,
                "allowed_capability_ids": frozenset({"cap.record"}),
            },
            {
                "allowed_adapter_identity_digests": [valid_digest],
                "allowed_capability_ids": frozenset({"cap.record"}),
            },
            {
                "allowed_adapter_identity_digests": frozenset({"not-a-digest"}),
                "allowed_capability_ids": frozenset({"cap.record"}),
            },
            {
                "allowed_adapter_identity_digests": frozenset({valid_digest}),
                "allowed_capability_ids": ["cap.record"],
            },
            {
                "allowed_adapter_identity_digests": frozenset({valid_digest}),
                "allowed_capability_ids": frozenset({""}),
            },
        ]
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "invalid trusted organism policy",
                ):
                    OrganismPolicy(
                        policy_version="organism-policy-v1",
                        **invalid,  # type: ignore[arg-type]
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

    def test_adapter_identity_authority_fields_are_validated_at_setup(self) -> None:
        identity = _identity("observer", "openai")
        for changes in (
            {"provider": []},
            {"origin": "\ud800"},
            {"schema_digest": "not-a-digest"},
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "invalid adapter identity"):
                    replace(identity, **changes)

    def test_missing_manifest_activation_holds_before_any_model_call(self) -> None:
        fixture = build_fixture(activation=False)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.HOLD, run.status)
        self.assertEqual(("MANIFEST_ACTIVATION_REQUIRED",), run.decision_reasons)
        self.assertEqual([], fixture["calls"])
        self.assertEqual([], fixture["executed"])

    def test_model_adapter_is_not_called_when_policy_denies_destination(self) -> None:
        fixture = build_fixture(deny_invocation=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(
            ("INVOCATION_POLICY_INCOMPATIBLE",),
            run.decision_reasons,
        )
        self.assertEqual([], fixture["calls"])
        self.assertEqual([], fixture["executed"])
        self.assertEqual((), run.cell_runs)

    def test_adapter_exception_is_effect_uncertain_and_stops_downstream(self) -> None:
        fixture = build_fixture(observer_mode="raise")
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.EFFECT_UNCERTAIN, run.status)
        self.assertEqual(("MODEL_EFFECT_UNCERTAIN",), run.decision_reasons)
        self.assertEqual(["observer"], [item[0] for item in fixture["calls"]])
        self.assertEqual(1, run.model_calls)
        self.assertEqual([], fixture["executed"])
        self.assertEqual("EXECUTOR_ERROR", run.cell_runs[0].observation["status"])

    def test_adapter_identity_change_after_call_is_effect_uncertain(self) -> None:
        fixture = build_fixture(mutate_observer_identity=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.EFFECT_UNCERTAIN, run.status)
        self.assertEqual(
            ("ADAPTER_PROVENANCE_CHANGED_EFFECT_UNCERTAIN",),
            run.decision_reasons,
        )
        self.assertEqual(1, run.model_calls)
        self.assertEqual(["observer"], [item[0] for item in fixture["calls"]])
        self.assertEqual([], fixture["executed"])

    def test_analyst_return_after_deadline_cannot_trigger_action(self) -> None:
        fixture = build_fixture(advance_during_analyst_seconds=31)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("ORGANISM_DEADLINE_EXCEEDED",), run.decision_reasons)
        self.assertEqual(2, run.model_calls)
        self.assertEqual([], fixture["executed"])
        self.assertIsNone(run.action_run)

    def test_activation_rotation_accepts_any_valid_exact_record(self) -> None:
        fixture = build_fixture(expired_activation_before_valid=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.COMPLETED, run.status)

    def test_activation_expiry_mid_run_stops_downstream(self) -> None:
        fixture = build_fixture(
            activation_expires_at="2026-08-27T21:00:20Z",
            advance_during_observer_seconds=21,
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(
            ("MANIFEST_ACTIVATION_EXPIRED",),
            run.decision_reasons,
        )
        self.assertEqual(["observer"], [item[0] for item in fixture["calls"]])
        self.assertEqual([], fixture["executed"])

    def test_per_cell_deadline_blocks_late_analyst_result(self) -> None:
        fixture = build_fixture(
            analyst_budget_seconds=1,
            advance_during_analyst_seconds=2,
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(
            ("MODEL_CALL_DEADLINE_EXCEEDED",),
            run.decision_reasons,
        )
        self.assertEqual(2, run.model_calls)
        self.assertEqual([], fixture["executed"])

    def test_provider_terminal_error_stops_downstream_cells(self) -> None:
        fixture = build_fixture(observer_mode="terminal")
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.FAILED, run.status)
        self.assertEqual(("MODEL_ERROR",), run.decision_reasons)
        self.assertEqual(1, run.model_calls)
        self.assertEqual([], fixture["executed"])

    def test_post_call_identity_lookup_failure_is_effect_uncertain(self) -> None:
        fixture = build_fixture(fail_identity_after_invoke=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.EFFECT_UNCERTAIN, run.status)
        self.assertEqual(
            ("ADAPTER_PROVENANCE_CHANGED_EFFECT_UNCERTAIN",),
            run.decision_reasons,
        )
        self.assertEqual(1, run.model_calls)
        self.assertEqual(9, run.reported_tokens)
        self.assertEqual(100, run.reported_cost_microunits)
        self.assertEqual([], fixture["executed"])

    def test_first_post_call_identity_failure_is_provenance_specific(self) -> None:
        fixture = build_fixture(fail_identity_on_first_post_read=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.EFFECT_UNCERTAIN, run.status)
        self.assertEqual(
            ("ADAPTER_PROVENANCE_CHANGED_EFFECT_UNCERTAIN",),
            run.decision_reasons,
        )
        self.assertEqual(1, run.model_calls)
        self.assertEqual(9, run.reported_tokens)
        self.assertEqual(100, run.reported_cost_microunits)
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
        self.assertEqual(201, run.reported_tokens)
        self.assertEqual(100, run.reported_cost_microunits)
        self.assertEqual([], fixture["executed"])

    def test_rejected_per_call_usage_is_still_reported(self) -> None:
        fixture = build_fixture(observer_mode="output_over_budget")
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(
            ("MODEL_OUTPUT_BUDGET_EXCEEDED",),
            run.decision_reasons,
        )
        self.assertEqual(1, run.model_calls)
        self.assertEqual(69, run.reported_tokens)
        self.assertEqual(100, run.reported_cost_microunits)
        self.assertEqual([], fixture["executed"])

    def test_malformed_model_usage_blocks_without_runtime_exception(self) -> None:
        fixture = build_fixture(observer_mode="invalid_usage")
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("MODEL_USAGE_INVALID",), run.decision_reasons)
        self.assertEqual(1, run.model_calls)
        self.assertEqual([], fixture["executed"])

    def test_hostile_model_result_scalar_subclasses_fail_closed(self) -> None:
        expected = {
            "evil_usage_scalar": "MODEL_USAGE_INVALID",
            "evil_identity_scalar": "MODEL_RESULT_INVALID",
            "oversized_metadata": "MODEL_RESULT_INVALID",
        }
        for mode, reason in expected.items():
            with self.subTest(mode=mode):
                fixture = build_fixture(observer_mode=mode)
                run = fixture["runner"].run(
                    fixture["root"],
                    fixture["context"],
                )

                self.assertEqual(OrganismStatus.BLOCK, run.status)
                self.assertIn(reason, run.decision_reasons)
                self.assertEqual(1, run.model_calls)
                self.assertEqual([], fixture["executed"])

    def test_model_output_must_be_a_detached_plain_json_snapshot(self) -> None:
        fixture = build_fixture(observer_mode="mutable_output")
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("MODEL_OUTPUT_INVALID",), run.decision_reasons)
        self.assertEqual(1, run.model_calls)
        self.assertEqual(9, run.reported_tokens)
        self.assertEqual(100, run.reported_cost_microunits)
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

    def test_same_semantic_effect_from_changed_root_executes_only_once(self) -> None:
        fixture = build_fixture()
        first = fixture["runner"].run(fixture["root"], fixture["context"])
        changed_root = {
            **fixture["root"],
            "irrelevant_untrusted_noise": "different",
        }
        second = fixture["runner"].run(changed_root, fixture["context"])

        self.assertEqual(OrganismStatus.COMPLETED, first.status)
        self.assertEqual(OrganismStatus.BLOCK, second.status)
        self.assertIsNotNone(second.action_run)
        assert second.action_run is not None
        self.assertIn("IDEMPOTENCY_REPLAYED", second.action_run.decision.reasons)
        self.assertEqual(2, second.model_calls)
        self.assertEqual(1, len(fixture["executed"]))

    def test_same_effect_across_manifest_revision_executes_only_once(self) -> None:
        nonces = InMemoryNonceStore()
        first_fixture = build_fixture(
            analyst_provider="anthropic",
            nonce_store=nonces,
        )
        second_fixture = build_fixture(
            analyst_provider="openai",
            nonce_store=nonces,
        )

        first = first_fixture["runner"].run(
            first_fixture["root"],
            first_fixture["context"],
        )
        second = second_fixture["runner"].run(
            second_fixture["root"],
            second_fixture["context"],
        )

        self.assertEqual(OrganismStatus.COMPLETED, first.status)
        self.assertEqual(OrganismStatus.BLOCK, second.status)
        self.assertIsNotNone(second.action_run)
        assert second.action_run is not None
        self.assertIn("IDEMPOTENCY_REPLAYED", second.action_run.decision.reasons)
        self.assertEqual(1, len(first_fixture["executed"]))
        self.assertEqual([], second_fixture["executed"])

    def test_https_aliases_share_one_semantic_effect_key(self) -> None:
        nonces = InMemoryNonceStore()
        first_fixture = build_fixture(
            network_capability=True,
            network_destination="https://PUBLIC-SINK.EXAMPLE.TEST:443",
            target_prefixes=("https://PUBLIC-SINK.EXAMPLE.TEST:443",),
            analyst_output=_action_draft(
                "cap.egress",
                target="https://PUBLIC-SINK.EXAMPLE.TEST:443/upload",
            ),
            nonce_store=nonces,
        )
        second_fixture = build_fixture(
            network_capability=True,
            network_destination="https://public-sink.example.test",
            target_prefixes=("https://public-sink.example.test",),
            analyst_output=_action_draft(
                "cap.egress",
                target="https://public-sink.example.test/upload",
            ),
            nonce_store=nonces,
        )

        first = first_fixture["runner"].run(
            first_fixture["root"], first_fixture["context"]
        )
        second = second_fixture["runner"].run(
            second_fixture["root"], second_fixture["context"]
        )

        self.assertEqual(OrganismStatus.COMPLETED, first.status)
        self.assertEqual(OrganismStatus.BLOCK, second.status)
        self.assertIsNotNone(second.action_run)
        assert second.action_run is not None
        self.assertIn("IDEMPOTENCY_REPLAYED", second.action_run.decision.reasons)
        self.assertEqual(1, len(first_fixture["executed"]))
        self.assertEqual([], second_fixture["executed"])

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
        self.assertEqual(("ACTION_EFFECT_UNCERTAIN",), run.decision_reasons)
        self.assertEqual(1, len(fixture["executed"]))
        self.assertTrue(run.executor_invoked)
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

    def test_secret_taint_cannot_be_dropped_before_public_egress(self) -> None:
        fixture = build_fixture(
            network_capability=True,
            sensitive_context=True,
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(
            ("ACTION_POLICY_INCOMPATIBLE",),
            run.decision_reasons,
        )
        self.assertEqual([], fixture["calls"])
        self.assertEqual([], fixture["executed"])
        self.assertIsNone(run.action_run)

    def test_observer_output_taint_blocks_unapproved_analyst_egress(self) -> None:
        fixture = build_fixture(observer_contains_secret=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertIn("SECRET_DESTINATION_DENIED", run.decision_reasons)
        self.assertEqual(["observer"], [item[0] for item in fixture["calls"]])
        self.assertEqual([], fixture["executed"])

    def test_analyst_output_taint_blocks_public_final_egress(self) -> None:
        fixture = build_fixture(
            network_capability=True,
            analyst_contains_secret=True,
            analyst_data_classification="confidential",
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertIn("SECRET_DESTINATION_DENIED", run.decision_reasons)
        self.assertEqual(
            ["observer", "analyst"],
            [item[0] for item in fixture["calls"]],
        )
        self.assertEqual([], fixture["executed"])

    def test_unknown_context_classification_blocks_first_model_egress(self) -> None:
        fixture = build_fixture()
        context = replace(
            fixture["context"],
            contains_secret=False,
            data_classification="unknown",
        )
        run = fixture["runner"].run(fixture["root"], context)

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(
            ("INVOCATION_POLICY_INCOMPATIBLE",),
            run.decision_reasons,
        )
        self.assertEqual([], fixture["calls"])

    def test_target_prefix_requires_a_segment_boundary(self) -> None:
        fixture = build_fixture(
            target_prefixes=("resource:tenant-a",),
            analyst_output=_action_draft(
                target="resource:tenant-attacker/secret",
            ),
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("TARGET_DENIED",), run.decision_reasons)
        self.assertEqual([], fixture["executed"])

    def test_target_prefix_accepts_a_real_descendant(self) -> None:
        fixture = build_fixture(
            target_prefixes=("resource:tenant-a",),
            analyst_output=_action_draft(
                target="resource:tenant-a/child",
            ),
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.COMPLETED, run.status)

    def test_target_validator_rejects_normalization_traversal(self) -> None:
        fixture = build_fixture(
            target_prefixes=("resource:tenant-a",),
            analyst_output=_action_draft(
                target="resource:tenant-a/../tenant-attacker",
            ),
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("TARGET_DENIED",), run.decision_reasons)
        self.assertEqual([], fixture["executed"])

    def test_url_prefix_does_not_accept_userinfo_host_confusion(self) -> None:
        fixture = build_fixture(
            target_prefixes=("https://trusted.example",),
            analyst_output=_action_draft(
                target="https://trusted.example:443@evil.example/x",
            ),
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("TARGET_DENIED",), run.decision_reasons)
        self.assertEqual([], fixture["executed"])

    def test_network_url_target_must_match_static_destination(self) -> None:
        fixture = build_fixture(
            network_capability=True,
            target_prefixes=("https://",),
            analyst_output=_action_draft(
                "cap.egress",
                target="https://unapproved.example.test/exfil",
            ),
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(
            ("TARGET_DESTINATION_MISMATCH",),
            run.decision_reasons,
        )
        self.assertEqual([], fixture["executed"])

    def test_same_origin_network_url_target_is_canonical_and_accepted(self) -> None:
        fixture = build_fixture(
            network_capability=True,
            network_destination="https://PUBLIC-SINK.EXAMPLE.TEST:443",
            target_prefixes=("https://PUBLIC-SINK.EXAMPLE.TEST:443",),
            analyst_output=_action_draft(
                "cap.egress",
                target="https://PUBLIC-SINK.EXAMPLE.TEST:443/upload",
            ),
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.COMPLETED, run.status)
        proposal = fixture["executed"][0]
        self.assertEqual(
            "https://public-sink.example.test",
            proposal["destination"],
        )
        self.assertEqual(
            "https://public-sink.example.test/upload",
            proposal["target"],
        )

    def test_trusted_argument_validator_rejects_wrong_value_shape(self) -> None:
        fixture = build_fixture(
            analyst_output=_action_draft(result=["model-shaped", "list"]),
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("ARGUMENTS_DENIED",), run.decision_reasons)
        self.assertEqual([], fixture["executed"])

    def test_approval_required_policy_is_rejected_before_calls(self) -> None:
        with self.assertRaisesRegex(ValueError, "excludes approval-required"):
            build_fixture(approval_required_config=True)

    def test_irreversible_capability_is_excluded_from_v01(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "invalid or approval-requiring capability",
        ):
            StaticProposalFactory(
                invocation_policy_version="invocation-policy-v1",
                action_policy_version="action-policy-v1",
                capabilities=[
                    replace(_capability(), reversibility="irreversible")
                ],
            )

    def test_capability_collection_strings_cannot_expand_to_characters(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid capability collections"):
            StaticProposalFactory(
                invocation_policy_version="invocation-policy-v1",
                action_policy_version="action-policy-v1",
                capabilities=[
                    replace(
                        _capability(),
                        allowed_target_prefixes="resource:",  # type: ignore[arg-type]
                    )
                ],
            )

    def test_factory_authority_scalars_fail_closed_at_setup(self) -> None:
        invalid_capabilities = [
            replace(_capability(), capability_id=[]),  # type: ignore[arg-type]
            replace(_capability(), action={"bad"}),  # type: ignore[arg-type]
            replace(_capability(), risk_tier=[]),  # type: ignore[arg-type]
            replace(_capability(), data_classification={}),  # type: ignore[arg-type]
            replace(_capability(), tool_origin="\ud800"),
        ]
        for capability in invalid_capabilities:
            with self.subTest(capability=capability):
                with self.assertRaisesRegex(
                    ValueError,
                    "invalid capability authority fields",
                ):
                    StaticProposalFactory(
                        invocation_policy_version="invocation-policy-v1",
                        action_policy_version="action-policy-v1",
                        capabilities=[capability],
                    )

        for invalid_version in ([], {"bad"}):
            with self.subTest(policy_version=invalid_version):
                with self.assertRaisesRegex(
                    ValueError,
                    "invalid proposal-factory policy version",
                ):
                    StaticProposalFactory(
                        invocation_policy_version=invalid_version,  # type: ignore[arg-type]
                        action_policy_version="action-policy-v1",
                        capabilities=[],
                    )

    def test_network_destination_is_a_canonicalizable_https_origin(self) -> None:
        for destination in (
            "http://public-sink.example.test",
            "https://public-sink.example.test/upload",
            "https://public-sink.example.test?route=upload",
            "https://.",
        ):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(
                    ValueError,
                    "invalid capability destination",
                ):
                    StaticProposalFactory(
                        invocation_policy_version="invocation-policy-v1",
                        action_policy_version="action-policy-v1",
                        capabilities=[
                            replace(
                                _capability(network=True),
                                destination=destination,
                            )
                        ],
                    )

    def test_capability_and_executor_registries_block_before_model_calls(self) -> None:
        for fixture, reason in (
            (build_fixture(omit_factory_capability=True), "CAPABILITY_UNKNOWN"),
            (build_fixture(omit_executor=True), "EXECUTOR_UNKNOWN"),
        ):
            with self.subTest(reason=reason):
                run = fixture["runner"].run(
                    fixture["root"],
                    fixture["context"],
                )
                self.assertEqual(OrganismStatus.BLOCK, run.status)
                self.assertEqual((reason,), run.decision_reasons)
                self.assertEqual(0, run.model_calls)
                self.assertEqual([], fixture["calls"])
                self.assertEqual([], fixture["executed"])

    def test_factory_policy_mismatch_blocks_before_model_calls(self) -> None:
        fixture = build_fixture(
            factory_action_policy_version="different-action-policy",
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(
            ("PROPOSAL_FACTORY_POLICY_MISMATCH",),
            run.decision_reasons,
        )
        self.assertEqual(0, run.model_calls)
        self.assertEqual([], fixture["calls"])
        self.assertEqual([], fixture["executed"])

    def test_static_policy_incompatibility_blocks_before_model_calls(self) -> None:
        for fixture, reason in (
            (
                build_fixture(deny_analyst_invocation_preflight=True),
                "INVOCATION_POLICY_INCOMPATIBLE",
            ),
            (
                build_fixture(omit_action_policy_capability=True),
                "ACTION_POLICY_INCOMPATIBLE",
            ),
        ):
            with self.subTest(reason=reason):
                run = fixture["runner"].run(
                    fixture["root"],
                    fixture["context"],
                )
                self.assertEqual(OrganismStatus.BLOCK, run.status)
                self.assertEqual((reason,), run.decision_reasons)
                self.assertEqual(0, run.model_calls)
                self.assertEqual([], fixture["calls"])
                self.assertEqual([], fixture["executed"])

    def test_static_policy_failure_does_not_consume_semantic_run(self) -> None:
        organisms = InMemoryOrganismStore()
        incompatible = build_fixture(
            omit_action_policy_capability=True,
            organism_store=organisms,
        )
        corrected = build_fixture(organism_store=organisms)

        first = incompatible["runner"].run(
            incompatible["root"],
            incompatible["context"],
        )
        second = corrected["runner"].run(
            corrected["root"],
            corrected["context"],
        )

        self.assertEqual(OrganismStatus.BLOCK, first.status)
        self.assertEqual(0, first.model_calls)
        self.assertEqual(OrganismStatus.COMPLETED, second.status)
        self.assertEqual(2, second.model_calls)

    def test_invalid_action_policy_is_rejected_at_setup(self) -> None:
        for schema_version in (2, True):
            with self.subTest(schema_version=schema_version):
                with self.assertRaisesRegex(
                    ValueError,
                    "invalid organism policy document",
                ):
                    build_fixture(
                        action_policy_schema_version=schema_version,
                    )

    def test_activation_registry_rejects_malformed_entries_at_setup(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid manifest activation"):
            StaticActivationRegistry([object()])  # type: ignore[list-item]
        with self.assertRaisesRegex(ValueError, "invalid manifest activation"):
            StaticActivationRegistry(
                [
                    ManifestActivation(
                        organism_id="organism:demo",
                        manifest_digest="sha256:" + "1" * 64,
                        subject="user:alice",
                        policy_version="organism-policy-v1",
                        expires_at="9999-12-31T23:59:59-23:59",
                    )
                ]
            )

    def test_model_evidence_failure_after_invoke_is_effect_uncertain(self) -> None:
        fixture = build_fixture(fail_model_evidence=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.EFFECT_UNCERTAIN, run.status)
        self.assertEqual(
            ("MODEL_CELL_RUNTIME_ERROR_EFFECT_UNCERTAIN",),
            run.decision_reasons,
        )
        self.assertEqual(["observer"], [item[0] for item in fixture["calls"]])
        self.assertEqual(1, run.model_calls)
        self.assertEqual(9, run.reported_tokens)
        self.assertEqual(100, run.reported_cost_microunits)
        self.assertEqual((), run.cell_runs)
        self.assertEqual([], fixture["executed"])

    def test_action_evidence_failure_after_executor_is_effect_uncertain(self) -> None:
        fixture = build_fixture(fail_action_evidence=True)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.EFFECT_UNCERTAIN, run.status)
        self.assertEqual(
            ("ACTION_CELL_RUNTIME_ERROR_EFFECT_UNCERTAIN",),
            run.decision_reasons,
        )
        self.assertEqual(1, len(fixture["executed"]))
        self.assertTrue(run.executor_invoked)
        self.assertTrue(run.action_effect_boundary_started)
        self.assertIsNone(run.action_run)

    def test_duplicate_manifest_capability_ids_are_blocked_preflight(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            capabilities = manifest["pipeline"]["executor"][
                "allowed_capability_ids"
            ]
            capabilities.append(capabilities[0])

        fixture = build_fixture(mutate_manifest=mutate)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertIn("CAPABILITY_DENIED", run.decision_reasons)
        self.assertEqual([], fixture["calls"])
        self.assertEqual([], fixture["executed"])

    def test_wrong_context_type_blocks_without_runtime_exception(self) -> None:
        fixture = build_fixture()
        run = fixture["runner"].run(
            fixture["root"],
            None,  # type: ignore[arg-type]
        )

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("CONTEXT_INVALID",), run.decision_reasons)
        self.assertEqual([], fixture["calls"])

    def test_root_mapping_subclass_is_rejected_before_calls(self) -> None:
        class RootSubclass(dict[str, Any]):
            pass

        fixture = build_fixture()
        run = fixture["runner"].run(
            RootSubclass(fixture["root"]),
            fixture["context"],
        )

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("CONTEXT_INVALID",), run.decision_reasons)
        self.assertEqual([], fixture["calls"])

    def test_deep_root_is_bounded_and_blocks_before_calls(self) -> None:
        deep_root: dict[str, Any] = {}
        for _ in range(129):
            deep_root = {"nested": deep_root}

        fixture = build_fixture()
        run = fixture["runner"].run(deep_root, fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("CONTEXT_INVALID",), run.decision_reasons)
        self.assertEqual([], fixture["calls"])

    def test_oversized_root_string_is_bounded_before_calls(self) -> None:
        fixture = build_fixture()
        run = fixture["runner"].run(
            {"oversized": "x" * 1_000_001},
            fixture["context"],
        )

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("CONTEXT_INVALID",), run.decision_reasons)
        self.assertEqual([], fixture["calls"])

    def test_action_draft_runtime_validator_is_strict(self) -> None:
        self.assertEqual((), validate_action_draft(_action_draft()))
        boolean_version = _action_draft()
        boolean_version["schema_version"] = True
        self.assertEqual(
            ("ACTION_DRAFT_INVALID",),
            validate_action_draft(boolean_version),
        )
        tuple_argument = _action_draft(result=("not", "json"))
        self.assertEqual(
            ("ACTION_DRAFT_INVALID",),
            validate_action_draft(tuple_argument),
        )
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

    def test_boolean_manifest_schema_version_is_blocked_preflight(self) -> None:
        def mutate(manifest: dict[str, Any]) -> None:
            manifest["schema_version"] = True

        fixture = build_fixture(mutate_manifest=mutate)
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertIn("MANIFEST_MALFORMED", run.decision_reasons)
        self.assertEqual([], fixture["calls"])

    def test_manifest_identity_scalars_block_without_unhashable_crash(self) -> None:
        invalid_values: tuple[Any, ...] = ([], {})
        for value in invalid_values:
            with self.subTest(value=value):
                def mutate(manifest: dict[str, Any], invalid: Any = value) -> None:
                    manifest["pipeline"]["observer"]["cell_id"] = invalid

                fixture = build_fixture(mutate_manifest=mutate)
                run = fixture["runner"].run(
                    fixture["root"],
                    fixture["context"],
                )
                self.assertEqual(OrganismStatus.BLOCK, run.status)
                self.assertIn("MANIFEST_CELL_INVALID", run.decision_reasons)
                self.assertEqual([], fixture["calls"])

    def test_context_classification_blocks_without_unhashable_crash(self) -> None:
        for value in ([], {}):
            with self.subTest(value=value):
                fixture = build_fixture()
                context = replace(
                    fixture["context"],
                    data_classification=value,  # type: ignore[arg-type]
                )
                run = fixture["runner"].run(fixture["root"], context)
                self.assertEqual(OrganismStatus.BLOCK, run.status)
                self.assertEqual(("CONTEXT_INVALID",), run.decision_reasons)
                self.assertEqual([], fixture["calls"])

    def test_context_timestamp_overflow_blocks_before_calls(self) -> None:
        for field, value in (
            ("issued_at", "0001-01-01T00:00:00+23:59"),
            ("expires_at", "9999-12-31T23:59:59-23:59"),
        ):
            with self.subTest(field=field):
                fixture = build_fixture()
                context = replace(fixture["context"], **{field: value})
                run = fixture["runner"].run(fixture["root"], context)
                self.assertEqual(OrganismStatus.BLOCK, run.status)
                self.assertEqual(("CONTEXT_INVALID",), run.decision_reasons)
                self.assertEqual([], fixture["calls"])

    def test_valid_near_max_timestamps_saturate_deadline_arithmetic(self) -> None:
        fixture = build_fixture(
            clock_now=datetime(9999, 12, 31, 23, 59, 40, tzinfo=UTC),
            context_issued_at="9999-12-31T23:59:30Z",
            context_expires_at="9999-12-31T23:59:59Z",
            activation_expires_at="9999-12-31T23:59:59Z",
        )
        run = fixture["runner"].run(fixture["root"], fixture["context"])

        self.assertEqual(OrganismStatus.COMPLETED, run.status)
        self.assertEqual(["observer", "analyst"], [item[0] for item in fixture["calls"]])

    def test_context_surrogate_string_blocks_before_calls(self) -> None:
        fixture = build_fixture()
        context = replace(fixture["context"], intent_id="\ud800")
        run = fixture["runner"].run(fixture["root"], context)

        self.assertEqual(OrganismStatus.BLOCK, run.status)
        self.assertEqual(("CONTEXT_INVALID",), run.decision_reasons)
        self.assertEqual([], fixture["calls"])

    def test_published_json_schemas_are_strict_and_parseable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        action_schema = json.loads(
            (root / "schemas" / "action-draft.v0.1.schema.json").read_text()
        )
        manifest_schema = json.loads(
            (root / "schemas" / "organism-manifest.v0.1.schema.json").read_text()
        )
        proposal_schema = json.loads(
            (root / "schemas" / "action-proposal.v0.1.schema.json").read_text()
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
        budget_schema = manifest_schema["$defs"]["resourceBudget"]["properties"]
        self.assertEqual(1, budget_schema["max_steps"]["minimum"])
        self.assertEqual(1, budget_schema["max_seconds"]["minimum"])
        self.assertTrue(
            manifest_schema["properties"]["pipeline"]["properties"][
                "executor"
            ]["properties"]["allowed_capability_ids"]["uniqueItems"]
        )
        context_variants = proposal_schema["properties"]["untrusted_context"][
            "items"
        ]["oneOf"]
        self.assertEqual(
            [
                ["source", "content"],
                ["source", "content_digest"],
            ],
            [branch["required"] for branch in context_variants],
        )
        self.assertTrue(
            all(branch["additionalProperties"] is False for branch in context_variants)
        )


if __name__ == "__main__":
    unittest.main()
