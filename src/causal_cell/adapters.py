"""Provider-neutral synchronous model-adapter contracts.

The adapter owns provider credentials. Calls and results contain no credentials and
provider correlation IDs are evidence only, never authorization.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping, Protocol

from .canonical import digest_json


class ModelResultStatus(StrEnum):
    RETURNED = "RETURNED"
    REFUSED = "REFUSED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class AdapterIdentity:
    adapter_id: str
    provider: str
    model: str
    origin: str
    version: str
    schema_digest: str
    destination: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "provider": self.provider,
            "model": self.model,
            "origin": self.origin,
            "version": self.version,
            "schema_digest": self.schema_digest,
            "destination": self.destination,
        }

    @property
    def identity_digest(self) -> str:
        return digest_json(self.to_record())


@dataclass(frozen=True, slots=True)
class ModelCall:
    run_id: str
    organism_id: str
    stage: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    intent_id: str
    parent_cause: str
    payload: Mapping[str, Any]
    payload_digest: str
    deadline_at: str
    max_output_tokens: int
    data_classification: str

    def to_record(self, *, include_payload: bool = False) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "profile": "org.causalcell.model-call.v0.1",
            "run_id": self.run_id,
            "organism_id": self.organism_id,
            "stage": self.stage,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "intent_id": self.intent_id,
            "parent_cause": self.parent_cause,
            "payload_digest": self.payload_digest,
            "deadline_at": self.deadline_at,
            "max_output_tokens": self.max_output_tokens,
            "data_classification": self.data_classification,
        }
        if include_payload:
            record["payload"] = copy.deepcopy(dict(self.payload))
        return record


@dataclass(frozen=True, slots=True)
class ModelResult:
    status: ModelResultStatus
    output: Mapping[str, Any] | None
    output_digest: str | None
    provider_request_id: str | None
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_microunits: int
    error_code: str | None

    @classmethod
    def returned(
        cls,
        identity: AdapterIdentity,
        output: Mapping[str, Any],
        *,
        provider_request_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_microunits: int = 0,
    ) -> "ModelResult":
        copied = copy.deepcopy(dict(output))
        return cls(
            status=ModelResultStatus.RETURNED,
            output=copied,
            output_digest=digest_json(copied),
            provider_request_id=provider_request_id,
            provider=identity.provider,
            model=identity.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_microunits=cost_microunits,
            error_code=None,
        )

    @classmethod
    def terminal(
        cls,
        identity: AdapterIdentity,
        status: ModelResultStatus,
        error_code: str,
        *,
        provider_request_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_microunits: int = 0,
    ) -> "ModelResult":
        if status is ModelResultStatus.RETURNED:
            raise ValueError("terminal result cannot use RETURNED")
        return cls(
            status=status,
            output=None,
            output_digest=None,
            provider_request_id=provider_request_id,
            provider=identity.provider,
            model=identity.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_microunits=cost_microunits,
            error_code=error_code,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "org.causalcell.model-result.v0.1",
            "status": self.status.value,
            "output_digest": self.output_digest,
            "provider_request_id": self.provider_request_id,
            "provider": self.provider,
            "model": self.model,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_microunits": self.cost_microunits,
            },
            "error_code": self.error_code,
            "claim_boundary": (
                "Provider-reported result and usage; not independent response truth "
                "or model-revision attestation."
            ),
        }


class ModelAdapter(Protocol):
    @property
    def identity(self) -> AdapterIdentity: ...

    def invoke(self, call: ModelCall) -> ModelResult: ...


class CallbackModelAdapter:
    """Synthetic/reference adapter. It is an in-process callback, not a sandbox."""

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
        result = self._callback(copy.deepcopy(call))
        if not isinstance(result, ModelResult):
            raise TypeError("adapter callback must return ModelResult")
        return result


class AdapterRegistry:
    def __init__(self, adapters: list[ModelAdapter] | tuple[ModelAdapter, ...]) -> None:
        self._adapters: dict[str, ModelAdapter] = {}
        for adapter in adapters:
            adapter_id = adapter.identity.adapter_id
            if adapter_id in self._adapters:
                raise ValueError(f"duplicate adapter_id: {adapter_id}")
            self._adapters[adapter_id] = adapter

    def get(self, adapter_id: str) -> ModelAdapter | None:
        return self._adapters.get(adapter_id)


def validate_model_result(
    result: ModelResult,
    identity: AdapterIdentity,
    call: ModelCall,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not isinstance(result.status, ModelResultStatus):
        reasons.append("MODEL_RESULT_INVALID")
    if result.provider != identity.provider or result.model != identity.model:
        reasons.append("ADAPTER_IDENTITY_MISMATCH")
    for value in (result.input_tokens, result.output_tokens, result.cost_microunits):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            reasons.append("MODEL_USAGE_INVALID")
            break
    if result.output_tokens > call.max_output_tokens:
        reasons.append("MODEL_OUTPUT_BUDGET_EXCEEDED")
    if result.status is ModelResultStatus.RETURNED:
        if not isinstance(result.output, Mapping):
            reasons.append("MODEL_OUTPUT_INVALID")
        else:
            try:
                if result.output_digest != digest_json(result.output):
                    reasons.append("MODEL_OUTPUT_DIGEST_MISMATCH")
            except (TypeError, ValueError):
                reasons.append("MODEL_OUTPUT_INVALID")
        if result.error_code is not None:
            reasons.append("MODEL_RESULT_INVALID")
    else:
        if result.output is not None or result.output_digest is not None:
            reasons.append("MODEL_RESULT_INVALID")
        if not isinstance(result.error_code, str) or not result.error_code:
            reasons.append("MODEL_RESULT_INVALID")
    if result.provider_request_id is not None and (
        not isinstance(result.provider_request_id, str) or not result.provider_request_id
    ):
        reasons.append("MODEL_RESULT_INVALID")
    return tuple(dict.fromkeys(reasons))
