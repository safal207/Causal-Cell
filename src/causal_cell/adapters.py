"""Provider-neutral synchronous model-adapter contracts.

The adapter owns provider credentials. Calls and results contain no credentials and
provider correlation IDs are evidence only, never authorization.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .canonical import digest_json, snapshot_json

DATA_CLASSIFICATIONS = {
    "public",
    "internal",
    "confidential",
    "restricted",
    "unknown",
}
DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


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

    def __post_init__(self) -> None:
        required_strings = (
            self.adapter_id,
            self.provider,
            self.model,
            self.origin,
            self.version,
        )
        try:
            strings_valid = all(
                type(value) is str
                and bool(value.strip())
                and value == value.strip()
                and snapshot_json(value) == value
                for value in required_strings
            )
            schema_valid = (
                type(self.schema_digest) is str
                and bool(DIGEST_RE.fullmatch(self.schema_digest))
            )
            destination_valid = self.destination is None or (
                type(self.destination) is str
                and bool(self.destination.strip())
                and self.destination == self.destination.strip()
                and snapshot_json(self.destination) == self.destination
            )
        except (TypeError, ValueError):
            strings_valid = schema_valid = destination_valid = False
        if not strings_valid or not schema_valid or not destination_valid:
            raise ValueError("invalid adapter identity")

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
    contains_secret: bool
    data_classification: str
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
        contains_secret: bool,
        data_classification: str,
    ) -> ModelResult:
        copied = snapshot_json(output)
        if not isinstance(copied, dict):
            raise TypeError("model output must be a JSON object")
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
            contains_secret=contains_secret,
            data_classification=data_classification,
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
        contains_secret: bool,
        data_classification: str,
    ) -> ModelResult:
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
            contains_secret=contains_secret,
            data_classification=data_classification,
            error_code=error_code,
        )

    def to_record(self) -> dict[str, Any]:
        record = {
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
            "contains_secret": self.contains_secret,
            "data_classification": self.data_classification,
            "error_code": self.error_code,
            "claim_boundary": (
                "Adapter-reported result, usage, and data labels; not independent "
                "response truth, content classification, or model-revision attestation."
            ),
        }
        detached = snapshot_json(record)
        if not isinstance(detached, dict):
            raise TypeError("model result record must be a JSON object")
        return detached


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
        if type(result) is not ModelResult:
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


def snapshot_model_result(result: ModelResult) -> ModelResult:
    """Detach provider data before validation and downstream authorization."""

    if type(result) is not ModelResult:
        raise TypeError("adapter result must be an exact ModelResult")
    output = None if result.output is None else snapshot_json(result.output)
    return ModelResult(
        status=result.status,
        output=output,
        output_digest=result.output_digest,
        provider_request_id=result.provider_request_id,
        provider=result.provider,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_microunits=result.cost_microunits,
        contains_secret=result.contains_secret,
        data_classification=result.data_classification,
        error_code=result.error_code,
    )


def validate_model_result(
    result: ModelResult,
    identity: AdapterIdentity,
    call: ModelCall,
) -> tuple[str, ...]:
    if type(result) is not ModelResult:
        return ("MODEL_RESULT_INVALID",)

    reasons: list[str] = []
    if type(result.status) is not ModelResultStatus:
        reasons.append("MODEL_RESULT_INVALID")
    identity_fields_valid = (
        type(result.provider) is str
        and bool(result.provider)
        and type(result.model) is str
        and bool(result.model)
    )
    if not identity_fields_valid:
        reasons.append("MODEL_RESULT_INVALID")
    elif result.provider != identity.provider or result.model != identity.model:
        reasons.append("ADAPTER_IDENTITY_MISMATCH")
    usage = (result.input_tokens, result.output_tokens, result.cost_microunits)
    usage_valid = all(
        type(value) is int
        and value >= 0
        and value.bit_length() <= 4_096
        for value in usage
    )
    if not usage_valid:
        reasons.append("MODEL_USAGE_INVALID")
    if usage_valid and result.output_tokens > call.max_output_tokens:
        reasons.append("MODEL_OUTPUT_BUDGET_EXCEEDED")
    if (
        type(result.contains_secret) is not bool
        or type(result.data_classification) is not str
        or result.data_classification not in DATA_CLASSIFICATIONS
    ):
        reasons.append("MODEL_RESULT_INVALID")

    if result.status is ModelResultStatus.RETURNED:
        if type(result.output_digest) is not str or not result.output_digest:
            reasons.append("MODEL_RESULT_INVALID")
        try:
            detached_output = snapshot_json(result.output)
        except (TypeError, ValueError):
            reasons.append("MODEL_OUTPUT_INVALID")
        else:
            if not isinstance(detached_output, dict):
                reasons.append("MODEL_OUTPUT_INVALID")
            elif (
                type(result.output_digest) is str
                and result.output_digest != digest_json(detached_output)
            ):
                reasons.append("MODEL_OUTPUT_DIGEST_MISMATCH")
        if result.error_code is not None:
            reasons.append("MODEL_RESULT_INVALID")
    elif type(result.status) is ModelResultStatus:
        if result.output is not None or result.output_digest is not None:
            reasons.append("MODEL_RESULT_INVALID")
        if type(result.error_code) is not str or not result.error_code:
            reasons.append("MODEL_RESULT_INVALID")
    if result.provider_request_id is not None and (
        type(result.provider_request_id) is not str
        or not result.provider_request_id
    ):
        reasons.append("MODEL_RESULT_INVALID")
    try:
        snapshot_json(
            {
                "output_digest": result.output_digest,
                "provider_request_id": result.provider_request_id,
                "provider": result.provider,
                "model": result.model,
                "error_code": result.error_code,
            }
        )
    except (TypeError, ValueError):
        reasons.append("MODEL_RESULT_INVALID")
    return tuple(dict.fromkeys(reasons))
