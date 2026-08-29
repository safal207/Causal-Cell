"""Deterministic JSON and digest helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

DIGEST_PREFIX = "sha256:"
JSON_SNAPSHOT_MAX_DEPTH = 128
JSON_SNAPSHOT_MAX_NODES = 100_000
JSON_SNAPSHOT_MAX_STRING_BYTES = 1_000_000
JSON_SNAPSHOT_MAX_TOTAL_STRING_BYTES = 4_000_000
JSON_SNAPSHOT_MAX_INTEGER_BITS = 4_096


def snapshot_json(value: Any) -> Any:
    """Return a detached tree containing only exact built-in JSON types.

    Rejecting subclasses prevents untrusted mappings from mutating while they are
    being hashed and later authorized. Explicit structural bounds keep hostile
    inputs from escaping the caller's fail-closed validation path via recursion.
    """

    nodes = 0
    string_bytes = 0

    def bounded_string(item: str) -> str:
        nonlocal string_bytes
        if len(item) > JSON_SNAPSHOT_MAX_STRING_BYTES:
            raise ValueError("JSON string exceeds maximum byte length")
        encoded_size = len(item.encode("utf-8"))
        if encoded_size > JSON_SNAPSHOT_MAX_STRING_BYTES:
            raise ValueError("JSON string exceeds maximum byte length")
        string_bytes += encoded_size
        if string_bytes > JSON_SNAPSHOT_MAX_TOTAL_STRING_BYTES:
            raise ValueError("JSON snapshot exceeds total string-byte limit")
        return item

    def visit(item: Any, depth: int) -> Any:
        nonlocal nodes
        if depth > JSON_SNAPSHOT_MAX_DEPTH:
            raise ValueError("JSON snapshot exceeds maximum depth")
        nodes += 1
        if nodes > JSON_SNAPSHOT_MAX_NODES:
            raise ValueError("JSON snapshot exceeds maximum node count")

        item_type = type(item)
        if item is None or item_type is bool:
            return item
        if item_type is str:
            return bounded_string(item)
        if item_type is int:
            if item.bit_length() > JSON_SNAPSHOT_MAX_INTEGER_BITS:
                raise ValueError("JSON integer exceeds maximum bit length")
            return item
        if item_type is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            return item
        if item_type is list:
            if nodes + len(item) > JSON_SNAPSHOT_MAX_NODES:
                raise ValueError("JSON snapshot exceeds maximum node count")
            return [visit(child, depth + 1) for child in tuple(item)]
        if item_type is dict:
            if nodes + len(item) > JSON_SNAPSHOT_MAX_NODES:
                raise ValueError("JSON snapshot exceeds maximum node count")
            entries = tuple(item.items())
            if any(type(key) is not str for key, _ in entries):
                raise TypeError("JSON object keys must be exact strings")
            return {
                bounded_string(key): visit(child, depth + 1)
                for key, child in entries
            }
        raise TypeError("value must contain exact built-in JSON types")

    try:
        return visit(value, 0)
    except UnicodeError as exc:
        raise ValueError("JSON strings must be valid UTF-8") from exc
    except RuntimeError as exc:
        raise ValueError("JSON value mutated during snapshot") from exc


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes.

    This is a local canonicalization profile, not RFC 8785 conformance.
    """

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("JSON strings must be valid UTF-8") from exc


def digest_bytes(value: bytes) -> str:
    return f"{DIGEST_PREFIX}{hashlib.sha256(value).hexdigest()}"


def digest_json(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def proposal_digest(proposal: Mapping[str, Any]) -> str:
    """Digest a proposal while excluding its self-referential digest field."""

    digestible = copy.deepcopy(dict(proposal))
    digestible.pop("proposal_digest", None)
    return digest_json(digestible)


def bind_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Bind application-supplied arguments and the complete proposal to digests.

    The helper does not invent intent, authority, identity, scope, or approval.
    Those values must already be supplied by the application/authority layer.
    """

    bound = copy.deepcopy(dict(proposal))
    bound["arguments_digest"] = digest_json(bound.get("arguments"))
    bound["proposal_digest"] = proposal_digest(bound)
    return bound


def parse_timestamp(value: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError("timestamp must be a non-empty string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except (OverflowError, ValueError) as exc:
        raise ValueError("timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include an explicit UTC offset")
    try:
        return parsed.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError("timestamp is outside the supported UTC range") from exc


def require_aware_utc(value: Any) -> datetime:
    """Return an exact, timezone-aware ``datetime`` normalized to UTC."""

    if type(value) is not datetime or value.tzinfo is None:
        raise ValueError("datetime must be an exact timezone-aware value")
    try:
        if value.utcoffset() is None:
            raise ValueError("datetime must be timezone-aware")
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError("datetime is outside the supported UTC range") from exc


def format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    normalized = value.astimezone(UTC)
    if normalized.microsecond:
        rendered = normalized.isoformat(timespec="microseconds")
    else:
        rendered = normalized.isoformat(timespec="seconds")
    return rendered.replace("+00:00", "Z")
