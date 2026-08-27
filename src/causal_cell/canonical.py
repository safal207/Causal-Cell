"""Deterministic JSON and digest helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Mapping


DIGEST_PREFIX = "sha256:"


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes.

    This is a local canonicalization profile, not RFC 8785 conformance.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a non-empty string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include an explicit UTC offset")
    return parsed.astimezone(UTC)


def format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    normalized = value.astimezone(UTC)
    if normalized.microsecond:
        rendered = normalized.isoformat(timespec="microseconds")
    else:
        rendered = normalized.isoformat(timespec="seconds")
    return rendered.replace("+00:00", "Z")
