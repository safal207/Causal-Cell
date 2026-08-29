"""Provider-neutral proposal normalization."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

SPAN_FIELDS = ("trace_id", "span_id", "parent_span_id")


def normalize_proposal(
    raw: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Normalize a provider span without allowing defaults to replace real data.

    Precedence is canonical top-level value, then exported span value, then
    application default. The returned object is still only a proposal.
    """

    normalized = copy.deepcopy(dict(raw))
    span = normalized.pop("span", {})
    if not isinstance(span, Mapping):
        raise ValueError("span must be an object")
    defaults = defaults or {}
    for field in SPAN_FIELDS:
        if field in normalized and normalized[field] not in (None, ""):
            continue
        if field in span and span[field] not in (None, ""):
            normalized[field] = span[field]
        elif field in defaults:
            normalized[field] = copy.deepcopy(defaults[field])
    return normalized
