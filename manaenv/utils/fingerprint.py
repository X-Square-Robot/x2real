"""Stable, JSON-canonicalised fingerprints for grouping pipeline orders.

The pipeline factories group orders by structural equality of their constraint
dictionaries (scene constraints, robot constraints, assembly constraints, ...).
Using ``json.dumps(..., sort_keys=True)`` + an md5 digest gives a deterministic
short string that is safe to use as a dict key and cheap to compare.

This is **not** a security primitive: md5 is only used because we need an
order-independent, hashable summary; collisions would only cause false dedup
which would surface as failing assertions in downstream scene/assembly logic.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def constraints_fingerprint(constraints: dict[str, Any]) -> str:
    """Return a stable hex-digest fingerprint for a constraint dict.

    ``default=str`` lets us include non-JSON values (paths, enums, ...)
    without crashing; the canonical form is sorted by key so dict ordering
    is irrelevant.
    """
    canonical = json.dumps(constraints, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()
