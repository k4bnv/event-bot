"""
Shared, pure feature-extraction helpers — no engine/strategy/storage
dependencies, so they're trivially unit-testable and reusable from
anywhere that has a price series and a timestamp.

`pct_change_over` is a near-duplicate of ai_prompt.py's private
`_pct_change_over` (kept there too, deliberately not re-imported from
here, to avoid coupling the engine's checkpoint-feature logging to one
specific strategy module) — if you're touching one, check whether the
other needs the same fix.
"""
from __future__ import annotations

from typing import Optional

from .models import PricePoint


def pct_change_over(points: list[PricePoint], now: float, window_sec: float) -> Optional[float]:
    """% change from the oldest sample within `window_sec` of `now` to the
    latest overall sample. None if fewer than 2 points fall in that
    window, or the oldest one is non-positive (can't take a % of it)."""
    window = [p for p in points if now - p.ts <= window_sec]
    if len(window) < 2 or window[0].price <= 0:
        return None
    return (window[-1].price - window[0].price) / window[0].price * 100
