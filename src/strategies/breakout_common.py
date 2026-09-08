"""Shared impulse-then-retest detection, used by both `breakout_retest`
(fixed % threshold) and `volatility_breakout` (threshold derived from
realized volatility) — they differ only in how they compute threshold_pct/
tolerance_pct, not in the breakout/retest mechanics itself.
"""
from __future__ import annotations

from typing import Optional

from ..models import Direction, PricePoint


def detect_breakout_retest(
    points: list[PricePoint], threshold_pct: float, tolerance_pct: float,
) -> Optional[tuple[Direction, float, float]]:
    """
    1. If price moved away from its starting level by >= threshold_pct,
       that's an "impulse" and the starting level is the "breakout level".
    2. Require price to have since pulled back close to that level (within
       tolerance_pct) while still holding on the breakout side — a
       continuation retest — before signalling.

    Returns (direction, breakout_move_pct, retest_dist_pct) or None.
    """
    if len(points) < 5:
        return None

    start_price = points[0].price
    max_price = max(p.price for p in points)
    min_price = min(p.price for p in points)
    current = points[-1].price

    up_move_pct = (max_price - start_price) / start_price * 100
    down_move_pct = (start_price - min_price) / start_price * 100

    if up_move_pct >= threshold_pct and up_move_pct >= down_move_pct:
        direction, level, move_pct = Direction.UP, start_price, up_move_pct
    elif down_move_pct >= threshold_pct:
        direction, level, move_pct = Direction.DOWN, start_price, down_move_pct
    else:
        return None

    retest_dist_pct = abs(current - level) / level * 100
    if retest_dist_pct > tolerance_pct:
        return None

    # must still be holding on the breakout side, not a full reversal back through it
    holding = (current >= level) if direction is Direction.UP else (current <= level)
    if not holding:
        return None

    return direction, move_pct, retest_dist_pct
