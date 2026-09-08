"""
Strategy D — Volatility-Adaptive Breakout & Retest.

Same impulse-then-retest mechanics as `breakout_retest`, but the threshold
is derived from *realized* volatility instead of a fixed percentage. A
fixed threshold is too tight in calm markets (noisy false positives) and
too loose in volatile ones (misses real breakouts); scaling by recent
realized volatility adapts to both regimes.

Important: the baseline volatility is measured over `vol_lookback_sec`
ENDING BEFORE the detection window (`breakout_lookback_sec`), not over the
same window used to look for the breakout. Measuring both from the same
window is self-referential — an actual sharp impulse inflates the very
volatility estimate used to decide whether it's significant, which can
suppress detection of the impulse it's supposed to catch.
"""
from __future__ import annotations

import math
import statistics
from typing import Optional

from ..models import PricePoint
from .base import BaseStrategy, Signal, StrategyContext
from .breakout_common import detect_breakout_retest


def _realized_vol_pct(points: list[PricePoint]) -> Optional[float]:
    """Stdev of consecutive log returns, expressed as a percentage."""
    if len(points) < 6:
        return None
    log_returns = []
    for i in range(1, len(points)):
        p0, p1 = points[i - 1].price, points[i].price
        if p0 > 0 and p1 > 0:
            log_returns.append(math.log(p1 / p0))
    if len(log_returns) < 5:
        return None
    sigma = statistics.pstdev(log_returns)
    return sigma * 100 if sigma > 0 else None


class VolatilityBreakoutStrategy(BaseStrategy):
    name = "volatility_breakout"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        breakout_lookback_sec = float(self.config.get("breakout_lookback_sec", 60))
        vol_lookback_sec = float(self.config.get("vol_lookback_sec", 300))
        vol_multiplier = float(self.config.get("vol_multiplier", 2.5))
        retest_tolerance_mult = float(self.config.get("retest_tolerance_mult", 0.4))
        min_threshold_pct = float(self.config.get("min_threshold_pct", 0.03))

        if not ctx.price_history:
            return None
        now = ctx.price_history[-1].ts

        # Baseline vol from the period BEFORE the detection window (see
        # module docstring) — deliberately excludes whatever is currently
        # happening in `breakout_lookback_sec`.
        baseline_points = [
            p for p in ctx.price_history
            if breakout_lookback_sec < now - p.ts <= breakout_lookback_sec + vol_lookback_sec
        ]
        vol_pct = _realized_vol_pct(baseline_points)
        if vol_pct is None:
            return None

        threshold_pct = max(vol_multiplier * vol_pct, min_threshold_pct)
        tolerance_pct = threshold_pct * retest_tolerance_mult

        detection_points = [p for p in ctx.price_history if now - p.ts <= breakout_lookback_sec]
        result = detect_breakout_retest(detection_points, threshold_pct, tolerance_pct)
        if result is None:
            return None
        direction, move_pct, retest_dist_pct = result

        return Signal(
            direction=direction,
            reason=(
                f"vol-adaptive breakout: baseline_vol={vol_pct:.4f}% -> "
                f"threshold={threshold_pct:.3f}%, move={move_pct:.3f}%, "
                f"retest within {retest_dist_pct:.3f}%"
            ),
            confidence=min(0.9, 0.4 + move_pct / (threshold_pct * 4)),
        )
