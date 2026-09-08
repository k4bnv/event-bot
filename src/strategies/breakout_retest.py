"""
Strategy A — Breakout & Retest (fixed % threshold).

See `breakout_common.detect_breakout_retest` for the shared mechanics; this
strategy's own job is just picking the lookback window and a fixed
percentage threshold from config. Compare with `volatility_breakout`, which
derives the threshold from realized volatility instead of a constant.
"""
from __future__ import annotations

from typing import Optional

from .base import BaseStrategy, Signal, StrategyContext
from .breakout_common import detect_breakout_retest


class BreakoutRetestStrategy(BaseStrategy):
    name = "breakout_retest"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        lookback_sec = float(self.config.get("impulse_lookback_sec", 90))
        threshold_pct = float(self.config.get("impulse_threshold_pct", 0.12))
        tolerance_pct = float(self.config.get("retest_tolerance_pct", 0.05))

        if not ctx.price_history:
            return None
        now = ctx.price_history[-1].ts
        points = [p for p in ctx.price_history if now - p.ts <= lookback_sec]

        result = detect_breakout_retest(points, threshold_pct, tolerance_pct)
        if result is None:
            return None
        direction, move_pct, retest_dist_pct = result

        return Signal(
            direction=direction,
            reason=(
                f"impulse {move_pct:.3f}% over {lookback_sec:.0f}s, "
                f"retest within {retest_dist_pct:.3f}% of level"
            ),
            confidence=min(0.9, 0.4 + move_pct / (threshold_pct * 4)),
        )
