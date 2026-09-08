"""
Strategy C — Mean Reversion.

Over `lookback_sec`, compute the rolling mean/stdev of price. If the current
price is an extreme outlier (|z-score| >= extreme_zscore) — typically
checked only in the closing minutes of a window via config
`entry_windows_min` — bet on reversion back toward the mean.
"""
from __future__ import annotations

import statistics
from typing import Optional

from ..models import Direction
from .base import BaseStrategy, Signal, StrategyContext


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        lookback_sec = float(self.config.get("lookback_sec", 180))
        extreme_z = float(self.config.get("extreme_zscore", 1.5))

        if not ctx.price_history:
            return None
        now = ctx.price_history[-1].ts
        points = [p.price for p in ctx.price_history if now - p.ts <= lookback_sec]
        if len(points) < 10:
            return None

        mean = statistics.fmean(points)
        stdev = statistics.pstdev(points)
        if stdev <= 0:
            return None

        current = points[-1]
        z = (current - mean) / stdev
        if abs(z) < extreme_z:
            return None

        # price is an outlier -> bet it reverts back toward the mean
        direction = Direction.DOWN if z > 0 else Direction.UP

        return Signal(
            direction=direction,
            reason=f"z-score={z:.2f} over last {lookback_sec:.0f}s (mean={mean:.2f}, std={stdev:.2f})",
            confidence=min(0.9, 0.4 + (abs(z) - extreme_z) / (extreme_z * 2)),
        )
