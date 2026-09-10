"""
Strategy C — Mean Reversion (now betting CONTINUATION, not reversion —
see below).

Over `lookback_sec`, compute the rolling mean/stdev of price. If the
current price is an extreme outlier (|z-score| >= extreme_zscore) —
typically checked only in the closing minutes of a window via config
`entry_windows_min` — bet the move CONTINUES in the same direction the
z-score is already pointing.

Direction flipped 2026-09-10 from the original "bet it reverts back
toward the mean" (kept the class/strategy name — same wallet, same
history, just a different thesis going forward): the first ~9h live
on this pair/timeframe went 0 wins out of 23 trades on the original
revert-to-mean logic — a result extreme enough (would happen by chance
well under 1% of the time even for a strategy with real negative edge)
to say the short-horizon reversion thesis is backwards here, not just
under-tuned. Framed as a live experiment, not a settled conclusion —
if continuation ALSO loses, that's its own real answer (no reliable
z-score signal either way at this timeframe), same spirit as
prior_window_momentum's control-group framing.
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

        # price is an outlier -> bet the move CONTINUES (see module
        # docstring for why this is now the opposite of the original
        # revert-to-mean call).
        direction = Direction.UP if z > 0 else Direction.DOWN

        return Signal(
            direction=direction,
            reason=f"z-score={z:.2f} over last {lookback_sec:.0f}s (mean={mean:.2f}, std={stdev:.2f}) — betting continuation",
            confidence=min(0.9, 0.4 + (abs(z) - extreme_z) / (extreme_z * 2)),
        )
