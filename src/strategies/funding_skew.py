"""
Strategy F — Funding Rate / Perp Skew (contrarian).

Uses the BTC perpetual funding rate as a proxy for crowd positioning —
a signal source independent of BTC/USDT price action itself, unlike the
other strategies. Strongly positive funding means longs are paying shorts
heavily (crowded long); strongly negative means the opposite. Crowded
positioning tends to mean-revert, so this strategy bets against the crowd.
"""
from __future__ import annotations

from typing import Optional

from ..models import Direction
from .base import BaseStrategy, Signal, StrategyContext


class FundingSkewStrategy(BaseStrategy):
    name = "funding_skew"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        rate = ctx.funding_rate
        if rate is None:
            return None

        threshold = float(self.config.get("funding_rate_threshold", 0.0003))  # fraction, e.g. 0.03%/8h

        if rate >= threshold:
            return Signal(
                direction=Direction.DOWN,
                reason=f"funding rate {rate * 100:.4f}% — longs crowded/paying up, betting mean-reversion DOWN",
                confidence=min(0.9, 0.4 + (rate - threshold) / (threshold * 2)),
            )
        if rate <= -threshold:
            return Signal(
                direction=Direction.UP,
                reason=f"funding rate {rate * 100:.4f}% — shorts crowded/paying up, betting mean-reversion UP",
                confidence=min(0.9, 0.4 + (-rate - threshold) / (threshold * 2)),
            )
        return None
