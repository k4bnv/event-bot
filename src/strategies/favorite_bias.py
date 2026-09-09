"""
Strategy H — Favorite Bias (crowd-momentum / longshot-bias), opt-in.

Two ideas the user found in other Kalshi/Polymarket bots turn out to be
the exact same trading rule under different names, folded into one
strategy here:

  * "favorite-longshot bias" (reedjacobp/kalshi-trading-bot) — a
    documented behavioral-finance observation in prediction/betting
    markets: the "favorite" side (priced above ~$0.70) tends to win MORE
    often than its own price implies, i.e. the market systematically
    underprices favorites and overprices longshots. No independent
    probability model at all — a pure statistical/behavioral bet on the
    market's OWN pricing being skewed.
  * "crowd-momentum confirmation" (confidence-surfing-bot, Polymarket) —
    the same rule again, framed as "ride the crowd's already-formed
    opinion" instead of a named pricing bias.
  * "Resolution Rider" (same repo as favorite-longshot bias) — the same
    rule again, just checked much later (~60-90s before expiry) instead
    of a single ~3-minutes-out checkpoint. This needs no new timing
    machinery of its own: entry_windows_min already supports multiple
    independent checkpoints per strategy (e.g. [3, 1]), so the "early"
    and "late" variants of this same rule are just two configured
    checkpoints on ONE strategy, not three separate ones.

This is the deliberate OPPOSITE of fair_value_edge: that strategy computes
an independent probability and only trades when it DISAGREES with the
market's own price. This one trusts the market's price outright and just
follows whichever side it already favors, once it's favored "enough" — an
intentional control group for whether "just follow the favorite" beats or
loses to the strategies that look for a disagreement instead.
"""
from __future__ import annotations

from typing import Optional

from ..models import Direction
from .base import BaseStrategy, Signal, StrategyContext


class FavoriteBiasStrategy(BaseStrategy):
    name = "favorite_bias"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        market = ctx.market
        if market.up_price is None:
            return None

        threshold = float(self.config.get("favorite_price_threshold", 0.70))
        up_price = market.up_price
        down_price = 1.0 - up_price

        # up_price/down_price sum to 1, so with the default threshold
        # (> 0.5) at most one side can ever qualify — the >= comparisons
        # below still resolve the tie cleanly if a lower threshold is
        # configured such that both sides technically qualify.
        if up_price >= threshold and up_price >= down_price:
            return Signal(
                direction=Direction.UP,
                reason=f"favorite bias: UP priced {up_price:.2f} >= {threshold:.2f}, riding the crowd",
                confidence=min(0.9, 0.4 + (up_price - threshold) / (1.0 - threshold + 1e-9) * 0.5),
            )
        if down_price >= threshold and down_price > up_price:
            return Signal(
                direction=Direction.DOWN,
                reason=f"favorite bias: DOWN priced {down_price:.2f} >= {threshold:.2f}, riding the crowd",
                confidence=min(0.9, 0.4 + (down_price - threshold) / (1.0 - threshold + 1e-9) * 0.5),
            )
        return None
