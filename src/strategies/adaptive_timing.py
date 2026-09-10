"""
Strategy J — Adaptive Timing (free-scanning Favorite Bias).

Every other strategy is told WHEN to look — the engine calls its
evaluate() only at a handful of configured entry_windows_min checkpoints
(e.g. "12 мин, 7 мин, 2 мин"), each firing exactly once and each getting
its own independent wallet (see Engine._wallet_key). Those fixed numbers
were originally guesses, and different series (5-min vs 15-min) give a
given checkpoint number a completely different amount of the market's
actual life.

This strategy instead gets called at a DENSE grid of checkpoints
(entry_windows_min: [10, 9, ..., 1] in config.yaml) and answers the
opposite question: not "is now a good moment for a fixed 2-minute bet",
but "of all the moments I've been asked about in this market's life so
far, is *this* the first one where one side has already become the
crowd's clear favorite" — i.e. it picks its own moment by scanning,
rather than having one assigned.

Underlying signal history (kept the class/strategy name/wallet through
all of it — same continuity as this session's other config-only
tweaks): started as a copy of Strategy E's (fair_value_edge) driftless
log-normal edge math; swapped 2026-09-10 to Strategy B's
(orderbook_momentum) bid/ask imbalance; both lost money over real live
stretches. Swapped again the same day to Strategy H's
(FavoriteBiasStrategy) crowd-momentum signal — picked over a plain
funding_rate_threshold copy of Strategy F specifically because funding
rate barely changes within one market's few-minute lifespan (it settles
every ~8h), so scanning it would mostly just fire-or-not at the first
checkpoint checked and sit there identically after — no real "moment"
to find. market.up_price, by contrast, is the single fastest-moving
input in the whole system, so continuous scanning here asks a genuinely
different, useful question from H's own fixed-checkpoint version: does
riding the favorite AS SOON AS it clears favorite_price_threshold (often
still well before expiry, arguably a less-decided market) do better or
worse than H's deliberately late (3м/1м) checkpoints, where the outcome
tends to already look close to settled?

Because it's called at many checkpoints per market but must place AT
MOST ONE trade per market, it relies on
`ctx.already_open_this_market` (computed by the engine from this
strategy's own wallet) rather than any internal bookkeeping: if an
earlier checkpoint's signal got rejected by the engine (no live quote,
low balance, slippage, ...) rather than actually opening a trade, this
naturally gets another chance at the next checkpoint instead of giving
up on the market for good. See StrategyConfig.dynamic_timing (must be set
for this strategy in config.yaml) — it collapses all these checkpoints
onto ONE shared wallet instead of splitting capital across a dozen
of them.
"""
from __future__ import annotations

from typing import Optional

from ..models import Direction
from .base import BaseStrategy, Signal, StrategyContext


class AdaptiveTimingStrategy(BaseStrategy):
    name = "adaptive_timing"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        if ctx.already_open_this_market:
            return None  # already committed to this market at an earlier checkpoint

        market = ctx.market
        if market.up_price is None:
            return None

        threshold = float(self.config.get("favorite_price_threshold", 0.70))
        up_price = market.up_price
        down_price = 1.0 - up_price
        reason_prefix = f"скан на {ctx.window_min}м: "

        # up_price/down_price sum to 1, so with the default threshold
        # (> 0.5) at most one side can ever qualify — the >= comparisons
        # below still resolve the tie cleanly if a lower threshold is
        # configured such that both sides technically qualify.
        if up_price >= threshold and up_price >= down_price:
            return Signal(
                direction=Direction.UP,
                reason=f"{reason_prefix}favorite bias: UP priced {up_price:.2f} >= {threshold:.2f}, riding the crowd",
                confidence=min(0.9, 0.4 + (up_price - threshold) / (1.0 - threshold + 1e-9) * 0.5),
            )
        if down_price >= threshold and down_price > up_price:
            return Signal(
                direction=Direction.DOWN,
                reason=(
                    f"{reason_prefix}favorite bias: DOWN priced {down_price:.2f} >= {threshold:.2f}, "
                    f"riding the crowd"
                ),
                confidence=min(0.9, 0.4 + (down_price - threshold) / (1.0 - threshold + 1e-9) * 0.5),
            )
        return None
