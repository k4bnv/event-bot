"""
Strategy J — Adaptive Timing (free-scanning Fair-Value edge).

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
far, is *this* the first one where the odds actually look wrong enough
to bet" — i.e. it picks its own moment by scanning, rather than having
one assigned. The underlying edge calculation is deliberately identical
to Strategy E (FairValueEdgeStrategy) — same driftless log-normal fair
probability vs OKX's own quoted price — so the two are an apples-to-apples
comparison of "bet at fixed checkpoints" vs "bet whenever the edge first
clears the bar", and, over many trades, the Combo/Leaderboard table
(which groups by the trade's *actual* entry_window_min) becomes a live,
empirical answer to "which minute-to-expiry is actually best for this
edge" — the exact question the ML checkpoint_features log was built to
eventually answer offline, just visible sooner and for free here.

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
from .fair_value_edge import (
    DEFAULT_MIN_SIGMA_PCT_PER_MIN, DEFAULT_UNFIXED_STRIKE_BASIS_PCT,
    basis_sigma_for_market, fair_probability_up, min_sigma_per_sec_from_pct,
)


class AdaptiveTimingStrategy(BaseStrategy):
    name = "adaptive_timing"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        if ctx.already_open_this_market:
            return None  # already committed to this market at an earlier checkpoint

        market = ctx.market
        if market.up_price is None or market.floor_strike is None:
            return None  # no live quote / no reference price to model against yet

        lookback_sec = float(self.config.get("lookback_sec", 120))
        min_edge = float(self.config.get("min_edge", 0.08))
        min_sigma_pct = float(self.config.get("min_sigma_pct_per_min", DEFAULT_MIN_SIGMA_PCT_PER_MIN))
        basis_pct = float(self.config.get("unfixed_strike_basis_pct", DEFAULT_UNFIXED_STRIKE_BASIS_PCT))

        if not ctx.price_history:
            return None
        now = ctx.price_history[-1].ts
        points = [p for p in ctx.price_history if now - p.ts <= lookback_sec]

        model_prob_up = fair_probability_up(
            points, points[-1].price, market.floor_strike, ctx.remaining_sec,
            min_sigma_per_sec=min_sigma_per_sec_from_pct(min_sigma_pct),
            basis_sigma=basis_sigma_for_market(market, basis_pct),
        )
        if model_prob_up is None:
            return None

        edge = model_prob_up - market.up_price
        reason_prefix = f"скан на {ctx.window_min}м: "
        if edge >= min_edge:
            return Signal(
                direction=Direction.UP,
                reason=f"{reason_prefix}model P(UP)={model_prob_up:.3f} vs market px={market.up_price:.3f}, edge={edge:+.3f}",
                confidence=min(0.9, 0.4 + (edge - min_edge) / (min_edge * 2)),
            )
        if -edge >= min_edge:
            return Signal(
                direction=Direction.DOWN,
                reason=(
                    f"{reason_prefix}model P(UP)={model_prob_up:.3f} vs market px={market.up_price:.3f}, "
                    f"edge={edge:+.3f} favors DOWN"
                ),
                confidence=min(0.9, 0.4 + (-edge - min_edge) / (min_edge * 2)),
            )
        return None
