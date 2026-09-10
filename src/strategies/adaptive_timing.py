"""
Strategy J — Adaptive Timing (free-scanning Orderbook Imbalance).

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
far, is *this* the first one where the book actually looks skewed enough
to bet" — i.e. it picks its own moment by scanning, rather than having
one assigned.

Underlying signal changed 2026-09-10 (kept the class/strategy name/
wallet — same continuity as this session's other config-only tweaks):
was a copy of Strategy E's (FairValueEdgeStrategy) driftless log-normal
edge calculation, deliberately identical so the two were an apples-to-
apples "fixed checkpoints vs scan for the edge" comparison. User asked
to swap it for a different underlying signal while keeping the scanning
shell — picked Strategy B's (OrderbookMomentumStrategy) resting bid/ask
imbalance instead (see that module's docstring), for the same reason E
was picked originally: B already exists as B's own fixed-checkpoint
version of this exact signal, so this strategy is now B's "scan for the
moment instead of guessing it" pairing — same apples-to-apples logic,
just against a different base strategy. Over many trades, the
Combo/Leaderboard table (which groups by the trade's *actual*
entry_window_min) becomes a live, empirical answer to "which
minute-to-expiry is actually best for an orderbook-imbalance edge".

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

        orderbook = ctx.orderbook
        if orderbook is None or not orderbook.bids or not orderbook.asks:
            return None

        depth = int(self.config.get("orderbook_depth_levels", 10))
        threshold = float(self.config.get("imbalance_threshold", 1.8))

        bid_vol = orderbook.bid_volume(depth)
        ask_vol = orderbook.ask_volume(depth)
        if bid_vol <= 0 or ask_vol <= 0:
            return None

        ratio = bid_vol / ask_vol
        if ratio >= threshold:
            direction, strength = Direction.UP, ratio
        elif ratio <= 1 / threshold:
            direction, strength = Direction.DOWN, 1 / ratio
        else:
            return None

        return Signal(
            direction=direction,
            reason=(
                f"скан на {ctx.window_min}м: orderbook imbalance {ratio:.2f}x "
                f"(bid={bid_vol:.2f} ask={ask_vol:.2f}, top {depth})"
            ),
            confidence=min(0.9, 0.4 + (strength - threshold) / (threshold * 2)),
        )
