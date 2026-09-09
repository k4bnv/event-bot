"""
Strategy I — Prior Window Momentum (trend-following baseline), opt-in.

Adapted from the simplest possible rule in DeweyMarco/simple-kalshi-bot:
bet the SAME direction that won the immediately preceding window on this
series — no price/volatility model, no order book, nothing but "what just
happened". Cheap to compute and easy to reason about, and — the actual
point of adding it — a genuine control group: if this simple baseline
beats (or even matches) the more sophisticated strategies, that says those
aren't adding real value; if it loses to a fair coin flip, that's equally
informative (no serial correlation between independent 5/15-minute
windows, consistent with a reasonably efficient short-horizon market).

Needs `ctx.previous_outcome`, which Engine._update_previous_outcomes()
populates the moment a window rolls over — see that method's docstring
for why this can't just reuse the normal settlement-polling path (that
one only ever learns an outcome for a window some strategy actually
traded; this needs every window's outcome, traded or not).
"""
from __future__ import annotations

from typing import Optional

from .base import BaseStrategy, Signal, StrategyContext


class PriorWindowMomentumStrategy(BaseStrategy):
    name = "prior_window_momentum"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        outcome = ctx.previous_outcome
        if outcome is None:
            return None  # no known prior window yet — engine just started, or a lookup gave up

        return Signal(
            direction=outcome,
            reason=f"prior window resolved {outcome.value.upper()} — betting the same side continues",
            confidence=0.5,
        )
