"""
Strategy E — Fair-Value / Probability Edge.

Unlike the other strategies (which bet on price *direction*), this one bets
on *mispricing of the contract itself*. `market.up_price` is OKX's own
market-implied probability that the primary side (UP/YES) wins. This
strategy independently estimates that same probability from recent realized
volatility (a driftless log-normal model of the underlying reaching
`market.floor_strike` by expiry) and only trades when its own estimate
disagrees with the market's price by more than `min_edge` — i.e. it trades
the *odds*, not a directional hunch. This is the one strategy that is
specific to event contracts being priced as probabilities rather than to
BTC price action in general.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional

from ..models import Direction, PricePoint
from .base import BaseStrategy, Signal, StrategyContext


@dataclass
class BarrierStats:
    """Intermediate values behind `fair_probability_up`'s single number —
    exposed separately because the ai_prompt strategy's barrier-style
    prompt template wants to show its own $z_score/$base_prob/$sigma_horizon
    rather than just the final probability, so an LLM can apply a small,
    bounded qualitative correction on top of a real statistical anchor."""
    z_score: float
    sigma_horizon_pct: float   # realized-vol sigma scaled to the remaining horizon, as a %
    base_prob: float           # normal CDF(z) -> P(underlying finishes >= strike), driftless


def compute_barrier_stats(
    points: list[PricePoint], current_price: float, strike: float, remaining_sec: float,
) -> Optional[BarrierStats]:
    """Driftless log-normal barrier model: estimates volatility from recent
    realized returns, scales it to the remaining time via the sqrt(time)
    rule, and derives the z-score / probability of finishing >= strike.
    None if there isn't enough price history to estimate volatility, or the
    estimate would be degenerate (zero variance)."""
    if len(points) < 10 or remaining_sec <= 0 or current_price <= 0 or strike <= 0:
        return None

    log_returns = []
    for i in range(1, len(points)):
        p0, p1 = points[i - 1].price, points[i].price
        if p0 > 0 and p1 > 0:
            log_returns.append(math.log(p1 / p0))
    if len(log_returns) < 8:
        return None

    sigma_sample = statistics.pstdev(log_returns)
    if sigma_sample <= 0:
        return None

    span_sec = points[-1].ts - points[0].ts
    n_intervals = len(points) - 1
    if span_sec <= 0 or n_intervals <= 0:
        return None
    avg_dt = span_sec / n_intervals
    if avg_dt <= 0:
        return None

    sigma_per_sec = sigma_sample / math.sqrt(avg_dt)
    sigma_remaining = sigma_per_sec * math.sqrt(remaining_sec)
    if sigma_remaining <= 1e-9:
        return None

    z = math.log(strike / current_price) / sigma_remaining
    # P(log-return over the remaining time >= log(strike/current)), driftless
    prob_up = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    prob_up = min(max(prob_up, 0.001), 0.999)
    return BarrierStats(z_score=z, sigma_horizon_pct=sigma_remaining * 100, base_prob=prob_up)


def fair_probability_up(
    points: list[PricePoint], current_price: float, strike: float, remaining_sec: float,
) -> Optional[float]:
    """P(underlying finishes >= strike) — see `compute_barrier_stats` for
    the intermediate z-score/sigma this is derived from."""
    stats = compute_barrier_stats(points, current_price, strike, remaining_sec)
    return stats.base_prob if stats else None


class FairValueEdgeStrategy(BaseStrategy):
    name = "fair_value_edge"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        market = ctx.market
        if market.up_price is None or market.floor_strike is None:
            return None  # no live quote / no reference price to model against yet

        lookback_sec = float(self.config.get("lookback_sec", 120))
        min_edge = float(self.config.get("min_edge", 0.08))

        if not ctx.price_history:
            return None
        now = ctx.price_history[-1].ts
        points = [p for p in ctx.price_history if now - p.ts <= lookback_sec]

        model_prob_up = fair_probability_up(points, points[-1].price, market.floor_strike, ctx.remaining_sec)
        if model_prob_up is None:
            return None

        edge = model_prob_up - market.up_price
        if edge >= min_edge:
            return Signal(
                direction=Direction.UP,
                reason=f"model P(UP)={model_prob_up:.3f} vs market px={market.up_price:.3f}, edge={edge:+.3f}",
                confidence=min(0.9, 0.4 + (edge - min_edge) / (min_edge * 2)),
            )
        if -edge >= min_edge:
            return Signal(
                direction=Direction.DOWN,
                reason=f"model P(UP)={model_prob_up:.3f} vs market px={market.up_price:.3f}, edge={edge:+.3f} favors DOWN",
                confidence=min(0.9, 0.4 + (-edge - min_edge) / (min_edge * 2)),
            )
        return None
