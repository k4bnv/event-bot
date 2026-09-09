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

Two robustness terms were adapted from a comparable open-source Kalshi
15-minute-BTC bot (github.com/preceptress/btc-15-minute-prediction-model,
`settlement_probability`/`minute_volatility` in its scripts/btc_bot.py),
whose own barrier model is the same driftless log-normal shape as this
one's:

  * a FLOOR on realized volatility (`min_sigma_per_sec`) — without one, a
    freak flat stretch in `ctx.price_history` estimates near-zero
    volatility, which makes the barrier model wildly overconfident (a
    probability that snaps to ~0 or ~1 on the next small tick, since a
    tiny sigma turns any nonzero distance-to-strike into a huge z-score).
    Their bot floors BTC's realized vol at 0.035%/minute; adapted here to
    the same per-second sigma this module already works in.

  * an added BASIS uncertainty term (`basis_sigma`) for when
    `market.strike_is_fixed` is False — OKX hasn't posted its own official
    reference price for this window yet (see market_data.py), so
    `floor_strike` is our own best guess (the first price we happened to
    observe), not a trustworthy anchor. Their bot has the exact same
    situation the other way around: Kalshi's settlement reference isn't
    always available, so they proxy it through Coinbase's own opening
    price and inflate the model's uncertainty (in quadrature with the
    volatility term) whenever they had to. Same fix, same shape here.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional

from ..models import Direction, EventMarket, PricePoint
from .base import BaseStrategy, Signal, StrategyContext

# BTC realized vol rarely sits meaningfully below this even during quiet
# stretches — matches the floor preceptress/btc-15-minute-prediction-model
# uses for the same underlying (0.035%/minute), just expressed per-second
# since that's the unit this module already normalizes to.
DEFAULT_MIN_SIGMA_PCT_PER_MIN = 0.035

# Extra log-scale uncertainty folded in (in quadrature with the volatility
# term) when floor_strike is our own proxy rather than OKX's fixed
# reference — matches PROXY_BASIS_SIGMA in the Kalshi bot this was adapted
# from, for the analogous Coinbase-proxied-reference situation.
DEFAULT_UNFIXED_STRIKE_BASIS_PCT = 0.075


def min_sigma_per_sec_from_pct(pct_per_min: float) -> float:
    """Convert a "minimum volatility, expressed as %/minute" config value
    into the per-second sigma floor `compute_barrier_stats` takes —
    variance scales linearly with time, so sigma scales with sqrt(time)."""
    return max(0.0, pct_per_min) / 100 / math.sqrt(60)


def basis_sigma_for_market(market: EventMarket, basis_pct: float) -> float:
    """0.0 when OKX has already fixed this window's own reference price
    (market.strike_is_fixed is True, or None/unknown — treated the same,
    i.e. trusted by default) — floor_strike IS the truth there, no proxy
    risk to model. basis_pct/100 when strike_is_fixed is explicitly False:
    floor_strike is our own stand-in (see EventMarket.strike_is_fixed's
    docstring), an extra source of uncertainty on top of realized
    volatility."""
    if market.strike_is_fixed is False:
        return max(0.0, basis_pct) / 100
    return 0.0


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
    min_sigma_per_sec: float = 0.0, basis_sigma: float = 0.0,
) -> Optional[BarrierStats]:
    """Driftless log-normal barrier model: estimates volatility from recent
    realized returns, scales it to the remaining time via the sqrt(time)
    rule, and derives the z-score / probability of finishing >= strike.
    None if there isn't enough price history to estimate volatility (or,
    with no floor requested, the estimate would be degenerate — exactly
    zero realized variance).

    `min_sigma_per_sec` and `basis_sigma` are both 0.0 (i.e. inert, exactly
    the original behavior) unless a caller opts in — see this module's
    docstring and `min_sigma_per_sec_from_pct`/`basis_sigma_for_market` for
    where FairValueEdgeStrategy/ai_prompt derive them from config and
    `market.strike_is_fixed`:
      * min_sigma_per_sec floors the realized-vol estimate so a freak flat
        stretch can't make the model absurdly overconfident.
      * basis_sigma adds extra uncertainty (in quadrature with the
        volatility term) when `strike` itself is a proxy, not OKX's fixed
        reference — a distance-to-strike computed against a shaky anchor
        shouldn't produce as sharp a probability as one computed against a
        trustworthy one.
    """
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
    if sigma_sample <= 0 and min_sigma_per_sec <= 0:
        return None

    span_sec = points[-1].ts - points[0].ts
    n_intervals = len(points) - 1
    if span_sec <= 0 or n_intervals <= 0:
        return None
    avg_dt = span_sec / n_intervals
    if avg_dt <= 0:
        return None

    sigma_per_sec = sigma_sample / math.sqrt(avg_dt) if sigma_sample > 0 else 0.0
    sigma_per_sec = max(sigma_per_sec, min_sigma_per_sec)
    vol_sigma_remaining = sigma_per_sec * math.sqrt(remaining_sec)
    sigma_remaining = math.sqrt(vol_sigma_remaining ** 2 + basis_sigma ** 2)
    if sigma_remaining <= 1e-9:
        return None

    z = math.log(strike / current_price) / sigma_remaining
    # P(log-return over the remaining time >= log(strike/current)), driftless
    prob_up = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    prob_up = min(max(prob_up, 0.001), 0.999)
    return BarrierStats(z_score=z, sigma_horizon_pct=sigma_remaining * 100, base_prob=prob_up)


def fair_probability_up(
    points: list[PricePoint], current_price: float, strike: float, remaining_sec: float,
    min_sigma_per_sec: float = 0.0, basis_sigma: float = 0.0,
) -> Optional[float]:
    """P(underlying finishes >= strike) — see `compute_barrier_stats` for
    the intermediate z-score/sigma this is derived from, and for what
    `min_sigma_per_sec`/`basis_sigma` do."""
    stats = compute_barrier_stats(points, current_price, strike, remaining_sec, min_sigma_per_sec, basis_sigma)
    return stats.base_prob if stats else None


class FairValueEdgeStrategy(BaseStrategy):
    name = "fair_value_edge"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
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
