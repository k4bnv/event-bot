"""
Strategy K — Absorption Reversal.

Idea (not original to this repo — a user-supplied write-up describing a
"failed attack -> reversal" microstructure setup): the market is being
sold aggressively, but price barely falls — someone is absorbing the
selling. Once sell pressure fades, the odds of an UP move rise. Mirror
logic for DOWN. This is deliberately about a DIVERGENCE between real
executed aggressor flow and the price's actual reaction to it, not just
an order-book imbalance (a resting wall can be pulled before it ever
trades) — see models.TradePrint's docstring for why this needed a new
data feed (OKX's trade-tape endpoint) this bot didn't have before.

Three phases, each a real (if simple, rule-based — see the note on the
ML question below) filter, not a single indicator crossing:

  Phase A — pressure + absorption. Trade-Flow-Imbalance (TFI) over
  `tfi_lookback_sec` (60-120s, per the original write-up):
      TFI = (buy_vol - sell_vol) / (buy_vol + sell_vol)
  from REAL executed prints (ctx.trade_prints), not resting orders.
  Needs |TFI| >= min_abs_tfi AND elevated volume vs a longer baseline
  (min_volume_multiple) AND the actual price move over that same window
  to diverge from what TFI-implied pressure would predict by more than
  min_residual_pct — that divergence (computed against a realized-vol
  scaled "expected move", see _expected_return_pct) IS the absorption
  signal. Heavy selling (TFI very negative) + a much smaller/positive
  actual move than expected = sell pressure absorbed = UP candidate.
  Heavy buying mirrors to a DOWN candidate.

  Phase B — confirmation. A one-off imbalanced print or a single big book
  wall proves nothing (it can be pulled). Requires the resting book on
  the SIDE that was under attack (bid for a UP candidate, ask for DOWN)
  to have stayed replenished rather than draining — see
  _book_side_replenished, using ctx.orderbook_history (NOT ctx.orderbook,
  which is only ever the latest snapshot).

  Phase C — trigger. Only fires the moment price actually breaks the
  local range of the last `confirm_lookback_sec` in the candidate
  direction (_breaks_local_range) — scanning continuously rather than
  waiting for one assigned checkpoint, same mechanism adaptive_timing
  uses (see StrategyConfig.dynamic_timing — required for this strategy
  too, and for the same reason: it places at most one trade per market,
  whenever this whole sequence lines up, not a bet at every checkpoint).

On the EV/p_BE framing and the ML question from the original write-up:
this strategy emits a raw directional Signal + confidence like
breakout_retest/orderbook_momentum/volatility_breakout do — NOT a
calibrated win probability like fair_value_edge/adaptive_timing. The
engine's existing max_coefficient gate (reject a signal priced too high)
is this strategy's practical "don't pay more than the setup is worth"
filter, same as every other directional strategy already relies on;
building a genuinely calibrated p_BE model (let alone the CatBoost/
LightGBM model the original write-up describes) needs a labeled
historical dataset this bot doesn't have yet — see storage.py's
checkpoint_features table, which is exactly the log meant to eventually
support that, once there's enough history logged. This module is the
honest, buildable slice of the idea today: a rule-based absorption
detector, not the full ML pipeline.
"""
from __future__ import annotations

import math
import statistics
from typing import Optional

from ..features import pct_change_over
from ..models import Direction, OrderBookSnapshot, PricePoint, TradePrint
from .base import BaseStrategy, Signal, StrategyContext


def compute_tfi(prints: list[TradePrint]) -> Optional[float]:
    """Trade-Flow-Imbalance over whatever prints are already filtered to
    the caller's lookback window. None if there's no volume to divide by
    (e.g. an empty/all-zero-size window)."""
    buy_vol = sum(p.size for p in prints if p.side == "buy")
    sell_vol = sum(p.size for p in prints if p.side == "sell")
    total = buy_vol + sell_vol
    if total <= 0:
        return None
    return (buy_vol - sell_vol) / total


def realized_vol_pct_per_sqrt_sec(points: list[PricePoint]) -> Optional[float]:
    """Realized volatility of `points` (already filtered to a lookback
    window by the caller), as a % per sqrt(second) — scale by
    sqrt(horizon_sec) to get an expected-move magnitude over a specific
    horizon. None if there isn't enough history to estimate from (same
    shape as fair_value_edge.compute_barrier_stats's own guard, but this
    is a plain unconditional vol estimate — no strike/barrier involved)."""
    if len(points) < 10:
        return None
    returns = []
    for i in range(1, len(points)):
        p0, p1 = points[i - 1].price, points[i].price
        if p0 > 0 and p1 > 0:
            returns.append(math.log(p1 / p0))
    if len(returns) < 8:
        return None
    sigma = statistics.pstdev(returns)
    if sigma <= 0:
        return None
    span_sec = points[-1].ts - points[0].ts
    n_intervals = len(points) - 1
    if span_sec <= 0 or n_intervals <= 0:
        return None
    avg_dt = span_sec / n_intervals
    if avg_dt <= 0:
        return None
    sigma_per_sec = sigma / math.sqrt(avg_dt)
    return sigma_per_sec * 100


def expected_return_pct(tfi: float, vol_pct_per_sqrt_sec: float, horizon_sec: float, sensitivity: float) -> float:
    """The move TFI's own pressure alone would "predict" over `horizon_sec`
    — TFI=+1 (pure one-sided buying) with `sensitivity`=1.0 means "expect
    about one full realized-vol sigma of upward move"; TFI=0 predicts no
    move either way. Compared against the ACTUAL move over the same
    window (see AbsorptionReversalStrategy.evaluate) — a big enough gap
    between the two, in the direction that says the move was suppressed
    rather than amplified, is what "absorption" means here."""
    return tfi * sensitivity * vol_pct_per_sqrt_sec * math.sqrt(horizon_sec)


def book_side_replenish_ratio(
    history: list[OrderBookSnapshot], now: float, lookback_sec: float, depth: int, side: str,
) -> Optional[float]:
    """current_volume / past_volume for the resting `side` ("bid" or
    "ask") over the last `lookback_sec` — None on too little history or a
    degenerate (zero) past volume to divide by. `history` must be
    chronological (oldest first — see market_data.py's
    btc_orderbook_history/_record_orderbook). Exposed separately from
    book_side_replenished (which just thresholds this) so the actual
    ratio can be logged as a diagnostic even when it doesn't clear the
    bar — see AbsorptionReversalStrategy.evaluate's ctx.diagnostics."""
    if len(history) < 2:
        return None
    past = next((book for book in history if now - book.ts <= lookback_sec), history[0])
    current = history[-1]
    past_vol = past.bid_volume(depth) if side == "bid" else past.ask_volume(depth)
    current_vol = current.bid_volume(depth) if side == "bid" else current.ask_volume(depth)
    if past_vol <= 0:
        return None
    return current_vol / past_vol


def book_side_replenished(
    history: list[OrderBookSnapshot], now: float, lookback_sec: float, depth: int, side: str, min_ratio: float,
) -> bool:
    """True if the resting `side` hasn't materially drained — current
    volume is still at least `min_ratio` of what it was back then. False
    (never "confirmed") on too little history — an absorption signal with
    no real confirmation data available is not a confirmed one."""
    ratio = book_side_replenish_ratio(history, now, lookback_sec, depth, side)
    return ratio is not None and ratio >= min_ratio


def breaks_local_range(points: list[PricePoint], direction: Direction) -> bool:
    """True the moment the LATEST point breaks outside the range of
    everything before it in `points` (already filtered to the caller's
    lookback window) — UP means a new local high, DOWN a new local low.
    This is Phase C's entry trigger: continuous scanning rather than one
    assigned checkpoint means the strategy just keeps checking "has it
    broken out YET" every time it's called, same as adaptive_timing."""
    if len(points) < 5:
        return False
    current = points[-1].price
    prior = [p.price for p in points[:-1]]
    return current > max(prior) if direction == Direction.UP else current < min(prior)


class AbsorptionReversalStrategy(BaseStrategy):
    name = "absorption_reversal"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        # Written to progressively as each phase below runs, whether or
        # not this call ends up returning a Signal — the engine logs
        # whatever's in here into checkpoint_features' extra_json column
        # (see Engine._record_checkpoint_features), so a no_signal row
        # still carries the actual numbers this decision was based on,
        # not just the generic barrier-model fields every strategy gets.
        diag = ctx.diagnostics

        if ctx.already_open_this_market:
            return None  # already committed to this market at an earlier checkpoint

        book = ctx.orderbook
        if book is None or not book.bids or not book.asks:
            return None
        best_bid, best_ask = book.best_bid(), book.best_ask()
        if not best_bid or not best_ask or best_bid <= 0:
            return None
        max_spread_pct = float(self.config.get("max_spread_pct", 0.15))
        spread_pct = (best_ask - best_bid) / best_bid * 100
        diag["spread_pct"] = spread_pct
        if spread_pct > max_spread_pct:
            return None  # book too wide/thin right now to trust anything below

        if not ctx.price_history or not ctx.trade_prints:
            return None
        now = max(ctx.price_history[-1].ts, ctx.trade_prints[-1].ts)

        tfi_lookback_sec = float(self.config.get("tfi_lookback_sec", 90))
        min_abs_tfi = float(self.config.get("min_abs_tfi", 0.5))
        min_prints = int(self.config.get("min_prints", 8))

        window_prints = [p for p in ctx.trade_prints if now - p.ts <= tfi_lookback_sec]
        diag["n_prints"] = len(window_prints)
        if len(window_prints) < min_prints:
            return None
        tfi = compute_tfi(window_prints)
        diag["tfi"] = tfi
        if tfi is None or abs(tfi) < min_abs_tfi:
            return None

        # Volume elevated vs a longer baseline — "продают активно" needs
        # real size behind it, not just a lopsided handful of dust prints.
        baseline_lookback_sec = float(self.config.get("baseline_lookback_sec", 600))
        min_volume_multiple = float(self.config.get("min_volume_multiple", 1.5))
        baseline_prints = [p for p in ctx.trade_prints if now - p.ts <= baseline_lookback_sec]
        baseline_rate = sum(p.size for p in baseline_prints) / baseline_lookback_sec if baseline_prints else 0.0
        current_rate = sum(p.size for p in window_prints) / tfi_lookback_sec
        volume_multiple = (current_rate / baseline_rate) if baseline_rate > 0 else None
        diag["volume_multiple"] = volume_multiple
        if baseline_rate <= 0 or current_rate < baseline_rate * min_volume_multiple:
            return None

        # Residual: actual move over the window vs what TFI's own
        # pressure alone would predict — the DIVERGENCE is the absorption.
        actual_return_pct = pct_change_over(list(ctx.price_history), now, tfi_lookback_sec)
        vol_window = [p for p in ctx.price_history if now - p.ts <= baseline_lookback_sec]
        vol_pct = realized_vol_pct_per_sqrt_sec(vol_window)
        if actual_return_pct is None or vol_pct is None:
            return None
        sensitivity = float(self.config.get("residual_sensitivity", 1.0))
        predicted_pct = expected_return_pct(tfi, vol_pct, tfi_lookback_sec, sensitivity)
        residual = actual_return_pct - predicted_pct
        diag["actual_return_pct"] = actual_return_pct
        diag["predicted_return_pct"] = predicted_pct
        diag["residual_pct"] = residual

        min_residual_pct = float(self.config.get("min_residual_pct", 0.03))
        if tfi <= -min_abs_tfi and residual >= min_residual_pct:
            candidate = Direction.UP
        elif tfi >= min_abs_tfi and residual <= -min_residual_pct:
            candidate = Direction.DOWN
        else:
            return None
        diag["candidate_direction"] = candidate.value

        # Phase B: the side that was under attack must have stayed
        # replenished, not just been a wall that's already been pulled.
        confirm_lookback_sec = float(self.config.get("confirm_lookback_sec", 30))
        min_replenish_ratio = float(self.config.get("min_replenish_ratio", 0.7))
        book_depth = int(self.config.get("book_depth_levels", 10))
        confirm_side = "bid" if candidate == Direction.UP else "ask"
        replenish_ratio = book_side_replenish_ratio(
            list(ctx.orderbook_history), now, confirm_lookback_sec, book_depth, confirm_side,
        )
        diag["replenish_ratio"] = replenish_ratio
        if replenish_ratio is None or replenish_ratio < min_replenish_ratio:
            return None

        # Phase C: only actually enter once price breaks the recent local
        # range in the candidate direction — not the moment absorption
        # was merely detected.
        trigger_lookback_sec = float(self.config.get("trigger_lookback_sec", 30))
        trigger_points = [p for p in ctx.price_history if now - p.ts <= trigger_lookback_sec]
        broke_out = breaks_local_range(trigger_points, candidate)
        diag["broke_local_range"] = broke_out
        if not broke_out:
            return None

        confidence = min(0.9, 0.4 + abs(residual) / (min_residual_pct * 4))
        return Signal(
            direction=candidate,
            reason=(
                f"absorption: TFI={tfi:+.2f} residual={residual:+.3f}% "
                f"(actual={actual_return_pct:+.3f}% vs expected={predicted_pct:+.3f}%), "
                f"vol_rate={current_rate:.3f} (baseline={baseline_rate:.3f})"
            ),
            confidence=confidence,
        )
