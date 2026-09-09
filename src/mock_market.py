"""
Fully offline synthetic market data provider.

Generates a random-walk BTC/USDT price (with occasional injected impulses so
the breakout strategy has something real to detect), a synthetic order book,
and rolling event-contract windows per configured series (duration inferred
from the series id, e.g. "...-15MIN" -> 15 minutes). Coefficients are derived
from a simple logistic model of (price vs strike). Ground truth for
settlement is known exactly (we generated it), so this doubles as a
lightweight strategy backtest harness with zero external dependencies.

Mirrors OKX's real Event Contract shape (one `inst_id` per window; UP/DOWN is
chosen at trade time, not via separate instruments) — see `models.EventMarket`
and `okx_client.py`'s module docstring for where that schema was verified.
"""
from __future__ import annotations

import datetime as dt
import math
import random
import time
from typing import Optional

from .market_data import MarketDataProvider
from .models import Direction, EventMarket, OrderBookLevel, OrderBookSnapshot, PricePoint, TradePrint


def _series_duration_sec(series_id: str) -> float:
    s = series_id.upper()
    if "15" in s:
        return 15 * 60
    if "1H" in s or "60" in s:
        return 60 * 60
    return 5 * 60  # default: 5-minute window


class MockMarketDataProvider(MarketDataProvider):
    def __init__(self, series_ids: list[str], start_price: float = 62000.0, seed: Optional[int] = None):
        super().__init__()
        self._rng = random.Random(seed)
        self.price = start_price
        self.series_ids = series_ids
        self._windows: dict[str, dict] = {}                    # series_id -> {strike, expiry_ts, duration}
        self._settlement_truth: dict[str, Direction] = {}        # inst_id -> winning Direction
        self._tick = 0
        self._funding_rate = 0.0    # synthetic BTC perp funding rate, mean-reverting around 0
        # Occasionally-injected synthetic "absorption" episode — see
        # _maybe_start_absorption/_step_trade_prints — so
        # absorption_reversal has a genuine pattern to detect in mock/dev
        # mode: heavy one-sided aggressor flow while price is deliberately
        # damped, i.e. exactly the setup that strategy looks for. None
        # when no episode is active.
        self._absorption_state: Optional[dict] = None
        for sid in series_ids:
            self._roll_window(sid)

    # -- window / instrument bookkeeping -------------------------------------------
    def _make_inst_id(self, series_id: str, start_ts: float, expiry_ts: float) -> str:
        start = dt.datetime.fromtimestamp(start_ts, dt.timezone.utc)
        end = dt.datetime.fromtimestamp(expiry_ts, dt.timezone.utc)
        return f"{series_id}-{start:%y%m%d}-{start:%H%M}-{end:%H%M}"

    def _roll_window(self, series_id: str) -> None:
        duration = _series_duration_sec(series_id)
        now = time.time()
        expiry_ts = now + duration
        strike = self.price
        self._windows[series_id] = {"strike": strike, "expiry_ts": expiry_ts, "duration": duration}
        inst_id = self._make_inst_id(series_id, now, expiry_ts)
        self._active_markets[series_id] = EventMarket(
            series_id=series_id, method="price_up_down", inst_id=inst_id,
            expiry_ts=expiry_ts, floor_strike=strike, state="live",
            strike_is_fixed=True,  # generated ground truth, never a proxy
        )

    def _finalize_window(self, series_id: str) -> None:
        market = self._active_markets[series_id]
        win = self._windows[series_id]
        outcome = Direction.UP if self.price >= win["strike"] else Direction.DOWN
        self._settlement_truth[market.inst_id] = outcome
        self._roll_window(series_id)

    # -- price model -----------------------------------------------------------------
    def _step_price(self, damp: bool = False) -> None:
        sigma = 0.0006
        if damp:
            # An absorption episode is, by construction, pressure the
            # price is (for now) RESISTING — heavily muted realized vol is
            # exactly the "price barely moves despite heavy flow" setup
            # absorption_reversal's Phase A looks for.
            sigma *= 0.15
        drift = self._rng.gauss(0, sigma)
        # ~1.5% chance per tick of an "impulse" move, gives breakout strategy
        # signal — suppressed during a damped tick, which would defeat the
        # whole point of it.
        if not damp and self._rng.random() < 0.015:
            drift += self._rng.choice([-1, 1]) * self._rng.uniform(0.004, 0.012)
        # Geometric (log-normal) step, NOT price *= (1 + drift): the naive
        # arithmetic version has a systematic downward "volatility drag" bias
        # over many ticks — E[log(1 + X)] = -Var(X)/2 < 0 even though
        # E[X] = 0 — which silently drags a long-running bot's synthetic BTC
        # price toward zero. Subtracting sigma^2/2 keeps this a driftless
        # (martingale) random walk no matter how long the bot runs.
        self.price *= math.exp(drift - (sigma ** 2) / 2)
        self._tick += 1

    def _maybe_start_absorption(self) -> None:
        """~1% chance per tick to start a new synthetic absorption episode
        (skipped while one's already running) — lasts 15-30 ticks. Picks
        which side dominates the aggressor flow for the whole episode;
        _step_price's damping and _step_trade_prints'/refresh()'s book
        skew all read this same state so price/flow/book all agree with
        each other for the episode's duration, like a real absorption
        setup would."""
        if self._absorption_state is not None or self._rng.random() >= 0.01:
            return
        self._absorption_state = {
            "dominant_side": self._rng.choice(["sell", "buy"]),
            "ticks_left": self._rng.randint(15, 30),
        }

    def _step_trade_prints(self, price_before: float) -> None:
        """Synthetic aggressor-classified trade prints for this tick — see
        models.TradePrint. Normal case: aggressor mix loosely tracks
        whichever way price just moved (a crude "trades caused the move"
        baseline) plus noise. During an active absorption episode, flow is
        deliberately heavier and far more one-sided than the (deliberately
        damped) price move alone would suggest — that divergence IS the
        pattern absorption_reversal looks for. Real OKX trade prints
        (market_data.py's OkxMarketDataProvider) replace this entirely in
        live mode; this only exists for mock/dev testing."""
        now = time.time()
        n_prints = self._rng.randint(3, 8)

        if self._absorption_state is not None:
            buy_prob = 0.15 if self._absorption_state["dominant_side"] == "sell" else 0.85
            size_mult = 2.5
            self._absorption_state["ticks_left"] -= 1
            if self._absorption_state["ticks_left"] <= 0:
                self._absorption_state = None
        else:
            price_delta_pct = (self.price - price_before) / price_before if price_before else 0.0
            bias = max(-1.0, min(1.0, price_delta_pct * 300 + self._rng.gauss(0, 0.35)))
            buy_prob = 0.5 + bias * 0.4
            size_mult = 1.0

        for _ in range(n_prints):
            side = "buy" if self._rng.random() < buy_prob else "sell"
            size = self._rng.uniform(0.01, 0.25) * size_mult
            self._trade_prints.append(TradePrint(ts=now, price=self.price, size=size, side=side))

    def _step_funding(self) -> None:
        """Mean-reverting synthetic funding rate with occasional larger
        excursions, so funding_skew has something to react to in mock mode."""
        self._funding_rate += self._rng.gauss(0, 0.00002) - self._funding_rate * 0.01
        if self._rng.random() < 0.005:
            self._funding_rate += self._rng.choice([-1, 1]) * self._rng.uniform(0.0002, 0.0006)

    def _coefficient(self, strike: float, remaining_sec: float, duration: float) -> float:
        """Logistic probability the price finishes >= strike, given current
        distance and remaining time (less time left -> price closer to 0/1)."""
        dist_pct = (self.price - strike) / strike
        time_frac = max(remaining_sec, 1.0) / duration       # 1 -> just opened, 0 -> expiring
        vol_scale = 0.01 * math.sqrt(max(time_frac, 0.02))    # shrinks as expiry nears
        z = max(min(dist_pct / vol_scale, 50.0), -50.0)       # clamp: math.exp overflows above ~709
        prob = 1 / (1 + math.exp(-z * 3))
        return min(max(prob, 0.02), 0.98)

    # -- MarketDataProvider interface ------------------------------------------------
    async def refresh(self) -> None:
        self._maybe_start_absorption()
        absorbing = self._absorption_state is not None
        price_before = self.price
        self._step_price(damp=absorbing)
        self._step_funding()
        self._price_history.append(PricePoint(ts=time.time(), price=self.price))

        # synthetic order book around current price, with a random imbalance
        # occasionally injected so the orderbook-momentum strategy has signal.
        if absorbing:
            # Heavy selling absorption -> resting bid stays thick
            # (replenished, not thinning) despite the aggressive flow
            # hitting it — mirror for a buying episode.
            skew = (0.5 if self._absorption_state["dominant_side"] == "sell" else -0.5) * self._rng.uniform(0.6, 1.6)
        else:
            skew = self._rng.uniform(-1, 1) if self._rng.random() < 0.2 else 0.0
        bids, asks = [], []
        for i in range(1, 11):
            bids.append(OrderBookLevel(price=self.price * (1 - 0.0002 * i), size=max(0.05, 1.0 + skew) * self._rng.uniform(0.5, 1.5)))
            asks.append(OrderBookLevel(price=self.price * (1 + 0.0002 * i), size=max(0.05, 1.0 - skew) * self._rng.uniform(0.5, 1.5)))
        self._record_orderbook(OrderBookSnapshot(ts=time.time(), bids=bids, asks=asks))

        # _step_trade_prints may clear self._absorption_state once its
        # ticks_left counts down — must run AFTER the skew above reads it.
        self._step_trade_prints(price_before)

        now = time.time()
        for series_id in self.series_ids:
            win = self._windows[series_id]
            if now >= win["expiry_ts"]:
                self._finalize_window(series_id)
                win = self._windows[series_id]
            market = self._active_markets[series_id]
            remaining = win["expiry_ts"] - now
            market.up_price = round(self._coefficient(win["strike"], remaining, win["duration"]), 3)

    async def check_settlement(self, series_id: str, inst_id: str) -> Optional[Direction]:
        return self._settlement_truth.get(inst_id)
