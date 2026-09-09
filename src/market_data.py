"""
Market data providers.

`MarketDataProvider` is the interface the strategy engine talks to. Two
implementations exist:

  * `OkxMarketDataProvider` — real OKX data, verified against OKX's own
    open-source reference client (https://github.com/okx/agent-trade-kit,
    packages/core/src/tools/event-trade.ts / event-helpers.ts). BTC/USDT
    spot ticker/orderbook use OKX's ordinary public market-data endpoints.
    Event-contract quotes/settlement use `/api/v5/public/event-contract/*`
    — note these require an authenticated (signed) request despite the
    "/public/" path, and each market has exactly one `instId` (direction
    is chosen at order time via `outcome`, not by picking a different
    instrument) — see `okx_client.py`'s module docstring.

  * `MockMarketDataProvider` (mock_market.py) — fully offline synthetic
    generator with the exact same interface, for strategy testing without
    any API access.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Optional

from .models import Direction, EventMarket, OrderBookLevel, OrderBookSnapshot, PricePoint
from .okx_client import OKXClient, OKXAPIError, OKXNetworkError

logger = logging.getLogger("okx_event_bot.market_data")

PRICE_HISTORY_MAXLEN = 1200  # ~1hr at 3s polling


class MarketDataProvider(ABC):
    def __init__(self) -> None:
        self._price_history: Deque[PricePoint] = deque(maxlen=PRICE_HISTORY_MAXLEN)
        self._orderbook: Optional[OrderBookSnapshot] = None
        self._active_markets: dict[str, EventMarket] = {}
        self._funding_rate: Optional[float] = None  # BTC perp funding rate, e.g. 0.0001 = 0.01%

    @abstractmethod
    async def refresh(self) -> None:
        """Pull the latest underlying price/orderbook and event-market quotes."""

    @abstractmethod
    async def check_settlement(self, series_id: str, inst_id: str) -> Optional[Direction]:
        """Return the Direction that won (UP or DOWN) once this instId has
        settled, else None (caller should retry)."""

    def btc_price_history(self) -> Deque[PricePoint]:
        return self._price_history

    def btc_orderbook(self) -> Optional[OrderBookSnapshot]:
        return self._orderbook

    def active_markets(self) -> dict[str, EventMarket]:
        return self._active_markets

    def funding_rate(self) -> Optional[float]:
        """Current BTC perpetual funding rate (fraction, e.g. 0.0001), or
        None if unavailable/not yet fetched. Used by the funding_skew
        strategy."""
        return self._funding_rate


def _to_epoch_sec(raw) -> Optional[float]:
    if raw is None or raw == "":
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v / 1000.0 if v > 1e12 else v  # OKX timestamps are ms


class OkxMarketDataProvider(MarketDataProvider):
    def __init__(
        self, client: OKXClient, underlying_inst_id: str, series_ids: list[str],
        funding_inst_id: str = "BTC-USDT-SWAP", funding_refresh_interval_sec: float = 300.0,
    ):
        super().__init__()
        self.client = client
        self.underlying_inst_id = underlying_inst_id
        self.series_ids = series_ids
        self.funding_inst_id = funding_inst_id
        self.funding_refresh_interval_sec = funding_refresh_interval_sec
        self._series_method: dict[str, str] = {}  # series_id -> settlement.method, cached
        self._last_funding_fetch_ts = 0.0

    async def refresh(self) -> None:
        await self._refresh_underlying()
        await self._maybe_refresh_funding()
        for series_id in self.series_ids:
            try:
                await self._refresh_series(series_id)
            except (OKXAPIError, OKXNetworkError) as exc:
                logger.error("Failed refreshing EVENTS series %s: %s", series_id, exc)

    async def _maybe_refresh_funding(self) -> None:
        # Funding settles every 8h in reality — no need to hammer this
        # endpoint every poll tick.
        now = time.time()
        if now - self._last_funding_fetch_ts < self.funding_refresh_interval_sec:
            return
        try:
            data = await self.client.get_funding_rate(self.funding_inst_id)
            if data:
                self._funding_rate = float(data[0]["fundingRate"])
            self._last_funding_fetch_ts = now
        except (OKXAPIError, OKXNetworkError, KeyError, ValueError) as exc:
            logger.warning("Failed refreshing funding rate for %s: %s", self.funding_inst_id, exc)

    async def _refresh_underlying(self) -> None:
        try:
            data = await self.client.get_ticker(self.underlying_inst_id)
            if data:
                px = float(data[0]["last"])
                self._price_history.append(PricePoint(ts=time.time(), price=px))

            book = await self.client.get_orderbook(self.underlying_inst_id, sz=20)
            if book:
                raw = book[0]
                bids = [OrderBookLevel(price=float(p), size=float(s)) for p, s, *_ in raw.get("bids", [])]
                asks = [OrderBookLevel(price=float(p), size=float(s)) for p, s, *_ in raw.get("asks", [])]
                self._orderbook = OrderBookSnapshot(ts=time.time(), bids=bids, asks=asks)
        except (OKXAPIError, OKXNetworkError) as exc:
            logger.error("Failed refreshing underlying %s: %s", self.underlying_inst_id, exc)

    async def _series_settlement_method(self, series_id: str) -> Optional[str]:
        if series_id in self._series_method:
            return self._series_method[series_id]
        rows = await self.client.get_event_series(series_id=series_id)
        for row in rows:
            if row.get("seriesId") == series_id:
                method = (row.get("settlement") or {}).get("method")
                if method:
                    self._series_method[series_id] = method
                    return method
        return None

    async def _fetch_event_price(self, inst_id: str) -> Optional[float]:
        """The event contract's own probability/price (0.01-0.99), from the
        standard market/ticker endpoint applied to its instId. Prefers
        `last` (most recently traded price); falls back to the bid/ask
        midpoint if there's a live spread but no trade yet. None if neither
        is available (e.g. genuinely no liquidity on this contract right
        now) — callers must treat that as "no live quote", not an error."""
        try:
            data = await self.client.get_ticker(inst_id)
        except (OKXAPIError, OKXNetworkError) as exc:
            logger.warning("Failed fetching event contract ticker for %s: %s", inst_id, exc)
            return None
        if not data:
            return None
        row = data[0]

        last_raw = row.get("last")
        if last_raw not in (None, ""):
            try:
                return float(last_raw)
            except ValueError:
                pass

        bid_raw, ask_raw = row.get("bidPx"), row.get("askPx")
        if bid_raw not in (None, "") and ask_raw not in (None, ""):
            try:
                return (float(bid_raw) + float(ask_raw)) / 2
            except ValueError:
                pass
        return None

    async def _fetch_event_book(self, inst_id: str) -> Optional[OrderBookSnapshot]:
        """This event contract's own live order book (not the underlying's)
        — lets `EventMarket.fill_price_for` simulate a real VWAP fill
        instead of trading at the naive last/mid price. Verified: on these
        thin books, a real $20 order routinely fills 40-1000%+ away from
        that naive price (see run.py's --check-liquidity and this repo's
        history). None on any failure/empty response — callers must treat
        that as "no depth data this tick", not an error, and fall back to
        the naive price (EventMarket.fill_price_for already does this)."""
        try:
            book = await self.client.get_orderbook(inst_id, sz=20)
        except (OKXAPIError, OKXNetworkError) as exc:
            logger.warning("Failed fetching event contract book for %s: %s", inst_id, exc)
            return None
        if not book:
            return None
        raw = book[0]
        try:
            bids = [OrderBookLevel(price=float(p), size=float(s)) for p, s, *_ in raw.get("bids", [])]
            asks = [OrderBookLevel(price=float(p), size=float(s)) for p, s, *_ in raw.get("asks", [])]
        except (TypeError, ValueError):
            return None
        return OrderBookSnapshot(ts=time.time(), bids=bids, asks=asks)

    async def _refresh_series(self, series_id: str) -> None:
        markets = await self.client.get_event_markets(series_id=series_id, state="live")
        if not markets:
            return  # nothing currently tradeable for this series right now

        now = time.time()
        candidates = []
        for m in markets:
            exp = _to_epoch_sec(m.get("expTime"))
            if exp is None or exp <= now:
                continue
            candidates.append((exp, m))
        if not candidates:
            return

        nearest_expiry = min(exp for exp, _ in candidates)
        same_expiry = [m for exp, m in candidates if exp == nearest_expiry]

        method = self._series_method.get(series_id) or await self._series_settlement_method(series_id)
        method = method or "price_up_down"

        chosen = same_expiry[0]
        if len(same_expiry) > 1:
            # Multiple strikes share this expiry (price_above / hit series
            # commonly list several strikes at once). Default to the strike nearest the
            # current underlying price ("at the money") as the one contract this window
            # trades — a defensible, deterministic pick for a bot that trades one
            # contract per rolling window. If you want a specific strike, filter
            # `series_ids`/extend this method to select by floorStrike yourself.
            underlying_px = self._price_history[-1].price if self._price_history else None
            if underlying_px is not None:
                def strike_distance(m: dict) -> float:
                    try:
                        return abs(float(m.get("floorStrike", underlying_px)) - underlying_px)
                    except (TypeError, ValueError):
                        return float("inf")
                chosen = min(same_expiry, key=strike_distance)

        inst_id = str(chosen.get("instId", ""))
        # Verified against a live (non-demo) response (2026-09): the
        # /event-contract/markets listing itself carries NO price field at
        # all (capStrike/floorStrike/outcome/state/etc., but nothing like
        # "px") — contradicts what OKX's own agent-trade-kit docs implied.
        # The actual live price/probability comes from the ordinary
        # `GET /api/v5/market/ticker?instId=...` endpoint, same one used
        # for spot, applied to the event contract's own instId — its `last`
        # field IS the 0.01-0.99 probability.
        up_price = await self._fetch_event_price(inst_id)
        book = await self._fetch_event_book(inst_id)

        # Verified against a live response (2026-09): OKX sends
        # floorStrike="0" as a PLACEHOLDER before the window's reference
        # price is actually fixed — fixTime is empty at the same time, and
        # both get populated together once fixing happens. Treating "0" as
        # a real $0 strike (float("0") == 0.0, not falsy-checked away by a
        # None/"" check) was a real bug: every UPDOWN window would silently
        # carry a bogus strike until OKX happened to fix it.
        is_fixed = str(chosen.get("fixTime") or "").strip() not in ("", "0")
        try:
            floor_strike_raw = chosen.get("floorStrike")
            floor_strike = float(floor_strike_raw) if floor_strike_raw not in (None, "") else None
        except (TypeError, ValueError):
            floor_strike = None
        if not is_fixed:
            floor_strike = None

        # price_up_down contracts have no fixed strike until OKX fixes one
        # (see above) — "UP" just means "higher than the window's reference
        # price". Until that happens we need SOME fixed reference of our
        # own to compute a fair-value probability against (fair_value_edge
        # strategy), so capture the underlying price the first time we see
        # this window and keep it for the window's whole lifetime —
        # recomputing it every tick would keep comparing "now" to "now".
        # Once OKX's own floorStrike is fixed, prefer that (more authoritative).
        existing = self._active_markets.get(series_id)
        is_same_window = existing is not None and existing.inst_id == inst_id
        if method == "price_up_down" and floor_strike is None:
            if is_same_window:
                floor_strike = existing.floor_strike
            elif self._price_history:
                floor_strike = self._price_history[-1].price

        self._active_markets[series_id] = EventMarket(
            series_id=series_id, method=method, inst_id=inst_id, expiry_ts=nearest_expiry,
            floor_strike=floor_strike, up_price=up_price, state=str(chosen.get("state", "live")),
            book=book,
        )

    async def check_settlement(self, series_id: str, inst_id: str) -> Optional[Direction]:
        try:
            rows = await self.client.get_event_markets(series_id=series_id, inst_id=inst_id, state="expired")
        except (OKXAPIError, OKXNetworkError) as exc:
            logger.warning("Settlement check failed for %s/%s: %s", series_id, inst_id, exc)
            return None
        for row in rows:
            if row.get("instId") != inst_id:
                continue
            outcome = str(row.get("outcome", "0"))
            if outcome == "1":
                return Direction.UP
            if outcome == "2":
                return Direction.DOWN
        return None  # not settled yet (outcome "0"/missing), or not found — retry
