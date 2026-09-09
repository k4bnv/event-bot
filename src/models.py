"""Shared data structures used across the whole bot."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Direction(str, Enum):
    UP = "up"      # betting the underlying finishes ABOVE the reference/strike
    DOWN = "down"  # betting the underlying finishes BELOW the reference/strike


class TradeStatus(str, Enum):
    OPEN = "open"
    WON = "won"
    LOST = "lost"
    UNRESOLVED = "unresolved"   # expiry passed but settlement could not be confirmed
    REJECTED = "rejected"       # signal fired but wallet couldn't afford it / API error


@dataclass
class PricePoint:
    ts: float
    price: float


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    ts: float
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)

    def bid_volume(self, depth: int) -> float:
        return sum(l.size for l in self.bids[:depth])

    def ask_volume(self, depth: int) -> float:
        return sum(l.size for l in self.asks[:depth])

    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None


def simulate_market_fill(levels: list[OrderBookLevel], budget_usd: float) -> tuple[Optional[float], float, float, bool]:
    """Walk order-book price levels (BEST PRICE FIRST — the order OKX's
    /market/books already returns them in) simulating a market order that
    spends up to budget_usd. Each contract at a level costs `price` (the
    0.01-0.99 probability IS the per-contract USDT cost), so a level of
    `size` contracts costs `price * size` USDT.

    Returns (vwap_price, contracts_filled, usd_spent, fully_filled).
    fully_filled=False means the visible book didn't have enough depth to
    absorb budget_usd at all — a real order that size would walk even
    deeper / partially fail, i.e. worse than what's computed here.

    Verified against live OKX data (run.py's --check-liquidity, same
    logic originally): on these thin event-contract books, a $20 market
    order routinely fills 40-1000%+ away from the top-of-book/last price —
    see that command's docstring and this repo's history for the numbers
    this was checked against before being wired into the engine itself."""
    contracts, spent = 0.0, 0.0
    for level in levels:
        price, size = level.price, level.size
        if price <= 0 or size <= 0:
            continue
        remaining_budget = budget_usd - spent
        if remaining_budget <= 0:
            break
        level_cost = price * size
        if level_cost <= remaining_budget:
            contracts += size
            spent += level_cost
        else:
            take = remaining_budget / price
            contracts += take
            spent += remaining_budget
            break
    fully_filled = spent >= budget_usd - 1e-6
    vwap = (spent / contracts) if contracts > 0 else None
    return vwap, contracts, spent, fully_filled


@dataclass
class EventMarket:
    """One rolling BTC event-contract window, matching OKX's real EVENTS
    schema (`GET /api/v5/public/event-contract/markets`): a single
    tradable `inst_id` per (series, expiry [, strike]) — NOT a pair of
    UP/DOWN instruments. Which side you bet is chosen at order time via the
    `outcome` field (UP/YES vs DOWN/NO map to the same instId), and `px` is
    the market-implied probability of the *primary* side (UP for
    `price_up_down` series, YES for `price_above`/`hit`).
    The complementary side's price is simply `1 - px` (see `price_for`).

    See README "Event Contracts — реальная схема API" and
    https://github.com/okx/agent-trade-kit (packages/core/src/tools/event-trade.ts)
    for the source this was verified against.
    """
    series_id: str
    method: str            # "price_up_down" | "price_above" | "hit"
    inst_id: str
    expiry_ts: float
    floor_strike: Optional[float] = None   # strike price, for price_above/hit
    up_price: Optional[float] = None       # px: probability the primary side (UP/YES) wins
    state: str = "live"                    # preopen | live | settling | expired
    book: Optional[OrderBookSnapshot] = None   # this instrument's own live order book (not the underlying's)

    def price_for(self, direction: Direction) -> Optional[float]:
        """Naive quoted price (last/mid, ignoring depth) — what strategies
        compare against to judge whether a contract looks cheap/mispriced
        (see e.g. fair_value_edge, which reads `up_price` directly for the
        same reason). NOT what a real order actually fills at on a thin
        book — see `fill_price_for` for that."""
        if self.up_price is None:
            return None
        return self.up_price if direction is Direction.UP else round(1 - self.up_price, 4)

    def fill_price_for(self, direction: Direction, stake_usd: float) -> Optional[float]:
        """Honest expected entry price for actually committing `stake_usd`
        right now: for UP, the real VWAP fill walking this instrument's own
        live ask depth (verified against OKX data — see
        `simulate_market_fill`'s docstring: routinely 40-1000%+ away from
        `up_price` on these thin books). For DOWN there's no public depth
        to walk (OKX doesn't expose a separate DOWN book, and treating the
        UP book's bids as sellable DOWN depth was tried and produced
        nonsense — see git history), so this falls back to the best real
        quote available: `1 - best_bid` (top-of-book only, NOT
        depth-adjusted — a real fill is only ever worse/higher than this).

        Falls back to the naive `price_for` when no book is attached at
        all (mock mode, or a tick where the book fetch came back empty) —
        callers get a usable price either way rather than silently
        blocking every trade whenever depth data is momentarily missing.
        """
        if self.book is None:
            return self.price_for(direction)

        if direction is Direction.UP:
            if not self.book.asks:
                return self.price_for(direction)
            vwap, _, _, _ = simulate_market_fill(self.book.asks, stake_usd)
            return round(vwap, 4) if vwap is not None else self.price_for(direction)

        best_bid = self.book.best_bid()
        if best_bid is None:
            return self.price_for(direction)
        return round(1 - best_bid, 4)

    def remaining_sec(self, now: Optional[float] = None) -> float:
        return self.expiry_ts - (now if now is not None else time.time())


@dataclass
class Trade:
    strategy: str
    entry_window_min: int
    series_id: str
    inst_id: str
    direction: Direction
    entry_price: float          # coefficient paid, e.g. 0.35
    stake_usd: float            # $ committed
    contracts: float            # stake_usd / entry_price
    opened_ts: float
    expiry_ts: float
    reason: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: TradeStatus = TradeStatus.OPEN
    closed_ts: Optional[float] = None
    pnl_usd: Optional[float] = None

    def payout_usd(self) -> float:
        """$1 per contract if this trade's direction won, else $0."""
        return self.contracts * 1.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["direction"] = self.direction.value
        d["status"] = self.status.value
        return d
