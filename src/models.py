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

    def price_for(self, direction: Direction) -> Optional[float]:
        if self.up_price is None:
            return None
        return self.up_price if direction is Direction.UP else round(1 - self.up_price, 4)

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
