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
class TradePrint:
    """One executed trade on the underlying (e.g. BTC-USDT spot), from
    OKX's public trade-tape endpoint (GET /market/trades) — NOT the same
    thing as OrderBookSnapshot's resting bid/ask levels. `side` is the
    AGGRESSOR's (taker's) side straight from OKX's own `side` field: "buy"
    means a market buy hit the ask (bullish pressure), "sell" means a
    market sell hit the bid (bearish pressure). This is what a real
    trade-flow-imbalance (TFI) signal needs — a resting order in the book
    can be pulled before it ever trades; an executed print is a fact that
    already happened. See absorption_reversal.py, the first strategy that
    uses this."""
    ts: float
    price: float
    size: float
    side: str  # "buy" | "sell"


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


def simulate_market_sell(levels: list[OrderBookLevel], contracts: float) -> tuple[Optional[float], float, float, bool]:
    """Mirror of `simulate_market_fill` for the EXIT side: walk bid levels
    (BEST PRICE FIRST, same order OKX returns) simulating a market SELL of
    `contracts` contracts. Each contract sold at a level yields `price`
    USDT, so a level of `size` contracts absorbs at most `price * size`.

    Returns (vwap_price, contracts_sold, usd_received, fully_filled).
    fully_filled=False means the visible bids couldn't absorb the whole
    position — the rest would walk deeper still, i.e. a real exit is only
    ever worse than what's computed here.

    The asymmetry with simulate_market_fill is deliberate and not
    cosmetic: a BUY is budget-limited (spend up to $X, receive however
    many contracts that buys), while a SELL is size-limited (you hold
    exactly N contracts and want the proceeds). Passing a dollar budget
    here instead would silently answer a different question.

    Why this matters for an early-exit strategy: entry already pays the
    spread on these thin books (see simulate_market_fill's docstring —
    40-1000%+ away from the naive price on live data), and exiting before
    expiry pays it a SECOND time. This function is what makes that
    round-trip cost measurable instead of assumed — see run.py's
    --check-liquidity, which prints both halves.
    """
    sold, received = 0.0, 0.0
    for level in levels:
        price, size = level.price, level.size
        if price <= 0 or size <= 0:
            continue
        remaining = contracts - sold
        if remaining <= 0:
            break
        take = size if size <= remaining else remaining
        sold += take
        received += take * price
    fully_filled = sold >= contracts - 1e-6
    vwap = (received / sold) if sold > 0 else None
    return vwap, sold, received, fully_filled


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
    # True: floor_strike is OKX's own fixed reference for this window
    # (fixTime set — see market_data.py). False: OKX hasn't fixed one yet
    # and floor_strike is our own proxy (the first price we happened to
    # observe for this window) — same basis risk as a Kalshi bot proxying
    # its settlement reference through Coinbase when the exchange's own
    # reference isn't available yet. None: unknown/not applicable (e.g.
    # mock data, or a method that doesn't use a window-relative strike at
    # all) — treated the same as True, i.e. no extra uncertainty assumed.
    strike_is_fixed: Optional[bool] = None

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

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        """Inverse of to_dict() — rebuilds a Trade from a row Storage
        handed back (e.g. get_trades()), converting direction/status back
        from their plain string form. Used to restore closed-trade
        history into a wallet at startup (see Engine._restore_or_create_wallet)
        so winrate/PnL-per-combo stats don't reset to zero on every
        redeploy the way they used to when only balance was restored."""
        d = dict(d)
        d["direction"] = Direction(d["direction"])
        d["status"] = TradeStatus(d["status"])
        return cls(**d)
