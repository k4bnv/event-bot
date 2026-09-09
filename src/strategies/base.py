"""Strategy interface. Every strategy is a function of market context -> Optional[Signal].

Strategies never touch wallets, orders or the network directly for trading
purposes — they only look at market context and say "I want to bet UP or
DOWN here, because X". The engine is responsible for sizing, affordability,
and execution. `evaluate()` is async because at least one strategy
(ai_prompt) needs to make a network call to an LLM API; the rest simply
don't `await` anything.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Deque, Optional

from ..models import Direction, EventMarket, OrderBookSnapshot, PricePoint


@dataclass
class Signal:
    direction: Direction
    reason: str
    confidence: float = 0.5  # 0..1, informational only (not used for sizing yet)


@dataclass
class StrategyContext:
    """Everything a strategy might need for one evaluate() call. Bundled
    into one object (rather than a growing positional-argument list) so
    adding a new data source doesn't require touching every strategy's
    signature."""
    price_history: Deque[PricePoint]     # underlying (e.g. BTC-USDT) price samples
    orderbook: Optional[OrderBookSnapshot]
    remaining_sec: float                  # time left to this contract's expiry
    window_min: int                       # which configured entry checkpoint fired
    market: EventMarket                    # the event contract itself (px, strike, method, ...)
    funding_rate: Optional[float] = None   # current BTC perp funding rate, if available
    # The winning Direction of the window immediately BEFORE this series'
    # current one, if known — Engine._update_previous_outcomes() looks
    # this up the moment a window rolls over, independent of whether any
    # strategy actually held a position in it. None if not known yet
    # (engine just started) or a lookup gave up after retrying.
    previous_outcome: Optional[Direction] = None
    # True when this strategy already has an OPEN trade in this EXACT
    # market (same series_id + expiry_ts). Most strategies fire at a few
    # fixed checkpoints and are fine stacking a separate bet at each one —
    # this only matters to a strategy meant to place AT MOST ONE trade per
    # market despite being called at many checkpoints (see
    # adaptive_timing, which scans a dense grid and enters at whichever
    # one first looks good); such a strategy checks this and returns None
    # once it's already positioned, so it naturally retries at the next
    # checkpoint if an earlier signal got rejected (no live quote, low
    # balance, ...) rather than giving up on the market for good.
    already_open_this_market: bool = False


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        """Called once per due entry-window checkpoint. Return a Signal to
        request opening a trade, or None to sit this checkpoint out."""

    async def aclose(self) -> None:
        """Release any resources (e.g. an HTTP session) held by this
        strategy instance. Default no-op; override if you allocate
        something in __init__/lazily in evaluate(). Called once during
        shutdown by run.py."""
