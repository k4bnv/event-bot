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
