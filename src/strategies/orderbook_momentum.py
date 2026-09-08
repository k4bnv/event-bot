"""
Strategy B — Momentum / Order Book Imbalance.

Compares resting bid vs ask volume in the top N order-book levels. A strong
skew toward bids suggests buyers are in control (bet UP); a strong skew
toward asks suggests sellers are in control (bet DOWN).
"""
from __future__ import annotations

from typing import Optional

from ..models import Direction
from .base import BaseStrategy, Signal, StrategyContext


class OrderbookMomentumStrategy(BaseStrategy):
    name = "orderbook_momentum"

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        orderbook = ctx.orderbook
        if orderbook is None or not orderbook.bids or not orderbook.asks:
            return None

        depth = int(self.config.get("orderbook_depth_levels", 10))
        threshold = float(self.config.get("imbalance_threshold", 1.8))

        bid_vol = orderbook.bid_volume(depth)
        ask_vol = orderbook.ask_volume(depth)
        if bid_vol <= 0 or ask_vol <= 0:
            return None

        ratio = bid_vol / ask_vol
        if ratio >= threshold:
            direction, strength = Direction.UP, ratio
        elif ratio <= 1 / threshold:
            direction, strength = Direction.DOWN, 1 / ratio
        else:
            return None

        return Signal(
            direction=direction,
            reason=f"orderbook imbalance {ratio:.2f}x (bid={bid_vol:.2f} ask={ask_vol:.2f}, top {depth})",
            confidence=min(0.9, 0.4 + (strength - threshold) / (threshold * 2)),
        )
