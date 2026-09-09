"""Per-(strategy, entry checkpoint) virtual wallet.

Each strategy owns its own isolated slice of the $100 demo capital — and,
one level deeper, EACH configured entry_windows_min checkpoint within that
strategy gets its OWN independent wallet too (e.g. breakout_retest's "12
мин" and "2 мин" checkpoints compound separately, never sharing or
competing for one stake_fraction-sized pool). OKX's Demo Trading account
itself is a single shared balance — it has no concept of "sub-wallets" at
all — so this isolation is implemented entirely inside the bot. This is
what lets strategy A's checkpoints (and A vs B vs C, ...) run
"independently" against the same OKX market data feed while never
touching each other's capital or being able to over-spend it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .models import Trade, TradeStatus


@dataclass
class VirtualWallet:
    strategy: str
    initial_balance: float
    # None only for a wallet that predates the per-checkpoint split (never
    # constructed that way going forward) — every wallet the engine builds
    # now always sets this to the checkpoint it belongs to.
    window_min: Optional[int] = None
    balance: float = field(init=False)
    reserved: float = 0.0          # capital currently locked in open trades
    trades: list[Trade] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.balance = self.initial_balance

    # -- capacity checks -----------------------------------------------------
    @property
    def free_balance(self) -> float:
        """Balance available for NEW trades (excludes money already staked)."""
        return self.balance

    @property
    def equity(self) -> float:
        """Free balance + capital tied up in still-open trades."""
        return self.balance + self.reserved

    @property
    def net_pnl(self) -> float:
        return self.equity - self.initial_balance

    def can_afford(self, stake_usd: float) -> bool:
        return stake_usd > 0 and stake_usd <= self.balance + 1e-9

    # -- trade lifecycle -------------------------------------------------------
    def open_trade(self, trade: Trade) -> bool:
        if not self.can_afford(trade.stake_usd):
            trade.status = TradeStatus.REJECTED
            self.trades.append(trade)
            return False
        self.balance -= trade.stake_usd
        self.reserved += trade.stake_usd
        self.trades.append(trade)
        return True

    def settle_trade(self, trade: Trade, won: bool) -> None:
        self.reserved -= trade.stake_usd
        payout = trade.payout_usd() if won else 0.0
        self.balance += payout
        trade.pnl_usd = payout - trade.stake_usd
        trade.status = TradeStatus.WON if won else TradeStatus.LOST
        trade.closed_ts = time.time()

    def mark_unresolved(self, trade: Trade) -> None:
        """Settlement couldn't be confirmed in time; return the stake so the
        strategy isn't permanently penalised by an API/data gap."""
        self.reserved -= trade.stake_usd
        self.balance += trade.stake_usd
        trade.pnl_usd = 0.0
        trade.status = TradeStatus.UNRESOLVED
        trade.closed_ts = time.time()

    def open_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.status == TradeStatus.OPEN]

    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.status in (TradeStatus.WON, TradeStatus.LOST)]
