"""
Metrics: per (strategy, entry-window) statistics and leaderboard detection.

Everything here is derived on demand from the wallets' trade lists — there's
no separate mutable counter state to keep in sync, which keeps this module
trivially correct and easy to unit test.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Trade, TradeStatus
from .wallet import VirtualWallet


@dataclass
class ComboStats:
    strategy: str
    window_min: int
    trades: int
    wins: int
    losses: int
    unresolved: int
    winrate_pct: float
    avg_entry_price: float
    total_staked: float
    net_pnl: float

    @property
    def roi_pct(self) -> float:
        return (self.net_pnl / self.total_staked * 100) if self.total_staked > 0 else 0.0


def build_combo_stats(wallets: dict[str, VirtualWallet]) -> dict[tuple[str, int], ComboStats]:
    buckets: dict[tuple[str, int], list[Trade]] = {}
    for wallet in wallets.values():
        for trade in wallet.trades:
            if trade.status == TradeStatus.REJECTED:
                continue  # never actually opened, not a meaningful data point
            key = (trade.strategy, trade.entry_window_min)
            buckets.setdefault(key, []).append(trade)

    out: dict[tuple[str, int], ComboStats] = {}
    for key, trades in buckets.items():
        closed = [t for t in trades if t.status in (TradeStatus.WON, TradeStatus.LOST)]
        wins = sum(1 for t in closed if t.status == TradeStatus.WON)
        losses = sum(1 for t in closed if t.status == TradeStatus.LOST)
        unresolved = sum(1 for t in trades if t.status == TradeStatus.UNRESOLVED)
        winrate = (wins / len(closed) * 100) if closed else 0.0
        avg_entry = sum(t.entry_price for t in trades) / len(trades) if trades else 0.0
        staked = sum(t.stake_usd for t in trades)
        pnl = sum(t.pnl_usd or 0.0 for t in trades if t.pnl_usd is not None)
        out[key] = ComboStats(
            strategy=key[0], window_min=key[1],
            trades=len(trades), wins=wins, losses=losses, unresolved=unresolved,
            winrate_pct=winrate, avg_entry_price=avg_entry,
            total_staked=staked, net_pnl=pnl,
        )
    return out


@dataclass
class Leaderboard:
    best_pnl: ComboStats | None
    best_winrate: ComboStats | None
    best_value: ComboStats | None   # highest ROI% per $ staked
    min_sample_size: int


def build_leaderboard(combo_stats: dict[tuple[str, int], ComboStats], min_sample_size: int = 5) -> Leaderboard:
    qualified = [c for c in combo_stats.values() if c.trades >= min_sample_size]
    pool = qualified or list(combo_stats.values())  # fall back if nothing has enough samples yet

    best_pnl = max(pool, key=lambda c: c.net_pnl, default=None)
    best_winrate = max(pool, key=lambda c: (c.winrate_pct, c.trades), default=None)
    best_value = max(pool, key=lambda c: c.roi_pct, default=None)

    return Leaderboard(
        best_pnl=best_pnl, best_winrate=best_winrate, best_value=best_value,
        min_sample_size=min_sample_size,
    )
