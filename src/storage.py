"""
Persistence: a real embedded database (SQLite, stdlib — no new dependency)
instead of ad-hoc CSV/JSON files. This is what makes two things possible:
  * resetting ONE strategy's history without touching the others
    (DELETE ... WHERE strategy = ?), and
  * cheap analytics queries for the dashboard's charts/trade table
    (equity curves, filtering by strategy, ordering by time) without
    re-parsing a growing CSV by hand.

The trades table is the durable log — deduplication is just the primary
key (`id`) doing an INSERT OR IGNORE, no in-memory "already logged" set
needed. `data/bot.db` can be inspected with any SQLite tool, e.g.:
    sqlite3 data/bot.db "SELECT * FROM trades ORDER BY closed_ts DESC LIMIT 20;"
    sqlite3 -header -csv data/bot.db "SELECT * FROM trades;" > trades.csv
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .wallet import VirtualWallet

logger = logging.getLogger("okx_event_bot.storage")

TRADE_FIELDS = [
    "id", "strategy", "entry_window_min", "series_id", "inst_id", "direction",
    "entry_price", "stake_usd", "contracts", "opened_ts", "expiry_ts",
    "closed_ts", "status", "pnl_usd", "reason",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    entry_window_min REAL,
    series_id TEXT,
    inst_id TEXT,
    direction TEXT,
    entry_price REAL,
    stake_usd REAL,
    contracts REAL,
    opened_ts REAL,
    expiry_ts REAL,
    closed_ts REAL,
    status TEXT,
    pnl_usd REAL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_closed_ts ON trades(closed_ts);

CREATE TABLE IF NOT EXISTS wallets (
    strategy TEXT PRIMARY KEY,
    initial_balance REAL,
    balance REAL,
    reserved REAL,
    updated_at REAL
);
"""


class Storage:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "bot.db"
        # check_same_thread=False: the engine's async tick loop and FastAPI
        # request handlers both run on the same asyncio event loop thread
        # (never truly concurrent Python bytecode), but may be scheduled
        # from different call stacks/tasks; sqlite3's default same-thread
        # check is stricter than we need here.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- writes ----------------------------------------------------------------
    def append_closed_trades(self, wallets: dict[str, VirtualWallet]) -> None:
        """Insert any newly-closed trades (won/lost/unresolved). Dedup is
        just the primary key — INSERT OR IGNORE silently skips a trade
        already written in an earlier tick."""
        rows = []
        for wallet in wallets.values():
            for trade in wallet.trades:
                if trade.closed_ts is None:
                    continue
                d = trade.to_dict()
                rows.append(tuple(d.get(f) for f in TRADE_FIELDS))
        if not rows:
            return
        placeholders = ",".join("?" * len(TRADE_FIELDS))
        self._conn.executemany(
            f"INSERT OR IGNORE INTO trades ({','.join(TRADE_FIELDS)}) VALUES ({placeholders})", rows
        )
        self._conn.commit()

    def write_snapshot(self, wallets: dict[str, VirtualWallet]) -> None:
        now = time.time()
        rows = [(w.strategy, w.initial_balance, w.balance, w.reserved, now) for w in wallets.values()]
        self._conn.executemany(
            "INSERT INTO wallets (strategy, initial_balance, balance, reserved, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(strategy) DO UPDATE SET "
            "initial_balance=excluded.initial_balance, balance=excluded.balance, "
            "reserved=excluded.reserved, updated_at=excluded.updated_at",
            rows,
        )
        self._conn.commit()

    # -- reset -------------------------------------------------------------------
    def reset(self) -> None:
        """Wipe ALL persisted history for every strategy. Used by the
        dashboard's Reset DB button / `run.py --reset-data`. Does not
        touch bot.log (kept as an audit trail of the reset itself)."""
        self._conn.execute("DELETE FROM trades")
        self._conn.execute("DELETE FROM wallets")
        self._conn.commit()
        logger.warning("Storage reset: all trades/wallets wiped from %s.", self.db_path)

    def reset_strategy(self, strategy: str) -> None:
        """Wipe persisted history for ONE strategy only — the others'
        rows are untouched. Used by the Settings tab's per-strategy Reset
        button."""
        self._conn.execute("DELETE FROM trades WHERE strategy = ?", (strategy,))
        self._conn.execute("DELETE FROM wallets WHERE strategy = ?", (strategy,))
        self._conn.commit()
        logger.warning("Storage reset for strategy '%s' only.", strategy)

    # -- reads (dashboard analytics) ------------------------------------------------
    def get_trades(self, strategy: Optional[str] = None, limit: int = 200) -> list[dict]:
        """Most-recently-closed trades first, optionally filtered to one
        strategy. Backs the Analytics tab's raw trade table."""
        if strategy:
            cur = self._conn.execute(
                "SELECT * FROM trades WHERE strategy = ? AND closed_ts IS NOT NULL "
                "ORDER BY closed_ts DESC LIMIT ?",
                (strategy, limit),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM trades WHERE closed_ts IS NOT NULL ORDER BY closed_ts DESC LIMIT ?",
                (limit,),
            )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
