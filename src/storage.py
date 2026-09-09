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

# Whitelisted for ORDER BY — these become raw SQL identifiers below (via an
# f-string), so only column names actually in the `trades` table may appear
# here. Never build sort_by from unvalidated input.
SORTABLE_TRADE_COLUMNS = {
    "closed_ts", "opened_ts", "strategy", "entry_window_min", "series_id",
    "inst_id", "direction", "entry_price", "stake_usd", "contracts",
    "expiry_ts", "status", "pnl_usd", "reason",
}

# A couple of sort keys the dashboard exposes that aren't raw columns —
# fixed literal SQL expressions (never built from request input), so
# splicing them into the query is as safe as a whitelisted column name.
_SORT_EXPRESSIONS = {
    "roi_pct": "(CASE WHEN stake_usd > 0 THEN pnl_usd / stake_usd ELSE 0 END)",
    "duration_sec": "(closed_ts - opened_ts)",
}

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

    def load_wallets(self) -> dict[str, dict]:
        """Every wallet row this DB currently has, keyed by strategy name —
        whatever write_snapshot() last wrote for it. Used by the engine at
        startup to resume each strategy's balance instead of restarting it
        at deposit_usd every time the process restarts (a redeploy, a
        crash, `docker compose up --build`, ...); a strategy with no row
        here (first-ever launch, or one just reset) simply gets no restore
        and starts fresh from config as before."""
        cur = self._conn.execute("SELECT strategy, initial_balance, balance, reserved FROM wallets")
        return {
            row[0]: {"initial_balance": row[1], "balance": row[2], "reserved": row[3]}
            for row in cur.fetchall()
        }

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
    def get_trades(
        self, strategy: Optional[str] = None, limit: Optional[int] = 200, offset: int = 0,
        sort_by: str = "closed_ts", sort_dir: str = "desc",
    ) -> list[dict]:
        """Closed trades (won/lost/unresolved), optionally filtered to one
        strategy, sorted/paginated. Backs the Analytics tab's trade table
        and its CSV export. `limit=None` returns every matching row (used
        by the CSV export, which shouldn't silently truncate history) —
        pass an int for a UI page. `sort_by` is checked against
        SORTABLE_TRADE_COLUMNS (falls back to closed_ts) before it's
        spliced into the query, so this is never raw user input reaching SQL."""
        if sort_by in _SORT_EXPRESSIONS:
            col = _SORT_EXPRESSIONS[sort_by]
        elif sort_by in SORTABLE_TRADE_COLUMNS:
            col = sort_by
        else:
            col = "closed_ts"
        direction = "ASC" if str(sort_dir).lower() == "asc" else "DESC"

        where = "WHERE closed_ts IS NOT NULL"
        params: list = []
        if strategy:
            where += " AND strategy = ?"
            params.append(strategy)

        # secondary key (id) keeps ties in a stable order across pages —
        # otherwise equal-timestamp rows could shuffle between requests.
        query = f"SELECT * FROM trades {where} ORDER BY {col} {direction}, id {direction}"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params += [limit, offset]

        cur = self._conn.execute(query, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count_trades(self, strategy: Optional[str] = None) -> int:
        """Total closed trades matching the filter — lets the dashboard
        compute page count without fetching every row."""
        if strategy:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM trades WHERE closed_ts IS NOT NULL AND strategy = ?", (strategy,)
            )
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM trades WHERE closed_ts IS NOT NULL")
        return cur.fetchone()[0]

    def trades_stats(self, strategy: Optional[str] = None) -> dict:
        """Aggregate win/loss/PnL over EVERY matching closed trade (not just
        the current page) — powers the small summary line under the
        Analytics trade table so filtering/paging doesn't hide the
        overall picture."""
        where = "WHERE closed_ts IS NOT NULL AND status IN ('won', 'lost')"
        params: list = []
        if strategy:
            where += " AND strategy = ?"
            params.append(strategy)
        cur = self._conn.execute(
            f"SELECT COUNT(*), "
            f"SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END), "
            f"COALESCE(SUM(pnl_usd), 0) "
            f"FROM trades {where}",
            params,
        )
        total, wins, losses, net_pnl = cur.fetchone()
        total, wins, losses = total or 0, wins or 0, losses or 0
        return {
            "total": total, "wins": wins, "losses": losses,
            "winrate_pct": (wins / total * 100) if total else 0.0,
            "net_pnl": net_pnl,
        }
