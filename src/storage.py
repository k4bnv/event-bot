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

`checkpoint_features` is the same idea for a future ML pass over this
bot's own history: one row per (strategy, entry checkpoint) EVALUATION —
not just the ones that opened a trade. See engine.py's
`_record_checkpoint_features` for what gets captured and why a row is
written on every outcome (no_signal/rejected/opened alike): a dataset
that only contains executed trades has no negative examples to learn
"why not" from. `trade_id` links a row to its trades-table entry (and
that row's eventual pnl_usd/status) when a trade actually opened; NULL
when the checkpoint didn't result in one. Same export pattern as trades:
    sqlite3 -header -csv data/bot.db "SELECT * FROM checkpoint_features;" > features.csv
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
    id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    window_min INTEGER,
    initial_balance REAL,
    balance REAL,
    reserved REAL,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_wallets_strategy ON wallets(strategy);

CREATE TABLE IF NOT EXISTS checkpoint_features (
    id TEXT PRIMARY KEY,
    ts REAL,
    strategy TEXT NOT NULL,
    series_id TEXT,
    inst_id TEXT,
    window_min INTEGER,
    remaining_sec REAL,
    market_method TEXT,
    up_price REAL,
    floor_strike REAL,
    strike_is_fixed INTEGER,
    spot REAL,
    drift_5m_pct REAL,
    mom_1m_pct REAL,
    z_score REAL,
    base_prob REAL,
    sigma_horizon_pct REAL,
    orderbook_bid_vol REAL,
    orderbook_ask_vol REAL,
    funding_rate REAL,
    previous_outcome TEXT,
    signal_direction TEXT,
    signal_confidence REAL,
    signal_reason TEXT,
    decision TEXT NOT NULL,
    fill_price REAL,
    stake_usd REAL,
    trade_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_checkpoint_features_strategy ON checkpoint_features(strategy);
CREATE INDEX IF NOT EXISTS idx_checkpoint_features_ts ON checkpoint_features(ts);
CREATE INDEX IF NOT EXISTS idx_checkpoint_features_trade_id ON checkpoint_features(trade_id);
"""

FEATURE_FIELDS = [
    "id", "ts", "strategy", "series_id", "inst_id", "window_min", "remaining_sec",
    "market_method", "up_price", "floor_strike", "strike_is_fixed", "spot",
    "drift_5m_pct", "mom_1m_pct", "z_score", "base_prob", "sigma_horizon_pct",
    "orderbook_bid_vol", "orderbook_ask_vol", "funding_rate", "previous_outcome",
    "signal_direction", "signal_confidence", "signal_reason", "decision",
    "fill_price", "stake_usd", "trade_id",
]

FEATURE_SORTABLE_COLUMNS = {
    "ts", "strategy", "series_id", "window_min", "remaining_sec", "decision", "trade_id",
}

# See get_trend_direction_stats — a starting heuristic, not a tuned
# constant. Below this magnitude, drift_5m_pct counts as "Флэт" rather
# than a real up/down trend.
TREND_DEADBAND_PCT = 0.03


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
        self._migrate_wallets_table()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _migrate_wallets_table(self) -> None:
        """Old schema: wallets(strategy TEXT PRIMARY KEY, ...) — one row
        per STRATEGY. New schema (see _SCHEMA above): one row per
        (strategy, window_min), since balances are now split per entry
        checkpoint. CREATE TABLE IF NOT EXISTS is a no-op against an
        existing old-schema table, so every write_snapshot() on an
        upgraded deployment would otherwise fail with "no such column:
        window_min" forever. Old balances aren't preserved — there's no
        principled way to map ONE old per-strategy balance onto N new
        per-checkpoint ones — so this renames the old table aside (never
        drops it) and every checkpoint just restarts from deposit_usd
        once, same as a strategy that's never run before. trades and
        checkpoint_features (the actual valuable accumulated history)
        are completely untouched by this."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(wallets)").fetchall()}
        if cols and "window_min" not in cols:
            self._conn.execute("ALTER TABLE wallets RENAME TO wallets_legacy_pre_checkpoint_split")
            self._conn.commit()
            logger.warning(
                "Migrated data/bot.db's wallets table to the new per-checkpoint schema "
                "(old data renamed to wallets_legacy_pre_checkpoint_split, not deleted). "
                "Every strategy's checkpoint balances restart from deposit_usd once — "
                "trades and checkpoint_features history is untouched."
            )

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
        """`wallets` is keyed by Engine's own composite wallet id
        (`"{strategy}:{window_min}"`, see Engine._wallet_key) — that key,
        not just wallet.strategy, is what's persisted as the primary key,
        since one strategy now has multiple independent wallets."""
        now = time.time()
        rows = [
            (key, w.strategy, w.window_min, w.initial_balance, w.balance, w.reserved, now)
            for key, w in wallets.items()
        ]
        self._conn.executemany(
            "INSERT INTO wallets (id, strategy, window_min, initial_balance, balance, reserved, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "initial_balance=excluded.initial_balance, balance=excluded.balance, "
            "reserved=excluded.reserved, updated_at=excluded.updated_at",
            rows,
        )
        self._conn.commit()

    def load_wallets(self) -> dict[str, dict]:
        """Every wallet row this DB currently has, keyed by the same
        composite wallet id write_snapshot() persisted it under
        ("{strategy}:{window_min}" — see Engine._wallet_key). Used by the
        engine at startup to resume each checkpoint's balance instead of
        restarting it at deposit_usd every time the process restarts (a
        redeploy, a crash, `docker compose up --build`, ...); a wallet
        with no row here (first-ever launch, a newly-added checkpoint, or
        one just reset) simply gets no restore and starts fresh from
        config as before."""
        cur = self._conn.execute("SELECT id, initial_balance, balance, reserved FROM wallets")
        return {
            row[0]: {"initial_balance": row[1], "balance": row[2], "reserved": row[3]}
            for row in cur.fetchall()
        }

    def log_checkpoint_features(self, row: dict) -> None:
        """One feature snapshot row — see this module's docstring and
        engine.py's `_record_checkpoint_features` for what it captures and
        why. `row` must carry every key in FEATURE_FIELDS (None for
        whichever don't apply this call) since this does a straight
        positional INSERT; INSERT OR IGNORE makes a duplicate `id` a
        silent no-op, same dedup approach as append_closed_trades."""
        values = tuple(row.get(f) for f in FEATURE_FIELDS)
        placeholders = ",".join("?" * len(FEATURE_FIELDS))
        self._conn.execute(
            f"INSERT OR IGNORE INTO checkpoint_features ({','.join(FEATURE_FIELDS)}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def get_checkpoint_features(
        self, strategy: Optional[str] = None, limit: Optional[int] = 200, offset: int = 0,
        sort_by: str = "ts", sort_dir: str = "desc",
    ) -> list[dict]:
        """Feature rows, optionally filtered to one strategy, newest
        first by default. `limit=None` returns every matching row (for a
        full export, e.g. before training something offline) — pass an
        int for a UI/inspection page."""
        col = sort_by if sort_by in FEATURE_SORTABLE_COLUMNS else "ts"
        direction = "ASC" if str(sort_dir).lower() == "asc" else "DESC"

        where = ""
        params: list = []
        if strategy:
            where = "WHERE strategy = ?"
            params.append(strategy)

        query = f"SELECT * FROM checkpoint_features {where} ORDER BY {col} {direction}, id {direction}"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params += [limit, offset]

        cur = self._conn.execute(query, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count_checkpoint_features(self, strategy: Optional[str] = None) -> int:
        if strategy:
            cur = self._conn.execute("SELECT COUNT(*) FROM checkpoint_features WHERE strategy = ?", (strategy,))
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM checkpoint_features")
        return cur.fetchone()[0]

    # -- reset -------------------------------------------------------------------
    def reset(self) -> None:
        """Wipe ALL persisted history for every strategy. Used by the
        dashboard's Reset DB button / `run.py --reset-data`. Does not
        touch bot.log (kept as an audit trail of the reset itself), and
        deliberately does not touch checkpoint_features either: those
        rows are a market-conditions log for future ML work, not trading
        state — wiping them every time someone resets balances while
        tuning a config would defeat the entire point of accumulating
        them. A dangling trade_id after a reset just means that
        particular row's eventual outcome link is gone; the feature
        snapshot itself is still valid history."""
        self._conn.execute("DELETE FROM trades")
        self._conn.execute("DELETE FROM wallets")
        self._conn.commit()
        logger.warning("Storage reset: all trades/wallets wiped from %s (checkpoint_features kept).", self.db_path)

    def reset_strategy(self, strategy: str) -> None:
        """Wipe persisted history for ONE strategy only — the others'
        rows are untouched. Used by the Settings tab's per-strategy Reset
        button. Also leaves checkpoint_features alone — see reset()'s
        docstring for why."""
        self._conn.execute("DELETE FROM trades WHERE strategy = ?", (strategy,))
        self._conn.execute("DELETE FROM wallets WHERE strategy = ?", (strategy,))
        self._conn.commit()
        logger.warning("Storage reset for strategy '%s' only (checkpoint_features kept).", strategy)

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

    @staticmethod
    def _bucket_rows(cur, ordered_labels: dict) -> list[dict]:
        """Shared shape for every "win/loss/PnL grouped by some bucket"
        query below: `cur` must have already run a query whose SELECT is
        exactly (bucket_key, COUNT(*), wins, losses, net_pnl).
        `ordered_labels` (e.g. {1: "Низкая", 2: "Средняя", 3: "Высокая"})
        both renames each raw SQL bucket value into the label the
        dashboard shows AND fixes the output's shape: every key in it
        gets exactly one row, zero-filled if that bucket had no matching
        trades — a bucket that's empty right now (e.g. too few trades so
        far for NTILE to fill all three) shows up as "0 сделок" rather
        than silently vanishing from the table."""
        by_bucket = {row[0]: row[1:] for row in cur.fetchall()}
        out = []
        for key, label in ordered_labels.items():
            total, wins, losses, net_pnl = by_bucket.get(key, (0, 0, 0, 0.0))
            out.append({
                "label": label, "trades": total, "wins": wins, "losses": losses,
                "winrate_pct": (wins / total * 100) if total else 0.0, "net_pnl": net_pnl,
            })
        return out

    def get_hourly_stats(self, strategy: Optional[str] = None) -> list[dict]:
        """Win rate / PnL bucketed by the hour-of-day (UTC) a trade was
        OPENED, over EVERY closed trade this DB has ever recorded (not
        just this process's in-memory wallets, which lose their trade
        history across a restart — see VirtualWallet/Engine's own
        docstrings) — surfaces whether some hours are systematically
        better/worse (e.g. thinner books during a particular session).
        Always returns all 24 hours, zero-filled for ones with no trades
        yet, so a quiet hour reads as "no data" rather than being
        silently absent from the table."""
        where = "WHERE closed_ts IS NOT NULL AND status IN ('won', 'lost') AND opened_ts IS NOT NULL"
        params: list = []
        if strategy:
            where += " AND strategy = ?"
            params.append(strategy)
        cur = self._conn.execute(
            f"SELECT CAST(strftime('%H', datetime(opened_ts, 'unixepoch')) AS INTEGER) AS hour, "
            f"COUNT(*), SUM(CASE WHEN status='won' THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END), COALESCE(SUM(pnl_usd), 0) "
            f"FROM trades {where} GROUP BY hour",
            params,
        )
        by_hour = {row[0]: row[1:] for row in cur.fetchall()}
        out = []
        for h in range(24):
            total, wins, losses, net_pnl = by_hour.get(h, (0, 0, 0, 0.0))
            out.append({
                "label": f"{h:02d}:00", "trades": total, "wins": wins, "losses": losses,
                "winrate_pct": (wins / total * 100) if total else 0.0, "net_pnl": net_pnl,
            })
        return out

    def get_volatility_regime_stats(self, strategy: Optional[str] = None) -> list[dict]:
        """Win rate / PnL split into THREE roughly-equal-sized buckets
        (terciles) of the realized-volatility estimate (checkpoint_features
        .sigma_horizon_pct, scaled to each trade's own remaining horizon —
        see fair_value_edge.py) captured at the exact moment each trade was
        decided. Terciles rather than a fixed % threshold — this bot has
        no prior idea what "high volatility" means in absolute terms for
        this market, so the split just tracks whatever spread has actually
        been observed. Only trades with a linked checkpoint_features row
        (trade_id set — i.e. every trade that ever actually opened) count.
        Always returns exactly 3 rows (Низкая/Средняя/Высокая), zero-filled
        for a bucket NTILE hasn't populated yet (too little data so far)."""
        where = "WHERE t.status IN ('won', 'lost') AND cf.sigma_horizon_pct IS NOT NULL"
        params: list = []
        if strategy:
            where += " AND t.strategy = ?"
            params.append(strategy)
        cur = self._conn.execute(
            "WITH ranked AS ("
            "  SELECT t.status, t.pnl_usd, "
            "         NTILE(3) OVER (ORDER BY cf.sigma_horizon_pct) AS bucket "
            "  FROM trades t JOIN checkpoint_features cf ON cf.trade_id = t.id "
            f" {where}"
            ") "
            "SELECT bucket, COUNT(*), SUM(CASE WHEN status='won' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END), COALESCE(SUM(pnl_usd), 0) "
            "FROM ranked GROUP BY bucket ORDER BY bucket",
            params,
        )
        return self._bucket_rows(cur, {1: "Низкая", 2: "Средняя", 3: "Высокая"})

    def get_trend_direction_stats(self, strategy: Optional[str] = None) -> list[dict]:
        """Win rate / PnL split by whether BTC was trending up, down, or
        flat (checkpoint_features.drift_5m_pct — % change over the 5
        minutes before the trade was decided) at the moment each trade was
        placed. TREND_DEADBAND_PCT is a starting heuristic (anything
        smaller in magnitude just counts as noise, not a real trend), not
        a tuned constant — revisit once enough live trades have
        accumulated to see whether it's splitting the data sensibly. Only
        trades with a linked checkpoint_features row count, same as
        get_volatility_regime_stats."""
        where = "WHERE t.status IN ('won', 'lost') AND cf.drift_5m_pct IS NOT NULL"
        params: list = [TREND_DEADBAND_PCT, -TREND_DEADBAND_PCT]
        if strategy:
            where += " AND t.strategy = ?"
            params.append(strategy)
        cur = self._conn.execute(
            "SELECT CASE WHEN cf.drift_5m_pct > ? THEN 'up' "
            "            WHEN cf.drift_5m_pct < ? THEN 'down' ELSE 'flat' END AS bucket, "
            "COUNT(*), SUM(CASE WHEN t.status='won' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN t.status='lost' THEN 1 ELSE 0 END), COALESCE(SUM(t.pnl_usd), 0) "
            "FROM trades t JOIN checkpoint_features cf ON cf.trade_id = t.id "
            f"{where} GROUP BY bucket",
            params,
        )
        return self._bucket_rows(cur, {"up": "Рост", "flat": "Флэт", "down": "Падение"})
