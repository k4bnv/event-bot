import datetime as dt
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Direction, Trade
from src.storage import FEATURE_FIELDS, Storage
from src.wallet import VirtualWallet


def make_settled_trade(strategy: str = "a") -> Trade:
    return Trade(
        strategy=strategy, entry_window_min=7, series_id="S", inst_id="I",
        direction=Direction.UP, entry_price=0.4, stake_usd=10.0, contracts=25.0,
        opened_ts=time.time(), expiry_ts=time.time(),
    )


class StorageResetTests(unittest.TestCase):
    def test_append_and_read_back(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            wallet = VirtualWallet(strategy="a", initial_balance=100.0)
            trade = make_settled_trade()
            wallet.open_trade(trade)
            wallet.settle_trade(trade, won=True)
            storage.append_closed_trades({"a": wallet})
            storage.write_snapshot({"a": wallet})

            rows = storage.get_trades()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], trade.id)
            self.assertEqual(rows[0]["strategy"], "a")
            storage.close()

    def test_append_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            wallet = VirtualWallet(strategy="a", initial_balance=100.0)
            trade = make_settled_trade()
            wallet.open_trade(trade)
            wallet.settle_trade(trade, won=False)
            storage.append_closed_trades({"a": wallet})
            storage.append_closed_trades({"a": wallet})  # same trade again, same tick pattern
            self.assertEqual(len(storage.get_trades()), 1)
            storage.close()

    def test_skip_ids_excludes_already_known_trades_from_the_sql_call(self):
        # Engine passes the ids of trades it RESTORED from this same table
        # (already safely persisted) so they're not resubmitted every
        # tick forever just because they now live in wallet.trades too —
        # verify the ACTUAL SQL statements executed exclude the skipped
        # trade's row, not just that the end state happens to look the
        # same (INSERT OR IGNORE would mask that either way). sqlite3.
        # Connection's own methods can't be mocked (a C-level read-only
        # attribute) — set_trace_callback is the supported hook for
        # observing exactly what SQL text actually ran.
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            wallet = VirtualWallet(strategy="a", initial_balance=100.0)
            old_trade = make_settled_trade("a")
            wallet.open_trade(old_trade)
            wallet.settle_trade(old_trade, won=True)
            storage.append_closed_trades({"a": wallet})  # "old_trade" is now genuinely already in the DB

            new_trade = make_settled_trade("a")
            wallet.open_trade(new_trade)
            wallet.settle_trade(new_trade, won=False)

            executed = []
            storage._conn.set_trace_callback(executed.append)
            try:
                storage.append_closed_trades({"a": wallet}, skip_ids={old_trade.id})
            finally:
                storage._conn.set_trace_callback(None)

            insert_statements = [sql for sql in executed if "INSERT" in sql and "INTO trades" in sql]
            self.assertEqual(len(insert_statements), 1)  # only new_trade's row was ever submitted
            self.assertIn(new_trade.id, insert_statements[0])
            self.assertNotIn(old_trade.id, insert_statements[0])

            rows = storage.get_trades()
            self.assertEqual({r["id"] for r in rows}, {old_trade.id, new_trade.id})  # both still there
            storage.close()

    def test_skip_ids_none_or_empty_behaves_like_before(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            wallet = VirtualWallet(strategy="a", initial_balance=100.0)
            trade = make_settled_trade("a")
            wallet.open_trade(trade)
            wallet.settle_trade(trade, won=True)
            storage.append_closed_trades({"a": wallet}, skip_ids=None)
            storage.append_closed_trades({"a": wallet}, skip_ids=set())
            self.assertEqual(len(storage.get_trades()), 1)
            storage.close()

    def test_reset_wipes_everything(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            wallet_a = VirtualWallet(strategy="a", initial_balance=100.0)
            wallet_b = VirtualWallet(strategy="b", initial_balance=100.0)
            trade_a, trade_b = make_settled_trade("a"), make_settled_trade("b")
            wallet_a.open_trade(trade_a)
            wallet_a.settle_trade(trade_a, won=True)
            wallet_b.open_trade(trade_b)
            wallet_b.settle_trade(trade_b, won=True)
            storage.append_closed_trades({"a": wallet_a, "b": wallet_b})
            storage.write_snapshot({"a": wallet_a, "b": wallet_b})

            storage.reset()
            self.assertEqual(storage.get_trades(), [])
            storage.close()

    def test_reset_strategy_only_touches_that_strategy(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            wallet_a = VirtualWallet(strategy="a", initial_balance=100.0)
            wallet_b = VirtualWallet(strategy="b", initial_balance=100.0)
            trade_a, trade_b = make_settled_trade("a"), make_settled_trade("b")
            wallet_a.open_trade(trade_a)
            wallet_a.settle_trade(trade_a, won=True)
            wallet_b.open_trade(trade_b)
            wallet_b.settle_trade(trade_b, won=True)
            storage.append_closed_trades({"a": wallet_a, "b": wallet_b})

            storage.reset_strategy("a")

            remaining = storage.get_trades()
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0]["strategy"], "b")
            storage.close()

    def test_get_trades_filters_by_strategy_and_orders_newest_first(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            wallet = VirtualWallet(strategy="a", initial_balance=100.0)
            t1 = make_settled_trade("a")
            wallet.open_trade(t1)
            wallet.settle_trade(t1, won=True)
            t1.closed_ts = 1000.0
            t2 = make_settled_trade("a")
            wallet.open_trade(t2)
            wallet.settle_trade(t2, won=False)
            t2.closed_ts = 2000.0
            storage.append_closed_trades({"a": wallet})

            rows = storage.get_trades(strategy="a")
            self.assertEqual([r["id"] for r in rows], [t2.id, t1.id])  # newest first
            self.assertEqual(storage.get_trades(strategy="nonexistent"), [])
            storage.close()

    def test_get_trades_sort_and_paginate(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            wallet = VirtualWallet(strategy="a", initial_balance=1000.0)
            trades = []
            for i, (won, stake) in enumerate([(True, 10.0), (False, 20.0), (True, 30.0)]):
                t = make_settled_trade("a")
                t.stake_usd = stake
                t.contracts = stake / t.entry_price
                wallet.open_trade(t)
                wallet.settle_trade(t, won=won)
                t.closed_ts = 1000.0 + i  # deterministic order
                trades.append(t)
            storage.append_closed_trades({"a": wallet})

            # ascending by pnl_usd: the loss (negative pnl) sorts first
            rows = storage.get_trades(sort_by="pnl_usd", sort_dir="asc")
            self.assertEqual(rows[0]["id"], trades[1].id)

            # an unknown/unsafe sort_by falls back to closed_ts, doesn't raise
            rows = storage.get_trades(sort_by="id; DROP TABLE trades;--", sort_dir="desc")
            self.assertEqual(len(rows), 3)

            # pagination: limit+offset walks the (default closed_ts DESC) order
            page1 = storage.get_trades(limit=2, offset=0)
            page2 = storage.get_trades(limit=2, offset=2)
            self.assertEqual([r["id"] for r in page1], [trades[2].id, trades[1].id])
            self.assertEqual([r["id"] for r in page2], [trades[0].id])

            self.assertEqual(storage.count_trades(), 3)
            self.assertEqual(storage.count_trades(strategy="nonexistent"), 0)
            storage.close()

    def test_trades_stats_aggregates_all_matching_rows(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            wallet = VirtualWallet(strategy="a", initial_balance=1000.0)
            t1 = make_settled_trade("a")
            wallet.open_trade(t1)
            wallet.settle_trade(t1, won=True)   # pnl = +15.0 (25 contracts - 10 stake)
            t2 = make_settled_trade("a")
            wallet.open_trade(t2)
            wallet.settle_trade(t2, won=False)  # pnl = -10.0
            storage.append_closed_trades({"a": wallet})

            stats = storage.trades_stats(strategy="a")
            self.assertEqual(stats["total"], 2)
            self.assertEqual(stats["wins"], 1)
            self.assertEqual(stats["losses"], 1)
            self.assertAlmostEqual(stats["winrate_pct"], 50.0)
            self.assertAlmostEqual(stats["net_pnl"], 5.0)  # +15 - 10

            empty = storage.trades_stats(strategy="nonexistent")
            self.assertEqual(empty, {"total": 0, "wins": 0, "losses": 0, "winrate_pct": 0.0, "net_pnl": 0})
            storage.close()


class PerCheckpointWalletsTests(unittest.TestCase):
    """Covers the wallets table's move from one-row-per-strategy to
    one-row-per-(strategy, window_min) — see Engine._wallet_key and
    storage.py's _migrate_wallets_table."""

    def test_two_checkpoints_of_the_same_strategy_persist_independently(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            w12 = VirtualWallet(strategy="breakout_retest", window_min=12, initial_balance=100.0)
            w2 = VirtualWallet(strategy="breakout_retest", window_min=2, initial_balance=100.0)
            w12.balance = 142.5
            w2.balance = 61.0
            storage.write_snapshot({"breakout_retest:12": w12, "breakout_retest:2": w2})

            loaded = storage.load_wallets()
            self.assertEqual(loaded["breakout_retest:12"]["balance"], 142.5)
            self.assertEqual(loaded["breakout_retest:2"]["balance"], 61.0)
            storage.close()

    def test_write_snapshot_updates_in_place_on_conflict(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            w = VirtualWallet(strategy="a", window_min=2, initial_balance=100.0)
            storage.write_snapshot({"a:2": w})
            w.balance = 88.0
            storage.write_snapshot({"a:2": w})  # same id again -> update, not a duplicate row

            loaded = storage.load_wallets()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded["a:2"]["balance"], 88.0)
            storage.close()

    def test_migrates_old_per_strategy_schema_without_crashing(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "bot.db"
            # Build the OLD schema by hand, as if this were a DB from before
            # the per-checkpoint split.
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE wallets (strategy TEXT PRIMARY KEY, initial_balance REAL, "
                "balance REAL, reserved REAL, updated_at REAL)"
            )
            conn.execute("INSERT INTO wallets VALUES ('breakout_retest', 100.0, 55.0, 0.0, 123.0)")
            conn.commit()
            conn.close()

            storage = Storage(Path(tmp))  # must not raise on the old schema

            # Old table renamed aside, not deleted, and not read as new-schema data.
            self.assertEqual(storage.load_wallets(), {})
            old_rows = storage._conn.execute(
                "SELECT * FROM wallets_legacy_pre_checkpoint_split"
            ).fetchall()
            self.assertEqual(len(old_rows), 1)

            # New-schema writes work fine on the now-migrated DB.
            w = VirtualWallet(strategy="breakout_retest", window_min=12, initial_balance=100.0)
            storage.write_snapshot({"breakout_retest:12": w})
            self.assertEqual(storage.load_wallets()["breakout_retest:12"]["balance"], 100.0)
            storage.close()

    def test_fresh_database_needs_no_migration(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))  # no pre-existing wallets table at all
            self.assertEqual(storage.load_wallets(), {})
            tables = {
                row[0] for row in
                storage._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            self.assertNotIn("wallets_legacy_pre_checkpoint_split", tables)
            storage.close()


def make_feature_row(id_="f1", strategy="a", ts=1000.0, decision="no_signal", **overrides) -> dict:
    row = {f: None for f in FEATURE_FIELDS}
    row.update({"id": id_, "strategy": strategy, "ts": ts, "decision": decision})
    row.update(overrides)
    return row


class CheckpointFeaturesTests(unittest.TestCase):
    """Covers the ML feature-logging table added for training on past
    outcomes — see storage.py's module docstring and engine.py's
    _record_checkpoint_features for what/why."""

    def test_log_and_read_back(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(strategy="a", up_price=0.42, decision="opened"))

            rows = storage.get_checkpoint_features()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["strategy"], "a")
            self.assertEqual(rows[0]["up_price"], 0.42)
            self.assertEqual(rows[0]["decision"], "opened")
            storage.close()

    def test_log_is_idempotent_on_duplicate_id(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(id_="dup"))
            storage.log_checkpoint_features(make_feature_row(id_="dup"))  # same id again
            self.assertEqual(storage.count_checkpoint_features(), 1)
            storage.close()

    def test_filters_by_strategy_and_counts(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(id_="1", strategy="a"))
            storage.log_checkpoint_features(make_feature_row(id_="2", strategy="a"))
            storage.log_checkpoint_features(make_feature_row(id_="3", strategy="b"))

            self.assertEqual(len(storage.get_checkpoint_features(strategy="a")), 2)
            self.assertEqual(len(storage.get_checkpoint_features(strategy="b")), 1)
            self.assertEqual(storage.count_checkpoint_features(strategy="a"), 2)
            self.assertEqual(storage.count_checkpoint_features(), 3)
            storage.close()

    def test_newest_first_by_default(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(id_="old", ts=1000.0))
            storage.log_checkpoint_features(make_feature_row(id_="new", ts=2000.0))
            rows = storage.get_checkpoint_features()
            self.assertEqual([r["id"] for r in rows], ["new", "old"])
            storage.close()

    def test_reset_does_not_wipe_checkpoint_features(self):
        # Deliberate: this table is a market-conditions log for future ML
        # work, not trading state — a config-tuning Reset shouldn't erase
        # months of accumulated feature history along with the wallets.
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(strategy="a"))

            storage.reset()

            self.assertEqual(storage.count_checkpoint_features(), 1)
            storage.close()

    def test_reset_strategy_does_not_wipe_checkpoint_features(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(strategy="a"))

            storage.reset_strategy("a")

            self.assertEqual(storage.count_checkpoint_features(strategy="a"), 1)
            storage.close()

    def test_extra_json_round_trips(self):
        # See StrategyContext.diagnostics/Engine._record_checkpoint_features
        # — a strategy's own numeric diagnostics (e.g. absorption_reversal's
        # tfi/residual_pct), serialized as JSON text, not parsed here —
        # storage.py just stores/returns the string as-is.
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            payload = '{"tfi": -0.74, "residual_pct": 0.073}'
            storage.log_checkpoint_features(make_feature_row(strategy="absorption_reversal", extra_json=payload))
            rows = storage.get_checkpoint_features(strategy="absorption_reversal")
            self.assertEqual(rows[0]["extra_json"], payload)
            storage.close()

    def test_extra_json_defaults_to_null_when_not_provided(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(strategy="breakout_retest"))
            rows = storage.get_checkpoint_features(strategy="breakout_retest")
            self.assertIsNone(rows[0]["extra_json"])
            storage.close()

    def test_migrates_old_schema_missing_extra_json_without_crashing(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "bot.db"
            import sqlite3
            conn = sqlite3.connect(db_path)
            old_cols = [f for f in FEATURE_FIELDS if f != "extra_json"]
            conn.execute(f"CREATE TABLE checkpoint_features ({', '.join(f'{c} TEXT' for c in old_cols)})")
            conn.execute(
                f"INSERT INTO checkpoint_features ({', '.join(old_cols)}) VALUES ({', '.join('?' * len(old_cols))})",
                ["old-row" if c == "id" else None for c in old_cols],
            )
            conn.commit()
            conn.close()

            storage = Storage(Path(tmp))  # must not raise despite the pre-existing table missing extra_json
            old_rows = storage.get_checkpoint_features()
            self.assertEqual(len(old_rows), 1)
            self.assertEqual(old_rows[0]["id"], "old-row")
            self.assertIsNone(old_rows[0]["extra_json"])  # migrated column, old row has nothing there

            storage.log_checkpoint_features(make_feature_row(id_="new-row", extra_json='{"tfi": 1.0}'))
            new_row = next(r for r in storage.get_checkpoint_features() if r["id"] == "new-row")
            self.assertEqual(new_row["extra_json"], '{"tfi": 1.0}')
            storage.close()

    def test_fresh_database_checkpoint_features_needs_no_migration(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            cols = {row[1] for row in storage._conn.execute("PRAGMA table_info(checkpoint_features)").fetchall()}
            self.assertIn("extra_json", cols)  # present from a fresh _SCHEMA, not via the migration path
            storage.close()


class DecisionBreakdownTests(unittest.TestCase):
    """Covers get_decision_breakdown/get_rejected_fill_price_stats — backs
    the dashboard's Диагностика tab (same data scripts/decision_breakdown.py
    prints from a terminal, see that script's docstring)."""

    def test_groups_counts_by_strategy_and_decision(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(id_="1", strategy="a", decision="no_signal"))
            storage.log_checkpoint_features(make_feature_row(id_="2", strategy="a", decision="no_signal"))
            storage.log_checkpoint_features(make_feature_row(id_="3", strategy="a", decision="opened"))
            storage.log_checkpoint_features(make_feature_row(id_="4", strategy="b", decision="rejected_max_coefficient"))

            rows = {r["strategy"]: r for r in storage.get_decision_breakdown()}
            self.assertEqual(rows["a"]["total"], 3)
            self.assertEqual(rows["a"]["decisions"], {"no_signal": 2, "opened": 1})
            self.assertEqual(rows["b"]["decisions"], {"rejected_max_coefficient": 1})
            storage.close()

    def test_busiest_strategy_first(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(id_="1", strategy="quiet"))
            for i in range(3):
                storage.log_checkpoint_features(make_feature_row(id_=f"b{i}", strategy="busy"))

            names = [r["strategy"] for r in storage.get_decision_breakdown()]
            self.assertEqual(names, ["busy", "quiet"])
            storage.close()

    def test_rejected_fill_price_stats_only_counts_that_decision_and_strategy(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            for i, price in enumerate([0.6, 0.7, 0.8]):
                storage.log_checkpoint_features(make_feature_row(
                    id_=f"r{i}", strategy="a", decision="rejected_max_coefficient", fill_price=price))
            storage.log_checkpoint_features(make_feature_row(id_="o1", strategy="a", decision="opened", fill_price=0.5))
            storage.log_checkpoint_features(make_feature_row(
                id_="r_other", strategy="b", decision="rejected_max_coefficient", fill_price=0.99))

            stats = storage.get_rejected_fill_price_stats("a")
            self.assertEqual(stats["n"], 3)
            self.assertAlmostEqual(stats["p50"], 0.7)
            self.assertAlmostEqual(stats["max"], 0.8)
            storage.close()

    def test_rejected_fill_price_stats_none_below_min_n(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(
                id_="r1", strategy="a", decision="rejected_max_coefficient", fill_price=0.6))
            self.assertIsNone(storage.get_rejected_fill_price_stats("a"))
            storage.close()

    def test_since_ts_excludes_stale_history(self):
        # Live bug this was added for: checkpoint_features is never wiped
        # by a Reset (unlike wallets/trades — see reset()'s docstring), so
        # an all-time breakdown right after a reset (or a strategy logic
        # change) silently counts evaluations from before it alongside
        # whatever's actually happened since. since_ts is how the
        # dashboard's Диагностика tab answers "is it alive RIGHT NOW"
        # instead of "ever, across however much stale history remains".
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(make_feature_row(id_="old", strategy="a", ts=1000.0, decision="opened"))
            storage.log_checkpoint_features(make_feature_row(id_="new", strategy="a", ts=5000.0, decision="no_signal"))

            all_time = {r["strategy"]: r for r in storage.get_decision_breakdown()}
            self.assertEqual(all_time["a"]["total"], 2)

            recent = {r["strategy"]: r for r in storage.get_decision_breakdown(since_ts=4000.0)}
            self.assertEqual(recent["a"]["total"], 1)
            self.assertEqual(recent["a"]["decisions"], {"no_signal": 1})
            storage.close()

    def test_rejected_fill_price_stats_since_ts_excludes_stale_history(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            for i, price in enumerate([0.6, 0.7, 0.8]):
                storage.log_checkpoint_features(make_feature_row(
                    id_=f"old{i}", strategy="a", ts=1000.0,
                    decision="rejected_max_coefficient", fill_price=price))

            self.assertIsNotNone(storage.get_rejected_fill_price_stats("a"))  # enough all-time
            self.assertIsNone(storage.get_rejected_fill_price_stats("a", since_ts=4000.0))  # none recent
            storage.close()


def make_closed_trade_with_features(
    storage: Storage, strategy: str = "a", won: bool = True, stake: float = 10.0,
    opened_ts: Optional[float] = None, link_features: bool = True, **feature_overrides,
) -> Trade:
    """A trade that's both in the trades table (closed, won/lost) AND has
    a linked checkpoint_features row (trade_id set) carrying whatever
    market-context fields the caller passes as feature_overrides (e.g.
    sigma_horizon_pct=0.5, drift_5m_pct=-0.2) — the exact shape
    get_volatility_regime_stats/get_trend_direction_stats join against.
    `link_features=False` closes the trade WITHOUT a checkpoint_features
    row at all, for the "trades that never opened via a logged checkpoint
    don't count" case."""
    wallet = VirtualWallet(strategy=strategy, initial_balance=1000.0)
    t = make_settled_trade(strategy)
    t.stake_usd = stake
    t.contracts = stake / t.entry_price
    if opened_ts is not None:
        t.opened_ts = opened_ts
    wallet.open_trade(t)
    wallet.settle_trade(t, won=won)
    t.closed_ts = t.opened_ts + 60.0
    storage.append_closed_trades({strategy: wallet})
    if link_features:
        storage.log_checkpoint_features(make_feature_row(
            id_=f"f-{t.id}", strategy=strategy, decision="opened", trade_id=t.id, **feature_overrides,
        ))
    return t


def utc_hour_ts(hour: int) -> float:
    """An arbitrary but fixed timestamp landing at exactly `hour` UTC —
    lets a test target a specific bucket of get_hourly_stats without
    depending on when the test happens to run."""
    return dt.datetime(2024, 1, 1, hour, 30, 0, tzinfo=dt.timezone.utc).timestamp()


class PatternBreakdownTests(unittest.TestCase):
    """Covers the Analytics tab's "Закономерности" (hour-of-day /
    volatility regime / trend direction) breakdowns — see
    Storage.get_hourly_stats/get_volatility_regime_stats/
    get_trend_direction_stats."""

    def test_get_hourly_stats_buckets_by_utc_hour_of_open_and_always_has_24_rows(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            make_closed_trade_with_features(storage, won=True, opened_ts=utc_hour_ts(5), link_features=False)
            make_closed_trade_with_features(storage, won=False, opened_ts=utc_hour_ts(5), link_features=False)
            make_closed_trade_with_features(storage, won=True, opened_ts=utc_hour_ts(17), link_features=False)

            rows = storage.get_hourly_stats()
            self.assertEqual(len(rows), 24)  # every hour present, not just the ones with data
            by_label = {r["label"]: r for r in rows}

            self.assertEqual(by_label["05:00"]["trades"], 2)
            self.assertEqual(by_label["05:00"]["wins"], 1)
            self.assertEqual(by_label["05:00"]["losses"], 1)
            self.assertEqual(by_label["05:00"]["winrate_pct"], 50.0)

            self.assertEqual(by_label["17:00"]["trades"], 1)
            self.assertEqual(by_label["17:00"]["wins"], 1)

            # An hour with no trades at all is zero-filled, not missing.
            self.assertEqual(by_label["03:00"]["trades"], 0)
            self.assertEqual(by_label["03:00"]["winrate_pct"], 0.0)
            self.assertEqual(by_label["03:00"]["net_pnl"], 0.0)
            storage.close()

    def test_get_hourly_stats_filters_by_strategy(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            make_closed_trade_with_features(storage, strategy="a", opened_ts=utc_hour_ts(9), link_features=False)
            make_closed_trade_with_features(storage, strategy="b", opened_ts=utc_hour_ts(9), link_features=False)

            rows = storage.get_hourly_stats(strategy="a")
            by_label = {r["label"]: r for r in rows}
            self.assertEqual(by_label["09:00"]["trades"], 1)
            storage.close()

    def test_get_volatility_regime_stats_splits_into_terciles_low_to_high(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            # Six trades, sigma strictly increasing — NTILE(3) over 6 rows
            # gives exactly 2 per bucket. Make the low-sigma pair lose and
            # the high-sigma pair win, so the buckets' PnL sign confirms
            # they were split in the right ORDER, not just into 3 groups.
            for sigma in (0.1, 0.2):
                make_closed_trade_with_features(storage, won=False, sigma_horizon_pct=sigma)
            for sigma in (0.3, 0.4):
                make_closed_trade_with_features(storage, won=True, sigma_horizon_pct=sigma)
            for sigma in (0.5, 0.6):
                make_closed_trade_with_features(storage, won=True, sigma_horizon_pct=sigma)

            rows = storage.get_volatility_regime_stats()
            self.assertEqual([r["label"] for r in rows], ["Низкая", "Средняя", "Высокая"])
            by_label = {r["label"]: r for r in rows}
            self.assertEqual(by_label["Низкая"]["trades"], 2)
            self.assertEqual(by_label["Низкая"]["wins"], 0)
            self.assertEqual(by_label["Средняя"]["wins"], 2)
            self.assertEqual(by_label["Высокая"]["wins"], 2)
            storage.close()

    def test_get_volatility_regime_stats_ignores_trades_without_a_feature_row(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            make_closed_trade_with_features(storage, won=True, link_features=False)  # no trade_id link at all

            rows = storage.get_volatility_regime_stats()
            self.assertEqual(sum(r["trades"] for r in rows), 0)  # nothing to bucket
            storage.close()

    def test_get_volatility_regime_stats_zero_fills_when_too_little_data(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            make_closed_trade_with_features(storage, won=True, sigma_horizon_pct=0.3)

            rows = storage.get_volatility_regime_stats()
            # Always exactly 3 rows even with just 1 trade — NTILE can't
            # populate all three buckets yet, the empty ones are 0, not absent.
            self.assertEqual([r["label"] for r in rows], ["Низкая", "Средняя", "Высокая"])
            self.assertEqual(sum(r["trades"] for r in rows), 1)
            storage.close()

    def test_get_trend_direction_stats_splits_up_flat_down(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            make_closed_trade_with_features(storage, won=True, drift_5m_pct=0.5)     # clearly up
            make_closed_trade_with_features(storage, won=False, drift_5m_pct=-0.5)  # clearly down
            make_closed_trade_with_features(storage, won=True, drift_5m_pct=0.001)   # inside the deadband -> flat

            rows = storage.get_trend_direction_stats()
            self.assertEqual([r["label"] for r in rows], ["Рост", "Флэт", "Падение"])
            by_label = {r["label"]: r for r in rows}
            self.assertEqual(by_label["Рост"]["trades"], 1)
            self.assertEqual(by_label["Рост"]["wins"], 1)
            self.assertEqual(by_label["Флэт"]["trades"], 1)
            self.assertEqual(by_label["Падение"]["trades"], 1)
            self.assertEqual(by_label["Падение"]["wins"], 0)
            storage.close()

    def test_get_trend_direction_stats_zero_fills_a_bucket_with_no_trades(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            make_closed_trade_with_features(storage, won=True, drift_5m_pct=0.5)  # only "up", ever

            rows = storage.get_trend_direction_stats()
            by_label = {r["label"]: r for r in rows}
            self.assertEqual(by_label["Рост"]["trades"], 1)
            self.assertEqual(by_label["Флэт"]["trades"], 0)   # zero-filled, not missing
            self.assertEqual(by_label["Падение"]["trades"], 0)
            storage.close()


if __name__ == "__main__":
    unittest.main()
