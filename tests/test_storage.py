import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


if __name__ == "__main__":
    unittest.main()
