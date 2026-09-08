import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Direction, Trade
from src.storage import Storage
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


if __name__ == "__main__":
    unittest.main()
