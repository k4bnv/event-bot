import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Direction, Trade, TradeStatus
from src.wallet import VirtualWallet


def make_trade(stake=10.0, price=0.4, direction=Direction.UP) -> Trade:
    return Trade(
        strategy="test", entry_window_min=7, series_id="BTC-USDT-5M", inst_id="X-UP-1",
        direction=direction, entry_price=price, stake_usd=stake, contracts=stake / price,
        opened_ts=time.time(), expiry_ts=time.time() + 300,
    )


class WalletTests(unittest.TestCase):
    def test_starts_with_initial_balance(self):
        w = VirtualWallet(strategy="a", initial_balance=33.3)
        self.assertEqual(w.balance, 33.3)
        self.assertEqual(w.equity, 33.3)
        self.assertEqual(w.net_pnl, 0.0)

    def test_cannot_overspend(self):
        w = VirtualWallet(strategy="a", initial_balance=10.0)
        t = make_trade(stake=20.0)
        opened = w.open_trade(t)
        self.assertFalse(opened)
        self.assertEqual(t.status, TradeStatus.REJECTED)
        self.assertEqual(w.balance, 10.0)

    def test_open_reserves_capital(self):
        w = VirtualWallet(strategy="a", initial_balance=33.3)
        t = make_trade(stake=10.0)
        self.assertTrue(w.open_trade(t))
        self.assertAlmostEqual(w.balance, 23.3)
        self.assertAlmostEqual(w.reserved, 10.0)
        self.assertAlmostEqual(w.equity, 33.3)  # nothing lost yet, just moved

    def test_settle_win_pays_one_dollar_per_contract(self):
        w = VirtualWallet(strategy="a", initial_balance=33.3)
        t = make_trade(stake=10.0, price=0.4)  # 25 contracts
        w.open_trade(t)
        w.settle_trade(t, won=True)
        self.assertEqual(t.status, TradeStatus.WON)
        self.assertAlmostEqual(t.pnl_usd, 25.0 - 10.0)
        self.assertAlmostEqual(w.balance, 23.3 + 25.0)
        self.assertEqual(w.reserved, 0.0)

    def test_settle_loss_forfeits_stake(self):
        w = VirtualWallet(strategy="a", initial_balance=33.3)
        t = make_trade(stake=10.0, price=0.4)
        w.open_trade(t)
        w.settle_trade(t, won=False)
        self.assertEqual(t.status, TradeStatus.LOST)
        self.assertAlmostEqual(t.pnl_usd, -10.0)
        self.assertAlmostEqual(w.balance, 23.3)
        self.assertAlmostEqual(w.net_pnl, -10.0)

    def test_unresolved_refunds_stake_without_pnl_impact(self):
        w = VirtualWallet(strategy="a", initial_balance=33.3)
        t = make_trade(stake=10.0)
        w.open_trade(t)
        w.mark_unresolved(t)
        self.assertEqual(t.status, TradeStatus.UNRESOLVED)
        self.assertAlmostEqual(w.balance, 33.3)
        self.assertAlmostEqual(w.net_pnl, 0.0)


if __name__ == "__main__":
    unittest.main()
