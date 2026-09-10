import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import build_combo_stats, build_leaderboard
from src.models import Direction, Trade
from src.wallet import VirtualWallet


def add_trade(wallet: VirtualWallet, window_min: int, price: float, won: bool, stake: float = 10.0) -> None:
    t = Trade(
        strategy=wallet.strategy, entry_window_min=window_min, series_id="S", inst_id="I",
        direction=Direction.UP, entry_price=price, stake_usd=stake, contracts=stake / price,
        opened_ts=time.time(), expiry_ts=time.time(),
    )
    wallet.open_trade(t)
    wallet.settle_trade(t, won=won)


class MetricsTests(unittest.TestCase):
    def test_combo_stats_winrate_and_pnl(self):
        w = VirtualWallet(strategy="breakout_retest", initial_balance=100.0)
        add_trade(w, 7, price=0.4, won=True)   # +15
        add_trade(w, 7, price=0.4, won=False)  # -10
        add_trade(w, 7, price=0.4, won=True)   # +15

        stats = build_combo_stats({"breakout_retest": w})
        combo = stats[("breakout_retest", 7)]
        self.assertEqual(combo.trades, 3)
        self.assertEqual(combo.wins, 2)
        self.assertEqual(combo.losses, 1)
        self.assertAlmostEqual(combo.winrate_pct, 2 / 3 * 100)
        self.assertAlmostEqual(combo.net_pnl, 15 - 10 + 15)
        self.assertAlmostEqual(combo.avg_entry_price, 0.4)

    def test_leaderboard_picks_best_pnl(self):
        good = VirtualWallet(strategy="good", initial_balance=100.0)
        bad = VirtualWallet(strategy="bad", initial_balance=100.0)
        for _ in range(5):
            add_trade(good, 2, price=0.3, won=True)
        for _ in range(5):
            add_trade(bad, 2, price=0.3, won=False)

        stats = build_combo_stats({"good": good, "bad": bad})
        lb = build_leaderboard(stats, min_sample_size=5)
        self.assertEqual(lb.best_pnl.strategy, "good")
        self.assertEqual(lb.best_winrate.strategy, "good")

    def test_single_qualified_combo_does_not_trivially_win_over_a_better_small_sample(self):
        # Live bug: a strategy that happened to be the ONLY one past
        # min_sample_size topped best_pnl/best_winrate/best_value even
        # though it had a losing PnL and a 20% winrate, simply because
        # max() over a one-element "qualified" list has nothing to lose
        # to — meanwhile combos with real positive PnL sat one trade
        # below the threshold and were invisible to the comparison.
        only_qualified = VirtualWallet(strategy="barely_qualifies", initial_balance=100.0)
        for _ in range(4):
            add_trade(only_qualified, 2, price=0.5, won=False)
        add_trade(only_qualified, 2, price=0.5, won=True)  # 1W/4L, net negative

        actually_good = VirtualWallet(strategy="actually_good", initial_balance=100.0)
        for _ in range(3):
            add_trade(actually_good, 2, price=0.3, won=True)  # below min_sample_size, but net positive

        stats = build_combo_stats({"barely_qualifies": only_qualified, "actually_good": actually_good})
        lb = build_leaderboard(stats, min_sample_size=5)
        self.assertEqual(lb.best_pnl.strategy, "actually_good")
        self.assertEqual(lb.best_winrate.strategy, "actually_good")

    def test_falls_back_below_min_sample_size(self):
        w = VirtualWallet(strategy="tiny", initial_balance=100.0)
        add_trade(w, 12, price=0.5, won=True)
        stats = build_combo_stats({"tiny": w})
        lb = build_leaderboard(stats, min_sample_size=5)
        # only 1 trade exists, below the sample threshold -> still returned as fallback
        self.assertIsNotNone(lb.best_pnl)
        self.assertEqual(lb.best_pnl.strategy, "tiny")


if __name__ == "__main__":
    unittest.main()
