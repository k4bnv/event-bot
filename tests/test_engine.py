import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig, DashboardConfig, OkxConfig, StorageConfig, StrategyConfig
from src.engine import Engine
from src.mock_market import MockMarketDataProvider
from src.models import Direction, EventMarket, OrderBookLevel, OrderBookSnapshot
from src.storage import Storage
from src.strategies.base import Signal


def make_config(data_dir: Path) -> AppConfig:
    okx = OkxConfig(
        api_key="", api_secret="", api_passphrase="", base_url="", demo_trading=True,
        underlying_inst_id="BTC-USDT", series_ids=["BTC-UPDOWN-5MIN"], poll_interval_sec=1,
        request_timeout_sec=5, max_retries=1, retry_backoff_base_sec=1,
        settlement_poll_attempts=3, settlement_poll_interval_sec=1,
        funding_inst_id="BTC-USDT-SWAP", funding_refresh_interval_sec=300,
    )
    strategies = [
        StrategyConfig(
            name="breakout_retest", display_name="A", enabled=True, deposit_usd=100.0,
            entry_windows_min=[12, 7, 2], max_coefficient=0.55, stake_fraction=0.08, extra={},
        ),
        StrategyConfig(
            name="mean_reversion", display_name="C", enabled=True, deposit_usd=100.0,
            entry_windows_min=[7, 2], max_coefficient=0.5, stake_fraction=0.08, extra={},
        ),
    ]
    dashboard = DashboardConfig(mode="none", refresh_sec=2, web_host="127.0.0.1", web_port=8000)
    storage_cfg = StorageConfig(data_dir=data_dir, snapshot_every_sec=15)
    return AppConfig(mock_mode=True, okx=okx, strategies=strategies, dashboard=dashboard, storage=storage_cfg)


class EngineResetStrategyTests(unittest.TestCase):
    def _make_engine(self, tmp: Path) -> Engine:
        cfg = make_config(tmp)
        provider = MockMarketDataProvider(series_ids=cfg.okx.series_ids, seed=1)
        storage = Storage(tmp)
        return Engine(cfg, provider, storage)

    def test_reset_strategy_leaves_others_untouched(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            wallet_a = engine.wallets["breakout_retest"]
            wallet_c = engine.wallets["mean_reversion"]
            wallet_a.balance = 42.0  # simulate some activity
            wallet_c.balance = 77.0

            engine.reset_strategy("breakout_retest")

            self.assertEqual(engine.wallets["breakout_retest"].balance, 100.0)  # reset to deposit
            self.assertIsNot(engine.wallets["breakout_retest"], wallet_a)       # fresh instance
            self.assertIs(engine.wallets["mean_reversion"], wallet_c)          # untouched, same object
            self.assertEqual(wallet_c.balance, 77.0)
            engine.storage.close()

    def test_reset_strategy_unknown_name_raises(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            with self.assertRaises(ValueError):
                engine.reset_strategy("does_not_exist")
            engine.storage.close()

    def test_update_strategy_settings_partial_and_validation(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))

            engine.update_strategy_settings({"breakout_retest": {"deposit_usd": 250.0}})
            self.assertEqual(engine.wallets["breakout_retest"].initial_balance, 250.0)
            self.assertEqual(engine.wallets["mean_reversion"].initial_balance, 100.0)  # untouched value

            with self.assertRaises(ValueError):
                engine.update_strategy_settings({"breakout_retest": {"stake_fraction": 2.0}})

            with self.assertRaises(ValueError):
                engine.update_strategy_settings({
                    "breakout_retest": {"enabled": False}, "mean_reversion": {"enabled": False},
                })
            engine.storage.close()


class EngineOpenTradeUsesHonestFillPriceTests(unittest.IsolatedAsyncioTestCase):
    """Confirms the actual wiring in _open_due_trades — not just
    EventMarket.fill_price_for in isolation — uses the honest,
    depth-simulated fill price for a real opened Trade whenever the
    market carries a live order book, instead of the naive last/mid
    price. See models.py's fill_price_for docstring for why this matters
    (verified against live OKX data: routinely 40-1000%+ apart)."""

    def _make_engine(self, tmp: Path) -> Engine:
        cfg = make_config(tmp)
        provider = MockMarketDataProvider(series_ids=cfg.okx.series_ids, seed=1)
        storage = Storage(tmp)
        return Engine(cfg, provider, storage)

    async def test_opened_trade_entry_price_is_book_vwap_not_naive_price(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]

            # Naive quote says a dirt-cheap 0.05 — but the real ask book is
            # thin, so a real stake-sized order actually walks much higher.
            expiry_ts = time.time() + 60  # ~1 min out, comfortably inside the "2" checkpoint
            # Second level stays under breakout_retest's max_coefficient
            # (0.55, see make_config) so the trade isn't rejected outright —
            # the point here is a materially worse honest price, not a
            # rejected one.
            book = OrderBookSnapshot(
                ts=time.time(),
                asks=[OrderBookLevel(price=0.05, size=0.01), OrderBookLevel(price=0.30, size=1000.0)],
            )
            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=expiry_ts, floor_strike=50000.0, up_price=0.05, state="live", book=book,
            )
            engine.provider._active_markets[series_id] = market

            # Force a deterministic UP signal from ONE strategy, skip the rest.
            strategy = engine.strategy_instances["breakout_retest"]
            strategy.evaluate = lambda ctx: _async_result(Signal(direction=Direction.UP, reason="test"))
            other = engine.strategy_instances["mean_reversion"]
            other.evaluate = lambda ctx: _async_result(None)

            await engine._open_due_trades()

            # Multiple configured checkpoints (12/7/2) can all be "due" at
            # once this close to expiry — assert on every trade opened
            # rather than assuming exactly one.
            trades = engine.wallets["breakout_retest"].trades
            self.assertGreater(len(trades), 0)
            for trade in trades:
                self.assertGreater(trade.entry_price, 0.05)  # NOT the naive quoted price
                self.assertNotEqual(trade.entry_price, market.up_price)
                self.assertEqual(trade.contracts, trade.stake_usd / trade.entry_price)
            self.assertEqual(engine.wallets["mean_reversion"].trades, [])  # stubbed to no-signal
            engine.storage.close()


async def _async_result(value):
    return value


if __name__ == "__main__":
    unittest.main()
