import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig, DashboardConfig, OkxConfig, StorageConfig, StrategyConfig
from src.engine import Engine
from src.mock_market import MockMarketDataProvider
from src.models import Direction, EventMarket, OrderBookLevel, OrderBookSnapshot, Trade, TradeStatus
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

    async def test_signal_rejected_when_slippage_exceeds_max_slippage_pct(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]

            s_cfg = next(s for s in engine.cfg.strategies if s.name == "breakout_retest")
            s_cfg.max_slippage_pct = 50.0  # reject anything more than 50% worse than quoted

            expiry_ts = time.time() + 60
            # Same book as the VWAP test above: quoted 0.05, real fill ~0.30
            # — ~500% worse, way over this strategy's 50% cap.
            book = OrderBookSnapshot(
                ts=time.time(),
                asks=[OrderBookLevel(price=0.05, size=0.01), OrderBookLevel(price=0.30, size=1000.0)],
            )
            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=expiry_ts, floor_strike=50000.0, up_price=0.05, state="live", book=book,
            )
            engine.provider._active_markets[series_id] = market

            strategy = engine.strategy_instances["breakout_retest"]
            strategy.evaluate = lambda ctx: _async_result(Signal(direction=Direction.UP, reason="test"))
            other = engine.strategy_instances["mean_reversion"]
            other.evaluate = lambda ctx: _async_result(None)

            await engine._open_due_trades()

            self.assertEqual(engine.wallets["breakout_retest"].trades, [])
            engine.storage.close()

    async def test_signal_allowed_when_slippage_within_max_slippage_pct(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]

            s_cfg = next(s for s in engine.cfg.strategies if s.name == "breakout_retest")
            s_cfg.max_slippage_pct = 1000.0  # generous cap — this fill should pass

            expiry_ts = time.time() + 60
            book = OrderBookSnapshot(
                ts=time.time(),
                asks=[OrderBookLevel(price=0.05, size=0.01), OrderBookLevel(price=0.30, size=1000.0)],
            )
            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=expiry_ts, floor_strike=50000.0, up_price=0.05, state="live", book=book,
            )
            engine.provider._active_markets[series_id] = market

            strategy = engine.strategy_instances["breakout_retest"]
            strategy.evaluate = lambda ctx: _async_result(Signal(direction=Direction.UP, reason="test"))
            other = engine.strategy_instances["mean_reversion"]
            other.evaluate = lambda ctx: _async_result(None)

            await engine._open_due_trades()

            self.assertGreater(len(engine.wallets["breakout_retest"].trades), 0)
            engine.storage.close()


async def _async_result(value):
    return value


class SettleExpiredTradesThrottleTests(unittest.IsolatedAsyncioTestCase):
    """Covers the settlement-polling fix: settlement_poll_interval_sec was
    parsed from config but never actually used — a check fired on every
    single engine tick regardless, so the real give-up window was
    `settlement_poll_attempts * poll_interval_sec`, not
    `* settlement_poll_interval_sec` as config.yaml implied. Live testing
    showed real trades going UNRESOLVED within that (too-short) window."""

    def _make_engine(self, tmp: Path) -> Engine:
        cfg = make_config(tmp)
        provider = MockMarketDataProvider(series_ids=cfg.okx.series_ids, seed=1)
        storage = Storage(tmp)
        return Engine(cfg, provider, storage)

    def _make_expired_trade(self) -> Trade:
        return Trade(
            strategy="breakout_retest", entry_window_min=7, series_id="S", inst_id="I",
            direction=Direction.UP, entry_price=0.4, stake_usd=10.0, contracts=25.0,
            opened_ts=time.time() - 120, expiry_ts=time.time() - 60,
        )

    def _expire_throttle_window(self, engine: Engine, trade_id: str) -> None:
        """Simulate settlement_poll_interval_sec having elapsed since the
        last check, without an actual sleep."""
        if trade_id in engine._last_settlement_check:
            engine._last_settlement_check[trade_id] -= engine.cfg.okx.settlement_poll_interval_sec + 0.01

    async def test_settlement_check_is_throttled_by_interval(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            wallet = engine.wallets["breakout_retest"]
            trade = self._make_expired_trade()
            wallet.open_trade(trade)

            call_count = 0

            async def fake_check_settlement(series_id, inst_id):
                nonlocal call_count
                call_count += 1
                return None

            engine.provider.check_settlement = fake_check_settlement

            # Three ticks in immediate succession — should only actually
            # call check_settlement once, not three times.
            await engine._settle_expired_trades()
            await engine._settle_expired_trades()
            await engine._settle_expired_trades()
            self.assertEqual(call_count, 1)

            self._expire_throttle_window(engine, trade.id)
            await engine._settle_expired_trades()
            self.assertEqual(call_count, 2)
            engine.storage.close()

    async def test_marks_unresolved_after_max_attempts_and_refunds_stake(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            wallet = engine.wallets["breakout_retest"]
            trade = self._make_expired_trade()
            wallet.open_trade(trade)
            balance_after_open = wallet.balance

            async def never_settles(series_id, inst_id):
                return None

            engine.provider.check_settlement = never_settles

            max_attempts = engine.cfg.okx.settlement_poll_attempts
            for _ in range(max_attempts):
                await engine._settle_expired_trades()
                self._expire_throttle_window(engine, trade.id)

            self.assertEqual(trade.status, TradeStatus.UNRESOLVED)
            self.assertEqual(trade.pnl_usd, 0.0)
            self.assertEqual(wallet.balance, balance_after_open + trade.stake_usd)  # stake refunded
            self.assertNotIn(trade.id, engine._settlement_attempts)
            self.assertNotIn(trade.id, engine._last_settlement_check)
            engine.storage.close()

    async def test_settles_normally_once_outcome_is_available(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            wallet = engine.wallets["breakout_retest"]
            trade = self._make_expired_trade()  # direction UP
            wallet.open_trade(trade)

            async def settles_up(series_id, inst_id):
                return Direction.UP

            engine.provider.check_settlement = settles_up
            await engine._settle_expired_trades()

            self.assertEqual(trade.status, TradeStatus.WON)
            self.assertNotIn(trade.id, engine._settlement_attempts)
            self.assertNotIn(trade.id, engine._last_settlement_check)
            engine.storage.close()


class ActivityFeedTests(unittest.IsolatedAsyncioTestCase):
    """The live per-strategy log (Dashboard's "Логи" tab) backing
    ActivityEvent/_log_activity/activity_since — covers that opening,
    rejecting, and settling a trade each produce the right `kind`, and
    that incremental polling (since_id) and the strategy filter work."""

    def _make_engine(self, tmp: Path) -> Engine:
        cfg = make_config(tmp)
        provider = MockMarketDataProvider(series_ids=cfg.okx.series_ids, seed=1)
        storage = Storage(tmp)
        return Engine(cfg, provider, storage)

    async def test_opened_trade_logs_opened_event(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]

            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=time.time() + 60, floor_strike=50000.0, up_price=0.4, state="live",
            )
            engine.provider._active_markets[series_id] = market
            engine.strategy_instances["breakout_retest"].evaluate = (
                lambda ctx: _async_result(Signal(direction=Direction.UP, reason="test"))
            )
            engine.strategy_instances["mean_reversion"].evaluate = lambda ctx: _async_result(None)

            await engine._open_due_trades()

            events = engine.activity_since()
            kinds = {e.kind for e in events if e.strategy == "breakout_retest"}
            self.assertIn("opened", kinds)
            opened = next(e for e in events if e.kind == "opened")
            self.assertEqual(opened.strategy, "breakout_retest")
            self.assertIn("UP", opened.message)

            # the stubbed no-signal strategy logged its own event too
            self.assertTrue(any(e.strategy == "mean_reversion" and e.kind == "no_signal" for e in events))
            engine.storage.close()

    async def test_rejected_by_max_coefficient_logs_rejected_event(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]
            s_cfg = next(s for s in engine.cfg.strategies if s.name == "breakout_retest")

            # naive/fill price (0.9) sits above this strategy's max_coefficient (0.55)
            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=time.time() + 60, floor_strike=50000.0, up_price=0.9, state="live",
            )
            engine.provider._active_markets[series_id] = market
            engine.strategy_instances["breakout_retest"].evaluate = (
                lambda ctx: _async_result(Signal(direction=Direction.UP, reason="test"))
            )
            engine.strategy_instances["mean_reversion"].evaluate = lambda ctx: _async_result(None)

            await engine._open_due_trades()

            self.assertEqual(engine.wallets["breakout_retest"].trades, [])
            rejected = [e for e in engine.activity_since() if e.strategy == "breakout_retest"]
            self.assertTrue(all(e.kind == "rejected" for e in rejected))
            self.assertTrue(any(f"{s_cfg.max_coefficient:.2f}" in e.message for e in rejected))
            engine.storage.close()

    async def test_settle_logs_won_and_lost(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            wallet = engine.wallets["breakout_retest"]
            t_won = Trade(
                strategy="breakout_retest", entry_window_min=7, series_id="S", inst_id="I1",
                direction=Direction.UP, entry_price=0.4, stake_usd=10.0, contracts=25.0,
                opened_ts=time.time() - 120, expiry_ts=time.time() - 60,
            )
            t_lost = Trade(
                strategy="breakout_retest", entry_window_min=7, series_id="S", inst_id="I2",
                direction=Direction.DOWN, entry_price=0.4, stake_usd=10.0, contracts=25.0,
                opened_ts=time.time() - 120, expiry_ts=time.time() - 60,
            )
            wallet.open_trade(t_won)
            wallet.open_trade(t_lost)

            async def settle(series_id, inst_id):
                return Direction.UP  # t_won's direction wins, t_lost's loses

            engine.provider.check_settlement = settle
            await engine._settle_expired_trades()

            events = engine.activity_since()
            self.assertTrue(any(e.kind == "won" and "I1" in e.message for e in events))
            self.assertTrue(any(e.kind == "lost" and "I2" in e.message for e in events))
            engine.storage.close()

    async def test_activity_since_filters_by_id_and_strategy(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            engine._log_activity("breakout_retest", "S", 7, "no_signal", "первое")
            marker_id = engine.latest_activity_id()
            engine._log_activity("mean_reversion", "S", 2, "no_signal", "второе")
            engine._log_activity("breakout_retest", "S", 7, "opened", "третье")

            since = engine.activity_since(since_id=marker_id)
            self.assertEqual([e.message for e in since], ["второе", "третье"])

            only_breakout = engine.activity_since(strategy="breakout_retest")
            self.assertEqual([e.message for e in only_breakout], ["первое", "третье"])
            engine.storage.close()


if __name__ == "__main__":
    unittest.main()
