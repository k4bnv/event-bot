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
            wallet_a_12 = engine.wallet_for("breakout_retest", 12)
            wallet_a_2 = engine.wallet_for("breakout_retest", 2)
            wallet_c = engine.wallet_for("mean_reversion", 7)
            wallet_a_12.balance = 42.0  # simulate some activity
            wallet_a_2.balance = 55.0
            wallet_c.balance = 77.0

            engine.reset_strategy("breakout_retest")

            # EVERY one of breakout_retest's checkpoint wallets resets, not just one.
            self.assertEqual(engine.wallet_for("breakout_retest", 12).balance, 100.0)
            self.assertEqual(engine.wallet_for("breakout_retest", 2).balance, 100.0)
            self.assertIsNot(engine.wallet_for("breakout_retest", 12), wallet_a_12)  # fresh instance
            self.assertIsNot(engine.wallet_for("breakout_retest", 2), wallet_a_2)
            self.assertIs(engine.wallet_for("mean_reversion", 7), wallet_c)  # untouched, same object
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
            self.assertEqual(engine.wallet_for("breakout_retest", 12).initial_balance, 250.0)
            self.assertEqual(engine.wallet_for("mean_reversion", 7).initial_balance, 100.0)  # untouched value

            with self.assertRaises(ValueError):
                engine.update_strategy_settings({"breakout_retest": {"stake_fraction": 2.0}})

            with self.assertRaises(ValueError):
                engine.update_strategy_settings({
                    "breakout_retest": {"enabled": False}, "mean_reversion": {"enabled": False},
                })
            engine.storage.close()


class WalletRestoreOnStartupTests(unittest.TestCase):
    """A redeploy/crash/restart used to silently reset every wallet back
    to deposit_usd — write_snapshot() persisted balances to data/bot.db,
    but nothing ever read them back on the next Engine(...). These cover
    the fix: a new Engine instance sharing the same Storage should resume
    where the last one left off, not restart from scratch."""

    def _make_engine(self, tmp: Path) -> Engine:
        cfg = make_config(tmp)
        provider = MockMarketDataProvider(series_ids=cfg.okx.series_ids, seed=1)
        storage = Storage(tmp)
        return Engine(cfg, provider, storage)

    def test_second_engine_resumes_balance_from_storage(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            engine1 = self._make_engine(tmp_path)
            engine1.wallet_for("breakout_retest", 12).balance = 142.5  # simulate accumulated profit
            engine1.storage.write_snapshot(engine1.wallets)
            engine1.storage.close()

            # A brand new Engine, same data dir/DB — simulates the process
            # restarting (redeploy) with the same persistent volume.
            storage2 = Storage(tmp_path)
            cfg2 = make_config(tmp_path)
            provider2 = MockMarketDataProvider(series_ids=cfg2.okx.series_ids, seed=1)
            engine2 = Engine(cfg2, provider2, storage2)

            self.assertEqual(engine2.wallet_for("breakout_retest", 12).balance, 142.5)
            self.assertEqual(engine2.wallet_for("breakout_retest", 12).initial_balance, 100.0)
            # A DIFFERENT checkpoint of the SAME strategy is a completely
            # separate wallet — untouched by the "12" one's saved balance.
            self.assertEqual(engine2.wallet_for("breakout_retest", 2).balance, 100.0)
            engine2.storage.close()

    def test_reserved_capital_is_refunded_to_balance_on_restore(self):
        # A trade was still open (stake reserved, not yet settled) at the
        # moment of the last snapshot — that specific Trade object is gone
        # after a restart, so nothing will ever settle it and return the
        # stake on its own. It must come back to balance, not vanish.
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            engine1 = self._make_engine(tmp_path)
            wallet = engine1.wallet_for("breakout_retest", 12)
            wallet.balance = 80.0
            wallet.reserved = 20.0  # e.g. one $20 stake still in flight
            engine1.storage.write_snapshot(engine1.wallets)
            engine1.storage.close()

            storage2 = Storage(tmp_path)
            cfg2 = make_config(tmp_path)
            provider2 = MockMarketDataProvider(series_ids=cfg2.okx.series_ids, seed=1)
            engine2 = Engine(cfg2, provider2, storage2)

            restored = engine2.wallet_for("breakout_retest", 12)
            self.assertEqual(restored.balance, 100.0)  # 80 + the refunded 20
            self.assertEqual(restored.reserved, 0.0)
            engine2.storage.close()

    def test_engine_reset_wipes_storage_before_rebuilding_wallets(self):
        # Regression guard for the ordering bug this restore feature could
        # introduce: reset() must clear storage BEFORE rebuilding wallets,
        # or the rebuild would immediately reload the very balance the
        # reset is supposed to erase (load_wallets() would still see it).
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            engine = self._make_engine(tmp_path)
            engine.wallet_for("breakout_retest", 12).balance = 55.0
            engine.storage.write_snapshot(engine.wallets)

            engine.reset()

            self.assertEqual(engine.wallet_for("breakout_retest", 12).balance, 100.0)
            self.assertEqual(engine.storage.load_wallets(), {})  # nothing left to restore either
            engine.storage.close()

    def test_no_saved_row_starts_fresh_from_config(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))  # nothing ever written to storage
            self.assertEqual(engine.wallet_for("breakout_retest", 12).balance, 100.0)
            self.assertEqual(engine.wallet_for("breakout_retest", 12).initial_balance, 100.0)
            engine.storage.close()

    def test_each_checkpoint_gets_its_own_independent_wallet(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            wallets = [engine.wallet_for("breakout_retest", w) for w in (12, 7, 2)]
            self.assertEqual(len({id(w) for w in wallets}), 3)  # three distinct objects
            for w in wallets:
                self.assertEqual(w.balance, 100.0)  # each gets the FULL deposit, not a split of it
                self.assertEqual(w.strategy, "breakout_retest")
            self.assertEqual({w.window_min for w in wallets}, {12, 7, 2})
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
            _prime_entry_window(engine, series_id, expiry_ts, "breakout_retest")

            # Force a deterministic UP signal from ONE strategy, skip the rest.
            strategy = engine.strategy_instances["breakout_retest"]
            strategy.evaluate = lambda ctx: _async_result(Signal(direction=Direction.UP, reason="test"))
            other = engine.strategy_instances["mean_reversion"]
            other.evaluate = lambda ctx: _async_result(None)

            await engine._open_due_trades()

            # Exactly the "2" checkpoint is reachable/due here (see
            # _prime_entry_window) -> exactly one trade, not "however many
            # of [12, 7, 2] happen to be due at once".
            trades = engine.wallet_for("breakout_retest", 2).trades
            self.assertEqual(len(trades), 1)
            trade = trades[0]
            self.assertGreater(trade.entry_price, 0.05)  # NOT the naive quoted price
            self.assertNotEqual(trade.entry_price, market.up_price)
            self.assertEqual(trade.contracts, trade.stake_usd / trade.entry_price)
            self.assertEqual(engine.wallet_for("mean_reversion", 2).trades, [])  # stubbed to no-signal
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
            _prime_entry_window(engine, series_id, expiry_ts, "breakout_retest")

            strategy = engine.strategy_instances["breakout_retest"]
            strategy.evaluate = lambda ctx: _async_result(Signal(direction=Direction.UP, reason="test"))
            other = engine.strategy_instances["mean_reversion"]
            other.evaluate = lambda ctx: _async_result(None)

            await engine._open_due_trades()

            self.assertEqual(engine.wallet_for("breakout_retest", 2).trades, [])
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
            _prime_entry_window(engine, series_id, expiry_ts, "breakout_retest")

            strategy = engine.strategy_instances["breakout_retest"]
            strategy.evaluate = lambda ctx: _async_result(Signal(direction=Direction.UP, reason="test"))
            other = engine.strategy_instances["mean_reversion"]
            other.evaluate = lambda ctx: _async_result(None)

            await engine._open_due_trades()

            self.assertEqual(len(engine.wallet_for("breakout_retest", 2).trades), 1)
            engine.storage.close()


async def _async_result(value):
    return value


def _prime_entry_window(engine: Engine, series_id: str, expiry_ts: float, strategy_name: str) -> None:
    """Establish this (series, expiry, strategy) window's "start" remaining
    time in EntryWindowManager BEFORE calling _open_due_trades() with a
    market that's already close to expiry — otherwise the very first
    observation IS the close-to-expiry one, and EntryWindowManager
    correctly refuses to fire any configured checkpoint the window could
    never have genuinely crossed from above (see timing.py's docstring).
    A real engine naturally "primes" every window this way just by
    polling it starting minutes earlier; these tests jump straight to
    a near-expiry snapshot; this line stands in for that earlier polling
    so exactly the checkpoints the test intends to be reachable are.
    """
    s_cfg = next(s for s in engine.cfg.strategies if s.name == strategy_name)
    engine.timing.due_windows(series_id, expiry_ts, strategy_name, 300.0, s_cfg.entry_windows_min)


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
            wallet = engine.wallet_for("breakout_retest", 7)
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
            wallet = engine.wallet_for("breakout_retest", 7)
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
            wallet = engine.wallet_for("breakout_retest", 7)
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

            expiry_ts = time.time() + 60
            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=expiry_ts, floor_strike=50000.0, up_price=0.4, state="live",
            )
            engine.provider._active_markets[series_id] = market
            # Both strategies get evaluated in this test -> both need their
            # own window primed (EntryWindowManager keys on strategy name too).
            _prime_entry_window(engine, series_id, expiry_ts, "breakout_retest")
            _prime_entry_window(engine, series_id, expiry_ts, "mean_reversion")
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
            expiry_ts = time.time() + 60
            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=expiry_ts, floor_strike=50000.0, up_price=0.9, state="live",
            )
            engine.provider._active_markets[series_id] = market
            _prime_entry_window(engine, series_id, expiry_ts, "breakout_retest")
            engine.strategy_instances["breakout_retest"].evaluate = (
                lambda ctx: _async_result(Signal(direction=Direction.UP, reason="test"))
            )
            engine.strategy_instances["mean_reversion"].evaluate = lambda ctx: _async_result(None)

            await engine._open_due_trades()

            self.assertEqual(engine.wallet_for("breakout_retest", 2).trades, [])
            rejected = [e for e in engine.activity_since() if e.strategy == "breakout_retest"]
            self.assertTrue(all(e.kind == "rejected" for e in rejected))
            self.assertTrue(any(f"{s_cfg.max_coefficient:.2f}" in e.message for e in rejected))
            engine.storage.close()

    async def test_settle_logs_won_and_lost(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            wallet = engine.wallet_for("breakout_retest", 7)
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


class PreviousOutcomeTrackingTests(unittest.IsolatedAsyncioTestCase):
    """Covers Engine._update_previous_outcomes() — the plumbing
    prior_window_momentum needs: learning a window's settlement outcome
    the moment it rolls over, independent of whether any strategy actually
    traded it (unlike the normal per-Trade settlement-polling path)."""

    def _make_engine(self, tmp: Path) -> Engine:
        cfg = make_config(tmp)
        provider = MockMarketDataProvider(series_ids=cfg.okx.series_ids, seed=1)
        storage = Storage(tmp)
        return Engine(cfg, provider, storage)

    @staticmethod
    def _set_market(engine: Engine, series_id: str, inst_id: str) -> None:
        engine.provider._active_markets[series_id] = EventMarket(
            series_id=series_id, method="price_up_down", inst_id=inst_id,
            expiry_ts=time.time() + 300, floor_strike=50000.0, up_price=0.5, state="live",
        )

    async def test_no_lookup_on_first_ever_observation(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]
            self._set_market(engine, series_id, "INST-A")

            calls = []
            engine.provider.check_settlement = lambda s, i: calls.append((s, i)) or _resolved(None)

            await engine._update_previous_outcomes()

            self.assertEqual(calls, [])  # nothing to look up yet — no prior window observed at all
            self.assertIsNone(engine._previous_outcome.get(series_id))
            self.assertEqual(engine._last_inst_id[series_id], "INST-A")
            engine.storage.close()

    async def test_rollover_resolves_immediately(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]
            self._set_market(engine, series_id, "INST-A")
            await engine._update_previous_outcomes()  # establishes INST-A as "last seen"

            self._set_market(engine, series_id, "INST-B")  # rollover
            engine.provider.check_settlement = lambda s, i: _resolved(Direction.UP if i == "INST-A" else None)

            await engine._update_previous_outcomes()

            self.assertEqual(engine._previous_outcome[series_id], Direction.UP)
            self.assertNotIn(series_id, engine._pending_outcome_inst_id)  # resolved immediately, nothing pending
            self.assertEqual(engine._last_inst_id[series_id], "INST-B")
            engine.storage.close()

    async def test_rollover_pending_then_resolves_on_retry(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]
            self._set_market(engine, series_id, "INST-A")
            await engine._update_previous_outcomes()

            self._set_market(engine, series_id, "INST-B")
            responses = iter([None, None, Direction.DOWN])  # not yet, not yet, resolved
            engine.provider.check_settlement = lambda s, i: _resolved(next(responses))

            await engine._update_previous_outcomes()  # rollover tick: first attempt -> None
            self.assertIsNone(engine._previous_outcome.get(series_id))
            self.assertEqual(engine._pending_outcome_inst_id[series_id], "INST-A")

            # advance past settlement_poll_interval_sec so the retry is due
            engine._last_outcome_check[series_id] -= engine.cfg.okx.settlement_poll_interval_sec + 1
            await engine._update_previous_outcomes()  # second attempt -> still None
            self.assertIsNone(engine._previous_outcome.get(series_id))

            engine._last_outcome_check[series_id] -= engine.cfg.okx.settlement_poll_interval_sec + 1
            await engine._update_previous_outcomes()  # third attempt -> resolved
            self.assertEqual(engine._previous_outcome[series_id], Direction.DOWN)
            self.assertNotIn(series_id, engine._pending_outcome_inst_id)
            engine.storage.close()

    async def test_gives_up_after_max_attempts_but_keeps_stale_reading(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]
            engine._previous_outcome[series_id] = Direction.UP  # a known-good earlier reading

            self._set_market(engine, series_id, "INST-A")
            await engine._update_previous_outcomes()

            self._set_market(engine, series_id, "INST-B")
            engine.provider.check_settlement = lambda s, i: _resolved(None)  # never resolves

            await engine._update_previous_outcomes()  # rollover attempt (1st)
            for _ in range(engine.cfg.okx.settlement_poll_attempts):
                engine._last_outcome_check[series_id] -= engine.cfg.okx.settlement_poll_interval_sec + 1
                await engine._update_previous_outcomes()

            self.assertNotIn(series_id, engine._pending_outcome_inst_id)  # gave up
            self.assertEqual(engine._previous_outcome[series_id], Direction.UP)  # stale reading kept, not wiped
            engine.storage.close()

    async def test_open_due_trades_passes_previous_outcome_into_context(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]
            engine._previous_outcome[series_id] = Direction.DOWN

            # _prime_entry_window() primes at a fixed remaining_sec=300 —
            # the real market here needs LESS remaining than that so a
            # checkpoint actually crosses (same pattern as the other
            # _open_due_trades tests above in this file).
            expiry_ts = time.time() + 60
            engine.provider._active_markets[series_id] = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="INST-CURRENT",
                expiry_ts=expiry_ts, floor_strike=50000.0, up_price=0.5, state="live",
            )
            _prime_entry_window(engine, series_id, expiry_ts, "breakout_retest")

            seen_ctx = []

            async def _capture(ctx):
                seen_ctx.append(ctx)
                return None

            engine.strategy_instances["breakout_retest"].evaluate = _capture
            engine.strategy_instances["mean_reversion"].evaluate = lambda ctx: _resolved(None)

            await engine._open_due_trades()

            self.assertEqual(len(seen_ctx), 1)
            self.assertEqual(seen_ctx[0].previous_outcome, Direction.DOWN)
            engine.storage.close()


class CheckpointFeatureLoggingTests(unittest.IsolatedAsyncioTestCase):
    """Covers Engine._record_checkpoint_features — every decision path in
    _open_due_trades (no_signal, each rejection reason, opened) must log a
    row, since a dataset for future ML work needs the negative examples
    too, not just executed trades. See storage.py's module docstring."""

    def _make_engine(self, tmp: Path) -> Engine:
        cfg = make_config(tmp)
        provider = MockMarketDataProvider(series_ids=cfg.okx.series_ids, seed=1)
        storage = Storage(tmp)
        return Engine(cfg, provider, storage)

    async def test_no_signal_logs_a_feature_row(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]
            expiry_ts = time.time() + 60
            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=expiry_ts, floor_strike=50000.0, up_price=0.4, state="live",
            )
            engine.provider._active_markets[series_id] = market
            _prime_entry_window(engine, series_id, expiry_ts, "breakout_retest")
            engine.strategy_instances["breakout_retest"].evaluate = lambda ctx: _resolved(None)
            engine.strategy_instances["mean_reversion"].evaluate = lambda ctx: _resolved(None)

            await engine._open_due_trades()

            rows = engine.storage.get_checkpoint_features(strategy="breakout_retest")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["decision"], "no_signal")
            self.assertIsNone(rows[0]["signal_direction"])
            self.assertIsNone(rows[0]["trade_id"])
            engine.storage.close()

    async def test_rejected_by_max_coefficient_logs_a_feature_row_with_fill_price(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]
            s_cfg = next(s for s in engine.cfg.strategies if s.name == "breakout_retest")
            expiry_ts = time.time() + 60
            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=expiry_ts, floor_strike=50000.0, up_price=0.9, state="live",
            )
            engine.provider._active_markets[series_id] = market
            _prime_entry_window(engine, series_id, expiry_ts, "breakout_retest")
            engine.strategy_instances["breakout_retest"].evaluate = (
                lambda ctx: _resolved(Signal(direction=Direction.UP, reason="test"))
            )
            engine.strategy_instances["mean_reversion"].evaluate = lambda ctx: _resolved(None)

            await engine._open_due_trades()

            rows = engine.storage.get_checkpoint_features(strategy="breakout_retest")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["decision"], "rejected_max_coefficient")
            self.assertEqual(rows[0]["signal_direction"], "up")
            self.assertAlmostEqual(rows[0]["fill_price"], 0.9)
            self.assertLess(s_cfg.max_coefficient, 0.9)  # sanity: this really is why it was rejected
            engine.storage.close()

    async def test_opened_trade_logs_a_feature_row_linked_to_the_trade(self):
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]
            expiry_ts = time.time() + 60
            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=expiry_ts, floor_strike=50000.0, up_price=0.4, state="live",
            )
            engine.provider._active_markets[series_id] = market
            _prime_entry_window(engine, series_id, expiry_ts, "breakout_retest")
            engine.strategy_instances["breakout_retest"].evaluate = (
                lambda ctx: _resolved(Signal(direction=Direction.UP, reason="test"))
            )
            engine.strategy_instances["mean_reversion"].evaluate = lambda ctx: _resolved(None)

            await engine._open_due_trades()

            trade = engine.wallet_for("breakout_retest", 2).trades[0]
            rows = engine.storage.get_checkpoint_features(strategy="breakout_retest")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["decision"], "opened")
            self.assertEqual(rows[0]["trade_id"], trade.id)
            self.assertEqual(rows[0]["signal_direction"], "up")
            self.assertAlmostEqual(rows[0]["stake_usd"], trade.stake_usd)
            engine.storage.close()

    async def test_a_logging_failure_never_blocks_the_real_trade(self):
        # storage.log_checkpoint_features raising must not stop
        # wallet.open_trade from actually happening — this is pure
        # instrumentation, never load-bearing for trading logic.
        with TemporaryDirectory() as tmp:
            engine = self._make_engine(Path(tmp))
            series_id = engine.cfg.okx.series_ids[0]
            expiry_ts = time.time() + 60
            market = EventMarket(
                series_id=series_id, method="price_up_down", inst_id="TEST-INST",
                expiry_ts=expiry_ts, floor_strike=50000.0, up_price=0.4, state="live",
            )
            engine.provider._active_markets[series_id] = market
            _prime_entry_window(engine, series_id, expiry_ts, "breakout_retest")
            engine.strategy_instances["breakout_retest"].evaluate = (
                lambda ctx: _resolved(Signal(direction=Direction.UP, reason="test"))
            )
            engine.strategy_instances["mean_reversion"].evaluate = lambda ctx: _resolved(None)
            engine.storage.log_checkpoint_features = lambda row: (_ for _ in ()).throw(RuntimeError("boom"))

            await engine._open_due_trades()  # must not raise

            self.assertEqual(len(engine.wallet_for("breakout_retest", 2).trades), 1)
            engine.storage.close()


def _resolved(value):
    async def _inner():
        return value
    return _inner()


if __name__ == "__main__":
    unittest.main()
