import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig, DashboardConfig, OkxConfig, StorageConfig, StrategyConfig
from src.engine import Engine
from src.mock_market import MockMarketDataProvider
from src.storage import Storage


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


if __name__ == "__main__":
    unittest.main()
