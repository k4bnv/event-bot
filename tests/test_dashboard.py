import csv
import io
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.config import load_config
from src.dashboard_web import build_app
from src.engine import Engine
from src.mock_market import MockMarketDataProvider
from src.models import Direction, Trade
from src.storage import FEATURE_FIELDS, Storage


def make_app(tmp_path: Path):
    os.environ.pop("DASHBOARD_PASSWORD", None)
    cfg = load_config("config.yaml")
    cfg.storage.data_dir = tmp_path
    provider = MockMarketDataProvider(series_ids=cfg.okx.series_ids, seed=1)
    storage = Storage(tmp_path)
    engine = Engine(cfg, provider, storage)
    return build_app(cfg, engine), engine


def make_feature_row(id_="f1", strategy="a", ts=1000.0, decision="no_signal", **overrides) -> dict:
    row = {f: None for f in FEATURE_FIELDS}
    row.update({"id": id_, "strategy": strategy, "ts": ts, "decision": decision})
    row.update(overrides)
    return row


def make_trade(strategy: str, window_min: int, series_id: str = "S") -> Trade:
    return Trade(
        strategy=strategy, entry_window_min=window_min, series_id=series_id, inst_id="I",
        direction=Direction.UP, entry_price=0.4, stake_usd=10.0, contracts=25.0,
        opened_ts=time.time(), expiry_ts=time.time(),
    )


class StrategySummaryTests(unittest.TestCase):
    """Covers /api/state's strategy_summary — one combined entry per
    STRATEGY (not per checkpoint wallet), which is what the Analytics
    tab's equity/PnL charts render (see the chart redesign this was
    added for: 20+ per-checkpoint lines/bars was unreadable)."""

    def test_aggregates_across_a_strategys_checkpoint_wallets(self):
        with TemporaryDirectory() as tmp:
            app, engine = make_app(Path(tmp))
            client = TestClient(app)

            wallet_12 = engine.wallet_for("breakout_retest", 12)
            wallet_2 = engine.wallet_for("breakout_retest", 2)
            t1 = make_trade("breakout_retest", 12)
            wallet_12.open_trade(t1)
            wallet_12.settle_trade(t1, won=True)   # pnl = +15.0
            t2 = make_trade("breakout_retest", 2)
            wallet_2.open_trade(t2)
            wallet_2.settle_trade(t2, won=False)   # pnl = -10.0

            resp = client.get("/api/state")
            self.assertEqual(resp.status_code, 200)
            summary = {s["strategy"]: s for s in resp.json()["strategy_summary"]}
            row = summary["breakout_retest"]

            # 3 checkpoints (12/7/2), each starting at its own full deposit_usd.
            self.assertEqual(row["initial_balance"], 300.0)
            self.assertAlmostEqual(row["net_pnl"], 5.0)  # +15 - 10
            self.assertAlmostEqual(row["equity"], 305.0)
            self.assertGreaterEqual(len(row["equity_curve"]), 2)  # not just a flat single point
            engine.storage.close()

    def test_dynamic_timing_strategy_still_gets_exactly_one_summary_row(self):
        with TemporaryDirectory() as tmp:
            app, engine = make_app(Path(tmp))
            client = TestClient(app)

            wallet = engine.wallet_for("adaptive_timing")
            t = make_trade("adaptive_timing", 5)
            wallet.open_trade(t)
            wallet.settle_trade(t, won=True)

            resp = client.get("/api/state")
            rows = [s for s in resp.json()["strategy_summary"] if s["strategy"] == "adaptive_timing"]
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["net_pnl"], 15.0)
            engine.storage.close()

    def test_untouched_strategy_has_zero_net_pnl_and_a_fallback_curve(self):
        with TemporaryDirectory() as tmp:
            app, engine = make_app(Path(tmp))
            client = TestClient(app)
            resp = client.get("/api/state")
            summary = {s["strategy"]: s for s in resp.json()["strategy_summary"]}
            row = summary["mean_reversion"]
            self.assertEqual(row["net_pnl"], 0.0)
            self.assertEqual(len(row["equity_curve"]), 1)  # the "no trades yet" fallback point
            engine.storage.close()


class FeaturesExportTests(unittest.TestCase):
    """Covers the "Данные для обучения (ML)" card's export button and row
    counter — /api/features/export.csv and /api/features/count."""

    def test_export_csv_has_the_right_header_and_rows(self):
        with TemporaryDirectory() as tmp:
            app, engine = make_app(Path(tmp))
            client = TestClient(app)
            engine.storage.log_checkpoint_features(make_feature_row(id_="f1", strategy="a", extra_json='{"tfi": -0.5}'))
            engine.storage.log_checkpoint_features(make_feature_row(id_="f2", strategy="b"))

            resp = client.get("/api/features/export.csv")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("attachment", resp.headers["content-disposition"])
            reader = csv.DictReader(io.StringIO(resp.text))
            self.assertEqual(reader.fieldnames, FEATURE_FIELDS)
            rows = list(reader)
            self.assertEqual({r["id"] for r in rows}, {"f1", "f2"})
            self.assertEqual(next(r for r in rows if r["id"] == "f1")["extra_json"], '{"tfi": -0.5}')
            engine.storage.close()

    def test_export_csv_filters_by_strategy(self):
        with TemporaryDirectory() as tmp:
            app, engine = make_app(Path(tmp))
            client = TestClient(app)
            engine.storage.log_checkpoint_features(make_feature_row(id_="f1", strategy="a"))
            engine.storage.log_checkpoint_features(make_feature_row(id_="f2", strategy="b"))

            resp = client.get("/api/features/export.csv", params={"strategy": "a"})
            rows = list(csv.DictReader(io.StringIO(resp.text)))
            self.assertEqual([r["id"] for r in rows], ["f1"])
            engine.storage.close()

    def test_count_matches_storage(self):
        with TemporaryDirectory() as tmp:
            app, engine = make_app(Path(tmp))
            client = TestClient(app)
            engine.storage.log_checkpoint_features(make_feature_row(id_="f1", strategy="a"))
            engine.storage.log_checkpoint_features(make_feature_row(id_="f2", strategy="a"))
            engine.storage.log_checkpoint_features(make_feature_row(id_="f3", strategy="b"))

            resp = client.get("/api/features/count")
            self.assertEqual(resp.json()["count"], 3)
            resp_a = client.get("/api/features/count", params={"strategy": "a"})
            self.assertEqual(resp_a.json()["count"], 2)
            engine.storage.close()


if __name__ == "__main__":
    unittest.main()
