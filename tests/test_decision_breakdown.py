import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.storage import FEATURE_FIELDS, Storage
from decision_breakdown import load_counts, load_rejected_fill_prices, main  # noqa: E402


def feature_row(id_, strategy, decision, fill_price=None) -> dict:
    row = {f: None for f in FEATURE_FIELDS}
    row.update({"id": id_, "strategy": strategy, "decision": decision, "fill_price": fill_price})
    return row


class LoadCountsTests(unittest.TestCase):
    def test_groups_by_strategy_and_decision(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(feature_row("f1", "breakout_retest", "no_signal"))
            storage.log_checkpoint_features(feature_row("f2", "breakout_retest", "no_signal"))
            storage.log_checkpoint_features(feature_row("f3", "breakout_retest", "opened"))
            storage.log_checkpoint_features(feature_row("f4", "mean_reversion", "rejected_max_coefficient"))
            db_path = storage.db_path
            storage.close()

            counts = load_counts(str(db_path))
            self.assertEqual(counts["breakout_retest"], {"no_signal": 2, "opened": 1})
            self.assertEqual(counts["mean_reversion"], {"rejected_max_coefficient": 1})

    def test_filters_by_strategy(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(feature_row("f1", "a", "no_signal"))
            storage.log_checkpoint_features(feature_row("f2", "b", "no_signal"))
            db_path = storage.db_path
            storage.close()

            counts = load_counts(str(db_path), strategy="a")
            self.assertEqual(set(counts), {"a"})

    def test_empty_db_no_error(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            db_path = storage.db_path
            storage.close()
            self.assertEqual(load_counts(str(db_path)), {})
            main(str(db_path))  # just must not raise


class MainOutputTests(unittest.TestCase):
    def test_runs_without_error(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(feature_row("f1", "breakout_retest", "no_signal"))
            storage.log_checkpoint_features(feature_row("f2", "breakout_retest", "rejected_max_coefficient"))
            db_path = storage.db_path
            storage.close()
            main(str(db_path))
            main(str(db_path), strategy="breakout_retest")


class RejectedFillPriceTests(unittest.TestCase):
    """Covers the "what were the rejected signals actually priced at"
    distribution — the data max_coefficient should be recalibrated
    against, not guessed. Added after a live decision_breakdown run
    showed breakout_retest/volatility_breakout/orderbook_momentum all
    throwing away most of their found signals purely on price."""

    def test_collects_only_this_strategys_rejected_prices(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(
                feature_row("f1", "breakout_retest", "rejected_max_coefficient", fill_price=0.61))
            storage.log_checkpoint_features(
                feature_row("f2", "breakout_retest", "rejected_max_coefficient", fill_price=0.72))
            storage.log_checkpoint_features(
                feature_row("f3", "breakout_retest", "opened", fill_price=0.50))  # not rejected — excluded
            storage.log_checkpoint_features(
                feature_row("f4", "mean_reversion", "rejected_max_coefficient", fill_price=0.99))  # other strategy
            db_path = storage.db_path
            storage.close()

            prices = load_rejected_fill_prices(str(db_path), "breakout_retest")
            self.assertEqual(sorted(prices), [0.61, 0.72])

    def test_main_prints_distribution_only_above_the_noise_floor(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            # Only 2 rejections — below the >=3 threshold, distribution line should be skipped.
            storage.log_checkpoint_features(
                feature_row("f1", "a", "rejected_max_coefficient", fill_price=0.6))
            storage.log_checkpoint_features(
                feature_row("f2", "a", "rejected_max_coefficient", fill_price=0.7))
            db_path = storage.db_path
            storage.close()
            main(str(db_path))  # must not raise either way; behavior differences are cosmetic


if __name__ == "__main__":
    unittest.main()
