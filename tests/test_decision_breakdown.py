import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.storage import FEATURE_FIELDS, Storage
from decision_breakdown import load_counts, main  # noqa: E402


def feature_row(id_, strategy, decision) -> dict:
    row = {f: None for f in FEATURE_FIELDS}
    row.update({"id": id_, "strategy": strategy, "decision": decision})
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


if __name__ == "__main__":
    unittest.main()
