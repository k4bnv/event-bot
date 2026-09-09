import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.storage import FEATURE_FIELDS
from analyze_absorption_funnel import load_rows, main  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        for row in rows:
            full = {f: "" for f in FEATURE_FIELDS}
            full.update(row)
            writer.writerow(full)


def feature_row(strategy="absorption_reversal", decision="no_signal", diag=None) -> dict:
    return {
        "id": "x", "strategy": strategy, "decision": decision,
        "extra_json": json.dumps(diag) if diag is not None else "",
    }


class LoadRowsTests(unittest.TestCase):
    def test_filters_by_strategy_and_parses_extra_json(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.csv"
            write_csv(path, [
                feature_row(strategy="absorption_reversal", diag={"spread_pct": 0.01, "tfi": -0.6}),
                feature_row(strategy="breakout_retest", diag={"foo": 1}),
                feature_row(strategy="absorption_reversal", diag=None),  # no diagnostics at all
            ])
            rows = load_rows(str(path))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["diag"]["tfi"], -0.6)
            self.assertEqual(rows[1]["diag"], {})

    def test_empty_file_no_matching_rows(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.csv"
            write_csv(path, [feature_row(strategy="breakout_retest")])
            self.assertEqual(load_rows(str(path)), [])


class MainFunnelOutputTests(unittest.TestCase):
    def test_runs_without_error_on_a_realistic_funnel(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.csv"
            write_csv(path, [
                # Died at the very first gate (spread too wide) — checked
                # before diag even gets n_prints.
                feature_row(diag={"spread_pct": 0.5}),
                # Died at min_prints.
                feature_row(diag={"spread_pct": 0.02, "n_prints": 3}),
                # Died at min_abs_tfi.
                feature_row(diag={"spread_pct": 0.02, "n_prints": 10, "tfi": -0.1}),
                # Made it to Phase A candidate, died at replenishment.
                feature_row(diag={
                    "spread_pct": 0.02, "n_prints": 12, "tfi": -0.6, "volume_multiple": 2.0,
                    "actual_return_pct": 0.05, "predicted_return_pct": -0.2, "residual_pct": 0.25,
                    "candidate_direction": "up", "replenish_ratio": 0.4,
                }),
                # Fired.
                feature_row(decision="opened", diag={
                    "spread_pct": 0.02, "n_prints": 15, "tfi": -0.7, "volume_multiple": 2.2,
                    "actual_return_pct": 0.06, "predicted_return_pct": -0.25, "residual_pct": 0.31,
                    "candidate_direction": "up", "replenish_ratio": 0.9, "broke_local_range": True,
                }),
                # An unrelated strategy's row should not affect the funnel.
                feature_row(strategy="breakout_retest", diag={"n_prints": 99}),
            ])
            # Just verify it runs end-to-end without raising, against a
            # funnel that actually narrows at each stage as expected.
            main(str(path))


if __name__ == "__main__":
    unittest.main()
