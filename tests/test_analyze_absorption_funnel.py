import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.storage import FEATURE_FIELDS, Storage
from analyze_absorption_funnel import load_rows, load_rows_from_db, main  # noqa: E402


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


class LoadRowsFromDbTests(unittest.TestCase):
    """The auth-proof path: read bot.db directly, no dashboard/HTTP
    involved — added after a live curl against a password-protected
    dashboard silently downloaded an unauthorized-error JSON body
    instead of a CSV, which load_rows_from_csv then quietly parsed as
    "0 matching rows" rather than an error (no "strategy"/"extra_json"
    header in the JSON body to match against)."""

    def test_reads_real_storage_db_and_dispatches_via_sniffing(self):
        with TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            storage.log_checkpoint_features(
                {**{f: None for f in FEATURE_FIELDS}, "id": "f1", "strategy": "absorption_reversal",
                 "decision": "no_signal", "extra_json": json.dumps({"tfi": -0.6})}
            )
            storage.log_checkpoint_features(
                {**{f: None for f in FEATURE_FIELDS}, "id": "f2", "strategy": "breakout_retest",
                 "decision": "no_signal", "extra_json": None}
            )
            db_path = storage.db_path
            storage.close()

            rows = load_rows_from_db(str(db_path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["diag"]["tfi"], -0.6)

            # load_rows (the dispatcher main() actually calls) must reach
            # the same result by sniffing the file's own header, not by
            # trusting a .db extension the user might not have used.
            self.assertEqual(load_rows(str(db_path)), rows)

    def test_a_real_csv_is_not_mistaken_for_a_database(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.csv"
            write_csv(path, [feature_row(diag={"tfi": -0.5})])
            rows = load_rows(str(path))
            self.assertEqual(len(rows), 1)


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
