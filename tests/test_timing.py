import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.timing import EntryWindowManager


class EntryWindowManagerTests(unittest.TestCase):
    def test_fires_each_window_once(self):
        mgr = EntryWindowManager()
        # remaining=6min crosses both 12 and 7 immediately, not 2
        due = mgr.due_windows("S", 1000.0, "strat", remaining_sec=6 * 60, configured_windows_min=[12, 7, 2])
        self.assertEqual(due, [12, 7])
        # calling again at the same remaining time must not re-fire
        due_again = mgr.due_windows("S", 1000.0, "strat", remaining_sec=6 * 60, configured_windows_min=[12, 7, 2])
        self.assertEqual(due_again, [])
        # later, crossing the 2-minute checkpoint fires just that one
        due_later = mgr.due_windows("S", 1000.0, "strat", remaining_sec=90, configured_windows_min=[12, 7, 2])
        self.assertEqual(due_later, [2])

    def test_different_strategies_are_independent(self):
        mgr = EntryWindowManager()
        mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=60, configured_windows_min=[2])
        due_b = mgr.due_windows("S", 1000.0, "strat_b", remaining_sec=60, configured_windows_min=[2])
        self.assertEqual(due_b, [2])  # strat_a firing didn't consume strat_b's checkpoint

    def test_reset_strategy_only_clears_that_strategy(self):
        mgr = EntryWindowManager()
        mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=60, configured_windows_min=[2])
        mgr.due_windows("S", 1000.0, "strat_b", remaining_sec=60, configured_windows_min=[2])

        mgr.reset_strategy("strat_a")

        # strat_a's checkpoint fires again (bookkeeping wiped)
        self.assertEqual(
            mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=60, configured_windows_min=[2]), [2]
        )
        # strat_b's bookkeeping is untouched -> does NOT re-fire
        self.assertEqual(
            mgr.due_windows("S", 1000.0, "strat_b", remaining_sec=60, configured_windows_min=[2]), []
        )


if __name__ == "__main__":
    unittest.main()
