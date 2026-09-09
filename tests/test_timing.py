import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.timing import EntryWindowManager


class EntryWindowManagerTests(unittest.TestCase):
    def test_fires_each_window_once_on_a_genuinely_long_window(self):
        # A 15-minute-ish window (starts with remaining well above every
        # configured checkpoint) crosses 12, then 7, then 2 for real as
        # time passes — this is the intended, working case the
        # [12, 7, 2] config default is designed around.
        mgr = EntryWindowManager()
        self.assertEqual(
            mgr.due_windows("S", 1000.0, "strat", remaining_sec=15 * 60, configured_windows_min=[12, 7, 2]), [],
        )
        due_12 = mgr.due_windows("S", 1000.0, "strat", remaining_sec=11 * 60, configured_windows_min=[12, 7, 2])
        self.assertEqual(due_12, [12])
        # calling again at the same remaining time must not re-fire
        self.assertEqual(
            mgr.due_windows("S", 1000.0, "strat", remaining_sec=11 * 60, configured_windows_min=[12, 7, 2]), [],
        )
        due_7 = mgr.due_windows("S", 1000.0, "strat", remaining_sec=6 * 60, configured_windows_min=[12, 7, 2])
        self.assertEqual(due_7, [7])
        due_2 = mgr.due_windows("S", 1000.0, "strat", remaining_sec=90, configured_windows_min=[12, 7, 2])
        self.assertEqual(due_2, [2])

    def test_checkpoints_above_the_windows_own_starting_time_never_fire(self):
        # Real bug this pins: a strategy's entry_windows_min list is shared
        # across series of different lengths in config.yaml (e.g. the same
        # [12, 7, 2] list applied to both a 5-minute and a 15-minute
        # series). On a 5-minute window remaining_min never exceeds ~5, so
        # 12 and 7 were BOTH already "at or below" on the very first poll
        # — the old code fired them TOGETHER in one due_windows() call,
        # and the engine opened two near-simultaneous duplicate trades on
        # the exact same instrument for it. Checkpoints the window could
        # never have genuinely crossed from above must never fire at all.
        mgr = EntryWindowManager()
        due_first_poll = mgr.due_windows(
            "BTC-UPDOWN-5MIN", 1000.0, "strat", remaining_sec=298.0, configured_windows_min=[12, 7, 2],
        )
        self.assertEqual(due_first_poll, [])  # NOT [12, 7]
        # the 2-minute checkpoint is still reachable (5min window did start above it) -> fires normally later
        due_later = mgr.due_windows(
            "BTC-UPDOWN-5MIN", 1000.0, "strat", remaining_sec=90.0, configured_windows_min=[12, 7, 2],
        )
        self.assertEqual(due_later, [2])

    def test_mixed_list_fires_only_the_reachable_checkpoints(self):
        # mean_reversion's config default, [7, 2], on a 5-minute window:
        # 7 is unreachable (window starts at ~5 < 7) but 2 genuinely is —
        # only the reachable one should ever fire, not both together.
        mgr = EntryWindowManager()
        due = mgr.due_windows("BTC-UPDOWN-5MIN", 1000.0, "strat", remaining_sec=298.0, configured_windows_min=[7, 2])
        self.assertEqual(due, [])
        due_later = mgr.due_windows(
            "BTC-UPDOWN-5MIN", 1000.0, "strat", remaining_sec=90.0, configured_windows_min=[7, 2],
        )
        self.assertEqual(due_later, [2])

    def test_different_strategies_are_independent(self):
        mgr = EntryWindowManager()
        # Prime both strategies' "window start" above the 2-minute checkpoint
        # first (a real window would have been polled earlier too — see
        # test_checkpoints_above_the_windows_own_starting_time_never_fire
        # for why a checkpoint can't fire on its very first observation
        # unless the window is known to have started above it).
        mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=200, configured_windows_min=[2])
        mgr.due_windows("S", 1000.0, "strat_b", remaining_sec=200, configured_windows_min=[2])

        mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=60, configured_windows_min=[2])
        due_b = mgr.due_windows("S", 1000.0, "strat_b", remaining_sec=60, configured_windows_min=[2])
        self.assertEqual(due_b, [2])  # strat_a firing didn't consume strat_b's checkpoint

    def test_reset_strategy_only_clears_that_strategy(self):
        mgr = EntryWindowManager()
        mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=200, configured_windows_min=[2])
        mgr.due_windows("S", 1000.0, "strat_b", remaining_sec=200, configured_windows_min=[2])
        mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=60, configured_windows_min=[2])
        mgr.due_windows("S", 1000.0, "strat_b", remaining_sec=60, configured_windows_min=[2])

        mgr.reset_strategy("strat_a")

        # strat_a's checkpoint fires again (bookkeeping wiped) — but only
        # once its window is re-primed above 2min, same as the first time.
        mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=200, configured_windows_min=[2])
        self.assertEqual(
            mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=60, configured_windows_min=[2]), [2]
        )
        # strat_b's bookkeeping is untouched -> does NOT re-fire
        self.assertEqual(
            mgr.due_windows("S", 1000.0, "strat_b", remaining_sec=60, configured_windows_min=[2]), []
        )

    def test_reset_strategy_clears_window_start_bookkeeping_too(self):
        mgr = EntryWindowManager()
        # This "window" starts already below the 2-minute checkpoint -> it
        # can never fire for strat_a until the bookkeeping is cleared.
        mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=30, configured_windows_min=[2])
        self.assertEqual(
            mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=10, configured_windows_min=[2]), [],
        )

        mgr.reset_strategy("strat_a")

        # A fresh observation that DOES start above 2min fires normally —
        # if _window_start_min hadn't been cleared by reset, the stale low
        # value from before would still be blocking it.
        mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=200, configured_windows_min=[2])
        self.assertEqual(
            mgr.due_windows("S", 1000.0, "strat_a", remaining_sec=60, configured_windows_min=[2]), [2],
        )


if __name__ == "__main__":
    unittest.main()
