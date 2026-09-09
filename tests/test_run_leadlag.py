"""Covers the pure functions behind `run.py --check-leadlag` and
`--check-leadlag-internal`: _detect_impulses (impulse detection on a
chronological price series), _measure_reaction_lag (how long a follower
series took to react, if at all), and _measure_upprice_reaction_lag (the
same idea for the event contract's own up_price, including rollover
exclusion) — the diagnostics' actual measurement logic, independent of
network I/O so it's fully testable offline."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import _detect_impulses, _measure_reaction_lag, _measure_upprice_reaction_lag


def _flat_then_jump(n: int, jump_at: int, jump_pct: float, base: float = 80000.0) -> list[tuple[float, float]]:
    """A (ts, price) series that's flat at `base` for ts < jump_at, then
    steps to base*(1+jump_pct/100) from jump_at onward."""
    out = []
    for t in range(n):
        price = base * (1 + jump_pct / 100) if t >= jump_at else base
        out.append((float(t), price))
    return out


class DetectImpulsesTests(unittest.TestCase):
    def test_detects_a_single_up_move(self):
        points = _flat_then_jump(100, jump_at=50, jump_pct=0.06)
        impulses = _detect_impulses(points, threshold_pct=0.03, window_sec=10, cooldown_sec=10)
        self.assertEqual(len(impulses), 1)
        self.assertEqual(impulses[0]["direction"], "up")
        self.assertAlmostEqual(impulses[0]["move_pct"], 0.06, places=3)
        self.assertEqual(impulses[0]["end_ts"], 50.0)

    def test_detects_a_down_move(self):
        points = _flat_then_jump(100, jump_at=50, jump_pct=-0.06)
        impulses = _detect_impulses(points, threshold_pct=0.03, window_sec=10, cooldown_sec=10)
        self.assertEqual(len(impulses), 1)
        self.assertEqual(impulses[0]["direction"], "down")

    def test_below_threshold_is_not_an_impulse(self):
        points = _flat_then_jump(100, jump_at=50, jump_pct=0.01)  # under the 0.03 threshold
        impulses = _detect_impulses(points, threshold_pct=0.03, window_sec=10, cooldown_sec=10)
        self.assertEqual(impulses, [])

    def test_cooldown_prevents_re_triggering_on_a_sustained_move(self):
        # Price steps up once and stays there — without a cooldown this
        # would re-trigger on every subsequent point that's still >=
        # threshold away from *some* earlier point in the window.
        points = _flat_then_jump(100, jump_at=50, jump_pct=0.06)
        impulses = _detect_impulses(points, threshold_pct=0.03, window_sec=10, cooldown_sec=10)
        self.assertEqual(len(impulses), 1)

    def test_empty_input_returns_empty(self):
        self.assertEqual(_detect_impulses([], 0.03, 10, 10), [])


class MeasureReactionLagTests(unittest.TestCase):
    def setUp(self):
        self.leader = _flat_then_jump(120, jump_at=50, jump_pct=0.06)
        self.impulses = _detect_impulses(self.leader, threshold_pct=0.03, window_sec=10, cooldown_sec=10)
        self.assertEqual(len(self.impulses), 1)  # sanity check on the fixture itself

    def test_real_lag_is_measured_correctly(self):
        # Follower jumps the same way, but 4 seconds after the leader.
        follower = _flat_then_jump(120, jump_at=54, jump_pct=0.06)
        lags = _measure_reaction_lag(self.impulses, follower, horizon_sec=20, react_threshold_pct=0.015)
        self.assertEqual(lags, [4.0])

    def test_zero_lag_is_distinguished_from_no_reaction(self):
        # Follower already shows the same move by the time the impulse
        # "ended" (e.g. the leader series itself) — this must come back as
        # 0.0, NOT None, or "reacted instantly" and "never reacted" become
        # indistinguishable in the report.
        lags = _measure_reaction_lag(self.impulses, self.leader, horizon_sec=20, react_threshold_pct=0.015)
        self.assertEqual(lags, [0.0])

    def test_no_reaction_within_horizon_is_none(self):
        flat = [(float(t), 80000.0) for t in range(120)]  # never moves at all
        lags = _measure_reaction_lag(self.impulses, flat, horizon_sec=20, react_threshold_pct=0.015)
        self.assertEqual(lags, [None])

    def test_reaction_outside_horizon_is_none(self):
        # Reacts, but 25s later — past the 20s horizon.
        follower = _flat_then_jump(120, jump_at=75, jump_pct=0.06)
        lags = _measure_reaction_lag(self.impulses, follower, horizon_sec=20, react_threshold_pct=0.015)
        self.assertEqual(lags, [None])

    def test_empty_reactor_series_is_none(self):
        self.assertEqual(_measure_reaction_lag(self.impulses, [], 20, 0.015), [None])

    def test_empty_impulses_returns_empty(self):
        self.assertEqual(_measure_reaction_lag([], self.leader, 20, 0.015), [])


class MeasureUppriceReactionLagTests(unittest.TestCase):
    def setUp(self):
        # Same shape leader fixture as the Binance case, but the "reactor"
        # here carries an inst_id per point (the event contract's own
        # up_price, which is bounded [0,1] and uses an absolute threshold).
        self.spot = _flat_then_jump(120, jump_at=50, jump_pct=0.06)
        self.impulses = _detect_impulses(self.spot, threshold_pct=0.03, window_sec=10, cooldown_sec=10)
        self.assertEqual(len(self.impulses), 1)

    @staticmethod
    def _upprice_series(n, jump_at, base=0.50, jump=0.05, inst_id="INST-A"):
        out = []
        for t in range(n):
            price = base + jump if t >= jump_at else base
            out.append((float(t), price, inst_id))
        return out

    def test_real_lag_is_measured_correctly(self):
        reactor = self._upprice_series(120, jump_at=54)  # 4s after the impulse ends at t=50
        results = _measure_upprice_reaction_lag(self.impulses, reactor, horizon_sec=20, react_threshold_abs=0.02)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], {"lag": 4.0, "excluded": False, "reason": ""})

    def test_zero_lag_is_distinguished_from_no_reaction(self):
        reactor = self._upprice_series(120, jump_at=50)  # already moved by the time the impulse "ends"
        results = _measure_upprice_reaction_lag(self.impulses, reactor, horizon_sec=20, react_threshold_abs=0.02)
        self.assertEqual(results[0]["lag"], 0.0)
        self.assertFalse(results[0]["excluded"])

    def test_no_reaction_within_horizon(self):
        reactor = [(float(t), 0.50, "INST-A") for t in range(120)]  # never moves
        results = _measure_upprice_reaction_lag(self.impulses, reactor, horizon_sec=20, react_threshold_abs=0.02)
        self.assertIsNone(results[0]["lag"])
        self.assertFalse(results[0]["excluded"])
        self.assertIn("no reaction", results[0]["reason"])

    def test_reaction_outside_horizon_is_none(self):
        reactor = self._upprice_series(120, jump_at=75)  # 25s later, past the 20s horizon
        results = _measure_upprice_reaction_lag(self.impulses, reactor, horizon_sec=20, react_threshold_abs=0.02)
        self.assertIsNone(results[0]["lag"])
        self.assertFalse(results[0]["excluded"])

    def test_rollover_between_impulse_start_and_end_is_excluded(self):
        # The instId changes partway through the impulse's own start..end
        # span — the "pre" price and the "base" price at end_ts belong to
        # two different instruments, so they're not comparable at all.
        reactor = []
        for t in range(120):
            inst = "INST-A" if t < 48 else "INST-B"  # impulse spans 40..50
            reactor.append((float(t), 0.50, inst))
        results = _measure_upprice_reaction_lag(self.impulses, reactor, horizon_sec=20, react_threshold_abs=0.02)
        self.assertTrue(results[0]["excluded"])
        self.assertIn("rolled over", results[0]["reason"])
        self.assertIsNone(results[0]["lag"])

    def test_rollover_during_horizon_is_excluded_not_no_reaction(self):
        # Instrument is stable through the impulse itself, but rolls over
        # partway through the reaction-measurement horizon, before any
        # matching move was found on the old instrument — this must be
        # reported as excluded, not conflated with "genuinely never reacted".
        reactor = []
        for t in range(120):
            inst = "INST-A" if t < 55 else "INST-B"  # impulse ends at t=50, rolls over at t=55
            reactor.append((float(t), 0.50, inst))
        results = _measure_upprice_reaction_lag(self.impulses, reactor, horizon_sec=20, react_threshold_abs=0.02)
        self.assertTrue(results[0]["excluded"])
        self.assertIn("horizon", results[0]["reason"])
        self.assertIsNone(results[0]["lag"])

    def test_empty_reactor_series_is_none_not_excluded(self):
        results = _measure_upprice_reaction_lag(self.impulses, [], horizon_sec=20, react_threshold_abs=0.02)
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["lag"])
        self.assertFalse(results[0]["excluded"])

    def test_empty_impulses_returns_empty(self):
        reactor = self._upprice_series(120, jump_at=54)
        self.assertEqual(_measure_upprice_reaction_lag([], reactor, 20, 0.02), [])


if __name__ == "__main__":
    unittest.main()
