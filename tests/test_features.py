import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import pct_change_over
from src.models import PricePoint


def make_points(prices: list[float], start_ts: float = 0.0, dt: float = 1.0) -> list[PricePoint]:
    return [PricePoint(ts=start_ts + i * dt, price=p) for i, p in enumerate(prices)]


class PctChangeOverTests(unittest.TestCase):
    def test_computes_percentage_change(self):
        points = make_points([100.0, 101.0, 102.0], start_ts=0.0, dt=30.0)  # 100 -> 102 over 60s
        self.assertAlmostEqual(pct_change_over(points, now=60.0, window_sec=60.0), 2.0)

    def test_none_with_single_point_in_window(self):
        self.assertIsNone(pct_change_over(make_points([100.0]), now=0.0, window_sec=60.0))

    def test_none_when_oldest_in_window_is_non_positive(self):
        points = make_points([0.0, 100.0], start_ts=0.0, dt=1.0)
        self.assertIsNone(pct_change_over(points, now=1.0, window_sec=60.0))

    def test_only_points_within_window_count(self):
        # Points far outside window_sec shouldn't shift the "oldest in window" anchor.
        points = make_points([50.0, 100.0, 110.0], start_ts=0.0, dt=100.0)  # ts=0,100,200
        pct = pct_change_over(points, now=200.0, window_sec=60.0)  # only ts=200 (110.0) is within 60s
        self.assertIsNone(pct)  # just one point in that window

    def test_negative_change(self):
        points = make_points([100.0, 95.0], start_ts=0.0, dt=30.0)
        self.assertAlmostEqual(pct_change_over(points, now=30.0, window_sec=60.0), -5.0)


if __name__ == "__main__":
    unittest.main()
