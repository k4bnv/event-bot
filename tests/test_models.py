import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Direction, EventMarket, OrderBookLevel, OrderBookSnapshot, simulate_market_fill


class SimulateMarketFillTests(unittest.TestCase):
    def test_single_level_fully_covers_budget(self):
        levels = [OrderBookLevel(price=0.5, size=100.0)]
        vwap, contracts, spent, full = simulate_market_fill(levels, 20.0)
        self.assertEqual(vwap, 0.5)
        self.assertAlmostEqual(contracts, 40.0)
        self.assertAlmostEqual(spent, 20.0)
        self.assertTrue(full)

    def test_walks_into_a_second_level_and_averages(self):
        # level 1: 10 contracts @ 0.5 = $5 max; remaining $15 spills into level 2 @ 0.7
        levels = [OrderBookLevel(price=0.5, size=10.0), OrderBookLevel(price=0.7, size=100.0)]
        vwap, contracts, spent, full = simulate_market_fill(levels, 20.0)
        self.assertAlmostEqual(spent, 20.0)
        self.assertTrue(full)
        # contracts = 10 (from level 1) + 15/0.7 (from level 2)
        expected_contracts = 10.0 + 15.0 / 0.7
        self.assertAlmostEqual(contracts, expected_contracts)
        self.assertAlmostEqual(vwap, 20.0 / expected_contracts)

    def test_empty_book_returns_none(self):
        vwap, contracts, spent, full = simulate_market_fill([], 20.0)
        self.assertIsNone(vwap)
        self.assertEqual(contracts, 0.0)
        self.assertEqual(spent, 0.0)
        self.assertFalse(full)

    def test_budget_exceeds_total_depth_reports_not_fully_filled(self):
        levels = [OrderBookLevel(price=0.5, size=2.0)]  # only $1 of depth available
        vwap, contracts, spent, full = simulate_market_fill(levels, 20.0)
        self.assertFalse(full)
        self.assertAlmostEqual(spent, 1.0)
        self.assertAlmostEqual(contracts, 2.0)
        self.assertAlmostEqual(vwap, 0.5)

    def test_zero_and_negative_price_levels_are_skipped(self):
        levels = [OrderBookLevel(price=0.0, size=10.0), OrderBookLevel(price=0.4, size=10.0)]
        vwap, contracts, spent, full = simulate_market_fill(levels, 2.0)
        self.assertAlmostEqual(vwap, 0.4)
        self.assertTrue(full)


def make_market(up_price=None, book=None) -> EventMarket:
    return EventMarket(
        series_id="S", method="price_up_down", inst_id="I", expiry_ts=0.0,
        up_price=up_price, book=book,
    )


class EventMarketFillPriceForTests(unittest.TestCase):
    def test_up_uses_real_book_vwap_not_naive_price(self):
        # naive up_price says 0.10 (cheap!), but the real ask book is thin
        # and a $20 order actually walks up to a much worse average price —
        # this is exactly the discrepancy observed against live OKX data.
        book = OrderBookSnapshot(
            ts=0.0,
            asks=[OrderBookLevel(price=0.10, size=0.5), OrderBookLevel(price=0.90, size=100.0)],
        )
        market = make_market(up_price=0.10, book=book)
        price = market.fill_price_for(Direction.UP, stake_usd=20.0)
        naive_price = market.price_for(Direction.UP)
        self.assertEqual(naive_price, 0.10)
        self.assertGreater(price, naive_price)  # honest price is materially worse
        self.assertNotEqual(price, naive_price)

    def test_down_uses_top_of_book_bid_not_naive_one_minus_last(self):
        book = OrderBookSnapshot(ts=0.0, bids=[OrderBookLevel(price=0.30, size=5.0)])
        market = make_market(up_price=0.80, book=book)  # naive DOWN = 1 - 0.80 = 0.20
        price = market.fill_price_for(Direction.DOWN, stake_usd=20.0)
        self.assertEqual(price, round(1 - 0.30, 4))  # 0.70, NOT the naive 0.20
        self.assertNotEqual(price, market.price_for(Direction.DOWN))

    def test_falls_back_to_naive_price_when_no_book_at_all(self):
        market = make_market(up_price=0.42, book=None)
        self.assertEqual(market.fill_price_for(Direction.UP, 20.0), market.price_for(Direction.UP))
        self.assertEqual(market.fill_price_for(Direction.DOWN, 20.0), market.price_for(Direction.DOWN))

    def test_falls_back_when_relevant_book_side_is_empty(self):
        book = OrderBookSnapshot(ts=0.0, asks=[], bids=[])
        market = make_market(up_price=0.42, book=book)
        self.assertEqual(market.fill_price_for(Direction.UP, 20.0), market.price_for(Direction.UP))
        self.assertEqual(market.fill_price_for(Direction.DOWN, 20.0), market.price_for(Direction.DOWN))

    def test_returns_none_when_nothing_is_available(self):
        market = make_market(up_price=None, book=None)
        self.assertIsNone(market.fill_price_for(Direction.UP, 20.0))
        self.assertIsNone(market.fill_price_for(Direction.DOWN, 20.0))


if __name__ == "__main__":
    unittest.main()
