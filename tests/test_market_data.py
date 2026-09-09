import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.market_data import OkxMarketDataProvider
from src.models import OrderBookLevel, OrderBookSnapshot


class FakeClient:
    """Stands in for OKXClient — only get_trades matters for these tests,
    the real HTTP transport (okx_client.py) has its own established
    "verified against a live response" boundary, not unit-mocked here."""

    def __init__(self, trade_pages: list[list[dict]]):
        self._pages = list(trade_pages)

    async def get_trades(self, inst_id: str, limit: int = 100) -> list[dict]:
        return self._pages.pop(0) if self._pages else []


def make_provider(trade_pages: list[list[dict]]) -> OkxMarketDataProvider:
    client = FakeClient(trade_pages)
    return OkxMarketDataProvider(client=client, underlying_inst_id="BTC-USDT", series_ids=[])


def trade_row(trade_id: str, px: float, sz: float, side: str, ts_ms: float) -> dict:
    return {"instId": "BTC-USDT", "tradeId": trade_id, "px": str(px), "sz": str(sz), "side": side, "ts": str(ts_ms)}


class TradePrintRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_poll_appends_in_chronological_order(self):
        # OKX returns newest-first ("3" is the most recent) — the deque
        # must end up oldest-first, matching _price_history's convention.
        page = [
            trade_row("3", 61000, 0.1, "sell", 1_700_000_003_000),
            trade_row("2", 60990, 0.2, "buy", 1_700_000_002_000),
            trade_row("1", 60980, 0.3, "buy", 1_700_000_001_000),
        ]
        provider = make_provider([page])
        await provider._refresh_trade_prints()

        prints = list(provider.btc_trade_prints())
        self.assertEqual([p.price for p in prints], [60980.0, 60990.0, 61000.0])
        self.assertEqual([p.side for p in prints], ["buy", "buy", "sell"])
        self.assertEqual([p.ts for p in prints], [1_700_000_001.0, 1_700_000_002.0, 1_700_000_003.0])  # ms -> sec

    async def test_second_poll_only_appends_genuinely_new_rows(self):
        page1 = [trade_row("2", 60990, 0.2, "buy", 2000), trade_row("1", 60980, 0.3, "buy", 1000)]
        # Next poll: "2" (already seen) plus two brand new trades on top.
        page2 = [
            trade_row("4", 61010, 0.1, "sell", 4000),
            trade_row("3", 61000, 0.1, "sell", 3000),
            trade_row("2", 60990, 0.2, "buy", 2000),
        ]
        provider = make_provider([page1, page2])
        await provider._refresh_trade_prints()
        await provider._refresh_trade_prints()

        prints = list(provider.btc_trade_prints())
        self.assertEqual([p.price for p in prints], [60980.0, 60990.0, 61000.0, 61010.0])
        ids_seen = {"1", "2", "3", "4"}
        self.assertEqual(len(prints), len(ids_seen))  # no duplicate of "2"

    async def test_no_overlap_means_nothing_new_appended(self):
        # Every row in the second poll is one we've already seen (e.g. a
        # very quiet market) — must be a no-op, not an error.
        page1 = [trade_row("1", 60980, 0.3, "buy", 1000)]
        page2 = [trade_row("1", 60980, 0.3, "buy", 1000)]
        provider = make_provider([page1, page2])
        await provider._refresh_trade_prints()
        await provider._refresh_trade_prints()

        self.assertEqual(len(provider.btc_trade_prints()), 1)

    async def test_malformed_row_is_skipped_without_dropping_the_rest(self):
        page = [
            trade_row("2", 60990, 0.2, "buy", 2000),
            {"instId": "BTC-USDT", "tradeId": "bad", "px": "not-a-number", "sz": "0.1", "side": "buy", "ts": "1500"},
            trade_row("1", 60980, 0.3, "buy", 1000),
        ]
        provider = make_provider([page])
        await provider._refresh_trade_prints()

        prints = list(provider.btc_trade_prints())
        self.assertEqual([p.price for p in prints], [60980.0, 60990.0])  # "bad" silently dropped

    async def test_empty_response_is_a_no_op(self):
        provider = make_provider([[]])
        await provider._refresh_trade_prints()  # must not raise
        self.assertEqual(len(provider.btc_trade_prints()), 0)


class RecordOrderbookTests(unittest.TestCase):
    def test_keeps_latest_and_history_in_sync(self):
        provider = make_provider([])
        book1 = OrderBookSnapshot(ts=1.0, bids=[OrderBookLevel(price=100.0, size=1.0)], asks=[])
        book2 = OrderBookSnapshot(ts=2.0, bids=[OrderBookLevel(price=101.0, size=1.0)], asks=[])

        provider._record_orderbook(book1)
        self.assertIs(provider.btc_orderbook(), book1)
        self.assertEqual(list(provider.btc_orderbook_history()), [book1])

        provider._record_orderbook(book2)
        self.assertIs(provider.btc_orderbook(), book2)  # latest always reflects the most recent
        self.assertEqual(list(provider.btc_orderbook_history()), [book1, book2])  # history keeps both


if __name__ == "__main__":
    unittest.main()
