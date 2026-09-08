import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import ChatAPIError
from src.models import Direction, EventMarket, PricePoint
from src.strategies.ai_prompt import (
    BARRIER_PROMPT_TEMPLATE, AIPromptStrategy, build_client_config, _guess_symbol, _pct_change_over,
)
from src.strategies.base import StrategyContext
from src.strategies.fair_value_edge import FairValueEdgeStrategy, fair_probability_up
from src.strategies.funding_skew import FundingSkewStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy


def make_points(prices: list[float], start_ts: float = 0.0, dt: float = 1.0) -> list[PricePoint]:
    return [PricePoint(ts=start_ts + i * dt, price=p) for i, p in enumerate(prices)]


def make_market(up_price=None, floor_strike=None, method="price_up_down", expiry_ts=None) -> EventMarket:
    return EventMarket(
        series_id="TEST-SERIES", method=method, inst_id="TEST-INST-1",
        expiry_ts=expiry_ts or time.time() + 300, floor_strike=floor_strike,
        up_price=up_price, state="live",
    )


def make_ctx(price_history, orderbook=None, remaining_sec=200.0, window_min=7, market=None, funding_rate=None):
    return StrategyContext(
        price_history=price_history, orderbook=orderbook, remaining_sec=remaining_sec,
        window_min=window_min, market=market or make_market(), funding_rate=funding_rate,
    )


class FairValueEdgeMathTests(unittest.TestCase):
    def test_symmetric_when_price_equals_strike(self):
        noisy = [100 + (0.05 if i % 2 == 0 else -0.05) for i in range(20)]
        points = make_points(noisy)
        prob = fair_probability_up(points, current_price=100.0, strike=100.0, remaining_sec=120)
        self.assertIsNotNone(prob)
        self.assertAlmostEqual(prob, 0.5, delta=0.01)

    def test_high_probability_when_far_above_strike(self):
        noisy = [100 + (0.05 if i % 2 == 0 else -0.05) for i in range(20)]
        points = make_points(noisy)
        prob = fair_probability_up(points, current_price=110.0, strike=100.0, remaining_sec=60)
        self.assertIsNotNone(prob)
        self.assertGreater(prob, 0.9)

    def test_none_with_insufficient_history(self):
        points = make_points([100, 100.1, 99.9])
        self.assertIsNone(fair_probability_up(points, 100.0, 100.0, 60))

    def test_none_with_zero_volatility(self):
        points = make_points([100.0] * 15)
        self.assertIsNone(fair_probability_up(points, 100.0, 100.0, 60))


class FairValueEdgeStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_signals_up_when_model_prob_beats_market_price(self):
        noisy = [100 + (0.05 if i % 2 == 0 else -0.05) for i in range(20)]
        points = make_points(noisy)
        # model P(UP) ~ 0.5 (price==strike); market prices it far lower -> edge to buy UP
        market = make_market(up_price=0.20, floor_strike=100.0)
        ctx = make_ctx(points, remaining_sec=120, market=market)
        strategy = FairValueEdgeStrategy(config={"min_edge": 0.08})
        signal = await strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)

    async def test_no_signal_when_market_price_matches_model(self):
        noisy = [100 + (0.05 if i % 2 == 0 else -0.05) for i in range(20)]
        points = make_points(noisy)
        market = make_market(up_price=0.5, floor_strike=100.0)
        ctx = make_ctx(points, remaining_sec=120, market=market)
        strategy = FairValueEdgeStrategy(config={"min_edge": 0.08})
        self.assertIsNone(await strategy.evaluate(ctx))

    async def test_no_signal_without_market_quote_or_strike(self):
        points = make_points([100 + (0.05 if i % 2 == 0 else -0.05) for i in range(20)])
        strategy = FairValueEdgeStrategy(config={})
        self.assertIsNone(await strategy.evaluate(make_ctx(points, market=make_market(up_price=None, floor_strike=100.0))))
        self.assertIsNone(await strategy.evaluate(make_ctx(points, market=make_market(up_price=0.5, floor_strike=None))))


class VolatilityBreakoutStrategyTests(unittest.IsolatedAsyncioTestCase):
    def _config(self):
        return {
            "breakout_lookback_sec": 15, "vol_lookback_sec": 60,
            "vol_multiplier": 2.5, "min_threshold_pct": 0.03, "retest_tolerance_mult": 0.4,
        }

    async def test_detects_impulse_and_retest(self):
        now = 1000.0
        # calm baseline: ts 940..984 (older than breakout_lookback_sec=15 before `now`)
        baseline_prices = [100 + (0.01 if i % 2 == 0 else -0.01) for i in range(45)]
        baseline = make_points(baseline_prices, start_ts=now - 60, dt=1.0)
        # detection window: last 15s -> sharp impulse then retest close to 100
        # baseline vol ~0.02% -> threshold ~0.05% -> tolerance ~0.02%; retest
        # price picked to land inside that tolerance.
        detection = [
            PricePoint(ts=now - 10, price=100.0),
            PricePoint(ts=now - 8, price=100.5),
            PricePoint(ts=now - 6, price=102.0),    # impulse: +2%
            PricePoint(ts=now - 2, price=100.015),  # retest, still holding above 100
            PricePoint(ts=now, price=100.015),
        ]
        history = baseline + detection
        ctx = make_ctx(history, remaining_sec=200)
        strategy = VolatilityBreakoutStrategy(config=self._config())
        signal = await strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)

    async def test_no_signal_when_calm(self):
        now = 1000.0
        prices = [100 + (0.01 if i % 2 == 0 else -0.01) for i in range(80)]
        history = make_points(prices, start_ts=now - 79, dt=1.0)
        ctx = make_ctx(history, remaining_sec=200)
        strategy = VolatilityBreakoutStrategy(config=self._config())
        self.assertIsNone(await strategy.evaluate(ctx))


class FundingSkewStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_positive_funding_bets_down(self):
        strategy = FundingSkewStrategy(config={"funding_rate_threshold": 0.0003})
        ctx = make_ctx([], funding_rate=0.0006)
        signal = await strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.DOWN)

    async def test_negative_funding_bets_up(self):
        strategy = FundingSkewStrategy(config={"funding_rate_threshold": 0.0003})
        signal = await strategy.evaluate(make_ctx([], funding_rate=-0.0006))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)

    async def test_within_band_no_signal(self):
        strategy = FundingSkewStrategy(config={"funding_rate_threshold": 0.0003})
        self.assertIsNone(await strategy.evaluate(make_ctx([], funding_rate=0.0001)))

    async def test_missing_funding_rate_no_signal(self):
        strategy = FundingSkewStrategy(config={})
        self.assertIsNone(await strategy.evaluate(make_ctx([], funding_rate=None)))


class FakeChatClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def chat_json(self, prompt, max_tokens=300, temperature=0.2):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        pass


class AIPromptStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_up_response_produces_signal(self):
        fake = FakeChatClient(['{"direction": "UP", "confidence": 0.8, "reason": "test"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        signal = await strategy.evaluate(make_ctx([PricePoint(ts=0, price=100)] * 12))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)
        self.assertAlmostEqual(signal.confidence, 0.8)
        self.assertEqual(fake.calls, 1)

    async def test_none_direction_yields_no_signal(self):
        fake = FakeChatClient(['{"direction": "NONE", "confidence": 0.5, "reason": "no edge"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([])))

    async def test_malformed_json_degrades_to_no_signal(self):
        fake = FakeChatClient(["not json at all"])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([])))

    async def test_api_error_degrades_to_no_signal(self):
        fake = FakeChatClient([ChatAPIError("boom")])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([])))

    async def test_confidence_is_clamped_to_0_1(self):
        fake = FakeChatClient(['{"direction": "DOWN", "confidence": 5, "reason": "x"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        signal = await strategy.evaluate(make_ctx([]))
        self.assertEqual(signal.confidence, 1.0)

    async def test_cooldown_skips_second_call(self):
        fake = FakeChatClient([
            '{"direction": "UP", "confidence": 0.6, "reason": "a"}',
            '{"direction": "UP", "confidence": 0.6, "reason": "b"}',
        ])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 9999}, client=fake)
        first = await strategy.evaluate(make_ctx([]))
        second = await strategy.evaluate(make_ctx([]))
        self.assertIsNotNone(first)
        self.assertIsNone(second)   # cooldown -> no signal, and no extra API call
        self.assertEqual(fake.calls, 1)

    async def test_daily_call_cap_blocks_further_calls(self):
        fake = FakeChatClient([
            '{"direction": "UP", "confidence": 0.6, "reason": "a"}',
            '{"direction": "UP", "confidence": 0.6, "reason": "b"}',
        ])
        strategy = AIPromptStrategy(
            config={"min_seconds_between_calls": 0, "max_calls_per_day": 1}, client=fake,
        )
        first = await strategy.evaluate(make_ctx([]))
        second = await strategy.evaluate(make_ctx([]))
        self.assertIsNotNone(first)
        self.assertIsNone(second)          # cap hit -> no signal, no extra API call
        self.assertEqual(fake.calls, 1)    # the cap check runs BEFORE spending anything

    async def test_no_client_configured_returns_none_without_raising(self):
        strategy = AIPromptStrategy(config={})  # no injected client, no env key set in test env
        result = await strategy.evaluate(make_ctx([]))
        self.assertIsNone(result)

    async def test_markdown_fenced_json_is_parsed(self):
        fake = FakeChatClient(['```json\n{"direction": "UP", "confidence": 0.7, "reason": "x"}\n```'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        signal = await strategy.evaluate(make_ctx([]))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)

    def test_custom_prompt_template_is_used(self):
        strategy = AIPromptStrategy(config={"prompt_template": "hello $series world"}, client=FakeChatClient([]))
        market = make_market(up_price=0.4, floor_strike=100.0)
        market.series_id = "MY-SERIES"
        prompt = strategy._build_prompt(make_ctx([], market=market))
        self.assertEqual(prompt, "hello MY-SERIES world")

    def test_prompt_template_typo_does_not_raise(self):
        strategy = AIPromptStrategy(
            config={"prompt_template": "value=$totally_unknown_placeholder!"}, client=FakeChatClient([]),
        )
        # safe_substitute leaves an unrecognized placeholder as literal text
        # instead of raising — must not crash the strategy.
        prompt = strategy._build_prompt(make_ctx([]))
        self.assertEqual(prompt, "value=$totally_unknown_placeholder!")

    def test_malformed_dollar_syntax_is_left_literal_not_raised(self):
        # string.Template.safe_substitute() never raises for bad $ syntax —
        # it leaves it as literal text, same as an unrecognized placeholder.
        # This test pins that behavior so the try/except ValueError fallback
        # in _build_prompt is understood as defensive, not load-bearing.
        strategy = AIPromptStrategy(
            config={"prompt_template": "trailing dollar sign: $"}, client=FakeChatClient([]),
        )
        prompt = strategy._build_prompt(make_ctx([]))
        self.assertEqual(prompt, "trailing dollar sign: $")

    async def test_prob_up_above_threshold_bets_up(self):
        fake = FakeChatClient(['{"base_prob":0.5,"adjustment":0.08,"prob_up":0.58,"reason":"drift up"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0, "min_edge": 0.05}, client=fake)
        signal = await strategy.evaluate(make_ctx([]))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)
        self.assertAlmostEqual(signal.confidence, 0.16, places=2)  # |0.58-0.5|*2

    async def test_prob_up_below_half_bets_down(self):
        fake = FakeChatClient(['{"prob_up":0.4,"reason":"drift down"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0, "min_edge": 0.05}, client=fake)
        signal = await strategy.evaluate(make_ctx([]))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.DOWN)

    async def test_prob_up_within_band_of_half_no_signal(self):
        fake = FakeChatClient(['{"prob_up":0.52,"reason":"barely off coinflip"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0, "min_edge": 0.05}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([])))

    async def test_prob_up_missing_or_non_numeric_no_signal(self):
        fake = FakeChatClient(['{"prob_up":"not-a-number","reason":"x"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([])))


class BarrierPromptTemplateTests(unittest.IsolatedAsyncioTestCase):
    """The exact template a user wanted to reuse from another bot (curly
    {PLACEHOLDER} converted to $placeholder — see ai_prompt.py module
    docstring for why $ was kept instead of adopting {})."""

    def test_pct_change_over_computes_percentage(self):
        points = make_points([100.0, 101.0, 102.0], start_ts=0.0, dt=30.0)  # 100 -> 102 over 60s
        pct = _pct_change_over(points, now=60.0, window_sec=60.0)
        self.assertAlmostEqual(pct, 2.0)

    def test_pct_change_over_none_with_single_point(self):
        self.assertIsNone(_pct_change_over(make_points([100.0]), now=0.0, window_sec=60.0))

    def test_guess_symbol_extracts_prefix(self):
        self.assertEqual(_guess_symbol("BTC-UPDOWN-15MIN"), "BTC")
        self.assertEqual(_guess_symbol(""), "")

    def test_full_barrier_template_renders_without_crashing(self):
        noisy = [100 + (0.05 if i % 2 == 0 else -0.05) for i in range(30)]
        points = make_points(noisy, start_ts=0.0, dt=10.0)  # 290s of history
        market = make_market(up_price=0.45, floor_strike=100.0)
        ctx = make_ctx(points, remaining_sec=120, market=market, funding_rate=0.0002)
        strategy = AIPromptStrategy(config={"prompt_template": BARRIER_PROMPT_TEMPLATE}, client=FakeChatClient([]))
        prompt = strategy._build_prompt(ctx)
        # every placeholder got substituted -> no leftover bare identifiers, and the
        # literal JSON schema at the end survived untouched (the whole point of $ vs {}).
        self.assertNotIn("$", prompt)
        self.assertIn('"base_prob":0.00,"adjustment":0.00,"prob_up":0.00', prompt)
        self.assertIn("TEST", prompt)  # symbol guessed from market_series_id="TEST-SERIES"
        self.assertNotIn("n/a", prompt)  # enough history -> z_score/base_prob/drift/mom all resolved

    async def test_full_barrier_template_end_to_end_produces_signal(self):
        noisy = [100 + (0.05 if i % 2 == 0 else -0.05) for i in range(30)]
        points = make_points(noisy, start_ts=0.0, dt=10.0)
        market = make_market(up_price=0.45, floor_strike=100.0)
        ctx = make_ctx(points, remaining_sec=120, market=market)
        fake = FakeChatClient(['{"base_prob":0.50,"adjustment":0.09,"prob_up":0.59,"reason":"momentum up"}'])
        strategy = AIPromptStrategy(
            config={"prompt_template": BARRIER_PROMPT_TEMPLATE, "min_seconds_between_calls": 0}, client=fake,
        )
        signal = await strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)


class BuildClientConfigTests(unittest.TestCase):
    """Provider -> base_url/model/api_key resolution, isolated from real env vars."""

    def _clear_llm_env(self):
        for var in (
            "REQUESTY_API_KEY", "REQUESTY_BASE_URL", "REQUESTY_MODEL",
            "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
            "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_MODEL",
        ):
            os.environ.pop(var, None)

    def setUp(self):
        self._clear_llm_env()

    def tearDown(self):
        self._clear_llm_env()

    def test_no_key_returns_none(self):
        self.assertIsNone(build_client_config({"provider": "requesty"}))

    def test_requesty_defaults(self):
        os.environ["REQUESTY_API_KEY"] = "rk-test"
        cfg = build_client_config({"provider": "requesty"})
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.api_key, "rk-test")
        self.assertEqual(cfg.base_url, "https://router.requesty.ai/v1")
        self.assertEqual(cfg.model, "anthropic/claude-haiku-4-5-20251001")

    def test_requesty_model_override_from_config_wins_over_env(self):
        os.environ["REQUESTY_API_KEY"] = "rk-test"
        os.environ["REQUESTY_MODEL"] = "google/gemini-2.5-flash-lite"
        cfg = build_client_config({"provider": "requesty", "model": "deepseek/deepseek-chat"})
        self.assertEqual(cfg.model, "deepseek/deepseek-chat")

    def test_deepseek_provider_defaults(self):
        os.environ["DEEPSEEK_API_KEY"] = "dk-test"
        cfg = build_client_config({"provider": "deepseek"})
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.base_url, "https://api.deepseek.com")
        self.assertEqual(cfg.model, "deepseek-chat")

    def test_unknown_provider_falls_back_to_requesty(self):
        os.environ["REQUESTY_API_KEY"] = "rk-test"
        cfg = build_client_config({"provider": "totally-not-a-provider"})
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.base_url, "https://router.requesty.ai/v1")


if __name__ == "__main__":
    unittest.main()
