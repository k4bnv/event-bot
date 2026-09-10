import math
import os
import random
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import ChatAPIError
from src.models import Direction, EventMarket, OrderBookLevel, OrderBookSnapshot, PricePoint, TradePrint
from src.strategies.absorption_reversal import (
    AbsorptionReversalStrategy, book_side_replenished, breaks_local_range, compute_tfi,
    expected_return_pct, realized_vol_pct_per_sqrt_sec,
)
from src.strategies.adaptive_timing import AdaptiveTimingStrategy
from src.strategies.ai_prompt import (
    BARRIER_PROMPT_TEMPLATE, AIPromptStrategy, build_client_config, _guess_symbol, _pct_change_over,
)
from src.strategies.base import StrategyContext
from src.strategies.fair_value_edge import (
    FairValueEdgeStrategy, basis_sigma_for_market, fair_probability_up, min_sigma_per_sec_from_pct,
)
from src.strategies.favorite_bias import FavoriteBiasStrategy
from src.strategies.funding_skew import FundingSkewStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.prior_window_momentum import PriorWindowMomentumStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy


def make_points(prices: list[float], start_ts: float = 0.0, dt: float = 1.0) -> list[PricePoint]:
    return [PricePoint(ts=start_ts + i * dt, price=p) for i, p in enumerate(prices)]


def make_market(
    up_price=None, floor_strike=None, method="price_up_down", expiry_ts=None, strike_is_fixed=None,
) -> EventMarket:
    return EventMarket(
        series_id="TEST-SERIES", method=method, inst_id="TEST-INST-1",
        expiry_ts=expiry_ts or time.time() + 300, floor_strike=floor_strike,
        up_price=up_price, state="live", strike_is_fixed=strike_is_fixed,
    )


def make_ctx(
    price_history, orderbook=None, remaining_sec=200.0, window_min=7, market=None, funding_rate=None,
    previous_outcome=None, already_open_this_market=False,
):
    return StrategyContext(
        price_history=price_history, orderbook=orderbook, remaining_sec=remaining_sec,
        window_min=window_min, market=market or make_market(), funding_rate=funding_rate,
        previous_outcome=previous_outcome, already_open_this_market=already_open_this_market,
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


class BarrierRobustnessTests(unittest.IsolatedAsyncioTestCase):
    """Covers the two robustness terms adapted from
    preceptress/btc-15-minute-prediction-model — see fair_value_edge.py's
    module docstring."""

    def test_min_sigma_per_sec_from_pct_scales_by_sqrt_time(self):
        # 0.06%/min floor -> per-second sigma is that /100 /sqrt(60).
        self.assertAlmostEqual(min_sigma_per_sec_from_pct(0.06), 0.0006 / math.sqrt(60), places=10)

    def test_min_sigma_per_sec_from_pct_clamps_negative_to_zero(self):
        self.assertEqual(min_sigma_per_sec_from_pct(-1.0), 0.0)

    def test_basis_sigma_is_zero_when_strike_is_fixed_or_unknown(self):
        self.assertEqual(basis_sigma_for_market(make_market(strike_is_fixed=True), 0.075), 0.0)
        self.assertEqual(basis_sigma_for_market(make_market(strike_is_fixed=None), 0.075), 0.0)

    def test_basis_sigma_applies_only_when_explicitly_unfixed(self):
        self.assertAlmostEqual(basis_sigma_for_market(make_market(strike_is_fixed=False), 0.075), 0.00075)

    def test_volatility_floor_turns_zero_variance_into_a_real_probability(self):
        # Same fixture as FairValueEdgeMathTests.test_none_with_zero_volatility
        # (flat prices -> exactly zero realized variance) — WITHOUT a floor
        # this still gives up (see that test); WITH one, price==strike
        # should resolve to ~0.5 instead of just refusing to answer.
        points = make_points([100.0] * 15)
        prob = fair_probability_up(
            points, 100.0, 100.0, 60, min_sigma_per_sec=min_sigma_per_sec_from_pct(0.035),
        )
        self.assertIsNotNone(prob)
        self.assertAlmostEqual(prob, 0.5, delta=0.01)

    def test_basis_sigma_pulls_an_off_strike_probability_toward_half(self):
        noisy = [100 + (0.05 if i % 2 == 0 else -0.05) for i in range(20)]
        points = make_points(noisy)
        without_basis = fair_probability_up(points, 101.0, 100.0, 60)
        with_basis = fair_probability_up(points, 101.0, 100.0, 60, basis_sigma=0.02)
        self.assertIsNotNone(without_basis)
        self.assertIsNotNone(with_basis)
        self.assertGreater(without_basis, 0.5)  # price above strike -> P(UP) > 0.5 either way
        # More uncertainty (unfixed/proxied strike) should make the estimate
        # LESS extreme, i.e. closer to the uninformative 0.5, not more.
        self.assertLess(with_basis, without_basis)

    async def test_unfixed_strike_can_suppress_a_signal_the_fixed_case_would_take(self):
        # Price has drifted to 101 against a strike of 100 (only the LAST
        # point matters as the strategy's anchor — see evaluate()'s
        # points[-1].price) — a real edge, not the price==strike symmetric
        # case where z stays 0 regardless of sigma and basis_sigma would
        # have no effect to test at all.
        noisy = [100 + (0.05 if i % 2 == 0 else -0.05) for i in range(19)] + [101.0]
        points = make_points(noisy)
        strategy = FairValueEdgeStrategy(config={"min_edge": 0.08, "unfixed_strike_basis_pct": 5.0})

        fixed_market = make_market(up_price=0.55, floor_strike=100.0, strike_is_fixed=True)
        signal = await strategy.evaluate(make_ctx(points, remaining_sec=120, market=fixed_market))
        self.assertIsNotNone(signal)  # model P(UP)=0.652 vs market 0.55 -> edge 0.102, above min_edge

        unfixed_market = make_market(up_price=0.55, floor_strike=100.0, strike_is_fixed=False)
        signal2 = await strategy.evaluate(make_ctx(points, remaining_sec=120, market=unfixed_market))
        self.assertIsNone(signal2)  # basis uncertainty pulls P(UP) to 0.570 -> edge 0.020, below min_edge


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


class AdaptiveTimingStrategyTests(unittest.IsolatedAsyncioTestCase):
    """Same crowd-favorite signal as FavoriteBiasStrategy (deliberately —
    see the module docstring for the full signal history: fair_value_edge
    -> orderbook_momentum -> this, all 2026-09-10, keeping the scanning
    shell each time), so the interesting behavior to test here is the
    scanning/one-shot-per-market part as much as the signal itself."""

    def _config(self):
        return {"favorite_price_threshold": 0.70}

    async def test_signals_up_when_up_is_favored(self):
        market = make_market(up_price=0.85)
        ctx = make_ctx([], market=market, remaining_sec=120)
        strategy = AdaptiveTimingStrategy(config=self._config())
        signal = await strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)

    async def test_signals_down_when_down_is_favored(self):
        market = make_market(up_price=0.10)  # down_price = 0.90
        ctx = make_ctx([], market=market, remaining_sec=120)
        strategy = AdaptiveTimingStrategy(config=self._config())
        signal = await strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.DOWN)

    async def test_no_signal_below_threshold(self):
        market = make_market(up_price=0.6)  # neither side reaches 0.70
        ctx = make_ctx([], market=market, remaining_sec=120)
        strategy = AdaptiveTimingStrategy(config=self._config())
        self.assertIsNone(await strategy.evaluate(ctx))

    async def test_no_signal_without_market_quote(self):
        strategy = AdaptiveTimingStrategy(config=self._config())
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=make_market(up_price=None))))

    async def test_already_open_this_market_suppresses_a_signal_it_would_otherwise_take(self):
        """The whole point of the scan: once it's placed one trade in a
        market, it must sit out every later checkpoint of that SAME
        market even if a side still (or again) looks favored — otherwise
        a strategy meant to enter at most once per market would stack
        bets exactly like the fixed-checkpoint strategies do on purpose."""
        market = make_market(up_price=0.85)
        strategy = AdaptiveTimingStrategy(config=self._config())

        # Without the flag, this exact same setup DOES signal (sanity check).
        ctx_free = make_ctx([], market=market, remaining_sec=120, already_open_this_market=False)
        self.assertIsNotNone(await strategy.evaluate(ctx_free))

        ctx_committed = make_ctx([], market=market, remaining_sec=90, already_open_this_market=True)
        self.assertIsNone(await strategy.evaluate(ctx_committed))


class AbsorptionReversalMathTests(unittest.TestCase):
    def test_compute_tfi_all_buy_is_plus_one(self):
        prints = [TradePrint(ts=0, price=100, size=1.0, side="buy") for _ in range(5)]
        self.assertEqual(compute_tfi(prints), 1.0)

    def test_compute_tfi_all_sell_is_minus_one(self):
        prints = [TradePrint(ts=0, price=100, size=1.0, side="sell") for _ in range(5)]
        self.assertEqual(compute_tfi(prints), -1.0)

    def test_compute_tfi_balanced_is_zero(self):
        prints = [TradePrint(ts=0, price=100, size=1.0, side=s) for s in ("buy", "sell")]
        self.assertEqual(compute_tfi(prints), 0.0)

    def test_compute_tfi_none_with_no_volume(self):
        self.assertIsNone(compute_tfi([]))
        self.assertIsNone(compute_tfi([TradePrint(ts=0, price=100, size=0.0, side="buy")]))

    def test_realized_vol_none_with_insufficient_points(self):
        points = make_points([100] * 5)
        self.assertIsNone(realized_vol_pct_per_sqrt_sec(points))

    def test_realized_vol_none_with_zero_variance(self):
        points = make_points([100.0] * 20)  # perfectly flat -> zero realized vol, no floor here
        self.assertIsNone(realized_vol_pct_per_sqrt_sec(points))

    def test_realized_vol_positive_with_real_fluctuation(self):
        points = make_points([100 + (0.05 if i % 2 == 0 else -0.05) for i in range(20)])
        vol = realized_vol_pct_per_sqrt_sec(points)
        self.assertIsNotNone(vol)
        self.assertGreater(vol, 0)

    def test_expected_return_pct_sign_follows_tfi(self):
        self.assertGreater(expected_return_pct(tfi=0.5, vol_pct_per_sqrt_sec=1.0, horizon_sec=100, sensitivity=1.0), 0)
        self.assertLess(expected_return_pct(tfi=-0.5, vol_pct_per_sqrt_sec=1.0, horizon_sec=100, sensitivity=1.0), 0)
        self.assertEqual(expected_return_pct(tfi=0.0, vol_pct_per_sqrt_sec=1.0, horizon_sec=100, sensitivity=1.0), 0.0)

    def test_book_side_replenished_true_when_depth_held_up(self):
        past = OrderBookSnapshot(ts=0, bids=[OrderBookLevel(price=100, size=5.0)], asks=[])
        current = OrderBookSnapshot(ts=30, bids=[OrderBookLevel(price=100, size=4.0)], asks=[])
        self.assertTrue(book_side_replenished([past, current], now=30, lookback_sec=60, depth=10, side="bid", min_ratio=0.7))

    def test_book_side_replenished_false_when_draining(self):
        past = OrderBookSnapshot(ts=0, bids=[OrderBookLevel(price=100, size=5.0)], asks=[])
        current = OrderBookSnapshot(ts=30, bids=[OrderBookLevel(price=100, size=1.0)], asks=[])
        self.assertFalse(book_side_replenished([past, current], now=30, lookback_sec=60, depth=10, side="bid", min_ratio=0.7))

    def test_book_side_replenished_false_with_too_little_history(self):
        current = OrderBookSnapshot(ts=30, bids=[OrderBookLevel(price=100, size=5.0)], asks=[])
        self.assertFalse(book_side_replenished([current], now=30, lookback_sec=60, depth=10, side="bid", min_ratio=0.7))

    def test_breaks_local_range_up(self):
        points = make_points([100, 101, 99, 100, 102])
        self.assertTrue(breaks_local_range(points, Direction.UP))
        self.assertFalse(breaks_local_range(points, Direction.DOWN))

    def test_breaks_local_range_false_with_too_few_points(self):
        self.assertFalse(breaks_local_range(make_points([100, 101]), Direction.UP))


def make_absorption_scenario(dominant_side: str, seed: int = 7, breakout: bool = True):
    """A synthetic scenario built to satisfy every phase of
    AbsorptionReversalStrategy at once, with default config — verified
    numerically (see the PR/commit this landed in) rather than guessed.
    dominant_side="sell": heavy aggressor selling, price barely falls (an
    absorption of sell pressure) -> UP candidate. dominant_side="buy" is
    the exact mirror -> DOWN candidate. breakout=False keeps phases A/B
    intact but never lets price actually break the recent range, so
    Phase C alone is what's being tested to fail."""
    rng = random.Random(seed)
    now = 1_000_000.0
    sign = -1 if dominant_side == "sell" else 1

    price_history = []
    base = 60000.0
    t, p = now - 700, base
    while t < now - 90:
        p *= (1 + rng.gauss(0, 0.00015))
        price_history.append(PricePoint(ts=t, price=p))
        t += 3.0
    while t < now - 3:
        # A tiny persistent move AGAINST what the dominant aggressor side
        # implies — heavy selling (sign=-1) still nudges price up a hair,
        # heavy buying nudges it down a hair. That divergence from the
        # TFI-implied direction is the absorption signature Phase A looks for.
        p *= (1 + sign * 0.00002)
        price_history.append(PricePoint(ts=t, price=p))
        t += 3.0
    if breakout:
        recent = [pt.price for pt in price_history[-10:]]
        p = (max(recent) * 1.0005) if dominant_side == "sell" else (min(recent) * 0.9995)
    price_history.append(PricePoint(ts=now, price=p))

    prints = []
    tp = now - 90
    while tp < now:
        side = dominant_side if rng.random() < 0.85 else ("buy" if dominant_side == "sell" else "sell")
        prints.append(TradePrint(ts=tp, price=p, size=rng.uniform(0.05, 0.2), side=side))
        tp += 2.0
    tp = now - 600
    while tp < now - 90:
        prints.append(TradePrint(ts=tp, price=base, size=rng.uniform(0.01, 0.05), side=rng.choice(["buy", "sell"])))
        tp += 15.0
    prints.sort(key=lambda x: x.ts)

    # The side under "attack" (bid for a sell-dominant/UP setup, ask for a
    # buy-dominant/DOWN one) stays replenished — current depth ~90% of
    # 30s-ago, comfortably above the default min_replenish_ratio (0.7).
    confirm_side = "bid" if dominant_side == "sell" else "ask"
    if confirm_side == "bid":
        book_old = OrderBookSnapshot(ts=now - 30, bids=[OrderBookLevel(price=p * 0.999, size=5.0)], asks=[OrderBookLevel(price=p * 1.001, size=5.0)])
        book_new = OrderBookSnapshot(ts=now, bids=[OrderBookLevel(price=p * 0.9995, size=4.5)], asks=[OrderBookLevel(price=p * 1.0005, size=5.0)])
    else:
        book_old = OrderBookSnapshot(ts=now - 30, bids=[OrderBookLevel(price=p * 0.999, size=5.0)], asks=[OrderBookLevel(price=p * 1.001, size=5.0)])
        book_new = OrderBookSnapshot(ts=now, bids=[OrderBookLevel(price=p * 0.9995, size=5.0)], asks=[OrderBookLevel(price=p * 1.0005, size=4.5)])

    return price_history, prints, [book_old, book_new], book_new, now


def make_absorption_ctx(dominant_side: str, **overrides):
    price_history, prints, orderbook_history, book_new, now = make_absorption_scenario(
        dominant_side, breakout=overrides.pop("breakout", True),
    )
    market = make_market(up_price=0.5, floor_strike=60000.0, expiry_ts=now + 300)
    defaults = dict(
        price_history=price_history, orderbook=book_new, remaining_sec=300, window_min=5,
        market=market, trade_prints=prints, orderbook_history=orderbook_history,
    )
    defaults.update(overrides)
    return StrategyContext(**defaults)


class AbsorptionReversalStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_setup_signals_up_on_absorbed_selling(self):
        strategy = AbsorptionReversalStrategy(config={})
        signal = await strategy.evaluate(make_absorption_ctx("sell"))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)

    async def test_full_pass_writes_every_diagnostic_stage(self):
        # ctx.diagnostics (see StrategyContext) is what ends up in
        # checkpoint_features' extra_json — a full pass through all three
        # phases should leave every stage's numbers behind, not just the
        # ones that happened to end up in signal.reason's free text.
        strategy = AbsorptionReversalStrategy(config={})
        ctx = make_absorption_ctx("sell")
        signal = await strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        for key in (
            "spread_pct", "n_prints", "tfi", "volume_multiple", "actual_return_pct",
            "predicted_return_pct", "residual_pct", "candidate_direction",
            "replenish_ratio", "broke_local_range",
        ):
            self.assertIn(key, ctx.diagnostics)
        self.assertEqual(ctx.diagnostics["candidate_direction"], "up")
        self.assertTrue(ctx.diagnostics["broke_local_range"])

    async def test_early_bail_still_writes_the_stages_it_reached(self):
        # Phase A rejects on TFI alone here — Phase B/C's diagnostics
        # (replenish_ratio, broke_local_range) must NOT appear, since
        # evaluate() never got that far, but everything up to and
        # including the tfi check should still be logged as "why not".
        strategy = AbsorptionReversalStrategy(config={"min_abs_tfi": 0.99})
        ctx = make_absorption_ctx("sell")
        signal = await strategy.evaluate(ctx)
        self.assertIsNone(signal)
        self.assertIn("spread_pct", ctx.diagnostics)
        self.assertIn("tfi", ctx.diagnostics)
        self.assertNotIn("residual_pct", ctx.diagnostics)
        self.assertNotIn("replenish_ratio", ctx.diagnostics)

    async def test_full_setup_signals_down_on_absorbed_buying(self):
        strategy = AbsorptionReversalStrategy(config={})
        signal = await strategy.evaluate(make_absorption_ctx("buy"))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.DOWN)

    async def test_already_open_this_market_suppresses_a_signal_it_would_otherwise_take(self):
        strategy = AbsorptionReversalStrategy(config={})
        ctx = make_absorption_ctx("sell", already_open_this_market=True)
        self.assertIsNone(await strategy.evaluate(ctx))

    async def test_no_signal_before_the_breakout_trigger(self):
        # Phases A and B are satisfied, but price hasn't actually broken
        # the local range yet — Phase C must still hold the line.
        strategy = AbsorptionReversalStrategy(config={})
        ctx = make_absorption_ctx("sell", breakout=False)
        self.assertIsNone(await strategy.evaluate(ctx))

    async def test_no_signal_when_tfi_not_extreme_enough(self):
        strategy = AbsorptionReversalStrategy(config={"min_abs_tfi": 0.99})  # real TFI here is ~0.74
        self.assertIsNone(await strategy.evaluate(make_absorption_ctx("sell")))

    async def test_no_signal_when_residual_too_small(self):
        strategy = AbsorptionReversalStrategy(config={"min_residual_pct": 5.0})  # real residual here is ~0.07%
        self.assertIsNone(await strategy.evaluate(make_absorption_ctx("sell")))

    async def test_no_signal_when_volume_not_elevated_enough(self):
        strategy = AbsorptionReversalStrategy(config={"min_volume_multiple": 1000.0})
        self.assertIsNone(await strategy.evaluate(make_absorption_ctx("sell")))

    async def test_no_signal_when_book_side_is_not_confirmed(self):
        strategy = AbsorptionReversalStrategy(config={"min_replenish_ratio": 1.5})  # impossible to satisfy
        self.assertIsNone(await strategy.evaluate(make_absorption_ctx("sell")))

    async def test_no_signal_with_too_few_prints(self):
        strategy = AbsorptionReversalStrategy(config={"min_prints": 10_000})
        self.assertIsNone(await strategy.evaluate(make_absorption_ctx("sell")))

    async def test_no_signal_without_orderbook_or_trade_prints(self):
        strategy = AbsorptionReversalStrategy(config={})
        ctx = make_absorption_ctx("sell")
        ctx.orderbook = None
        self.assertIsNone(await strategy.evaluate(ctx))

        ctx2 = make_absorption_ctx("sell")
        ctx2.trade_prints = []
        self.assertIsNone(await strategy.evaluate(ctx2))

    async def test_no_signal_when_spread_too_wide(self):
        strategy = AbsorptionReversalStrategy(config={"max_spread_pct": 0.001})  # far stricter than the fixture's book
        self.assertIsNone(await strategy.evaluate(make_absorption_ctx("sell")))


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


class MeanReversionStrategyTests(unittest.IsolatedAsyncioTestCase):
    """Direction flipped 2026-09-10 to bet CONTINUATION instead of
    reversion — see the strategy's module docstring for why (0/23 live
    on the original revert-to-mean call). These lock in the flipped
    behavior so a future edit can't silently flip it back."""

    def _config(self):
        return {"lookback_sec": 60, "extreme_zscore": 1.5}

    async def test_price_spike_up_bets_continuation_up(self):
        now = 1000.0
        calm = [100 + (0.01 if i % 2 == 0 else -0.01) for i in range(50)]
        history = make_points(calm, start_ts=now - 59, dt=1.0)
        history[-1] = PricePoint(ts=now, price=103.0)  # sharp outlier above the mean
        ctx = make_ctx(history, remaining_sec=200)
        strategy = MeanReversionStrategy(config=self._config())
        signal = await strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)  # continuation, not reversion

    async def test_price_spike_down_bets_continuation_down(self):
        now = 1000.0
        calm = [100 + (0.01 if i % 2 == 0 else -0.01) for i in range(50)]
        history = make_points(calm, start_ts=now - 59, dt=1.0)
        history[-1] = PricePoint(ts=now, price=97.0)  # sharp outlier below the mean
        ctx = make_ctx(history, remaining_sec=200)
        strategy = MeanReversionStrategy(config=self._config())
        signal = await strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.DOWN)  # continuation, not reversion

    async def test_no_signal_when_calm(self):
        now = 1000.0
        prices = [100 + (0.01 if i % 2 == 0 else -0.01) for i in range(60)]
        history = make_points(prices, start_ts=now - 59, dt=1.0)
        ctx = make_ctx(history, remaining_sec=200)
        strategy = MeanReversionStrategy(config=self._config())
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
    # A market with a live up_price — evaluate() now skips the call
    # entirely (and returns None before ever reaching the LLM) when
    # market.up_price is None, since there'd be nothing sound to compare
    # the model's estimate against. Tests exercising what happens AFTER a
    # call need a live price so they actually reach that code, not just
    # the new early-exit gate.
    LIVE_MARKET = staticmethod(lambda: make_market(up_price=0.5))

    async def test_valid_up_response_produces_signal(self):
        fake = FakeChatClient(['{"direction": "UP", "confidence": 0.8, "reason": "test"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        ctx = make_ctx([PricePoint(ts=0, price=100)] * 12, market=self.LIVE_MARKET())
        signal = await strategy.evaluate(ctx)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)
        self.assertAlmostEqual(signal.confidence, 0.8)
        self.assertEqual(fake.calls, 1)

    async def test_no_signal_and_no_call_when_market_has_no_live_price(self):
        # The gate itself: no up_price -> skip before spending an API call.
        fake = FakeChatClient(['{"direction": "UP", "confidence": 0.8, "reason": "test"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        signal = await strategy.evaluate(make_ctx([], market=make_market(up_price=None)))
        self.assertIsNone(signal)
        self.assertEqual(fake.calls, 0)

    async def test_none_direction_yields_no_signal(self):
        fake = FakeChatClient(['{"direction": "NONE", "confidence": 0.5, "reason": "no edge"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET())))

    async def test_malformed_json_degrades_to_no_signal(self):
        fake = FakeChatClient(["not json at all"])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET())))

    async def test_api_error_degrades_to_no_signal(self):
        fake = FakeChatClient([ChatAPIError("boom")])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET())))

    async def test_confidence_is_clamped_to_0_1(self):
        fake = FakeChatClient(['{"direction": "DOWN", "confidence": 5, "reason": "x"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        signal = await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET()))
        self.assertEqual(signal.confidence, 1.0)

    async def test_cooldown_skips_second_call(self):
        fake = FakeChatClient([
            '{"direction": "UP", "confidence": 0.6, "reason": "a"}',
            '{"direction": "UP", "confidence": 0.6, "reason": "b"}',
        ])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 9999}, client=fake)
        first = await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET()))
        second = await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET()))
        self.assertIsNotNone(first)
        self.assertIsNone(second)   # cooldown -> no signal, and no extra API call
        self.assertEqual(fake.calls, 1)

    async def test_cooldown_is_independent_per_series(self):
        # Real bug this fixes: a single shared cooldown meant one series
        # firing could starve every OTHER series for min_seconds_between_calls,
        # even though EntryWindowManager already guarantees each is only
        # ever due once per window on its own.
        fake = FakeChatClient([
            '{"direction": "UP", "confidence": 0.6, "reason": "5min"}',
            '{"direction": "UP", "confidence": 0.6, "reason": "15min"}',
        ])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 9999}, client=fake)
        market_5min = make_market(up_price=0.5)
        market_5min.series_id = "BTC-UPDOWN-5MIN"
        market_15min = make_market(up_price=0.5)
        market_15min.series_id = "BTC-UPDOWN-15MIN"

        first = await strategy.evaluate(make_ctx([], market=market_5min, window_min=2))
        second = await strategy.evaluate(make_ctx([], market=market_15min, window_min=2))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)  # different series -> NOT blocked by the other's cooldown
        self.assertEqual(fake.calls, 2)

    async def test_cooldown_now_blocks_across_different_checkpoints_of_the_same_series(self):
        # Changed 2026-09-10 when this strategy became dynamic_timing
        # (dense scanning grid, like J/K — see ai_prompt.py's module
        # docstring): the cooldown key dropped window_min and is now just
        # the series_id, specifically so repeated checkpoints of the SAME
        # still-open market collapse onto one cooldown timer instead of
        # each getting its own independent shot — otherwise a market that
        # never signals could burn through most of max_calls_per_day on
        # its own before ever closing.
        fake = FakeChatClient([
            '{"direction": "UP", "confidence": 0.6, "reason": "a"}',
            '{"direction": "UP", "confidence": 0.6, "reason": "b"}',
        ])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 9999}, client=fake)
        market = self.LIVE_MARKET()

        first = await strategy.evaluate(make_ctx([], market=market, window_min=4))
        second = await strategy.evaluate(make_ctx([], market=market, window_min=2))
        self.assertIsNotNone(first)
        self.assertIsNone(second)  # same series, different checkpoint -> still cooled down now
        self.assertEqual(fake.calls, 1)

    async def test_already_open_this_market_suppresses_a_call_it_would_otherwise_make(self):
        """Checked FIRST, before the cooldown/budget bookkeeping (see
        evaluate()) — once a trade's opened in a market, every later
        checkpoint of that SAME market must cost nothing, not just skip
        trading. Same already_open_this_market mechanism adaptive_timing/
        absorption_reversal use for their own dense scanning grids."""
        fake = FakeChatClient(['{"direction": "UP", "confidence": 0.6, "reason": "a"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)

        signal = await strategy.evaluate(
            make_ctx([], market=self.LIVE_MARKET(), already_open_this_market=True)
        )
        self.assertIsNone(signal)
        self.assertEqual(fake.calls, 0)  # never even reached the client

    async def test_daily_call_cap_blocks_further_calls(self):
        fake = FakeChatClient([
            '{"direction": "UP", "confidence": 0.6, "reason": "a"}',
            '{"direction": "UP", "confidence": 0.6, "reason": "b"}',
        ])
        strategy = AIPromptStrategy(
            config={"min_seconds_between_calls": 0, "max_calls_per_day": 1}, client=fake,
        )
        first = await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET()))
        second = await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET()))
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
        signal = await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET()))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)

    def test_custom_prompt_template_is_used(self):
        strategy = AIPromptStrategy(config={"prompt_template": "hello $series world"}, client=FakeChatClient([]))
        market = make_market(up_price=0.4, floor_strike=100.0)
        market.series_id = "MY-SERIES"
        prompt = strategy._build_prompt(make_ctx([], market=market))
        self.assertEqual(prompt, "hello MY-SERIES world")

    def test_confluence_placeholders_describe_the_barrier_margin(self):
        # $strike_distance_pct/$side_now back CONFLUENCE_PROMPT_TEMPLATE's
        # "safety margin in plain terms" framing (see that constant's
        # comment) — the deliberate alternative to handing the model a
        # normal-CDF number to fudge.
        strategy = AIPromptStrategy(
            config={"prompt_template": "d=${strike_distance_pct} side=$side_now need=$min_agree", "min_agree": 3},
            client=FakeChatClient([]),
        )
        market = make_market(up_price=0.4, floor_strike=100.0)
        ctx = make_ctx([PricePoint(ts=0, price=100.5)] * 12, market=market)
        self.assertEqual(strategy._build_prompt(ctx), "d=+0.500 side=UP need=3")

        below = make_ctx([PricePoint(ts=0, price=99.5)] * 12, market=market)
        self.assertEqual(strategy._build_prompt(below), "d=-0.500 side=DOWN need=3")

    def test_confluence_placeholders_are_na_without_a_strike(self):
        strategy = AIPromptStrategy(
            config={"prompt_template": "d=${strike_distance_pct} side=$side_now"}, client=FakeChatClient([]),
        )
        ctx = make_ctx([PricePoint(ts=0, price=100.0)] * 12, market=make_market(up_price=0.4, floor_strike=None))
        self.assertEqual(strategy._build_prompt(ctx), "d=n/a side=n/a")

    async def test_min_confidence_drops_a_hedged_call(self):
        fake = FakeChatClient([
            '{"direction": "UP", "agree": 4, "confidence": 0.55, "reason": "meh"}',
            '{"direction": "UP", "agree": 4, "confidence": 0.80, "reason": "clean"}',
        ])
        strategy = AIPromptStrategy(
            config={"min_seconds_between_calls": 0, "min_confidence": 0.7}, client=fake,
        )
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET())))
        signal = await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET()))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)

    async def test_min_agree_drops_a_thin_confluence(self):
        fake = FakeChatClient([
            '{"direction": "DOWN", "agree": 2, "confidence": 0.9, "reason": "half"}',
            '{"direction": "DOWN", "agree": 3, "confidence": 0.9, "reason": "enough"}',
        ])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0, "min_agree": 3}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET())))
        signal = await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET()))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.DOWN)

    async def test_min_agree_does_not_block_a_schema_without_that_field(self):
        # A response carrying no "agree" at all (e.g. DEFAULT_PROMPT_TEMPLATE's
        # schema) must not be silently blocked by a gate meant for the
        # confluence template — see _signal_from_direction's docstring.
        fake = FakeChatClient(['{"direction": "UP", "confidence": 0.9, "reason": "other schema"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0, "min_agree": 3}, client=fake)
        self.assertIsNotNone(await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET())))

    async def test_gates_are_off_by_default(self):
        # Unconfigured => the plain default template behaves exactly as before.
        fake = FakeChatClient(['{"direction": "UP", "agree": 1, "confidence": 0.3, "reason": "weak"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        self.assertIsNotNone(await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET())))

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
        # market up_price=0.5 makes prob_up-market_px degenerate to the old
        # prob_up-0.5 comparison for this test's own numbers specifically.
        fake = FakeChatClient(['{"base_prob":0.5,"adjustment":0.08,"prob_up":0.58,"reason":"drift up"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0, "min_edge": 0.05}, client=fake)
        signal = await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET()))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)
        self.assertAlmostEqual(signal.confidence, 0.16, places=2)  # |0.58-0.5|*2

    async def test_prob_up_below_half_bets_down(self):
        fake = FakeChatClient(['{"prob_up":0.4,"reason":"drift down"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0, "min_edge": 0.05}, client=fake)
        signal = await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET()))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.DOWN)

    async def test_prob_up_within_band_of_half_no_signal(self):
        fake = FakeChatClient(['{"prob_up":0.52,"reason":"barely off coinflip"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0, "min_edge": 0.05}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET())))

    async def test_prob_up_missing_or_non_numeric_no_signal(self):
        fake = FakeChatClient(['{"prob_up":"not-a-number","reason":"x"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0}, client=fake)
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=self.LIVE_MARKET())))

    async def test_edge_is_measured_against_market_price_not_a_flat_half(self):
        # The actual bug this fixes: a naive prob_up-0.5 comparison would
        # bet UP here (0.60 > 0.5) even though the market is ALREADY at
        # 0.85 — i.e. the model thinks UP is LESS likely than the market
        # does. That's a real disagreement favoring DOWN, not UP.
        fake = FakeChatClient(['{"prob_up":0.60,"reason":"model still bullish but less than market"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0, "min_edge": 0.05}, client=fake)
        market = make_market(up_price=0.85, floor_strike=100.0)
        signal = await strategy.evaluate(make_ctx([], market=market))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.DOWN)  # not UP, despite prob_up > 0.5

    async def test_no_signal_when_prob_up_matches_market_price(self):
        # Model's estimate agrees with the market almost exactly -> no
        # edge, regardless of how far prob_up itself sits from 0.5.
        fake = FakeChatClient(['{"prob_up":0.83,"reason":"agrees with market"}'])
        strategy = AIPromptStrategy(config={"min_seconds_between_calls": 0, "min_edge": 0.05}, client=fake)
        market = make_market(up_price=0.85, floor_strike=100.0)
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=market)))

    async def test_adjustment_beyond_max_is_clamped_server_side(self):
        # The model ignores "не более ±0.10" and returns a 0.30 adjustment
        # anyway — the final prob_up must reflect the CLAMPED adjustment
        # (0.5+0.10=0.60), not the model's own unclamped arithmetic (0.80),
        # regardless of what prob_up field the model itself echoed back.
        fake = FakeChatClient(['{"base_prob":0.5,"adjustment":0.30,"prob_up":0.80,"reason":"overconfident"}'])
        strategy = AIPromptStrategy(
            config={"min_seconds_between_calls": 0, "min_edge": 0.05, "max_adjustment": 0.10}, client=fake,
        )
        market = make_market(up_price=0.5, floor_strike=100.0)
        signal = await strategy.evaluate(make_ctx([], market=market))
        self.assertIsNotNone(signal)
        self.assertIn("prob_up=0.600", signal.reason)  # clamped to 0.5+0.10, not the model's 0.80


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
        self.assertEqual(cfg.model, "anthropic/claude-haiku-4-5")

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


class FavoriteBiasStrategyTests(unittest.IsolatedAsyncioTestCase):
    """The deliberate opposite of fair_value_edge — see the module
    docstring for the favorite-longshot-bias / crowd-momentum /
    Resolution Rider ideas merged into this one rule."""

    async def test_no_signal_below_threshold(self):
        strategy = FavoriteBiasStrategy(config={"favorite_price_threshold": 0.70})
        market = make_market(up_price=0.6)  # neither side reaches 0.70
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=market)))

    async def test_up_favored_bets_up(self):
        strategy = FavoriteBiasStrategy(config={"favorite_price_threshold": 0.70})
        market = make_market(up_price=0.85)
        signal = await strategy.evaluate(make_ctx([], market=market))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)
        self.assertIn("0.85", signal.reason)

    async def test_down_favored_bets_down(self):
        strategy = FavoriteBiasStrategy(config={"favorite_price_threshold": 0.70})
        market = make_market(up_price=0.10)  # down_price = 0.90
        signal = await strategy.evaluate(make_ctx([], market=market))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.DOWN)

    async def test_exactly_at_threshold_still_signals(self):
        strategy = FavoriteBiasStrategy(config={"favorite_price_threshold": 0.70})
        market = make_market(up_price=0.70)
        signal = await strategy.evaluate(make_ctx([], market=market))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)

    async def test_no_signal_without_market_quote(self):
        strategy = FavoriteBiasStrategy(config={})
        market = make_market(up_price=None)
        self.assertIsNone(await strategy.evaluate(make_ctx([], market=market)))

    async def test_confidence_increases_toward_certainty(self):
        strategy = FavoriteBiasStrategy(config={"favorite_price_threshold": 0.70})
        near_threshold = await strategy.evaluate(make_ctx([], market=make_market(up_price=0.71)))
        near_certain = await strategy.evaluate(make_ctx([], market=make_market(up_price=0.98)))
        self.assertLess(near_threshold.confidence, near_certain.confidence)

    async def test_custom_threshold_is_respected(self):
        strategy = FavoriteBiasStrategy(config={"favorite_price_threshold": 0.55})
        signal = await strategy.evaluate(make_ctx([], market=make_market(up_price=0.60)))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)


class PriorWindowMomentumStrategyTests(unittest.IsolatedAsyncioTestCase):
    """Trend-following baseline — see the module docstring for why this
    exists as a deliberate control group, not a "real" strategy."""

    async def test_no_signal_when_previous_outcome_unknown(self):
        strategy = PriorWindowMomentumStrategy(config={})
        self.assertIsNone(await strategy.evaluate(make_ctx([], previous_outcome=None)))

    async def test_bets_up_after_up(self):
        strategy = PriorWindowMomentumStrategy(config={})
        signal = await strategy.evaluate(make_ctx([], previous_outcome=Direction.UP))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.UP)

    async def test_bets_down_after_down(self):
        strategy = PriorWindowMomentumStrategy(config={})
        signal = await strategy.evaluate(make_ctx([], previous_outcome=Direction.DOWN))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.DOWN)


if __name__ == "__main__":
    unittest.main()
