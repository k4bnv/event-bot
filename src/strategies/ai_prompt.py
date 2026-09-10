"""
Strategy G — AI Prompt, opt-in.

Sends a compact text summary of the contract, recent price action, order
book and funding rate — plus an optional user-supplied `news_context`
string — to an LLM, and asks for a strict JSON verdict. This is the one
strategy meant to carry *qualitative* context (news, narrative) that the
purely numeric strategies can't use; feeding it only numbers is a weak use
of an LLM (plain statistics do that better/cheaper/faster), so
`news_context` is the whole point — wire a real news feed into it if you
want this strategy to earn its keep.

Provider: `strategies.ai_prompt.provider` in config.yaml selects which
OpenAI-compatible gateway/env vars to use —
  * "requesty" (default) — https://router.requesty.ai/v1, one API key
    (REQUESTY_API_KEY) routes to 600+ models from any provider, addressed
    as "provider/model" (e.g. "anthropic/claude-haiku-4-5",
    "google/gemini-2.5-flash-lite", "deepseek/deepseek-chat"). See
    https://docs.requesty.ai/features/supported-models for the current
    catalog/pricing before picking a model.
  * "deepseek" — talks to DeepSeek directly (DEEPSEEK_API_KEY), no router.
  * "openai_compatible" — generic escape hatch for any other gateway via
    OPENAI_COMPATIBLE_{API_KEY,BASE_URL,MODEL}.

Disabled by default in config.yaml: it costs money per call and needs an
API key. Every failure mode (missing key, network error, timeout,
malformed JSON) degrades to "no signal" — this strategy must never crash
the engine just because an LLM had a bad day.

Converted to dynamic_timing (free-scanning, like J/K) 2026-09-10, at the
user's request, after G-2м's first live stretch came back 0 wins out of
18 — no obvious bug found in the edge-direction code (the prob_up/edge
math already had its own documented fix, see _signal_from_prob_up), so
rather than keep guessing which of a couple of fixed checkpoints (was
[4, 2]) suits an LLM verdict best, it now scans a dense grid and picks
its own moment per market, same as J/K already do (see
adaptive_timing.py/absorption_reversal.py's own module docstrings for
that mechanism — ctx.already_open_this_market, one shared wallet). Two
knock-on changes this needed, both about keeping API spend sane once a
market gets evaluated at up to 10 checkpoints instead of 2:
  1. `already_open_this_market` is now checked FIRST, before even the
     cooldown/budget bookkeeping — once a trade's opened in a market,
     every later checkpoint of that SAME market must cost nothing, not
     just skip trading.
  2. The per-call cooldown (`min_seconds_between_calls`) used to be
     keyed per (series, checkpoint) specifically so two fixed checkpoints
     minutes apart wouldn't starve each other (see __init__). With a
     dense scanning grid that reasoning flips: many checkpoints of the
     SAME still-open market firing within the same short cooldown window
     is exactly the case to collapse together, or a slow LLM day could
     burn most of max_calls_per_day on markets that were never going to
     signal. Keyed by series_id alone now.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from string import Template
from typing import Optional

from ..models import Direction, EventMarket, PricePoint
from .base import BaseStrategy, Signal, StrategyContext
from .fair_value_edge import (
    DEFAULT_MIN_SIGMA_PCT_PER_MIN, DEFAULT_UNFIXED_STRIKE_BASIS_PCT,
    basis_sigma_for_market, compute_barrier_stats, min_sigma_per_sec_from_pct,
)

try:
    from ..llm_client import ChatAPIError, ChatClient, ChatClientConfig
except ImportError:  # pragma: no cover - llm_client has no extra deps, this is just defensive
    ChatClient = None  # type: ignore[assignment]

logger = logging.getLogger("okx_event_bot.strategies.ai_prompt")

# Editable via config.yaml -> strategies.ai_prompt.prompt_template (shows up
# as a textarea in the dashboard's Settings tab, since it's just another
# `extra` field). Uses string.Template ($name / ${name}) rather than
# str.format({name}) specifically because prompts that ask for JSON back
# (this one included) contain literal {...} — with str.format that's a
# ValueError/garbled output waiting to happen; $ never collides with JSON
# braces, and a typo'd or removed $placeholder is left as literal text
# instead of crashing the strategy (safe_substitute()).
#
# Available placeholders (all pre-computed each call, in _build_prompt):
#   $series $method $strike $target (alias of $strike) $spot (current
#   underlying price) $symbol (underlying ticker guessed from $series)
#   $remaining $seconds_left (alias of $remaining) $market_px $lookback
#   $price_series $orderbook_line $funding_line $context_line (these three
#   are pre-built as full lines, each already ending in "\n" when they have
#   content, "" when they don't — avoids stray blank lines e.g. with no
#   orderbook) $z_score $base_prob $sigma_horizon (the barrier-model z-score/
#   CDF-probability/vol-scaled-to-horizon behind the fair_value_edge
#   strategy — "n/a" for all three if there isn't enough price history yet)
#   $drift_5m $mom_1m (price % change over the last 5min/1min — "n/a" if
#   not enough history). $max_adjustment (the config.yaml max_adjustment
#   value, formatted — see BARRIER_PROMPT_TEMPLATE and
#   _signal_from_prob_up for why this is enforced in code, not just text).
DEFAULT_PROMPT_TEMPLATE = (
    "BTC event contract, decide UP/DOWN/NONE before expiry.\n"
    "series=$series method=$method strike=$strike remaining=${remaining}s market_px_up=$market_px\n"
    "prices(${lookback}s,oldest-first)=[$price_series]\n"
    "$orderbook_line$funding_line$context_line"
    'JSON only, no other text: {"direction":"UP"|"DOWN"|"NONE","confidence":0-1,"reason":"<=6 words"}'
)

# Alternative template: statistical barrier estimate (z-score / normal CDF)
# as an anchor, with the LLM applying a small BOUNDED correction from
# short-term drift/momentum/orderbook/funding — copy into config.yaml's
# prompt_template (or paste via the dashboard's Settings tab) to use this
# instead. Expects the {"base_prob":...,"adjustment":...,"prob_up":...}
# response schema — see _parse_response/_signal_from_prob_up. The "не более
# ±$max_adjustment" instruction here is not just wording: _signal_from_prob_up
# recomputes prob_up itself from base_prob + a server-side-clamped
# adjustment whenever both fields are present, rather than trusting
# whatever prob_up number the model echoes back — free text doesn't
# reliably bind an LLM's own arithmetic, so the bound is enforced in code,
# not just requested in the prompt.
BARRIER_PROMPT_TEMPLATE = (
    "$symbol, барьер $target, спот $spot, market_px_up=$market_px, осталось $seconds_left сек.\n"
    "\n"
    "Предрасчитано в коде:\n"
    "  z-score: $z_score\n"
    "  нормальная CDF(z): $base_prob\n"
    "  sigma на оставшийся горизонт: ${sigma_horizon}%\n"
    "Дрейф 5 мин: ${drift_5m}% | Импульс 1 мин: ${mom_1m}%\n"
    "$orderbook_line$funding_line$context_line"
    "\n"
    "BASE_PROB — чисто статистическая оценка без учёта направления рынка.\n"
    "Скорректируй её на дрейф, импульс, стакан и funding. Коррекция не более ±${max_adjustment}.\n"
    "Если сигналы разнонаправлены — коррекция близка к нулю.\n"
    "Торгуем расхождением между твоей итоговой prob_up и market_px_up, а не отклонением от 0.5 —\n"
    "рынок мог уже частично отразить движение.\n"
    "\n"
    '{"base_prob":0.00,"adjustment":0.00,"prob_up":0.00,\n'
    '"reason":"<=6 слов"}'
)

# provider -> (api_key env var, default base_url, default model)
_PROVIDER_DEFAULTS = {
    "requesty": ("REQUESTY_API_KEY", "https://router.requesty.ai/v1", "anthropic/claude-haiku-4-5"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    "openai_compatible": ("OPENAI_COMPATIBLE_API_KEY", "", ""),
}


def _pct_change_over(points: list[PricePoint], now: float, window_sec: float) -> Optional[float]:
    """% change from the oldest sample within `window_sec` to the latest
    overall sample. None if fewer than 2 points fall in that window."""
    window = [p for p in points if now - p.ts <= window_sec]
    if len(window) < 2 or window[0].price <= 0:
        return None
    return (window[-1].price - window[0].price) / window[0].price * 100


def _guess_symbol(series_id: str) -> str:
    """e.g. 'BTC-UPDOWN-15MIN' -> 'BTC'. Falls back to the whole seriesId
    if it doesn't look like the usual UNDERLYING-METHOD-... shape."""
    return series_id.split("-")[0] if series_id else series_id


def build_client_config(config: dict) -> Optional["ChatClientConfig"]:
    """Resolve provider/model/base_url/api_key from `config` + environment.
    Returns None if the selected provider has no API key set (caller should
    then skip the strategy rather than crash). Pure function — kept
    separate from the strategy class so provider-selection logic can be
    unit tested without spinning up a whole strategy/engine."""
    if ChatClient is None:
        return None

    provider = str(config.get("provider", "requesty")).lower()
    if provider not in _PROVIDER_DEFAULTS:
        logger.warning("ai_prompt: unknown provider '%s', falling back to 'requesty'", provider)
        provider = "requesty"

    key_env, default_base_url, default_model = _PROVIDER_DEFAULTS[provider]
    api_key = os.getenv(key_env, "")
    if not api_key:
        return None

    base_url_env = f"{provider.upper()}_BASE_URL"
    model_env = f"{provider.upper()}_MODEL"
    return ChatClientConfig(
        api_key=api_key,
        base_url=os.getenv(base_url_env, default_base_url) or default_base_url,
        model=config.get("model") or os.getenv(model_env, default_model) or default_model,
        timeout_sec=float(config.get("timeout_sec", 20.0)),
        use_json_response_format=bool(config.get("use_json_response_format", True)),
    )


class AIPromptStrategy(BaseStrategy):
    name = "ai_prompt"

    def __init__(self, config: dict, client: Optional["ChatClient"] = None):
        super().__init__(config)
        # Keyed by series_id alone, NOT a single shared timestamp and NOT
        # per-checkpoint either (see the module docstring's "Converted to
        # dynamic_timing" note for why that changed) — a single float
        # cooldown would let one series' market silently eat the whole
        # strategy's budget and block every OTHER series for
        # min_seconds_between_calls, so different series still need
        # independent keys. But within ONE series, this now scans a dense
        # checkpoint grid per market (entry_windows_min), and collapsing
        # all of a still-open market's repeated "still no edge" calls onto
        # one cooldown timer is exactly the point: max_calls_per_day is
        # the real, hard cost ceiling regardless (see below), but this
        # cooldown is what keeps one slow-to-signal market from burning
        # through most of it before ctx.already_open_this_market even gets
        # a chance to stop the calls for good.
        self._last_call_ts: dict[str, float] = {}
        self._warned_no_key = False
        self._warned_budget = False
        self._call_times: deque[float] = deque()  # for the rolling 24h call-count cap

        if client is not None:
            self._client = client  # injected — used by tests, bypasses env/key lookup
            return
        client_cfg = build_client_config(config)
        self._client = ChatClient(client_cfg) if client_cfg is not None else None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

    async def evaluate(self, ctx: StrategyContext) -> Optional[Signal]:
        if ctx.already_open_this_market:
            return None  # already committed to this market at an earlier checkpoint — costs nothing to check first

        if self._client is None:
            if not self._warned_no_key:
                provider = str(self.config.get("provider", "requesty"))
                logger.warning(
                    "ai_prompt strategy is enabled but no API key is set for provider='%s' "
                    "(see that provider's *_API_KEY in .env) — every checkpoint will be skipped.",
                    provider,
                )
                self._warned_no_key = True
            return None

        if ctx.market.up_price is None:
            return None  # nothing to show the model or compare its estimate against — save the call

        min_gap = float(self.config.get("min_seconds_between_calls", 300))
        now = time.time()
        cooldown_key = ctx.market.series_id
        if now - self._last_call_ts.get(cooldown_key, 0.0) < min_gap:
            return None

        # Hard ceiling on API spend, independent of the cooldown above and
        # of any assumption about how many series/checkpoints are configured
        # — this is the actual safety net for a small prepaid balance.
        # Counts every attempted call (success or failure), checked BEFORE
        # spending anything.
        max_per_day = int(self.config.get("max_calls_per_day", 100))
        while self._call_times and now - self._call_times[0] > 86400:
            self._call_times.popleft()
        if len(self._call_times) >= max_per_day:
            if not self._warned_budget:
                logger.warning(
                    "ai_prompt: hit max_calls_per_day=%d — skipping further calls until "
                    "the 24h window rolls off. Raise this in config.yaml if you want more.",
                    max_per_day,
                )
                self._warned_budget = True
            return None
        self._warned_budget = False
        self._call_times.append(now)

        prompt = self._build_prompt(ctx)
        try:
            raw = await self._client.chat_json(
                prompt,
                max_tokens=int(self.config.get("max_tokens", 80)),
                temperature=float(self.config.get("temperature", 0.2)),
            )
        except ChatAPIError as exc:
            logger.warning("ai_prompt: LLM call failed, skipping this checkpoint: %s", exc)
            return None
        finally:
            self._last_call_ts[cooldown_key] = now

        logger.info("ai_prompt raw response: %s", raw[:500])
        return self._parse_response(raw, ctx.market)

    # -- prompt / parsing -----------------------------------------------------------
    # Kept deliberately terse by default — every extra word is input tokens
    # on every single call. `price_sample_count` / `max_tokens` /
    # `min_seconds_between_calls` / `max_calls_per_day` in config.yaml are
    # the actual spend controls; `prompt_template` just controls wording.
    def _build_prompt(self, ctx: StrategyContext) -> str:
        market = ctx.market
        lookback_sec = float(self.config.get("lookback_sec", 180))
        news_context = str(self.config.get("news_context", "")).strip()
        sample_count = int(self.config.get("price_sample_count", 8))

        now = ctx.price_history[-1].ts if ctx.price_history else time.time()
        recent = [p for p in ctx.price_history if now - p.ts <= lookback_sec]
        step = max(1, len(recent) // sample_count)
        sampled = recent[::step]
        price_series = ",".join(f"{p.price:.1f}" for p in sampled)

        ob = ctx.orderbook
        bid_vol = ob.bid_volume(10) if ob else None
        ask_vol = ob.ask_volume(10) if ob else None

        spot = ctx.price_history[-1].price if ctx.price_history else None
        barrier = (
            compute_barrier_stats(
                recent, spot, market.floor_strike, ctx.remaining_sec,
                min_sigma_per_sec=min_sigma_per_sec_from_pct(DEFAULT_MIN_SIGMA_PCT_PER_MIN),
                basis_sigma=basis_sigma_for_market(market, DEFAULT_UNFIXED_STRIKE_BASIS_PCT),
            )
            if spot is not None and market.floor_strike is not None else None
        )
        drift_5m = _pct_change_over(ctx.price_history, now, 300) if ctx.price_history else None
        mom_1m = _pct_change_over(ctx.price_history, now, 60) if ctx.price_history else None
        max_adjustment = float(self.config.get("max_adjustment", 0.10))

        values = {
            "series": market.series_id, "method": market.method, "strike": market.floor_strike,
            "target": market.floor_strike,
            "remaining": f"{ctx.remaining_sec:.0f}", "seconds_left": f"{ctx.remaining_sec:.0f}",
            "market_px": f"{market.up_price:.3f}" if market.up_price is not None else "n/a",
            "spot": f"{spot:.2f}" if spot is not None else "n/a",
            "symbol": _guess_symbol(market.series_id),
            "lookback": f"{lookback_sec:.0f}", "price_series": price_series,
            "orderbook_line": (
                f"orderbook_top10: bid={bid_vol:.2f} ask={ask_vol:.2f}\n"
                if bid_vol is not None and ask_vol is not None else ""
            ),
            "funding_line": (
                f"funding_rate={ctx.funding_rate * 100:.4f}% (+=longs pay)\n"
                if ctx.funding_rate is not None else ""
            ),
            "context_line": f"context: {news_context}\n" if news_context else "",
            "z_score": f"{barrier.z_score:.3f}" if barrier else "n/a",
            "base_prob": f"{barrier.base_prob:.3f}" if barrier else "n/a",
            "sigma_horizon": f"{barrier.sigma_horizon_pct:.3f}" if barrier else "n/a",
            "max_adjustment": f"{max_adjustment:.2f}",
            "drift_5m": f"{drift_5m:.3f}" if drift_5m is not None else "n/a",
            "mom_1m": f"{mom_1m:.3f}" if mom_1m is not None else "n/a",
        }

        template_str = str(self.config.get("prompt_template") or DEFAULT_PROMPT_TEMPLATE)
        try:
            # safe_substitute() never raises for a bad/unknown placeholder —
            # it leaves it as literal text (verified: test_strategies.py) —
            # so this except is a defensive belt-and-suspenders for any
            # other unexpected failure, not something safe_substitute is
            # actually known to trigger today.
            return Template(template_str).safe_substitute(values)
        except ValueError as exc:
            logger.warning(
                "ai_prompt: prompt_template raised on substitution (%s) — using the default template this call.", exc
            )
            return Template(DEFAULT_PROMPT_TEMPLATE).safe_substitute(values)

    @staticmethod
    def _strip_markdown_fence(raw: str) -> str:
        """Not every model/gateway combination honors response_format
        perfectly — some still wrap JSON in ```json ... ``` fences. Strip
        those defensively before parsing."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        return text.strip()

    def _parse_response(self, raw: str, market: EventMarket) -> Optional[Signal]:
        cleaned = self._strip_markdown_fence(raw)
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            logger.warning("ai_prompt: could not parse JSON from response: %r", raw[:200])
            return None

        # Two supported response schemas, detected by which key is present:
        #   {"direction": "UP"|"DOWN"|"NONE", "confidence": 0-1, "reason": ...}   (default template)
        #   {"prob_up": 0-1, "reason": ...}   (BARRIER_PROMPT_TEMPLATE and similar)
        if "prob_up" in data:
            return self._signal_from_prob_up(data, market)
        return self._signal_from_direction(data)

    def _signal_from_direction(self, data: dict) -> Optional[Signal]:
        direction_raw = str(data.get("direction", "")).strip().upper()
        if direction_raw not in ("UP", "DOWN"):
            return None  # "NONE" or anything unrecognized -> no trade, not an error

        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(max(confidence, 0.0), 1.0)

        reason = str(data.get("reason", "AI analysis"))[:200]
        direction = Direction.UP if direction_raw == "UP" else Direction.DOWN
        return Signal(direction=direction, reason=f"AI: {reason}", confidence=confidence)

    def _signal_from_prob_up(self, data: dict, market: EventMarket) -> Optional[Signal]:
        """Barrier-style schema: the model returns an adjusted P(UP)
        instead of a discrete direction. Two guardrails enforced HERE,
        not just requested in the prompt — free text doesn't reliably
        constrain an LLM's own arithmetic:

        1. The edge that decides whether/which side to trade is
           `prob_up - market.up_price` — the market's own current price —
           NOT `prob_up - 0.5`. A probability simply being > 50% means
           nothing on its own; only a DISAGREEMENT with what the market
           already prices in is a mispricing worth betting on. (This used
           to compare against a flat 0.5, which meant it could bet UP on a
           market already trading at 0.85 as long as the model said
           anything above 50% — the wrong side of a genuine edge.) Mirrors
           fair_value_edge's own edge definition exactly.

        2. When the response includes base_prob/adjustment separately
           (BARRIER_PROMPT_TEMPLATE's schema), the final probability is
           RECOMPUTED here as base_prob + a server-side-clamped
           adjustment (±max_adjustment) — rather than trusting whatever
           prob_up number the model echoed back, which could silently
           ignore the prompt's own "не более ±X" instruction. Falls back
           to trusting prob_up verbatim only when base_prob/adjustment
           aren't both present (e.g. a custom, simpler prompt_template).
        """
        max_adjustment = float(self.config.get("max_adjustment", 0.10))
        try:
            base_prob = float(data.get("base_prob"))
            adjustment = float(data.get("adjustment"))
        except (TypeError, ValueError):
            base_prob = adjustment = None

        if base_prob is not None and adjustment is not None:
            clamped_adjustment = min(max(adjustment, -max_adjustment), max_adjustment)
            prob_up = min(max(base_prob + clamped_adjustment, 0.0), 1.0)
        else:
            try:
                prob_up = float(data.get("prob_up"))
            except (TypeError, ValueError):
                logger.warning("ai_prompt: prob_up missing/non-numeric in response: %r", data)
                return None
            prob_up = min(max(prob_up, 0.0), 1.0)

        if market.up_price is None:
            return None  # nothing to compare the model's estimate against

        min_edge = float(self.config.get("min_edge", 0.05))
        edge = prob_up - market.up_price
        if abs(edge) < min_edge:
            return None

        reason = str(data.get("reason", "AI barrier adjustment"))[:200]
        direction = Direction.UP if edge > 0 else Direction.DOWN
        confidence = min(abs(edge) * 2, 1.0)
        return Signal(
            direction=direction,
            reason=f"AI: {reason} (prob_up={prob_up:.3f}, market={market.up_price:.3f}, edge={edge:+.3f})",
            confidence=confidence,
        )
