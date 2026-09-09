#!/usr/bin/env python3
"""
Entry point.

    python run.py                     # run with config.yaml (mock or live per mock_mode)
    python run.py --config other.yaml # use a different config file
    python run.py --discover-series   # print live OKX EVENTS series/instruments and exit
                                       # (helps you fill in okx.series_ids in config.yaml)

Ctrl+C (and, since wallet balances now survive a redeploy, a `docker stop`/
`docker compose up --build` SIGTERM too) stops cleanly: the engine finishes
its current tick, writes a final wallet snapshot to data/bot.db, and exits.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import aiohttp

from src.config import load_config
from src.engine import Engine
from src.logger import setup_logging
from src.market_data import OkxMarketDataProvider
from src.mock_market import MockMarketDataProvider
from src.models import EventMarket, OrderBookLevel, PricePoint, simulate_market_fill
from src.okx_client import OKXClient, OKXClientConfig
from src.storage import Storage
from src.strategies.ai_prompt import AIPromptStrategy, build_client_config
from src.strategies.base import StrategyContext

logger = logging.getLogger("okx_event_bot.run")


async def discover_series(cfg) -> None:
    """Utility mode: list whatever Event Contract series OKX currently
    exposes on your account, so you can copy the right seriesId values into
    config.yaml. NOTE: unlike ordinary spot market data, OKX's event-contract
    endpoints require a signed (authenticated) request even to just browse
    series/markets — so this needs valid API keys even though it's read-only."""
    if not (cfg.okx.api_key and cfg.okx.api_secret and cfg.okx.api_passphrase):
        print(
            "--discover-series needs OKX Demo Trading API keys (event-contract "
            "endpoints require authentication even for read-only browsing). "
            "Copy .env.example to .env and fill in OKX_API_KEY / OKX_API_SECRET / "
            "OKX_API_PASSPHRASE first.",
            file=sys.stderr,
        )
        sys.exit(1)

    client_cfg = OKXClientConfig(
        base_url=cfg.okx.base_url, api_key=cfg.okx.api_key, api_secret=cfg.okx.api_secret,
        api_passphrase=cfg.okx.api_passphrase, demo_trading=cfg.okx.demo_trading,
        timeout_sec=cfg.okx.request_timeout_sec, max_retries=cfg.okx.max_retries,
    )
    async with OKXClient(client_cfg) as client:
        series = await client.get_event_series()
        if not series:
            print("OKX returned no Event Contract series right now (or your region/account "
                  "doesn't have Event Contracts enabled). Check the OKX app to confirm the "
                  "product is available to you, then retry.")
            return

        print(f"\nFound {len(series)} event-contract series:\n")
        for s in series:
            sid = s.get("seriesId", "?")
            settlement = s.get("settlement") or {}
            method = settlement.get("method", "?")
            underlying = settlement.get("underlying", "?")
            freq = s.get("freq", "?")
            print(f"  seriesId: {sid:<28} method={method:<16} underlying={underlying:<10} freq={freq}")

            try:
                markets = await client.get_event_markets(series_id=sid, state="live")
            except Exception:
                markets = []
            live = [m.get("instId", "?") for m in markets[:3]]
            if live:
                print(f"      live now: {', '.join(live)}" + (" …" if len(markets) > 3 else ""))

        print("\nCopy the seriesId values you want into config.yaml -> okx.series_ids\n")


def _parse_book_levels(raw_levels: list) -> list[OrderBookLevel]:
    """Convert OKX's raw [priceStr, sizeStr, ...] rows (from /market/books)
    into OrderBookLevel objects, silently skipping any malformed row."""
    out = []
    for level in raw_levels:
        try:
            out.append(OrderBookLevel(price=float(level[0]), size=float(level[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


async def _measure_series_liquidity(client: OKXClient, series_id: str, test_stake_usd: float) -> Optional[dict]:
    """One fetch+simulate pass for a single series' current live
    instrument. Prints the human-readable detail (same as before) AND
    returns a flat dict of the numeric fields worth aggregating across
    repeated samples — None if this pass produced nothing usable (no live
    market / empty ticker / empty book)."""

    def _exp_ms(m: dict) -> float:
        raw = m.get("expTime")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float("inf")

    try:
        markets = await client.get_event_markets(series_id=series_id, state="live")
    except Exception as exc:
        print(f"{series_id}: failed to list live markets ({exc})")
        return None
    if not markets:
        print(f"{series_id}: no live markets right now")
        return None

    chosen = min(markets, key=_exp_ms)  # nearest-expiry instrument, same pick the engine itself trades
    inst_id = str(chosen.get("instId", "?"))
    exp_ms = _exp_ms(chosen)
    remaining_sec = (exp_ms / 1000.0 - time.time()) if exp_ms != float("inf") else None

    try:
        ticker = await client.get_ticker(inst_id)
        book = await client.get_orderbook(inst_id, sz=20)
    except Exception as exc:
        print(f"{series_id} ({inst_id}): failed to fetch ticker/book ({exc})")
        return None

    remaining_note = f"  (~{remaining_sec:.0f}s to expiry)" if remaining_sec is not None else ""
    print(f"\n{series_id}  ->  {inst_id}{remaining_note}")

    last, spread_pct = None, None
    if not ticker:
        print("  ticker: <empty — no live quote>")
    else:
        row = ticker[0]
        last, bid, ask = row.get("last"), row.get("bidPx"), row.get("askPx")
        bid_sz, ask_sz = row.get("bidSz"), row.get("askSz")
        print(f"  ticker: last={last}  bid={bid}({bid_sz})  ask={ask}({ask_sz})")
        try:
            bid_f, ask_f = float(bid), float(ask)
            mid = (bid_f + ask_f) / 2
            spread_abs = ask_f - bid_f
            spread_pct = (spread_abs / mid * 100) if mid else None
            print(f"  top-of-book spread: {spread_abs:.4f} absolute  ({spread_pct:.1f}% of mid {mid:.4f})")
        except (TypeError, ValueError):
            pass

    if not book:
        print("  book: <empty> — can't simulate a fill")
        return None

    b = book[0]
    asks, bids = _parse_book_levels(b.get("asks", [])), _parse_book_levels(b.get("bids", []))
    print(f"  book depth: {len(bids)} bid levels / {len(asks)} ask levels")

    # UP: walking real ask depth for a real BUY UP is a faithful
    # simulation (you're buying exactly the instrument those asks quote) —
    # same function the engine itself now uses (EventMarket.fill_price_for).
    up_vwap, up_contracts, up_spent, up_full = simulate_market_fill(asks, test_stake_usd)
    up_slippage_pct = None
    print(f"  simulated ${test_stake_usd:.2f} market BUY UP:")
    if up_vwap is None:
        print("    <no ask liquidity at all>")
    else:
        print(f"    vwap_fill={up_vwap:.4f}  contracts={up_contracts:.2f}  "
              f"spent=${up_spent:.2f}  {'(fully filled)' if up_full else '(BOOK RAN OUT — worse in reality)'}")
        try:
            engine_price = float(last) if last not in (None, "") else asks[0].price
            up_slippage_pct = (up_vwap - engine_price) / engine_price * 100
            print(f"    vs engine's simulated entry ({engine_price:.4f}): {up_slippage_pct:+.1f}% slippage")
        except (TypeError, ValueError, IndexError):
            pass

    # DOWN: bids are OTHER traders' resting buy-UP orders, not a depth of
    # offers to sell you DOWN — walking multiple levels like we do for UP
    # is not a faithful simulation (OKX doesn't publicly document how a
    # DOWN/"no" order actually matches internally), and produces nonsense
    # once the top level is thin. Report only the top-of-book estimate
    # (1 - best_bid) as a best-case floor, explicitly NOT a depth simulation.
    down_top_estimate, down_diff_pct = None, None
    print("  DOWN top-of-book estimate (best case only — NOT a depth simulation, see docstring):")
    if not bids:
        print("    <no bid liquidity at all>")
    else:
        try:
            best_bid = bids[0].price
            down_top_estimate = round(1 - best_bid, 4)
            engine_price_down = round(1 - float(last), 4) if last not in (None, "") else down_top_estimate
            if engine_price_down:
                down_diff_pct = (down_top_estimate - engine_price_down) / engine_price_down * 100
                print(f"    best case ~{down_top_estimate:.4f} (vs engine's {engine_price_down:.4f}: "
                      f"{down_diff_pct:+.1f}%) — a real fill only gets WORSE (higher) than this the "
                      f"deeper the order has to walk; how much worse isn't something the public book "
                      f"tells us for this side.")
            else:
                print(f"    best case ~{down_top_estimate:.4f}")
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            pass

    try:
        last_f = float(last) if last not in (None, "") else None
    except (TypeError, ValueError):
        last_f = None

    return {
        "ts": time.time(), "series_id": series_id, "inst_id": inst_id,
        "remaining_sec": remaining_sec, "last": last_f, "spread_pct": spread_pct,
        "up_vwap": up_vwap, "up_slippage_pct": up_slippage_pct,
        "up_book_ran_out": (not up_full) if up_vwap is not None else None,
        "down_top_estimate": down_top_estimate, "down_diff_pct": down_diff_pct,
    }


def _print_liquidity_summary(rows: list[dict]) -> None:
    print(f"\n{'=' * 64}\nSummary over {len(rows)} samples\n{'=' * 64}")
    by_series: dict[str, list[dict]] = {}
    for r in rows:
        by_series.setdefault(r["series_id"], []).append(r)

    for series_id, group in by_series.items():
        slips = sorted(r["up_slippage_pct"] for r in group if r["up_slippage_pct"] is not None)
        spreads = [r["spread_pct"] for r in group if r["spread_pct"] is not None]
        remaining = [r["remaining_sec"] for r in group if r["remaining_sec"] is not None]
        ran_out = sum(1 for r in group if r["up_book_ran_out"])

        print(f"\n{series_id}  ({len(group)} samples)")
        if slips:
            median = slips[len(slips) // 2]
            print(f"  UP slippage %:  min={slips[0]:+.1f}  median={median:+.1f}  "
                  f"mean={sum(slips) / len(slips):+.1f}  max={slips[-1]:+.1f}")
        else:
            print("  UP slippage %: no usable samples")
        if spreads:
            print(f"  top-of-book spread %:  min={min(spreads):.1f}  "
                  f"mean={sum(spreads) / len(spreads):.1f}  max={max(spreads):.1f}")
        if remaining:
            print(f"  remaining_sec at sample time:  min={min(remaining):.0f}  max={max(remaining):.0f}")
        if ran_out:
            print(f"  ⚠ book couldn't fully absorb the test order in {ran_out}/{len(group)} samples "
                  f"— real slippage there is WORSE than what's reported")


def _save_liquidity_csv(cfg, rows: list[dict]) -> Optional[Path]:
    if not rows:
        return None
    data_dir = Path(cfg.storage.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"liquidity_samples_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "ts", "series_id", "inst_id", "remaining_sec", "last", "spread_pct",
        "up_vwap", "up_slippage_pct", "up_book_ran_out", "down_top_estimate", "down_diff_pct",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRaw samples saved to {path} (mounted volume — readable from the host too)")
    return path


async def check_liquidity(
    cfg, test_stake_usd: float = 20.0, samples: int = 1, interval_sec: float = 30.0,
) -> None:
    """Utility mode: for every live Event Contract instrument currently open
    on the configured series, print its real ticker (last/bidPx/askPx),
    the computed bid/ask spread, AND a simulated market-order fill for a
    `test_stake_usd`-sized order walking the real order book depth —
    a genuine VWAP-based slippage estimate, not just a top-of-book number.

    Why this exists: the strategy engine simulates a fill at `last` (or the
    bid/ask midpoint as a fallback — see market_data.py's
    `_fetch_event_price`), with NO slippage/spread cost applied. On a thin
    market that can be well away from what a real market order would
    actually fill at. OKX's v5 API has no dry-run/"test order" endpoint
    (unlike e.g. Binance) — there's no way to ask the exchange itself "what
    would this order fill at" without actually placing one — so this
    command reconstructs the answer from the real, live, PUBLIC order book
    instead. No auth needed for ticker/books themselves (unlike the
    series/markets discovery calls, which do need a signed request — see
    okx_client.py), and no order is ever placed — zero execution risk.

    With `samples > 1`, repeats every `interval_sec` and prints a summary
    (min/median/mean/max UP slippage %, spread %, observed remaining_sec
    range) plus writes every raw sample to a CSV under the data dir — the
    point being to see whether slippage is a one-off or a consistent
    pattern before changing anything in the engine itself.

    Caveat: the book/ticker only clearly represents the UP/YES side's own
    bid-ask (see models.py EventMarket — OKX gives one px per instrument,
    not separate UP/DOWN books). The DOWN estimate is top-of-book only
    (1 - best_bid), explicitly NOT a depth simulation — OKX doesn't
    publicly document the exact internal matching for the non-primary
    side, and walking multiple bid levels there produces nonsense (see
    git history — an earlier version of this tool did that and was wrong).
    """
    if not (cfg.okx.api_key and cfg.okx.api_secret and cfg.okx.api_passphrase):
        print(
            "--check-liquidity needs OKX API keys to discover which instId is "
            "currently live per series (browsing event-contract markets requires "
            "authentication even though it's read-only). Copy .env.example to .env "
            "and fill in OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE first.",
            file=sys.stderr,
        )
        sys.exit(1)
    if samples < 1:
        print("--samples must be >= 1", file=sys.stderr)
        sys.exit(1)

    client_cfg = OKXClientConfig(
        base_url=cfg.okx.base_url, api_key=cfg.okx.api_key, api_secret=cfg.okx.api_secret,
        api_passphrase=cfg.okx.api_passphrase, demo_trading=cfg.okx.demo_trading,
        timeout_sec=cfg.okx.request_timeout_sec, max_retries=cfg.okx.max_retries,
    )
    all_rows: list[dict] = []
    async with OKXClient(client_cfg) as client:
        for i in range(samples):
            if samples > 1:
                print(f"\n{'#' * 64}\n# Sample {i + 1}/{samples}  —  {time.strftime('%H:%M:%S')}\n{'#' * 64}")
            for series_id in cfg.okx.series_ids:
                row = await _measure_series_liquidity(client, series_id, test_stake_usd)
                if row is not None:
                    all_rows.append(row)
            if samples > 1 and i < samples - 1:
                await asyncio.sleep(interval_sec)

    if samples > 1:
        _print_liquidity_summary(all_rows)
        _save_liquidity_csv(cfg, all_rows)


async def check_ai_prompt(cfg) -> None:
    """Utility mode: send ONE real test call through the ai_prompt
    strategy's actual client/prompt-building code (same config.yaml
    provider/model it uses live) against a synthetic-but-plausible market
    context, and print exactly what came back.

    Why this exists: ai_prompt only fires on its configured
    entry_windows_min checkpoints, at most once per min_seconds_between_calls
    — waiting for that live, then digging through data/bot.log to see what
    happened, is slow. This bypasses all of that: no live OKX data needed
    (works even with mock_mode/no OKX keys — the synthetic context below
    has everything the prompt template needs), and a freshly-constructed
    AIPromptStrategy's cooldown starts at 0, so the very first call always
    goes through regardless of min_seconds_between_calls. Answers "is this
    even configured right" (key/provider/model) separately from "did it
    actually decide to trade on live conditions" (a real market judgment
    call, not something this test can substitute for).

    Every possible reason "not opening trades" turns out to be something
    OTHER than a broken LLM call — enabled: false, no API key, cooldown/
    daily-cap not elapsed yet, entry_windows_min just not due, or the
    resulting signal getting rejected downstream by max_coefficient/
    max_slippage_pct — is called out explicitly in the output so this
    doesn't get misread as "the LLM path is broken" when it isn't.
    """
    ai_cfg = next((s for s in cfg.strategies if s.name == "ai_prompt"), None)
    if ai_cfg is None:
        print("No 'ai_prompt' entry under strategies: in config.yaml.", file=sys.stderr)
        sys.exit(1)

    print(f"enabled in config.yaml: {ai_cfg.enabled}")
    if not ai_cfg.enabled:
        print("  -> this is almost certainly why it's not opening trades live: flip to "
              "'enabled: true' (dashboard Settings tab, or config.yaml + restart).")

    merged_config = dict(ai_cfg.extra)
    provider = str(merged_config.get("provider", "requesty")).lower()
    client_cfg = build_client_config(merged_config)
    if client_cfg is None:
        key_env = {
            "requesty": "REQUESTY_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
            "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
        }.get(provider, f"{provider.upper()}_API_KEY")
        print(
            f"\nNo API key found for provider='{provider}' ({key_env} unset in .env).\n"
            "-> This is the other most likely reason for zero trades: the strategy silently "
            "skips every single checkpoint (one WARNING logged the first time, then quiet) "
            "rather than crash. Set the key in .env (copy from .env.example) and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"provider={provider}  model={client_cfg.model}  base_url={client_cfg.base_url}")
    print(
        f"cost controls: min_seconds_between_calls={merged_config.get('min_seconds_between_calls', 300)}  "
        f"max_calls_per_day={merged_config.get('max_calls_per_day', 100)}  "
        f"entry_windows_min={ai_cfg.entry_windows_min}  "
        f"(live, it only ever fires AT one of these checkpoints, at most once per cooldown — "
        f"quiet in between is normal, not broken)"
    )

    # Synthetic-but-plausible context — a connectivity/parsing smoke test,
    # not a live trading decision, so no real OKX data is fetched here.
    now = time.time()
    price_history = deque(PricePoint(ts=now - (30 - i) * 3, price=80000.0 + i * 5) for i in range(10))
    market = EventMarket(
        series_id="TEST-SERIES", method="price_up_down", inst_id="TEST-INST",
        expiry_ts=now + 120, floor_strike=80000.0, up_price=0.5, state="live",
    )
    ctx = StrategyContext(
        price_history=price_history, orderbook=None, remaining_sec=120.0, window_min=2,
        market=market, funding_rate=0.0001,
    )

    print("\nSending one test prompt (bypasses cooldown — fresh instance)...")
    strategy = AIPromptStrategy(config=merged_config)
    try:
        signal = await strategy.evaluate(ctx)
    finally:
        await strategy.aclose()

    if signal is None:
        print(
            "\nResult: no signal (None). Check the WARNING/INFO lines above (or "
            "data/bot.log, search for 'ai_prompt') for what actually happened — could be "
            "a genuine NONE/low-confidence verdict from the model (working as intended), "
            "or a real call failure (bad key, timeout, malformed JSON)."
        )
    else:
        print(
            f"\nResult: signal = {signal.direction.value.upper()}  "
            f"confidence={signal.confidence:.2f}  reason={signal.reason!r}"
        )
        print(
            "-> The LLM call/parse path works end-to-end. If it's still not trading live, "
            "the cause is elsewhere: enabled: false, the cooldown/daily-cap not elapsed, "
            "entry_windows_min checkpoint not due yet, or the signal getting rejected "
            "downstream by max_coefficient/max_slippage_pct (see engine.py logs)."
        )


BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"


async def _fetch_binance_price(session: aiohttp.ClientSession, symbol: str = "BTCUSDT") -> Optional[float]:
    """Binance's public spot ticker — no API key, no auth, same shape of
    call as OKX's own unauthenticated market/ticker endpoint. Binance is
    used as the "leader" reference here purely because it's the deepest,
    most liquid BTC spot market by a wide margin — if ANY venue is setting
    the pace of price discovery rather than following it, it's the most
    likely candidate."""
    try:
        async with session.get(
            BINANCE_TICKER_URL, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
            return float(data["price"])
    except (aiohttp.ClientError, TimeoutError, KeyError, ValueError, TypeError):
        return None


async def _fetch_okx_spot_price(client: OKXClient, inst_id: str) -> Optional[float]:
    try:
        data = await client.get_ticker(inst_id)
    except Exception:
        return None
    if not data:
        return None
    try:
        return float(data[0].get("last"))
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _detect_impulses(
    points: list[tuple[float, float]], threshold_pct: float, window_sec: float, cooldown_sec: float,
) -> list[dict]:
    """Walk a chronological (ts, price) series and flag "impulses": a move
    of at least threshold_pct within a trailing window_sec window. After a
    detected impulse, waits cooldown_sec before looking for the next one,
    so one sustained move isn't counted dozens of times as price keeps
    drifting through it. O(n) — the trailing window's left edge (`j`) only
    ever moves forward, since points are chronological and the window is
    fixed-width."""
    impulses: list[dict] = []
    last_impulse_ts = float("-inf")
    j = 0
    for i, (ts_i, price_i) in enumerate(points):
        while points[j][0] < ts_i - window_sec:
            j += 1
        if ts_i - last_impulse_ts < cooldown_sec or j == i:
            continue
        ts_j, price_j = points[j]
        if price_j <= 0:
            continue
        move_pct = (price_i - price_j) / price_j * 100
        if abs(move_pct) >= threshold_pct:
            impulses.append({
                "start_ts": ts_j, "end_ts": ts_i,
                "direction": "up" if move_pct > 0 else "down", "move_pct": move_pct,
            })
            last_impulse_ts = ts_i
    return impulses


def _measure_reaction_lag(
    impulses: list[dict], reactor_points: list[tuple[float, float]], horizon_sec: float, react_threshold_pct: float,
) -> list[Optional[float]]:
    """For each impulse (detected on the LEADER series), find how many
    seconds after it ended the reactor's own price shows a comparable
    same-direction move. Returns 0.0 if the reactor had ALREADY moved by
    the time the impulse ended (checked against the reactor's own price
    at the impulse's START — real zero-lag/no-lag case, the actual
    "nothing to exploit" outcome we're hoping to distinguish from "never
    reacted"), the delay in seconds if it reacted later within
    horizon_sec, or None if it never moved that much within horizon_sec.

    Getting the zero-lag case right matters: without it, "OKX reacted
    instantly" and "OKX never reacted at all" both come out as None —
    conflating the two most different possible answers into one."""
    lags: list[Optional[float]] = []
    for imp in impulses:
        start_ts, end_ts, direction = imp["start_ts"], imp["end_ts"], imp["direction"]

        pre_price, base_price = None, None
        for ts, price in reactor_points:
            if ts <= start_ts:
                pre_price = price
            if ts <= end_ts:
                base_price = price
            else:
                break
        if base_price is None or base_price <= 0:
            lags.append(None)
            continue

        if pre_price is not None and pre_price > 0:
            already_pct = (base_price - pre_price) / pre_price * 100
            if (direction == "up" and already_pct >= react_threshold_pct) or (
                direction == "down" and already_pct <= -react_threshold_pct
            ):
                lags.append(0.0)
                continue

        found = None
        for ts, price in reactor_points:
            if ts <= end_ts:
                continue
            if ts - end_ts > horizon_sec:
                break
            move_pct = (price - base_price) / base_price * 100
            if (direction == "up" and move_pct >= react_threshold_pct) or (
                direction == "down" and move_pct <= -react_threshold_pct
            ):
                found = ts - end_ts
                break
        lags.append(found)
    return lags


def _lag_percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list. Used
    alongside min/median/mean/max in the leadlag summaries: a median alone
    hides a skewed distribution — e.g. "most reactions are instant but a
    meaningful minority take several seconds" reads as "no lag" if you only
    look at the median, even though that slow tail might be the whole
    story worth investigating."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


async def check_leadlag(
    cfg, duration_sec: float = 300.0, poll_interval_sec: float = 1.0,
    impulse_threshold_pct: float = 0.03, window_sec: float = 10.0,
    lag_horizon_sec: float = 20.0, react_threshold_pct: Optional[float] = None,
) -> None:
    """Utility mode: measure whether OKX's own spot BTC-USDT ticker lags
    Binance's (the deepest/most liquid BTC market) — the first, cleanest
    question to answer before building any cross-exchange lead-lag
    strategy. No API keys needed at all: both are public, unauthenticated
    tickers.

    Method: poll both venues every poll_interval_sec for duration_sec,
    detect "impulse" moves on Binance (>= impulse_threshold_pct within a
    trailing window_sec window — same impulse+retest style detection as
    breakout_common.py, applied here to find events rather than trade
    signals), then measure how many seconds later (if at all, within
    lag_horizon_sec) OKX's own spot price makes a comparable same-direction
    move. This is a REAL measurement, not a guess — reports "no usable
    lag" just as readily as "here's a real lag", and says so explicitly.

    Deliberately scoped to spot-vs-spot only, NOT the event contract's own
    `up_price` — that's a separate, nonlinear quantity (function of both
    price distance from strike AND time-to-expiry, not price alone), and
    conflating the two in one measurement would make the result ambiguous.
    If this shows a real, exploitable spot lag, measuring up_price's own
    reaction time is the natural follow-up — worth a second command, not
    bolted onto this one.
    """
    if react_threshold_pct is None:
        react_threshold_pct = impulse_threshold_pct * 0.5

    client_cfg = OKXClientConfig(
        base_url=cfg.okx.base_url, api_key=cfg.okx.api_key, api_secret=cfg.okx.api_secret,
        api_passphrase=cfg.okx.api_passphrase, demo_trading=cfg.okx.demo_trading,
        timeout_sec=cfg.okx.request_timeout_sec, max_retries=cfg.okx.max_retries,
    )

    data_dir = Path(cfg.storage.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    samples_path = data_dir / f"leadlag_samples_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    print(
        f"Collecting {duration_sec:.0f}s of Binance vs OKX spot BTC-USDT ({cfg.okx.underlying_inst_id}), "
        f"polling every {poll_interval_sec:.1f}s — no API keys needed, both are public tickers.\n"
        f"Raw samples are written incrementally to {samples_path} as they come in — if this "
        f"session gets disconnected partway through (e.g. a dropped SSH/console), that file "
        f"still has everything collected up to the disconnect; run without `nohup` at your "
        f"own risk on a flaky connection.\n"
    )

    binance_series: list[tuple[float, float]] = []
    okx_series: list[tuple[float, float]] = []

    # Opened before the loop and flushed after every row — this is what
    # makes the run survive a dropped connection killing the process
    # mid-collection: whatever was written before the drop is safe on
    # disk, readable from the host too (data/ is a mounted volume).
    with open(samples_path, "w", newline="") as samples_file:
        samples_writer = csv.DictWriter(samples_file, fieldnames=["ts", "binance_price", "okx_price"])
        samples_writer.writeheader()

        async with OKXClient(client_cfg) as okx_client, aiohttp.ClientSession() as binance_session:
            end_at = time.time() + duration_sec
            n = 0
            while time.time() < end_at:
                tick_start = time.time()
                binance_price, okx_price = await asyncio.gather(
                    _fetch_binance_price(binance_session),
                    _fetch_okx_spot_price(okx_client, cfg.okx.underlying_inst_id),
                )
                now = time.time()
                if binance_price is not None:
                    binance_series.append((now, binance_price))
                if okx_price is not None:
                    okx_series.append((now, okx_price))
                samples_writer.writerow({"ts": now, "binance_price": binance_price, "okx_price": okx_price})
                samples_file.flush()
                n += 1
                if n % 30 == 0:
                    remaining = max(0.0, end_at - time.time())
                    print(f"  ...{n} samples so far, ~{remaining:.0f}s left (saved to {samples_path.name})")
                elapsed = time.time() - tick_start
                await asyncio.sleep(max(0.0, poll_interval_sec - elapsed))

    print(f"\nCollected {len(binance_series)} Binance samples, {len(okx_series)} OKX samples "
          f"(raw data in {samples_path}).")
    if len(binance_series) < 10 or len(okx_series) < 10:
        print("Not enough data to analyze (too many failed requests?) — check network/API access and retry.")
        return

    # Every line from here on is both printed live AND accumulated, then
    # written to a report file in `finally` — regardless of which return
    # point below is hit (no impulses / no reaction / full result), so a
    # dropped connection right at the end still leaves the conclusion on
    # disk, not just whatever scrolled past on a terminal that's now gone.
    report_lines: list[str] = []

    def out(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    report_path = data_dir / f"leadlag_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        impulses = _detect_impulses(binance_series, impulse_threshold_pct, window_sec, cooldown_sec=window_sec)
        out(
            f"\nDetected {len(impulses)} impulses on Binance "
            f"(>= {impulse_threshold_pct:.3f}% move within {window_sec:.0f}s, "
            f"{window_sec:.0f}s cooldown between detections):"
        )
        for imp in impulses:
            t = time.strftime("%H:%M:%S", time.localtime(imp["end_ts"]))
            out(f"  {t}  {imp['direction'].upper():5s}  move={imp['move_pct']:+.3f}%")

        if not impulses:
            out(
                "\nNo impulses detected in this window — try a longer --duration-sec, run during a more "
                "volatile period, or lower --impulse-threshold-pct. Can't measure a lag with no events to "
                "measure it from."
            )
            return

        lags = _measure_reaction_lag(impulses, okx_series, lag_horizon_sec, react_threshold_pct)
        out(
            f"\nOKX spot reaction (>= {react_threshold_pct:.3f}% same-direction move within "
            f"{lag_horizon_sec:.0f}s of the Binance impulse):"
        )
        for imp, lag in zip(impulses, lags):
            lag_str = f"{lag:.1f}s later" if lag is not None else f"no reaction within {lag_horizon_sec:.0f}s"
            out(f"  Binance {imp['direction'].upper():5s} {imp['move_pct']:+.3f}%  ->  OKX: {lag_str}")

        reacted = sorted(l for l in lags if l is not None)
        out(f"\n{'=' * 64}\nSummary\n{'=' * 64}")
        if not reacted:
            out(
                f"0/{len(impulses)} impulses got any OKX reaction within {lag_horizon_sec:.0f}s — either OKX "
                f"didn't move at all, or it reacted too fast/too small to separate from noise at this "
                f"threshold. Inconclusive either way — try a lower --react-threshold-pct or a longer "
                f"--lag-horizon-sec before concluding there's nothing here."
            )
            return

        median = reacted[len(reacted) // 2]
        p75 = _lag_percentile(reacted, 0.75)
        tail_threshold = max(poll_interval_sec * 3, 2.0)
        tail_n = sum(1 for l in reacted if l >= tail_threshold)
        out(
            f"{len(reacted)}/{len(impulses)} impulses got an OKX reaction within {lag_horizon_sec:.0f}s.\n"
            f"lag (seconds):  min={min(reacted):.1f}  median={median:.1f}  p75={p75:.1f}  "
            f"mean={sum(reacted) / len(reacted):.1f}  max={max(reacted):.1f}"
        )
        if median < poll_interval_sec * 1.5 and p75 < poll_interval_sec * 1.5:
            out(
                "\n-> OKX reacts about as fast as our own polling resolution can even distinguish — no "
                "usable lag visible at this measurement granularity. Consistent with OKX spot being just "
                "as fast/liquid as Binance (expected for a top-tier BTC/USDT pair — real arbitrageurs keep "
                "them in sync to milliseconds, well below what REST polling can see)."
            )
        elif tail_n == 0:
            out(
                f"\n-> median/p75 are elevated but no individual reaction reached the "
                f"{tail_threshold:.1f}s tail cutoff — likely polling-interval noise pushing the stats "
                f"around with a small sample, not a real lag. Rerun with a longer --duration-sec before "
                f"drawing a conclusion either way."
            )
        else:
            out(
                f"\n-> mixed picture: median={median:.1f}s looks fast, but {tail_n}/{len(reacted)} "
                f"({tail_n / len(reacted) * 100:.0f}%) reactions took >= {tail_threshold:.1f}s — a real, "
                f"if inconsistent, tail of slow reactions rather than a fast, uniform one. Don't read the "
                f"median alone as \"no lag\" here. Caveats before building anything on this: (1) this "
                f"measured raw SPOT price only, NOT the event contract's own up_price (a separate, "
                f"nonlinear quantity — measure that specifically with --check-leadlag-internal); (2) a "
                f"handful of tail cases isn't a lot of samples — rerun with a longer --duration-sec / "
                f"during more volatile periods to see if the tail rate holds up; (3) even a real occasional "
                f"lag may not survive execution latency + the slippage measured with --check-liquidity, and "
                f"since it doesn't happen every time you'd need to detect it live (impulse + no reaction "
                f"yet) rather than assume it."
            )
    finally:
        if report_lines:
            report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            print(f"\n(report saved to {report_path})")


def _measure_upprice_reaction_lag(
    impulses: list[dict], reactor_points: list[tuple[float, float, str]],
    horizon_sec: float, react_threshold_abs: float,
) -> list[dict]:
    """Like _measure_reaction_lag, but for the event contract's own
    up_price against impulses detected on a SEPARATE spot series. Two
    differences that matter enough to need a dedicated function rather than
    reusing _measure_reaction_lag directly:

    1. up_price is a bounded probability [0, 1], not a raw price — a
       percentage move means something very different near 0.02 than near
       0.50, so the reaction threshold here (react_threshold_abs) is an
       ABSOLUTE probability-point difference, not a percentage.

    2. The event contract itself expires and rolls to a brand new instId
       every window (see market_data.py) — a new instId is a different
       strike/expiry, so its up_price is simply not the same quantity as
       the old instId's. reactor_points carries (ts, up_price, inst_id) so
       any impulse whose measurement span (its own start .. end of its
       reaction horizon) crosses a rollover gets EXCLUDED from the result
       rather than silently mis-reported as "no reaction" — the two mean
       very different things and conflating them would make the summary
       stats meaningless.

    Returns one dict per impulse: {"lag": float|None, "excluded": bool,
    "reason": str}. lag mirrors _measure_reaction_lag's convention (0.0 =
    already reacted by impulse end, a positive float = seconds later, None
    = no usable reaction found)."""
    results: list[dict] = []
    for imp in impulses:
        start_ts, end_ts, direction = imp["start_ts"], imp["end_ts"], imp["direction"]

        pre_price = pre_inst = None
        base_price = base_inst = None
        for ts, price, inst_id in reactor_points:
            if ts <= start_ts:
                pre_price, pre_inst = price, inst_id
            if ts <= end_ts:
                base_price, base_inst = price, inst_id
            else:
                break

        if base_price is None:
            results.append({"lag": None, "excluded": False, "reason": "no up_price data at impulse time"})
            continue

        if pre_inst is not None and base_inst is not None and pre_inst != base_inst:
            results.append({
                "lag": None, "excluded": True,
                "reason": "contract rolled over between impulse start and end",
            })
            continue

        if pre_price is not None:
            already = base_price - pre_price
            if (direction == "up" and already >= react_threshold_abs) or (
                direction == "down" and already <= -react_threshold_abs
            ):
                results.append({"lag": 0.0, "excluded": False, "reason": ""})
                continue

        found = None
        rolled = False
        for ts, price, inst_id in reactor_points:
            if ts <= end_ts:
                continue
            if ts - end_ts > horizon_sec:
                break
            if inst_id != base_inst:
                rolled = True
                break
            move = price - base_price
            if (direction == "up" and move >= react_threshold_abs) or (
                direction == "down" and move <= -react_threshold_abs
            ):
                found = ts - end_ts
                break

        if found is not None:
            results.append({"lag": found, "excluded": False, "reason": ""})
        elif rolled:
            results.append({
                "lag": None, "excluded": True,
                "reason": "contract rolled over during the reaction-measurement horizon",
            })
        else:
            results.append({"lag": None, "excluded": False, "reason": f"no reaction within {horizon_sec:.0f}s"})
    return results


async def check_internal_leadlag(
    cfg, duration_sec: float = 300.0, poll_interval_sec: float = 1.0,
    impulse_threshold_pct: float = 0.03, window_sec: float = 10.0,
    lag_horizon_sec: float = 20.0, upprice_react_threshold: float = 0.02,
    series_id: Optional[str] = None, resolve_interval_sec: float = 15.0,
) -> None:
    """Utility mode: measure whether OKX's own event-contract up_price lags
    OKX's own spot BTC-USDT ticker — both on OKX, no Binance involved. The
    natural follow-up to --check-leadlag: even if OKX spot itself keeps up
    with Binance just fine, the CONTRACT's up_price is a separate, less
    liquid, derived quantity (probability, not price) — its own market
    makers could still re-quote slower than spot moves, which would be a
    cleaner edge than cross-exchange lag since there's no extra network hop
    to a second exchange involved in exploiting it.

    Unlike --check-leadlag this needs OKX API keys: finding the currently
    live instId requires the signed get_event_markets call (see
    discover_series) even though the ticker polls themselves are public.

    Method: poll OKX spot + the live event contract's own ticker every
    poll_interval_sec, detect impulses on the SPOT series (same detector as
    --check-leadlag), then measure how long up_price took to move by
    >= upprice_react_threshold probability points in the matching
    direction. Any impulse whose measurement window spans a contract
    rollover (a new instId — different strike/expiry, not a comparable
    quantity) is EXCLUDED from the stats rather than mis-reported as "no
    reaction" — see _measure_upprice_reaction_lag.
    """
    if not (cfg.okx.api_key and cfg.okx.api_secret and cfg.okx.api_passphrase):
        print(
            "--check-leadlag-internal needs OKX Demo Trading API keys (event-contract "
            "endpoints require authentication even for read-only browsing, to resolve the "
            "live instId). Copy .env.example to .env and fill in OKX_API_KEY / "
            "OKX_API_SECRET / OKX_API_PASSPHRASE first.",
            file=sys.stderr,
        )
        sys.exit(1)

    sid = series_id or (cfg.okx.series_ids[0] if cfg.okx.series_ids else None)
    if not sid:
        print("No --series-id given and config.yaml has no okx.series_ids configured.", file=sys.stderr)
        sys.exit(1)

    client_cfg = OKXClientConfig(
        base_url=cfg.okx.base_url, api_key=cfg.okx.api_key, api_secret=cfg.okx.api_secret,
        api_passphrase=cfg.okx.api_passphrase, demo_trading=cfg.okx.demo_trading,
        timeout_sec=cfg.okx.request_timeout_sec, max_retries=cfg.okx.max_retries,
    )

    data_dir = Path(cfg.storage.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    samples_path = data_dir / f"leadlag_internal_samples_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    print(
        f"Collecting {duration_sec:.0f}s of OKX spot BTC-USDT ({cfg.okx.underlying_inst_id}) vs series "
        f"'{sid}''s own live event-contract up_price, polling every {poll_interval_sec:.1f}s.\n"
        f"Raw samples are written incrementally to {samples_path} as they come in — if this "
        f"session gets disconnected partway through, that file still has everything collected "
        f"up to the disconnect.\n"
    )

    def _exp_ms(m: dict) -> float:
        raw = m.get("expTime")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float("inf")

    async def _resolve_inst_id(client: OKXClient) -> Optional[str]:
        try:
            markets = await client.get_event_markets(series_id=sid, state="live")
        except Exception:
            return None
        if not markets:
            return None
        chosen = min(markets, key=_exp_ms)
        inst = chosen.get("instId")
        return str(inst) if inst else None

    spot_series: list[tuple[float, float]] = []
    up_series: list[tuple[float, float, str]] = []

    with open(samples_path, "w", newline="") as samples_file:
        samples_writer = csv.DictWriter(samples_file, fieldnames=["ts", "spot_price", "up_price", "inst_id"])
        samples_writer.writeheader()

        async with OKXClient(client_cfg) as client:
            inst_id = await _resolve_inst_id(client)
            if not inst_id:
                print(
                    f"No live instrument found for series '{sid}' right now — try again later "
                    f"or check config.yaml -> okx.series_ids.", file=sys.stderr,
                )
                return
            print(f"Tracking live instrument: {inst_id}\n")

            end_at = time.time() + duration_sec
            last_resolve = time.time()
            n = 0
            while time.time() < end_at:
                tick_start = time.time()

                if tick_start - last_resolve >= resolve_interval_sec:
                    new_inst_id = await _resolve_inst_id(client)
                    last_resolve = tick_start
                    if new_inst_id and new_inst_id != inst_id:
                        print(f"  ...instrument rolled over: {inst_id} -> {new_inst_id}")
                        inst_id = new_inst_id

                spot_price, up_price = await asyncio.gather(
                    _fetch_okx_spot_price(client, cfg.okx.underlying_inst_id),
                    _fetch_okx_spot_price(client, inst_id),
                )
                now = time.time()
                if spot_price is not None:
                    spot_series.append((now, spot_price))
                if up_price is not None:
                    up_series.append((now, up_price, inst_id))
                samples_writer.writerow(
                    {"ts": now, "spot_price": spot_price, "up_price": up_price, "inst_id": inst_id}
                )
                samples_file.flush()
                n += 1
                if n % 30 == 0:
                    remaining = max(0.0, end_at - time.time())
                    print(f"  ...{n} samples so far, ~{remaining:.0f}s left (saved to {samples_path.name})")
                elapsed = time.time() - tick_start
                await asyncio.sleep(max(0.0, poll_interval_sec - elapsed))

    print(f"\nCollected {len(spot_series)} spot samples, {len(up_series)} up_price samples "
          f"(raw data in {samples_path}).")
    if len(spot_series) < 10 or len(up_series) < 10:
        print("Not enough data to analyze (too many failed requests?) — check network/API access and retry.")
        return

    report_lines: list[str] = []

    def out(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    report_path = data_dir / f"leadlag_internal_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        impulses = _detect_impulses(spot_series, impulse_threshold_pct, window_sec, cooldown_sec=window_sec)
        out(
            f"\nDetected {len(impulses)} impulses on OKX spot BTC-USDT "
            f"(>= {impulse_threshold_pct:.3f}% move within {window_sec:.0f}s, "
            f"{window_sec:.0f}s cooldown between detections):"
        )
        for imp in impulses:
            t = time.strftime("%H:%M:%S", time.localtime(imp["end_ts"]))
            out(f"  {t}  {imp['direction'].upper():5s}  move={imp['move_pct']:+.3f}%")

        if not impulses:
            out(
                "\nNo impulses detected in this window — try a longer --duration-sec, run during a more "
                "volatile period, or lower --impulse-threshold-pct. Can't measure a lag with no events to "
                "measure it from."
            )
            return

        results = _measure_upprice_reaction_lag(impulses, up_series, lag_horizon_sec, upprice_react_threshold)
        out(
            f"\nEvent contract up_price reaction (>= {upprice_react_threshold:.3f} probability-point "
            f"same-direction move within {lag_horizon_sec:.0f}s of the spot impulse):"
        )
        for imp, res in zip(impulses, results):
            if res["excluded"]:
                status = f"EXCLUDED ({res['reason']})"
            elif res["lag"] is not None:
                status = f"{res['lag']:.1f}s later"
            else:
                status = res["reason"] or f"no reaction within {lag_horizon_sec:.0f}s"
            out(f"  spot {imp['direction'].upper():5s} {imp['move_pct']:+.3f}%  ->  up_price: {status}")

        excluded_n = sum(1 for r in results if r["excluded"])
        usable = [r for r in results if not r["excluded"]]
        reacted = sorted(r["lag"] for r in usable if r["lag"] is not None)

        out(f"\n{'=' * 64}\nSummary\n{'=' * 64}")
        if excluded_n:
            out(
                f"{excluded_n}/{len(impulses)} impulses excluded from the stats below (a contract rollover "
                f"fell inside their measurement window, making before/after up_price not comparable)."
            )
        if not usable:
            out("No usable (non-excluded) impulses left — rerun with a longer --duration-sec.")
            return
        if not reacted:
            out(
                f"0/{len(usable)} usable impulses got any up_price reaction within {lag_horizon_sec:.0f}s — "
                f"either it didn't move at all, or reacted too fast/small to separate from noise at this "
                f"threshold. Try a lower --upprice-react-threshold or a longer --lag-horizon-sec before "
                f"concluding there's nothing here."
            )
            return

        median = reacted[len(reacted) // 2]
        p75 = _lag_percentile(reacted, 0.75)
        tail_threshold = max(poll_interval_sec * 3, 2.0)
        tail_n = sum(1 for l in reacted if l >= tail_threshold)
        out(
            f"{len(reacted)}/{len(usable)} usable impulses got an up_price reaction within "
            f"{lag_horizon_sec:.0f}s.\nlag (seconds):  min={min(reacted):.1f}  median={median:.1f}  "
            f"p75={p75:.1f}  mean={sum(reacted) / len(reacted):.1f}  max={max(reacted):.1f}"
        )
        if median < poll_interval_sec * 1.5 and p75 < poll_interval_sec * 1.5:
            out(
                "\n-> up_price reacts about as fast as our own polling resolution can even distinguish — "
                "no usable internal lag visible at this measurement granularity. Consistent with the "
                "contract's market makers re-quoting essentially in lockstep with spot."
            )
        elif tail_n == 0:
            out(
                f"\n-> median/p75 are elevated but no individual reaction reached the "
                f"{tail_threshold:.1f}s tail cutoff — likely polling-interval noise pushing the stats "
                f"around with a small sample, not a real lag. Rerun with a longer --duration-sec before "
                f"drawing a conclusion either way."
            )
        else:
            out(
                f"\n-> mixed picture, NOT \"no lag\": median={median:.1f}s looks fast (most reactions are "
                f"near-instant), but {tail_n}/{len(reacted)} ({tail_n / len(reacted) * 100:.0f}%) reactions "
                f"took >= {tail_threshold:.1f}s to show up in up_price — a real, if inconsistent, tail of "
                f"stale pricing rather than a uniformly fast one. This is the interesting part: after a spot "
                f"impulse, the side that just became more likely (UP after a spot jump up, DOWN after a "
                f"drop) is SOMETIMES still priced at its OLD, cheaper probability for several seconds — "
                f"buying it in that window would be +EV, IF this tail is real and not just noise from a "
                f"small sample. This is inherently an event-driven signal (react to a detected impulse and "
                f"check whether up_price has caught up yet), not a blanket \"always faster\" edge — most of "
                f"the time there's nothing there. Before building anything on this: (1) rerun with a longer "
                f"--duration-sec to get more tail samples and confirm the {tail_n / len(reacted) * 100:.0f}% "
                f"rate holds up rather than being a fluke of this one run; (2) check whether the tail cases "
                f"cluster around specific conditions (e.g. near contract expiry, or during faster spot moves) "
                f"— that's in the raw CSV (ts, spot_price, up_price, inst_id); (3) this still has to survive "
                f"the slippage measured with --check-liquidity, and by definition you'd only know a reaction "
                f"is 'slow' in hindsight — a live strategy would need to guess it's in the slow case before "
                f"the window closes; (4) rollover-excluded impulses aren't counted here."
            )
    finally:
        if report_lines:
            report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            print(f"\n(report saved to {report_path})")


def _rows_to_series(
    rows: list[dict],
) -> tuple[list[tuple[float, float]], list[tuple[float, float, str]]]:
    """Parse the CSV rows a --check-leadlag-internal run wrote (columns ts,
    spot_price, up_price, inst_id — all strings, any of them possibly empty
    if that particular fetch failed at the time) back into the
    (spot_series, up_series) shapes _detect_impulses and
    _measure_upprice_reaction_lag expect. A row with an unparseable ts is
    dropped entirely (it's not usable for either series); spot_price and
    up_price/inst_id are otherwise independent — a row missing one can
    still contribute the other."""
    spot: list[tuple[float, float]] = []
    up: list[tuple[float, float, str]] = []
    for row in rows:
        try:
            ts = float(row.get("ts"))
        except (TypeError, ValueError):
            continue

        sp = row.get("spot_price")
        if sp not in (None, ""):
            try:
                spot.append((ts, float(sp)))
            except (TypeError, ValueError):
                pass

        upp, inst = row.get("up_price"), row.get("inst_id")
        if upp not in (None, "") and inst:
            try:
                up.append((ts, float(upp), inst))
            except (TypeError, ValueError):
                pass
    return spot, up


def _seconds_since_rollover(
    ts: float, up_series: list[tuple[float, float, str]], inst_id: str,
) -> Optional[float]:
    """How long `inst_id` had already been the live instrument at time
    `ts` — ts minus the first up_series sample we recorded for it at or
    before ts. None if we never saw an earlier sample of this inst_id
    (most commonly: it's the very first instrument in the recording, so we
    genuinely don't know when its own window started)."""
    first_ts = None
    for t, _, inst in up_series:
        if inst != inst_id:
            continue
        if t > ts:
            break
        if first_ts is None:
            first_ts = t
    return (ts - first_ts) if first_ts is not None else None


def _latest_leadlag_csv(cfg) -> Optional[Path]:
    data_dir = Path(cfg.storage.data_dir)
    candidates = sorted(data_dir.glob("leadlag_internal_samples_*.csv"))  # filenames sort chronologically
    return candidates[-1] if candidates else None


def analyze_internal_leadlag(
    csv_path: Path, impulse_threshold_pct: float = 0.03, window_sec: float = 10.0,
    lag_horizon_sec: float = 20.0, upprice_react_threshold: float = 0.02, tail_threshold_sec: float = 3.0,
) -> None:
    """Utility mode: re-analyze an already-collected --check-leadlag-internal
    CSV for WHEN the slow-reaction tail happens, instead of just the
    min/median/p75/mean/max summary --check-leadlag-internal itself prints.
    Pure offline analysis — no network calls, no API keys, re-reads the
    same raw samples file. The question this answers: is the slow tail
    predictable (clusters with bigger impulses, or with how long the
    current contract has been live) or does it look basically random? A
    predictable tail is something a live strategy could plausibly act on;
    a random one means you'd only ever know a reaction was "slow" in
    hindsight, after the window to act on it already closed.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    spot_series, up_series = _rows_to_series(rows)
    print(f"Loaded {csv_path} — {len(spot_series)} spot samples, {len(up_series)} up_price samples.\n")
    if len(spot_series) < 10 or len(up_series) < 10:
        print("Not enough parsable rows to analyze.")
        return

    impulses = _detect_impulses(spot_series, impulse_threshold_pct, window_sec, cooldown_sec=window_sec)
    results = _measure_upprice_reaction_lag(impulses, up_series, lag_horizon_sec, upprice_react_threshold)

    records = []
    for imp, res in zip(impulses, results):
        if res["excluded"] or res["lag"] is None:
            continue
        base_inst = None
        for t, _, inst in up_series:
            if t <= imp["end_ts"]:
                base_inst = inst
            else:
                break
        since_rollover = _seconds_since_rollover(imp["end_ts"], up_series, base_inst) if base_inst else None
        records.append({
            "ts": imp["end_ts"], "direction": imp["direction"],
            "move_pct": abs(imp["move_pct"]), "lag": res["lag"], "since_rollover": since_rollover,
        })

    if not records:
        print("No reacted (non-excluded) impulses in this file to analyze — nothing to break down.")
        return

    def tail_rate(subset: list[dict]) -> str:
        if not subset:
            return "no samples"
        n_tail = sum(1 for r in subset if r["lag"] >= tail_threshold_sec)
        return f"{n_tail}/{len(subset)} ({n_tail / len(subset) * 100:.0f}%) tail (>= {tail_threshold_sec:.1f}s)"

    moves_sorted = sorted(r["move_pct"] for r in records)
    move_median = moves_sorted[len(moves_sorted) // 2]
    print(f"{len(records)} reacted impulses total.\n")
    print(f"By impulse size (median move_pct={move_median:.3f}%):")
    print(f"  smaller moves: {tail_rate([r for r in records if r['move_pct'] < move_median])}")
    print(f"  larger  moves: {tail_rate([r for r in records if r['move_pct'] >= move_median])}")

    since_vals = sorted(r["since_rollover"] for r in records if r["since_rollover"] is not None)
    if since_vals:
        since_median = since_vals[len(since_vals) // 2]
        with_since = [r for r in records if r["since_rollover"] is not None]
        print(f"\nBy time already spent on the current contract (median={since_median:.0f}s; "
              f"{len(records) - len(with_since)} impulse(s) excluded — their contract's own start "
              f"wasn't captured in this recording):")
        print(f"  earlier in the contract's life: "
              f"{tail_rate([r for r in with_since if r['since_rollover'] < since_median])}")
        print(f"  later in the contract's life:   "
              f"{tail_rate([r for r in with_since if r['since_rollover'] >= since_median])}")
    else:
        print("\nNo time-since-rollover data available (every reacted impulse's contract was "
              "already live at the very start of this recording).")

    print(f"\nAll {len(records)} reacted impulses, slowest first (eyeball for clustering by time of day):")
    for r in sorted(records, key=lambda r: -r["lag"]):
        t = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        since_str = f"{r['since_rollover']:.0f}s into its contract" if r["since_rollover"] is not None else "contract start unknown"
        print(f"  {t}  {r['direction'].upper():5s} move={r['move_pct']:.3f}%  lag={r['lag']:5.1f}s  {since_str}")


async def run_bot(cfg) -> None:
    storage = Storage(cfg.storage.data_dir)

    if cfg.mock_mode:
        logger.warning("Running in MOCK_MODE — synthetic data only, no OKX network calls.")
        provider = MockMarketDataProvider(series_ids=cfg.okx.series_ids)
        await _run_with_provider(cfg, provider, storage)
        return

    if not (cfg.okx.api_key and cfg.okx.api_secret and cfg.okx.api_passphrase):
        print(
            "mock_mode is false but OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE are "
            "not set. Copy .env.example to .env and fill in your API keys (Demo Trading "
            "keys, or a live read-only key with OKX_DEMO_TRADING=0 — see README), "
            "or set mock_mode: true in config.yaml to test offline first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if cfg.okx.demo_trading:
        logger.warning("Reading OKX Demo Trading data (x-simulated-trading:1). No orders are ever placed.")
    else:
        logger.warning(
            "OKX_DEMO_TRADING=0: reading LIVE production market data (real quotes/liquidity), "
            "not the demo sandbox. Trades stay 100%s virtual regardless — this bot has no code "
            "path that places a real order (see README 'Event Contracts — реальная схема API').",
            "%",
        )

    client_cfg = OKXClientConfig(
        base_url=cfg.okx.base_url, api_key=cfg.okx.api_key, api_secret=cfg.okx.api_secret,
        api_passphrase=cfg.okx.api_passphrase, demo_trading=cfg.okx.demo_trading,
        timeout_sec=cfg.okx.request_timeout_sec, max_retries=cfg.okx.max_retries,
        retry_backoff_base_sec=cfg.okx.retry_backoff_base_sec,
    )
    async with OKXClient(client_cfg) as client:
        provider = OkxMarketDataProvider(
            client, cfg.okx.underlying_inst_id, cfg.okx.series_ids,
            funding_inst_id=cfg.okx.funding_inst_id,
            funding_refresh_interval_sec=cfg.okx.funding_refresh_interval_sec,
        )
        await _run_with_provider(cfg, provider, storage)


async def _run_with_provider(cfg, provider, storage: Storage) -> None:
    engine = Engine(cfg, provider, storage)
    tasks = [asyncio.create_task(engine.run_forever(), name="engine")]

    if cfg.dashboard.mode in ("console", "both"):
        from src.dashboard_console import run_console_dashboard
        tasks.append(asyncio.create_task(run_console_dashboard(cfg, engine), name="console_dashboard"))

    if cfg.dashboard.mode in ("web", "both"):
        tasks.append(asyncio.create_task(_run_web_dashboard(cfg, engine), name="web_dashboard"))

    if cfg.dashboard.mode == "web":
        print(f"Web dashboard: http://{cfg.dashboard.web_host}:{cfg.dashboard.web_port}")

    # Docker sends SIGTERM (not SIGINT) to stop a container — e.g. every
    # `docker compose up --build` on redeploy — and Python installs no
    # default handler for it, so without this the process would just die
    # mid-tick, skipping the `finally` block below entirely and losing up
    # to snapshot_every_sec seconds of wallet state (closed trades are
    # written every tick regardless, via _maybe_persist — that's why the
    # Analytics tab never lost history even before this fix, only the
    # wallet balances did). Wiring SIGTERM to the same cancellation path
    # SIGINT/Ctrl+C already takes makes a redeploy behave like a clean
    # stop: engine.stop() + one final storage.write_snapshot() before exit.
    loop = asyncio.get_running_loop()

    def _cancel_all(*_args) -> None:
        for t in tasks:
            t.cancel()

    try:
        loop.add_signal_handler(signal.SIGTERM, _cancel_all)
    except (NotImplementedError, RuntimeError):
        pass  # e.g. Windows — no graceful SIGTERM handling there, same as before this change

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        engine.stop()
        storage.write_snapshot(engine.wallets)
        await engine.aclose_strategies()
        storage.close()
        for t in tasks:
            t.cancel()


async def _run_web_dashboard(cfg, engine: Engine) -> None:
    import uvicorn
    from src.dashboard_web import build_app

    app = build_app(cfg, engine)
    server_cfg = uvicorn.Config(
        app, host=cfg.dashboard.web_host, port=cfg.dashboard.web_port, log_level="warning"
    )
    server = uvicorn.Server(server_cfg)
    await server.serve()


def main() -> None:
    parser = argparse.ArgumentParser(description="OKX Event-Contract paper-trading strategy bot")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--discover-series", action="store_true",
        help="list live OKX EVENTS seriesId/instId values and exit (needs API keys or public access)",
    )
    parser.add_argument(
        "--check-liquidity", action="store_true",
        help="print real ticker + a simulated market-order fill (walking the real order "
             "book, no order ever placed) for each series' current live instrument, and "
             "exit — shows how far a real fill would be from the last/mid price the "
             "engine simulates trades at",
    )
    parser.add_argument(
        "--test-stake", type=float, default=20.0, metavar="USD",
        help="order size (USDT) to simulate in --check-liquidity (default: 20, "
             "roughly matching a typical live stake)",
    )
    parser.add_argument(
        "--samples", type=int, default=1, metavar="N",
        help="repeat --check-liquidity N times (default: 1, i.e. a single snapshot) "
             "and print a min/median/mean/max summary at the end, plus save every raw "
             "sample to a CSV under the data dir — use this to see whether slippage is "
             "a one-off or a consistent pattern before changing the engine",
    )
    parser.add_argument(
        "--interval-sec", type=float, default=30.0, metavar="SEC",
        help="seconds to wait between --check-liquidity samples when --samples > 1 "
             "(default: 30)",
    )
    parser.add_argument(
        "--check-ai-prompt", action="store_true",
        help="send ONE test call through the ai_prompt (strategy G) LLM client with a "
             "synthetic market context, print what came back, and exit — fast way to check "
             "'is this even configured right' without waiting on live entry-window "
             "checkpoints or the call cooldown",
    )
    parser.add_argument(
        "--check-leadlag", action="store_true",
        help="measure whether OKX's own spot BTC-USDT ticker lags Binance's — no API keys "
             "needed (both public tickers). Prints detected Binance impulses and how long OKX "
             "took to react to each, then a summary; exits after --duration-sec",
    )
    parser.add_argument(
        "--duration-sec", type=float, default=300.0, metavar="SEC",
        help="how long to collect data for --check-leadlag (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--poll-interval-sec", type=float, default=1.0, metavar="SEC",
        help="polling interval for --check-leadlag (default: 1.0 — this is the measurement's "
             "own time resolution, a detected lag shorter than this isn't distinguishable from noise)",
    )
    parser.add_argument(
        "--impulse-threshold-pct", type=float, default=0.03, metavar="PCT",
        help="minimum %% move within --window-sec on Binance to count as an impulse worth "
             "measuring a reaction to, for --check-leadlag (default: 0.03)",
    )
    parser.add_argument(
        "--window-sec", type=float, default=10.0, metavar="SEC",
        help="trailing window --check-leadlag looks for an impulse within, and the cooldown "
             "before it looks for the next one (default: 10)",
    )
    parser.add_argument(
        "--lag-horizon-sec", type=float, default=20.0, metavar="SEC",
        help="how long after a Binance impulse --check-leadlag keeps watching OKX for a "
             "reaction before giving up on that impulse (default: 20)",
    )
    parser.add_argument(
        "--check-leadlag-internal", action="store_true",
        help="measure whether OKX's own event-contract up_price lags OKX's own spot "
             "BTC-USDT ticker (both on OKX — the follow-up question to --check-leadlag). "
             "Needs OKX API keys, unlike --check-leadlag, to resolve the live instId",
    )
    parser.add_argument(
        "--series-id", metavar="SERIES",
        help="override which okx.series_ids entry --check-leadlag-internal tracks "
             "(default: the first one in config.yaml)",
    )
    parser.add_argument(
        "--upprice-react-threshold", type=float, default=0.02, metavar="PROB",
        help="minimum absolute probability-point move in up_price to count as a reaction "
             "for --check-leadlag-internal (default: 0.02, i.e. 2 probability points — "
             "up_price is bounded [0,1] so this is absolute, not a percentage)",
    )
    parser.add_argument(
        "--resolve-interval-sec", type=float, default=15.0, metavar="SEC",
        help="how often --check-leadlag-internal re-checks which instId is live (default: "
             "15 — this is the heavier signed get_event_markets call, so it's not re-checked "
             "every poll tick like the ticker fetches are)",
    )
    parser.add_argument(
        "--analyze-leadlag-internal", action="store_true",
        help="re-analyze an already-collected --check-leadlag-internal CSV for WHEN the "
             "slow-reaction tail happens (impulse size, time since the contract's own "
             "rollover) instead of just the summary numbers — pure offline analysis, no "
             "network calls, no API keys needed",
    )
    parser.add_argument(
        "--leadlag-csv", metavar="PATH",
        help="which leadlag_internal_samples_*.csv to analyze for --analyze-leadlag-internal "
             "(default: the newest one in the data dir)",
    )
    parser.add_argument(
        "--tail-threshold-sec", type=float, default=3.0, metavar="SEC",
        help="lag (seconds) at/above which a reaction counts as 'tail' rather than fast, for "
             "--analyze-leadlag-internal (default: 3.0, matching the live diagnostics' default)",
    )
    parser.add_argument(
        "--reset-data", action="store_true",
        help="wipe data/bot.db (all strategies), then exit "
             "(equivalent to the web dashboard's Reset DB button, for console-mode users)",
    )
    parser.add_argument(
        "--reset-strategy", metavar="NAME",
        help="wipe data/bot.db rows for ONE strategy only, then exit "
             "(equivalent to the Settings tab's per-strategy Reset button)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # --check-ai-prompt wants the ai_prompt logger's own "raw response"/
    # warning lines visible on the console (normally file-only, WARNING+
    # on console, so the console dashboard isn't spammed) — everyone else
    # keeps the quiet default.
    console_level = logging.INFO if args.check_ai_prompt else logging.WARNING
    setup_logging(cfg.storage.data_dir, console_level=console_level)

    if args.discover_series:
        asyncio.run(discover_series(cfg))
        return

    if args.check_liquidity:
        asyncio.run(check_liquidity(
            cfg, test_stake_usd=args.test_stake, samples=args.samples, interval_sec=args.interval_sec,
        ))
        return

    if args.check_ai_prompt:
        asyncio.run(check_ai_prompt(cfg))
        return

    if args.check_leadlag:
        asyncio.run(check_leadlag(
            cfg, duration_sec=args.duration_sec, poll_interval_sec=args.poll_interval_sec,
            impulse_threshold_pct=args.impulse_threshold_pct, window_sec=args.window_sec,
            lag_horizon_sec=args.lag_horizon_sec,
        ))
        return

    if args.check_leadlag_internal:
        asyncio.run(check_internal_leadlag(
            cfg, duration_sec=args.duration_sec, poll_interval_sec=args.poll_interval_sec,
            impulse_threshold_pct=args.impulse_threshold_pct, window_sec=args.window_sec,
            lag_horizon_sec=args.lag_horizon_sec, upprice_react_threshold=args.upprice_react_threshold,
            series_id=args.series_id, resolve_interval_sec=args.resolve_interval_sec,
        ))
        return

    if args.analyze_leadlag_internal:
        csv_path = Path(args.leadlag_csv) if args.leadlag_csv else _latest_leadlag_csv(cfg)
        if not csv_path or not csv_path.exists():
            print(
                "No leadlag_internal_samples_*.csv found in the data dir — run "
                "--check-leadlag-internal first, or pass --leadlag-csv PATH.", file=sys.stderr,
            )
            sys.exit(1)
        analyze_internal_leadlag(
            csv_path, impulse_threshold_pct=args.impulse_threshold_pct, window_sec=args.window_sec,
            lag_horizon_sec=args.lag_horizon_sec, upprice_react_threshold=args.upprice_react_threshold,
            tail_threshold_sec=args.tail_threshold_sec,
        )
        return

    if args.reset_data:
        storage = Storage(cfg.storage.data_dir)
        storage.reset()
        storage.close()
        print("Data reset: all strategies wiped from data/bot.db.")
        return

    if args.reset_strategy:
        known = {s.name for s in cfg.strategies}
        if args.reset_strategy not in known:
            print(f"Unknown strategy '{args.reset_strategy}'. Known: {', '.join(sorted(known))}", file=sys.stderr)
            sys.exit(1)
        storage = Storage(cfg.storage.data_dir)
        storage.reset_strategy(args.reset_strategy)
        storage.close()
        print(f"Data reset for strategy '{args.reset_strategy}' only in data/bot.db.")
        return

    try:
        asyncio.run(run_bot(cfg))
    except KeyboardInterrupt:
        print("\nStopped by user. Final wallet/trade state saved to data/bot.db.")


if __name__ == "__main__":
    main()
