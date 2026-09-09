#!/usr/bin/env python3
"""
Entry point.

    python run.py                     # run with config.yaml (mock or live per mock_mode)
    python run.py --config other.yaml # use a different config file
    python run.py --discover-series   # print live OKX EVENTS series/instruments and exit
                                       # (helps you fill in okx.series_ids in config.yaml)

Ctrl+C stops cleanly: the engine finishes its current tick, writes a final
state snapshot to data/state_snapshot.json, and exits.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

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
        print("\nStopped by user. Final state saved to data/state_snapshot.json and data/trades.csv.")


if __name__ == "__main__":
    main()
