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
import logging
import sys

from src.config import load_config
from src.engine import Engine
from src.logger import setup_logging
from src.market_data import OkxMarketDataProvider
from src.mock_market import MockMarketDataProvider
from src.okx_client import OKXClient, OKXClientConfig
from src.storage import Storage

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
    setup_logging(cfg.storage.data_dir, console_level=logging.WARNING)

    if args.discover_series:
        asyncio.run(discover_series(cfg))
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
