"""
Configuration loading.

Merges config.yaml (strategy/behaviour knobs) with .env (secrets). Nothing in
this module talks to the network — it only produces typed, validated config
objects that the rest of the bot consumes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class OkxConfig:
    api_key: str
    api_secret: str
    api_passphrase: str
    base_url: str
    demo_trading: bool
    underlying_inst_id: str
    series_ids: list[str]
    poll_interval_sec: float
    request_timeout_sec: float
    max_retries: int
    retry_backoff_base_sec: float
    settlement_poll_attempts: int
    settlement_poll_interval_sec: float
    funding_inst_id: str
    funding_refresh_interval_sec: float


@dataclass
class StrategyConfig:
    name: str
    display_name: str
    enabled: bool
    deposit_usd: float                # this strategy's OWN independent virtual capital
    entry_windows_min: list[int]
    max_coefficient: float
    stake_fraction: float
    # Reject a signal if the honest book-simulated fill price (see
    # EventMarket.fill_price_for) is more than this many PERCENT worse
    # than the naive quoted price (up_price/1-up_price) — the paper-
    # trading analog of the "max slippage" guard a real OKX order lets you
    # set before it refuses to fill. None (default) = no limit, i.e. the
    # OLD behavior (only max_coefficient's absolute ceiling applies).
    # Measured on live data (see README "Ограничения"): median slippage
    # alone is already +66-109%, so a strict value here will reject a LOT
    # of signals — pick deliberately, this isn't a small tuning knob.
    max_slippage_pct: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardConfig:
    mode: str
    refresh_sec: float
    web_host: str
    web_port: int


@dataclass
class StorageConfig:
    data_dir: Path
    snapshot_every_sec: float


@dataclass
class AppConfig:
    mock_mode: bool
    okx: OkxConfig
    strategies: list[StrategyConfig]
    dashboard: DashboardConfig
    storage: StorageConfig

    def enabled_strategy_names(self) -> list[str]:
        return [s.name for s in self.strategies if s.enabled]


_KNOWN_STRATEGY_FIELDS = {
    "enabled",
    "display_name",
    "deposit_usd",
    "entry_windows_min",
    "max_coefficient",
    "stake_fraction",
    "max_slippage_pct",
}

DEFAULT_DEPOSIT_USD = 100.0


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    load_dotenv(ROOT_DIR / ".env")

    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT_DIR / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    okx_raw = raw.get("okx", {})
    okx = OkxConfig(
        api_key=os.getenv("OKX_API_KEY", ""),
        api_secret=os.getenv("OKX_API_SECRET", ""),
        api_passphrase=os.getenv("OKX_API_PASSPHRASE", ""),
        base_url=os.getenv("OKX_BASE_URL", "https://www.okx.com"),
        demo_trading=os.getenv("OKX_DEMO_TRADING", "1") == "1",
        underlying_inst_id=okx_raw.get("underlying_inst_id", "BTC-USDT"),
        series_ids=list(okx_raw.get("series_ids", [])),
        poll_interval_sec=float(okx_raw.get("poll_interval_sec", 3)),
        request_timeout_sec=float(okx_raw.get("request_timeout_sec", 10)),
        max_retries=int(okx_raw.get("max_retries", 5)),
        retry_backoff_base_sec=float(okx_raw.get("retry_backoff_base_sec", 1.5)),
        settlement_poll_attempts=int(okx_raw.get("settlement_poll_attempts", 10)),
        settlement_poll_interval_sec=float(okx_raw.get("settlement_poll_interval_sec", 2)),
        funding_inst_id=okx_raw.get("funding_inst_id", "BTC-USDT-SWAP"),
        funding_refresh_interval_sec=float(okx_raw.get("funding_refresh_interval_sec", 300)),
    )

    strategies: list[StrategyConfig] = []
    for name, s_raw in (raw.get("strategies") or {}).items():
        extra = {k: v for k, v in s_raw.items() if k not in _KNOWN_STRATEGY_FIELDS}
        strategies.append(
            StrategyConfig(
                name=name,
                display_name=s_raw.get("display_name", name),
                enabled=bool(s_raw.get("enabled", True)),
                deposit_usd=float(s_raw.get("deposit_usd", DEFAULT_DEPOSIT_USD)),
                entry_windows_min=list(s_raw.get("entry_windows_min", [12, 7, 2])),
                max_coefficient=float(s_raw.get("max_coefficient", 0.55)),
                stake_fraction=float(s_raw.get("stake_fraction", 0.08)),
                max_slippage_pct=(
                    float(s_raw["max_slippage_pct"]) if s_raw.get("max_slippage_pct") is not None else None
                ),
                extra=extra,
            )
        )

    dash_raw = raw.get("dashboard", {})
    dashboard = DashboardConfig(
        mode=dash_raw.get("mode", "console"),
        refresh_sec=float(dash_raw.get("refresh_sec", 2)),
        # DASHBOARD_WEB_HOST env var overrides config.yaml — lets a container
        # bind 0.0.0.0 (reachable via `-p 8000:8000`) without changing the
        # safe 127.0.0.1-only default for a plain local run.
        web_host=os.getenv("DASHBOARD_WEB_HOST", dash_raw.get("web_host", "127.0.0.1")),
        web_port=int(dash_raw.get("web_port", 8000)),
    )

    store_raw = raw.get("storage", {})
    data_dir = ROOT_DIR / store_raw.get("data_dir", "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    storage = StorageConfig(
        data_dir=data_dir,
        snapshot_every_sec=float(store_raw.get("snapshot_every_sec", 15)),
    )

    return AppConfig(
        mock_mode=bool(raw.get("mock_mode", True)),
        okx=okx,
        strategies=strategies,
        dashboard=dashboard,
        storage=storage,
    )
