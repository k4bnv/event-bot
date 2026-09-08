"""Live ASCII dashboard (rich). Read-only view onto Engine.snapshot()."""
from __future__ import annotations

import asyncio
import datetime as dt

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import AppConfig
from .engine import Engine, EngineSnapshot


def _fmt_money(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}${v:,.2f}"


def _pnl_style(v: float) -> str:
    return "bold green" if v > 0 else ("bold red" if v < 0 else "dim")


def _wallet_table(cfg: AppConfig, snapshot: EngineSnapshot) -> Table:
    table = Table(title="Виртуальные кошельки стратегий (у каждой свой депозит)", expand=True)
    table.add_column("Стратегия")
    table.add_column("Депозит", justify="right")
    table.add_column("Баланс", justify="right")
    table.add_column("В сделках", justify="right")
    table.add_column("Equity", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("Открытых сделок", justify="right")

    for s_cfg in cfg.strategies:
        if not s_cfg.enabled:
            continue
        w = snapshot.wallets.get(s_cfg.name)
        if w is None:
            continue
        table.add_row(
            s_cfg.display_name,
            f"${w.initial_balance:,.2f}",
            f"${w.balance:,.2f}",
            f"${w.reserved:,.2f}",
            f"${w.equity:,.2f}",
            Text(_fmt_money(w.net_pnl), style=_pnl_style(w.net_pnl)),
            str(len(w.open_trades())),
        )
    return table


def _combo_table(cfg: AppConfig, snapshot: EngineSnapshot) -> Table:
    table = Table(title="Стратегия и время входа (детальная статистика)", expand=True)
    table.add_column("Стратегия")
    table.add_column("Окно, мин", justify="right")
    table.add_column("Сделок", justify="right")
    table.add_column("Winrate", justify="right")
    table.add_column("Ср. коэфф.", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("ROI", justify="right")

    display_names = {s.name: s.display_name for s in cfg.strategies}
    rows = sorted(snapshot.combo_stats.values(), key=lambda c: (c.strategy, -c.window_min))
    for c in rows:
        table.add_row(
            display_names.get(c.strategy, c.strategy),
            str(c.window_min),
            f"{c.trades} ({c.wins}W/{c.losses}L)",
            f"{c.winrate_pct:.1f}%",
            f"${c.avg_entry_price:.3f}",
            Text(_fmt_money(c.net_pnl), style=_pnl_style(c.net_pnl)),
            Text(f"{c.roi_pct:+.1f}%", style=_pnl_style(c.roi_pct)),
        )
    if not rows:
        table.add_row("—", "—", "—", "—", "—", "—", "—")
    return table


def _open_trades_table(snapshot: EngineSnapshot) -> Table:
    table = Table(title="Активные сделки", expand=True)
    table.add_column("Стратегия")
    table.add_column("Инструмент")
    table.add_column("Направление")
    table.add_column("Вход")
    table.add_column("Стейк")
    table.add_column("До экспирации")

    now = dt.datetime.now().timestamp()
    any_open = False
    for w in snapshot.wallets.values():
        for t in w.open_trades():
            any_open = True
            remaining = max(0, t.expiry_ts - now)
            table.add_row(
                t.strategy, t.inst_id, t.direction.value.upper(),
                f"${t.entry_price:.3f}", f"${t.stake_usd:.2f}", f"{remaining:.0f}s",
            )
    if not any_open:
        table.add_row("—", "—", "—", "—", "—", "—")
    return table


def _leaderboard_panel(cfg: AppConfig, snapshot: EngineSnapshot) -> Panel:
    lb = snapshot.leaderboard
    display_names = {s.name: s.display_name for s in cfg.strategies}

    def line(label: str, c) -> str:
        if c is None:
            return f"{label}: недостаточно данных"
        name = display_names.get(c.strategy, c.strategy)
        return (
            f"{label}: [bold]{name}[/bold] @ {c.window_min} мин  "
            f"— winrate {c.winrate_pct:.1f}%, ср.коэфф ${c.avg_entry_price:.3f}, "
            f"PnL {_fmt_money(c.net_pnl)} ({c.trades} сделок)"
        )

    body = Text.from_markup(
        "\n".join(
            [
                line("[BEST PnL] Лучшая связка по PnL", lb.best_pnl),
                line("[BEST WINRATE] Лучший Winrate", lb.best_winrate),
                line("[BEST ROI] Лучшая ROI/ценность", lb.best_value),
                f"\n(мин. выборка для 'лучшая связка': {lb.min_sample_size} сделок; "
                "иначе показан лидер по всем данным)",
            ]
        )
    )
    return Panel(body, title="Лидерборд", border_style="yellow")


def build_renderable(cfg: AppConfig, snapshot: EngineSnapshot):
    price_txt = f"{snapshot.underlying_price:,.2f}" if snapshot.underlying_price else "—"
    header = Text.from_markup(
        f"[bold]OKX Event-Contract Paper Bot[/bold]   "
        f"BTC/USDT: [cyan]{price_txt}[/cyan]   "
        f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   "
        f"[{'MOCK DATA' if cfg.mock_mode else 'OKX DEMO TRADING'}]"
    )
    return Group(
        header,
        _wallet_table(cfg, snapshot),
        _combo_table(cfg, snapshot),
        _open_trades_table(snapshot),
        _leaderboard_panel(cfg, snapshot),
    )


async def run_console_dashboard(cfg: AppConfig, engine: Engine) -> None:
    console = Console()
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            snapshot = engine.snapshot()
            live.update(build_renderable(cfg, snapshot))
            await asyncio.sleep(cfg.dashboard.refresh_sec)
